from typing import Union, Optional
from abc import ABC, abstractmethod
from pydantic import BaseModel, Field
import warnings


class UsageData(BaseModel):
    """Standard usage data structure for all agent types."""

    requests: int = Field(ge=0, description="Total number of API requests made")
    request_tokens: int = Field(ge=0, description="Total number of tokens in the input/prompts")
    response_tokens: int = Field(ge=0, description="Total number of tokens in the model's responses")
    thoughts_tokens: int = Field(ge=0, description="Total number of tokens used for internal reasoning")
    total_tokens: int = Field(ge=0, description="Total number of tokens used (request + response + thoughts)")


class CostConfig(BaseModel):
    """Cost configuration for AI models."""

    input_ppm_cost: float = Field(ge=0, description="Cost per million input tokens")
    output_ppm_cost: float = Field(ge=0, description="Cost per million output tokens")
    thought_ppm_cost: float = Field(ge=0, description="Cost per million thought/reasoning tokens")


# Centralized cost registry - single source of truth for all model pricing
MODEL_COSTS = {
    # OpenAI models
    "gpt-5": CostConfig(input_ppm_cost=1.25, output_ppm_cost=10.0, thought_ppm_cost=10.0),
    "gpt-5-chat-latest": CostConfig(input_ppm_cost=1.0, output_ppm_cost=5.0, thought_ppm_cost=5.0),
    "gpt-4o": CostConfig(input_ppm_cost=2.5, output_ppm_cost=10.0, thought_ppm_cost=10.0),
    "gpt-4o-mini": CostConfig(input_ppm_cost=0.15, output_ppm_cost=0.6, thought_ppm_cost=0.6),
    "gpt-4": CostConfig(input_ppm_cost=30.0, output_ppm_cost=60.0, thought_ppm_cost=60.0),
    "gpt-4-turbo": CostConfig(input_ppm_cost=10.0, output_ppm_cost=30.0, thought_ppm_cost=30.0),
    "gpt-3.5-turbo": CostConfig(input_ppm_cost=0.5, output_ppm_cost=1.5, thought_ppm_cost=1.5),
    
    # Anthropic models
    "claude-3-5-sonnet-latest": CostConfig(input_ppm_cost=3.0, output_ppm_cost=15.0, thought_ppm_cost=15.0),
    "claude-3-5-haiku-latest": CostConfig(input_ppm_cost=1.0, output_ppm_cost=5.0, thought_ppm_cost=5.0),
    "claude-3-opus-latest": CostConfig(input_ppm_cost=15.0, output_ppm_cost=75.0, thought_ppm_cost=75.0),
    "claude-3-7-sonnet-latest": CostConfig(input_ppm_cost=3.0, output_ppm_cost=15.0, thought_ppm_cost=15.0),
    
    # Gemini models
    "google-vertex:gemini-2.5-flash": CostConfig(input_ppm_cost=0.30, output_ppm_cost=2.50, thought_ppm_cost=2.50),
    "google-vertex:gemini-2.5-pro": CostConfig(input_ppm_cost=1.25, output_ppm_cost=10.0, thought_ppm_cost=10.0),
}


class EZAgent:
    """
    Simplified AI agent wrapper with automatic model selection and unified API.

    EZAgent provides a simple interface that automatically detects the appropriate AI provider
    based on the model name and creates the underlying pydantic-ai agent with default settings.

    Parameters:
        instructions (str): The instructions/system prompt to use for the agent.
        model_name (str | None): The name of the model to use. If None, defaults to 
            'google-vertex:gemini-2.5-flash'. Supported models include:
            - OpenAI: gpt-4o, gpt-4o-mini, gpt-4, gpt-4-turbo, gpt-3.5-turbo, etc.
            - Anthropic: claude-3-5-sonnet-latest, claude-3-5-haiku-latest, etc.
            - Gemini: google-vertex:gemini-2.5-flash, google-vertex:gemini-2.5-pro, etc.
        retries (int): The number of retries to attempt if agent calls fail. Defaults to 1.
        validate_model_name (bool): Whether to validate the model name against allowed 
            models. Defaults to True.
        max_tokens (int | None): Maximum number of tokens to generate. Defaults to None 
            (no limit). Supported by all agent types (OpenAI, Anthropic, Gemini).
        system_prompt (str | None): DEPRECATED. Use 'instructions' instead.

    Examples:
        # Basic usage - automatically selects correct provider
        agent = EZAgent("You are a helpful assistant", model_name="gpt-4o", max_tokens=100)
        agent = EZAgent("You are helpful", model_name="claude-3-5-sonnet-latest", max_tokens=200)
        agent = EZAgent("You are helpful", max_tokens=300)  # Uses default Gemini model

    Attributes:
        model_name (str): The name of the model being used
        cost_config (CostConfig | None): Cost configuration for the model, automatically 
            set from MODEL_COSTS registry. None if model not found in registry.
        agent: The underlying pydantic-ai Agent instance

    Note:
        This class automatically handles model-to-provider mapping and uses default
        settings optimized for each provider. Cost configuration is pre-computed during
        initialization for efficient cost calculations.
    """

    @classmethod
    def list_models(cls) -> list[str]:
        """
        Return a list of all supported model names.

        Returns:
            Sorted list of all model names supported across all providers
        """
        all_models = []
        for provider_config in PROVIDER_REGISTRY.values():
            all_models.extend(provider_config.models)
        return sorted(list(set(all_models)))

    def __init__(
        self, 
        instructions: str | None = None,
        model_name: str | None = None, 
        retries: int = 1, 
        validate_model_name: bool = True, 
        max_tokens: int | None = None,
        system_prompt: str | None = None,
        **kwargs
    ) -> None:
        """
        Initialize an EZAgent with the specified instructions, model, and retry settings.

        Args:
            instructions: The instructions/system prompt to use for the agent.
            model_name: The name of the model to use. Defaults to 'google-vertex:gemini-2.5-flash'.
            retries: The number of retries to attempt if agent calls fail. Defaults to 1.
            validate_model_name: Whether to validate the model name against allowed models. Defaults to True.
            max_tokens: Maximum number of tokens to generate. Defaults to None (no limit).
            system_prompt: DEPRECATED. Use 'instructions' instead. Will be removed in a future version.
            **kwargs: Additional arguments passed to the underlying pydantic-ai Agent constructor.

        Raises:
            ValueError: If the model_name is not supported and validate_model_name is True.
            ValueError: If both instructions and system_prompt are provided, or if neither are provided.
        """
        # Handle instructions vs system_prompt parameter logic
        if instructions is not None and system_prompt is not None:
            raise ValueError("Cannot specify both 'instructions' and 'system_prompt'. Use 'instructions' only.")
        
        if instructions is None and system_prompt is None:
            raise ValueError("Must specify either 'instructions' or 'system_prompt' parameter.")
        
        if system_prompt is not None:
            warnings.warn(
                "The 'system_prompt' parameter is deprecated and will be removed in a future version. "
                "Use 'instructions' instead.",
                DeprecationWarning,
                stacklevel=2
            )
            resolved_instructions = system_prompt
        else:
            resolved_instructions = instructions
        
        # Set default model if none provided
        if model_name is None:
            model_name = "google-vertex:gemini-2.5-flash"
        
        # Store model name for cost calculations
        self.model_name = model_name
        
        # Store cost config for this model (None if not in registry)
        self.cost_config = MODEL_COSTS.get(model_name)
        
        # Validate model if requested
        if validate_model_name and model_name not in MODEL_COSTS:
            all_models = self.list_models()
            raise ValueError(f"Model '{model_name}' not supported. Available models: {', '.join(all_models)}")
        
        # Create the underlying agent
        if validate_model_name:
            # Use ez_pydantic_agent for validated models
            self.agent = ez_pydantic_agent(
                model_name=model_name,
                instructions=resolved_instructions,
                retries=retries,
                max_tokens=max_tokens,
                **kwargs
            )
        else:
            # For unvalidated models, create agent directly with pydantic-ai
            from pydantic_ai import Agent
            
            # Remove max_tokens from kwargs if present to avoid passing it to Agent constructor
            agent_kwargs = kwargs.copy()
            agent_kwargs.pop('max_tokens', max_tokens)
            
            self.agent = Agent(
                model_name,
                instructions=resolved_instructions,
                retries=retries,
                **agent_kwargs
            )

    async def run(
        self,
        user_prompt: str,
        output_type: Optional["pydantic.main.BaseModel"] = None,  # type: ignore
    ) -> Union["pydantic_ai.agent.AgentRunResult", None]:  # type: ignore
        """
        Run the agent with the given user prompt and expected output type.

        Args:
            user_prompt: The prompt to send to the agent.
            output_type: Optional Pydantic model for structured output.

        Returns:
            The agent run result, or None if the run fails.
        """
        result = await self.agent.run(
            user_prompt,
            output_type=output_type,
        )  # type: ignore
        return result

    def _validate_usage(self, usage_dict: dict) -> dict:
        """
        Validate and standardize usage data using Pydantic.

        Args:
            usage_dict: Dictionary containing usage statistics

        Returns:
            Validated dictionary with standardized structure

        Raises:
            ValidationError: If the usage data doesn't match the expected structure
        """
        validated_usage = UsageData(**usage_dict)
        return validated_usage.model_dump()

    def get_cost(self, *results: "pydantic_ai.agent.AgentRunResult") -> "pd.Series":  # type: ignore
        """
        Calculate cost information from usage results using pre-computed cost configuration.

        Args:
            *results: One or more result objects returned from the agent's run method.

        Returns:
            A pandas Series containing cost breakdown:
            - 'input_cost': Cost for input/request tokens
            - 'output_cost': Cost for output/response tokens
            - 'thoughts_cost': Cost for reasoning/thoughts tokens
            - 'total_cost': Total cost across all token types

        Raises:
            KeyError: If the agent's model cost configuration is not available (cost_config is None)
            AttributeError: If any result object doesn't have a usage() method.
            ValueError: If no results are provided.
        """
        # Get usage data using existing method
        usage = self.get_usage(*results)

        # Basic validation
        if not hasattr(self, 'model_name'):
            raise KeyError("Model 'unknown' not found in MODEL_COSTS registry")
        
        if self.cost_config is None:
            raise KeyError(f"Model '{self.model_name}' not found in MODEL_COSTS registry")
        
        # Use pre-computed cost config directly
        cost_config = self.cost_config

        # Calculate costs (convert from per-million to actual costs)
        input_cost = usage["request_tokens"] * cost_config.input_ppm_cost / 1_000_000
        output_cost = usage["response_tokens"] * cost_config.output_ppm_cost / 1_000_000
        thoughts_cost = usage["thoughts_tokens"] * cost_config.thought_ppm_cost / 1_000_000
        total_cost = input_cost + output_cost + thoughts_cost

        import pandas as pd

        return pd.Series(
            {
                "input_cost": input_cost,
                "output_cost": output_cost,
                "thoughts_cost": thoughts_cost,
                "total_cost": total_cost,
            }
        )


    def _aggregate_usage_data(self, *results: "pydantic_ai.agent.AgentRunResult") -> dict:  # type: ignore
        """
        Common usage data aggregation logic for all agent types.

        Args:
            *results: One or more result objects returned from the agent's run method

        Returns:
            Dictionary with aggregated usage data (before thoughts_tokens extraction)

        Raises:
            AttributeError: If any result object doesn't have a usage() method
            ValueError: If no results are provided
        """
        if not results:
            raise ValueError("At least one result must be provided.")

        try:
            # Initialize totals
            total_requests = 0
            total_request_tokens = 0
            total_response_tokens = 0
            total_tokens = 0

            # Aggregate usage from all results (skip None results from failed calls)
            for result in results:
                if result is not None:
                    usage = result.usage()
                    total_requests += usage.requests
                    total_request_tokens += usage.request_tokens
                    total_response_tokens += usage.response_tokens
                    total_tokens += usage.total_tokens

            return {
                "requests": total_requests,
                "request_tokens": total_request_tokens,
                "response_tokens": total_response_tokens,
                "total_tokens": total_tokens,
            }
        except AttributeError:
            raise AttributeError(
                "One or more result objects do not have a usage() method. Make sure you're passing valid agent run results."
            )

    def get_usage(self, *results: "pydantic_ai.agent.AgentRunResult") -> "pd.Series":  # type: ignore
        """
        Extract usage information from one or more agent run results and return as a pandas Series.

        Args:
            *results: One or more result objects returned from the agent's run method.

        Returns:
            A pandas Series containing aggregated token usage statistics with keys:
            - 'requests': Total number of API requests made
            - 'request_tokens': Total number of tokens in the input/prompts
            - 'response_tokens': Total number of tokens in the model's responses
            - 'thoughts_tokens': Total number of tokens used for internal reasoning
            - 'total_tokens': Total number of tokens used (request + response + thoughts)

        Raises:
            AttributeError: If any result object doesn't have a usage() method.
            ValueError: If no results are provided.
        """
        import pandas as pd

        # Use base class aggregation for common fields
        data = self._aggregate_usage_data(*results)

        # Extract thoughts tokens from provider-specific details
        total_thoughts_tokens = 0
        for result in results:
            if result is not None:
                usage = result.usage()
                if hasattr(usage, "details") and usage.details:
                    # OpenAI uses 'reasoning_tokens', others use 'thoughts_tokens'
                    total_thoughts_tokens += usage.details.get("reasoning_tokens", 0)
                    total_thoughts_tokens += usage.details.get("thoughts_tokens", 0)

        data["thoughts_tokens"] = total_thoughts_tokens

        # Validate data structure using Pydantic
        validated_data = self._validate_usage(data)
        return pd.Series(validated_data)




class ProviderConfig(ABC):
    """
    Abstract base class for provider-specific configurations.
    
    Each provider config handles model detection, default settings, and agent creation
    for a specific AI provider (OpenAI, Anthropic, Gemini, etc.).
    """
    
    def __init__(self, models: list[str]):
        """
        Initialize provider config with supported models.
        
        Args:
            models: List of model names this provider supports
        """
        self.models = models
    
    @abstractmethod
    def create_agent(self, model_name: str, instructions: str, **kwargs) -> "pydantic_ai.agent.Agent":  # type: ignore
        """
        Create a pydantic-ai Agent with provider-specific configuration.
        
        Args:
            model_name: The model name to use
            instructions: Instructions/system prompt for the agent  
            **kwargs: Additional arguments passed to Agent constructor
            
        Returns:
            Configured pydantic-ai Agent instance
        """
        raise NotImplementedError


class OpenAIConfig(ProviderConfig):
    """Provider configuration for OpenAI models."""
    
    def __init__(self):
        models = [
            "gpt-5",
            "gpt-5-chat-latest", 
            "gpt-4o",
            "gpt-4o-mini",
            "gpt-4",
            "gpt-4-turbo",
            "gpt-3.5-turbo",
        ]
        super().__init__(models)
    
    def create_agent(self, model_name: str, instructions: str, **kwargs) -> "pydantic_ai.agent.Agent":  # type: ignore
        """Create OpenAI agent with default settings."""
        from pydantic_ai import Agent
        from pydantic_ai.models.openai import OpenAIModelSettings
        
        # Extract model_settings if provided, otherwise use defaults
        model_settings = kwargs.pop('model_settings', None)
        
        # Always extract max_tokens from kwargs to prevent it from being passed to Agent
        max_tokens = kwargs.pop('max_tokens', None)
        
        if model_settings is None:
            # Create default model settings
            reasoning_models = {"gpt-5"}
            temperature = 1 if model_name in reasoning_models else 0
            
            if max_tokens is not None:
                model_settings = OpenAIModelSettings(
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
            else:
                model_settings = OpenAIModelSettings(
                    temperature=temperature,
                )
        
        return Agent(
            f"openai:{model_name}",
            instructions=instructions,
            model_settings=model_settings,
            **kwargs
        )


class AnthropicConfig(ProviderConfig):
    """Provider configuration for Anthropic models."""
    
    def __init__(self):
        models = [
            "claude-3-5-sonnet-latest",
            "claude-3-5-haiku-latest", 
            "claude-3-opus-latest",
            "claude-3-7-sonnet-latest",
        ]
        super().__init__(models)
    
    def create_agent(self, model_name: str, instructions: str, **kwargs) -> "pydantic_ai.agent.Agent":  # type: ignore
        """Create Anthropic agent with default settings."""
        from pydantic_ai import Agent
        from pydantic_ai.models.anthropic import AnthropicModelSettings
        
        # Extract model_settings if provided, otherwise use defaults
        model_settings = kwargs.pop('model_settings', None)
        
        # Always extract max_tokens from kwargs to prevent it from being passed to Agent
        max_tokens = kwargs.pop('max_tokens', None)
        
        if model_settings is None:
            # Create default model settings
            reasoning_models = {"gpt-5"}  # Currently no Anthropic reasoning models
            temperature = 1 if model_name in reasoning_models else 0
            
            if max_tokens is not None:
                model_settings = AnthropicModelSettings(
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
            else:
                model_settings = AnthropicModelSettings(
                    temperature=temperature,
                )
        
        return Agent(
            f"anthropic:{model_name}",
            instructions=instructions,
            model_settings=model_settings,
            **kwargs
        )


class GeminiConfig(ProviderConfig):
    """Provider configuration for Gemini models."""
    
    def __init__(self):
        models = [
            "google-vertex:gemini-2.5-flash",
            "google-vertex:gemini-2.5-pro",
        ]
        super().__init__(models)
    
    def create_agent(self, model_name: str, instructions: str, **kwargs) -> "pydantic_ai.agent.Agent":  # type: ignore
        """Create Gemini agent with default settings."""
        from pydantic_ai import Agent
        from pydantic_ai.models.google import GoogleModelSettings
        
        # Extract model_settings if provided, otherwise use defaults
        model_settings = kwargs.pop('model_settings', None)
        
        # Always extract max_tokens from kwargs to prevent it from being passed to Agent
        max_tokens = kwargs.pop('max_tokens', None)
        
        if model_settings is None:
            # Create default model settings
            reasoning_models = {"gpt-5"}  # Currently no Gemini reasoning models
            temperature = 1 if model_name in reasoning_models else 0
            
            gemini_safety_settings = [
                {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "OFF"},
                {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "OFF"}, 
                {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "OFF"},
                {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "OFF"},
            ]
            
            if max_tokens is not None:
                model_settings = GoogleModelSettings(
                    temperature=temperature,
                    max_tokens=max_tokens,
                    google_safety_settings=gemini_safety_settings,
                )
            else:
                model_settings = GoogleModelSettings(
                    temperature=temperature,
                    google_safety_settings=gemini_safety_settings,
                )
        
        return Agent(
            model_name,  # Gemini doesn't need provider prefix
            instructions=instructions,
            model_settings=model_settings,
            **kwargs
        )


# Provider registry - single source of truth for model->provider mapping
PROVIDER_REGISTRY = {
    'openai': OpenAIConfig(),
    'anthropic': AnthropicConfig(), 
    'gemini': GeminiConfig()
}


def _get_provider_for_model(model_name: str) -> str:
    """
    Determine which provider supports the given model.
    
    Args:
        model_name: The model name to look up
        
    Returns:
        Provider name (key in PROVIDER_REGISTRY)
        
    Raises:
        ValueError: If no provider supports the model
    """
    for provider_name, config in PROVIDER_REGISTRY.items():
        if model_name in config.models:
            return provider_name
    
    # Collect all available models for error message
    all_models = []
    for config in PROVIDER_REGISTRY.values():
        all_models.extend(config.models)
    
    raise ValueError(f"Model '{model_name}' not supported. Available models: {', '.join(sorted(all_models))}")


def ez_pydantic_agent(model_name: str | None = None, instructions: str | None = None, **kwargs) -> "pydantic_ai.agent.Agent":  # type: ignore
    """
    Create a pydantic-ai Agent with automatic provider detection and default settings.
    
    This function automatically detects the appropriate AI provider based on the model name
    and creates a properly configured pydantic-ai Agent with provider-specific default settings.
    
    Args:
        model_name: The model name to use. Defaults to 'google-vertex:gemini-2.5-flash' if None.
        instructions: Instructions/system prompt for the agent. 
        **kwargs: Additional arguments passed to the pydantic-ai Agent constructor.
                 Common kwargs include: retries, deps, result_type, model_settings, max_tokens
    
    Returns:
        Configured pydantic-ai Agent instance
        
    Raises:
        ValueError: If the model_name is not supported by any provider
        
    Examples:
        # Basic usage with default model
        agent = ez_pydantic_agent(instructions="You are a helpful assistant")
        
        # Specify model and max tokens
        agent = ez_pydantic_agent(
            model_name="gpt-4o", 
            instructions="You are helpful", 
            max_tokens=1000
        )
        
        # With custom model settings (overrides defaults)
        from pydantic_ai.models.openai import OpenAIModelSettings
        settings = OpenAIModelSettings(temperature=0.5)
        agent = ez_pydantic_agent(
            model_name="gpt-4o",
            instructions="You are helpful",
            model_settings=settings
        )
    """
    # Set default model if none provided
    if model_name is None:
        model_name = "google-vertex:gemini-2.5-flash"
    
    # Set default instructions if none provided
    if instructions is None:
        instructions = "You are a helpful AI assistant."
    
    # Find appropriate provider
    provider_name = _get_provider_for_model(model_name)
    provider_config = PROVIDER_REGISTRY[provider_name]
    
    # Delegate to provider-specific creation logic
    return provider_config.create_agent(model_name, instructions, **kwargs)
