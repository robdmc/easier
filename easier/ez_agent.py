from typing import Union, Optional
from abc import ABC, abstractmethod
from pydantic import BaseModel, Field


class UsageData(BaseModel):
    """Standard usage data structure for all agent types."""

    requests: int = Field(ge=0, description="Total number of API requests made")
    request_tokens: int = Field(ge=0, description="Total number of tokens in the input/prompts")
    response_tokens: int = Field(ge=0, description="Total number of tokens in the model's responses")
    thoughts_tokens: int = Field(ge=0, description="Total number of tokens used for internal reasoning")
    total_tokens: int = Field(ge=0, description="Total number of tokens used (request + response + thoughts)")


class ModelConfig(BaseModel):
    """Configuration for a specific model including cost information."""
    
    input_ppm_cost: float = Field(ge=0, description="Cost per million input tokens")
    output_ppm_cost: float = Field(ge=0, description="Cost per million output tokens")
    thought_ppm_cost: float = Field(ge=0, description="Cost per million thought/reasoning tokens")


class EZAgent(ABC):
    # Base class attribute that subclasses should override
    allowed_models = {}

    @classmethod
    def list_models(cls) -> list[str]:
        """
        Return a list of supported model names.
        
        When called on EZAgent base class: returns union of all models from all subclasses
        When called on child classes: returns only that class's models
        
        Returns:
            Sorted list of model names supported by this class or all classes
        """
        if cls is EZAgent:
            # Base class: return union of all models from all subclasses
            all_models = set()
            for subclass in cls.__subclasses__():
                if (
                    hasattr(subclass, "allowed_models")
                    and subclass.allowed_models
                    and subclass.allowed_models != cls.allowed_models
                ):
                    all_models.update(subclass.allowed_models.keys())
            return sorted(list(all_models))
        else:
            # Child class: return only this class's models
            return sorted(list(cls.allowed_models.keys()))

    def __new__(cls, system_prompt: str, model_name: str = None, **kwargs):
        """
        Factory method that automatically selects the correct agent subclass based on model_name.

        Args:
            system_prompt: The system prompt for the agent
            model_name: The model name to determine which agent type to use. 
                        Defaults to 'google-vertex:gemini-2.5-flash' if None.
            **kwargs: Additional arguments passed to the agent constructor

        Returns:
            Instance of the appropriate agent subclass (OpenAIAgent, GeminiAgent, etc.)

        Raises:
            ValueError: If no agent supports the specified model_name
        """
        if cls is not EZAgent:
            # Direct instantiation of a subclass, use normal construction
            return super().__new__(cls)

        # Set default model if none provided
        if model_name is None:
            model_name = "google-vertex:gemini-2.5-flash"

        # Factory logic: find first subclass that supports this model
        for subclass in cls.__subclasses__():
            # Check that subclass properly implements allowed_models (not empty base class dict)
            if (
                hasattr(subclass, "allowed_models")
                and subclass.allowed_models  # Not empty
                and subclass.allowed_models != cls.allowed_models  # Not the base class empty dict
                and model_name in subclass.allowed_models.keys()
            ):
                # Create instance and store resolved model name for __init__ to use
                instance = super().__new__(subclass)
                instance._resolved_model_name = model_name
                return instance

        # No matching subclass found - use list_models() for error message
        all_models = cls.list_models()
        raise ValueError(f"Model '{model_name}' not supported. Available models: {', '.join(all_models)}")

    @abstractmethod
    async def run(
        self,
        user_prompt: str,
        output_type: Optional["pydantic.main.BaseModel"] = None,  # type: ignore
    ) -> Union["pydantic_ai.agent.AgentRunResult", None]:  # type: ignore
        raise NotImplementedError

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
        Calculate cost information from usage results.
        
        Args:
            *results: One or more result objects returned from the agent's run method.
            
        Returns:
            A pandas Series containing cost breakdown:
            - 'input_cost': Cost for input/request tokens
            - 'output_cost': Cost for output/response tokens  
            - 'thoughts_cost': Cost for reasoning/thoughts tokens
            - 'total_cost': Total cost across all token types
            
        Raises:
            KeyError: If the agent's model_name is not found in allowed_models
            AttributeError: If any result object doesn't have a usage() method.
            ValueError: If no results are provided.
        """
        # Get usage data using existing method
        usage = self.get_usage(*results)
        
        # Get model config for current model
        if not hasattr(self, 'model_name') or self.model_name not in self.allowed_models:
            raise KeyError(f"Model '{getattr(self, 'model_name', 'unknown')}' not found in allowed_models")
        
        model_config = self.allowed_models[self.model_name]
        
        # Calculate costs (convert from per-million to actual costs)
        input_cost = usage['request_tokens'] * model_config.input_ppm_cost / 1_000_000
        output_cost = usage['response_tokens'] * model_config.output_ppm_cost / 1_000_000  
        thoughts_cost = usage['thoughts_tokens'] * model_config.thought_ppm_cost / 1_000_000
        total_cost = input_cost + output_cost + thoughts_cost
        
        import pandas as pd
        return pd.Series({
            'input_cost': input_cost,
            'output_cost': output_cost, 
            'thoughts_cost': thoughts_cost,
            'total_cost': total_cost
        })

    @abstractmethod
    def get_usage(self, *results: "pydantic_ai.agent.AgentRunResult") -> "pd.Series":  # type: ignore
        raise NotImplementedError


class OpenAIAgent(EZAgent):
    """
    A wrapper class for creating and configuring an OpenAI Agent with simplified initialization.

    This class provides an easy way to create an agent with predefined settings
    and system prompts for OpenAI models.
    """

    allowed_models = {
        "gpt-5": ModelConfig(input_ppm_cost=1.0, output_ppm_cost=5.0, thought_ppm_cost=5.0),
        "gpt-5-mini": ModelConfig(input_ppm_cost=0.5, output_ppm_cost=2.5, thought_ppm_cost=2.5),
        "gpt-5-nano": ModelConfig(input_ppm_cost=0.25, output_ppm_cost=1.25, thought_ppm_cost=1.25),
        "gpt-5-chat-latest": ModelConfig(input_ppm_cost=1.0, output_ppm_cost=5.0, thought_ppm_cost=5.0),
        "gpt-4o": ModelConfig(input_ppm_cost=2.5, output_ppm_cost=10.0, thought_ppm_cost=10.0),
        "gpt-4o-mini": ModelConfig(input_ppm_cost=0.15, output_ppm_cost=0.6, thought_ppm_cost=0.6),
        "gpt-4": ModelConfig(input_ppm_cost=30.0, output_ppm_cost=60.0, thought_ppm_cost=60.0),
        "gpt-4-turbo": ModelConfig(input_ppm_cost=10.0, output_ppm_cost=30.0, thought_ppm_cost=30.0),
        "gpt-3.5-turbo": ModelConfig(input_ppm_cost=0.5, output_ppm_cost=1.5, thought_ppm_cost=1.5),
        "o1": ModelConfig(input_ppm_cost=15.0, output_ppm_cost=60.0, thought_ppm_cost=60.0),
        "o1-mini": ModelConfig(input_ppm_cost=3.0, output_ppm_cost=12.0, thought_ppm_cost=12.0),
    }

    def __init__(
        self,
        system_prompt: str,
        model_name: str = "gpt-4o",
        retries: int = 1,
        validate_model_name: bool = True,
    ) -> None:
        """
        Initialize an OpenAIAgent with the specified system prompt, model, and retry settings.

        Args:
            system_prompt: The system prompt to use for the agent.
            model_name: The name of the OpenAI model to use. Defaults to 'gpt-4o'.
            retries: The number of retries to attempt if agent calls fail. Defaults to 1.
            validate_model_name: Whether to validate the model name against allowed models. Defaults to True.

        Raises:
            ValueError: If the model_name is not one of the allowed models and validate_model_name is True.
        """
        # Use resolved model name from factory if available
        if hasattr(self, '_resolved_model_name'):
            model_name = self._resolved_model_name
            delattr(self, '_resolved_model_name')
        
        if validate_model_name and model_name not in self.allowed_models.keys():
            raise ValueError(f"Model {model_name} not supported. Please use one of: {', '.join(list(self.allowed_models.keys()))}")

        # Store model name as public attribute
        self.model_name = model_name

        from pydantic_ai import Agent
        from pydantic_ai.models.openai import OpenAIModel, OpenAIModelSettings

        self._OpenAIModelSettings = OpenAIModelSettings

        # Store model configuration as attributes
        self._temperature = 0
        self._max_tokens = None

        # Create OpenAI model and agent
        openai_model = OpenAIModel(model_name)
        self.agent = Agent(
            openai_model,
            instructions=system_prompt,
            retries=retries,
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
        # Create model settings with stored configuration
        model_settings = self._OpenAIModelSettings(
            temperature=self._temperature,
            max_tokens=self._max_tokens,
        )

        result = await self.agent.run(
            user_prompt,
            output_type=output_type,
            model_settings=model_settings,
        )  # type: ignore
        return result

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
        if not results:
            raise ValueError("At least one result must be provided.")

        try:
            import pandas as pd

            # Initialize totals
            total_requests = 0
            total_request_tokens = 0
            total_response_tokens = 0
            total_thoughts_tokens = 0
            total_tokens = 0

            # Aggregate usage from all results
            for result in results:
                usage = result.usage()
                total_requests += usage.requests
                total_request_tokens += usage.request_tokens
                total_response_tokens += usage.response_tokens
                total_tokens += usage.total_tokens

                # Extract thoughts tokens from OpenAI details (map reasoning_tokens to thoughts_tokens for consistency)
                if hasattr(usage, "details") and usage.details:
                    total_thoughts_tokens += usage.details.get("reasoning_tokens", 0)

            data = {
                "requests": total_requests,
                "request_tokens": total_request_tokens,
                "response_tokens": total_response_tokens,
                "thoughts_tokens": total_thoughts_tokens,
                "total_tokens": total_tokens,
            }
            # Validate data structure using Pydantic
            validated_data = self._validate_usage(data)
            return pd.Series(validated_data)
        except AttributeError:
            raise AttributeError(
                "One or more result objects do not have a usage() method. Make sure you're passing valid agent run results."
            )


class AnthropicAgent(EZAgent):
    """
    A wrapper class for creating and configuring an Anthropic Agent with simplified initialization.

    This class provides an easy way to create an agent with predefined settings
    and system prompts for Anthropic Claude models.
    """

    allowed_models = {
        "claude-4": ModelConfig(input_ppm_cost=15.0, output_ppm_cost=75.0, thought_ppm_cost=75.0),
        "claude-sonnet-4": ModelConfig(input_ppm_cost=3.0, output_ppm_cost=15.0, thought_ppm_cost=15.0),
        "claude-3-5-sonnet-latest": ModelConfig(input_ppm_cost=3.0, output_ppm_cost=15.0, thought_ppm_cost=15.0),
        "claude-3-5-haiku-latest": ModelConfig(input_ppm_cost=1.0, output_ppm_cost=5.0, thought_ppm_cost=5.0),
        "claude-3-opus-latest": ModelConfig(input_ppm_cost=15.0, output_ppm_cost=75.0, thought_ppm_cost=75.0),
        "claude-3-7-sonnet-latest": ModelConfig(input_ppm_cost=3.0, output_ppm_cost=15.0, thought_ppm_cost=15.0),
    }

    def __init__(
        self,
        system_prompt: str,
        model_name: str = "claude-3-5-sonnet-latest",
        retries: int = 1,
        validate_model_name: bool = True,
    ) -> None:
        """
        Initialize an AnthropicAgent with the specified system prompt, model, and retry settings.

        Args:
            system_prompt: The system prompt to use for the agent.
            model_name: The name of the Anthropic model to use. Defaults to 'claude-3-5-sonnet-latest'.
            retries: The number of retries to attempt if agent calls fail. Defaults to 1.
            validate_model_name: Whether to validate the model name against allowed models. Defaults to True.

        Raises:
            ValueError: If the model_name is not one of the allowed models and validate_model_name is True.
        """
        # Use resolved model name from factory if available
        if hasattr(self, '_resolved_model_name'):
            model_name = self._resolved_model_name
            delattr(self, '_resolved_model_name')
        
        if validate_model_name and model_name not in self.allowed_models.keys():
            raise ValueError(f"Model {model_name} not supported. Please use one of: {', '.join(list(self.allowed_models.keys()))}")

        # Store model name as public attribute
        self.model_name = model_name

        from pydantic_ai import Agent
        from pydantic_ai.models.anthropic import AnthropicModel, AnthropicModelSettings

        self._AnthropicModelSettings = AnthropicModelSettings

        # Store model configuration as attributes
        self._temperature = 0
        self._max_tokens = None

        # Create Anthropic model and agent
        anthropic_model = AnthropicModel(model_name)
        self.agent = Agent(
            anthropic_model,
            instructions=system_prompt,
            retries=retries,
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
        # Create model settings with stored configuration
        model_settings = self._AnthropicModelSettings(
            temperature=self._temperature,
        )

        result = await self.agent.run(
            user_prompt,
            output_type=output_type,
            model_settings=model_settings,
        )  # type: ignore
        return result

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
        if not results:
            raise ValueError("At least one result must be provided.")

        try:
            import pandas as pd

            # Initialize totals
            total_requests = 0
            total_request_tokens = 0
            total_response_tokens = 0
            total_thoughts_tokens = 0
            total_tokens = 0

            # Aggregate usage from all results
            for result in results:
                usage = result.usage()
                total_requests += usage.requests
                total_request_tokens += usage.request_tokens
                total_response_tokens += usage.response_tokens
                total_tokens += usage.total_tokens

                # Extract thoughts tokens from Anthropic details if available
                if hasattr(usage, "details") and usage.details:
                    total_thoughts_tokens += usage.details.get("thoughts_tokens", 0)

            data = {
                "requests": total_requests,
                "request_tokens": total_request_tokens,
                "response_tokens": total_response_tokens,
                "thoughts_tokens": total_thoughts_tokens,
                "total_tokens": total_tokens,
            }
            # Validate data structure using Pydantic
            validated_data = self._validate_usage(data)
            return pd.Series(validated_data)
        except AttributeError:
            raise AttributeError(
                "One or more result objects do not have a usage() method. Make sure you're passing valid agent run results."
            )


class GeminiAgent(EZAgent):
    """
    A wrapper class for creating and configuring an Agent with simplified initialization.

    This class provides an easy way to create an agent with predefined safety settings
    and system prompts.
    """

    allowed_models = {
        "google-vertex:gemini-2.5-flash": ModelConfig(input_ppm_cost=0.5, output_ppm_cost=3.0, thought_ppm_cost=3.0),
        "google-vertex:gemini-2.5-pro": ModelConfig(input_ppm_cost=1.25, output_ppm_cost=10.0, thought_ppm_cost=10.0),
    }

    def __init__(
        self,
        system_prompt: str,
        model_name: str = "google-vertex:gemini-2.5-flash",
        retries: int = 1,
        validate_model_name: bool = True,
    ) -> None:
        """
        Initialize a GeminiAgent with the specified system prompt, model, and retry settings.

        Args:
            system_prompt: The system prompt to use for the agent.
            model_name: The name of the model to use for the agent. Defaults to 'google-vertex:gemini-2.5-flash'.
            retries: The number of retries to attempt if agent calls fail. Defaults to 1.
            validate_model_name: Whether to validate the model name against allowed models. Defaults to True.

        Raises:
            ValueError: If the model_name is not one of the allowed models and validate_model_name is True.
        """
        # Use resolved model name from factory if available
        if hasattr(self, '_resolved_model_name'):
            model_name = self._resolved_model_name
            delattr(self, '_resolved_model_name')
        
        if validate_model_name and model_name not in self.allowed_models.keys():
            raise ValueError(f"Model {model_name} not supported. Please use one of: {', '.join(list(self.allowed_models.keys()))}")

        # Store model name as public attribute
        self.model_name = model_name

        from pydantic_ai import Agent
        from pydantic_ai.models.gemini import GeminiModelSettings

        self._GeminiModelSettings = GeminiModelSettings

        # Store model configuration as attributes instead of creating instance
        self._temperature = 0
        self._gemini_safety_settings = [
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "OFF"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "OFF"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "OFF"},
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "OFF"},
        ]

        self.agent = Agent(
            model_name,
            instructions=system_prompt,
            retries=retries,
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
        # Create model settings with all stored configuration
        model_settings = self._GeminiModelSettings(
            temperature=self._temperature,
            gemini_safety_settings=self._gemini_safety_settings.copy(),  # type: ignore
        )

        result = await self.agent.run(
            user_prompt,
            output_type=output_type,
            model_settings=model_settings,
        )  # type: ignore
        return result

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
        if not results:
            raise ValueError("At least one result must be provided.")

        try:
            import pandas as pd

            # Initialize totals
            total_requests = 0
            total_request_tokens = 0
            total_response_tokens = 0
            total_thoughts_tokens = 0
            total_tokens = 0

            # Aggregate usage from all results
            for result in results:
                usage = result.usage()
                total_requests += usage.requests
                total_request_tokens += usage.request_tokens
                total_response_tokens += usage.response_tokens
                total_tokens += usage.total_tokens

                # Extract thoughts_tokens from details if available
                if hasattr(usage, "details") and usage.details:
                    total_thoughts_tokens += usage.details.get("thoughts_tokens", 0)

            data = {
                "requests": total_requests,
                "request_tokens": total_request_tokens,
                "response_tokens": total_response_tokens,
                "thoughts_tokens": total_thoughts_tokens,
                "total_tokens": total_tokens,
            }
            # Validate data structure using Pydantic
            validated_data = self._validate_usage(data)
            return pd.Series(validated_data)
        except AttributeError:
            raise AttributeError(
                "One or more result objects do not have a usage() method. Make sure you're passing valid agent run results."
            )
