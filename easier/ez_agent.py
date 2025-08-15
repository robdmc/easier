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


class AgentBase(ABC):
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

    @abstractmethod
    def get_usage(self, *results: "pydantic_ai.agent.AgentRunResult") -> "pd.Series":  # type: ignore
        raise NotImplementedError


class OpenAIAgent(AgentBase):
    """
    A wrapper class for creating and configuring an OpenAI Agent with simplified initialization.

    This class provides an easy way to create an agent with predefined settings
    and system prompts for OpenAI models.
    """

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
        allowed_models = [
            "gpt-5",
            "gpt-5-mini",
            "gpt-5-nano",
            "gpt-5-chat-latest",
            "gpt-4o",
            "gpt-4o-mini",
            "gpt-4",
            "gpt-4-turbo",
            "gpt-3.5-turbo",
            "o1",
            "o1-mini",
        ]

        if validate_model_name and model_name not in allowed_models:
            raise ValueError(f"Model {model_name} not supported. Please use one of: {', '.join(allowed_models)}")

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


class GeminiAgent(AgentBase):
    """
    A wrapper class for creating and configuring an Agent with simplified initialization.

    This class provides an easy way to create an agent with predefined safety settings
    and system prompts.
    """

    def __init__(
        self,
        system_prompt: str,
        model_name: str = "google-vertex:gemini-2.5-flash",
        retries: int = 1,
        validate_model_name: bool = True,
    ) -> None:
        """
        Initialize an EZAgent with the specified system prompt, model, and retry settings.

        Args:
            system_prompt: The system prompt to use for the agent.
            model_name: The name of the model to use for the agent. Defaults to 'google-vertex:gemini-2.5-flash'.
            retries: The number of retries to attempt if agent calls fail. Defaults to 1.
            validate_model_name: Whether to validate the model name against allowed models. Defaults to True.

        Raises:
            ValueError: If the model_name is not one of the allowed models and validate_model_name is True.
        """
        allowed_models = [
            "google-vertex:gemini-2.5-flash",
            "google-vertex:gemini-2.5-pro",
        ]

        if validate_model_name and model_name not in allowed_models:
            raise ValueError(f"Model {model_name} not supported. Please use one of: {', '.join(allowed_models)}")

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


EZAgent = GeminiAgent
