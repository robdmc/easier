from typing import Union, Optional


class EZAgent:
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
            model_name: The name of the model to use for the agent. Defaults to 'google-vertex:gemini-2.5-pro'.
            retries: The number of retries to attempt if agent calls fail. Defaults to 1.
            validate_model_name: Whether to validate the model name against allowed models. Defaults to True.

        Raises:
            ValueError: If the model_name is not one of the allowed models and validate_model_name is True.
        """
        allowed_models = [
            "google-vertex:gemini-2.0-flash",
            "google-vertex:gemini-2.5-flash",
            "google-vertex:gemini-2.5-pro",
        ]

        if validate_model_name and model_name not in allowed_models:
            raise ValueError(f"Model {model_name} not supported. Please use one of: {', '.join(allowed_models)}")

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
            - 'total_tokens': Total number of tokens used

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
            total_tokens = 0

            # Aggregate usage from all results
            for result in results:
                usage = result.usage()
                total_requests += usage.requests
                total_request_tokens += usage.request_tokens
                total_response_tokens += usage.response_tokens
                total_tokens += usage.total_tokens

            data = {
                "requests": total_requests,
                "request_tokens": total_request_tokens,
                "response_tokens": total_response_tokens,
                "total_tokens": total_tokens,
            }
            return pd.Series(data)
        except AttributeError:
            raise AttributeError(
                "One or more result objects do not have a usage() method. Make sure you're passing valid agent run results."
            )
