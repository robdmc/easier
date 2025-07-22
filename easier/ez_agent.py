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

        self.model_settings = GeminiModelSettings(
            temperature=0,
            gemini_safety_settings=[
                {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "OFF"},
                {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "OFF"},
                {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "OFF"},
                {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "OFF"},
            ],
        )
        self.agent = Agent(
            model_name,
            instructions=system_prompt,
            retries=retries,
        )

    async def run(self, user_prompt: str, output_type: Optional["BaseModel"] = None) -> Union["BaseModel", None]:  # type: ignore
        """
        Run the agent with the given user prompt and expected output type.
        """
        result = await self.agent.run(
            user_prompt,
            output_type=output_type,
            model_settings=self.model_settings,
        )  # type: ignore
        return result.output

    def run_sync(self, user_prompt: str, output_type: Optional["BaseModel"] = None) -> Union["BaseModel", None]:  # type: ignore
        """
        Run the agent synchronously with the given user prompt and expected output type.
        """
        result = self.agent.run_sync(
            user_prompt,
            output_type=output_type,
            model_settings=self.model_settings,
        )  # type: ignore
        return result.output
