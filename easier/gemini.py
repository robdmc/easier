try:
    import google.generativeai as genai
    from google.generativeai import types
except ImportError:
    genai = None
    types = None
import os

class Gemini:
    """
    A class for interacting with Google's Gemini AI models through the Vertex AI API.

    This class provides a simplified interface to generate text responses using Gemini models.
    It handles authentication, model configuration, and response generation in a streamlined way.

    ==================================================================================
    Note:  I just discovered the mirascope project.  Here are notes on how to do it.
        from mirascope import llm
        from pydantic import BaseModel
        gem = ezr.Gemini()

        class Response(BaseModel):
            my_message: str

        @llm.call(provider='google', model=gem.model, client=gem.client, response_model=Response)
        def doit(prompt):
            return prompt

        resp = doit('what is wikipedia')
        print(resp.my_message)
        print(resp._response.input_tokens)
        print(resp._response.output_tokens)
    ==================================================================================


    Attributes:
        COMPUTE_LOCATION (str): The Google Cloud compute location to use for API calls.
    """
    COMPUTE_LOCATION = 'us-central1'

    def __init__(self, model='gemini-2.0-flash', temperature=0.01, top_p=0.95, max_output_tokens=8192, use_staging=True):
        """
        Initialize the Gemini client with specified configuration.

        Args:
            model (str, optional): The Gemini model to use. Defaults to "gemini-2.0-flash-001".
            temperature (float, optional): Controls randomness in the output. Lower values make
                the output more deterministic. Defaults to 0.01.
            top_p (float, optional): Controls diversity via nucleus sampling. Defaults to 0.95.
            max_output_tokens (int, optional): Maximum number of tokens to generate. Defaults to 8192.
            use_staging (bool, optional): Whether to use staging environment. Defaults to True.

        Note:
            Requires GOOGLE_CLOUD_PROJECT or GOOGLE_CLOUD_PROJECT_STAGING environment variables to be set.
        """
        self.model = model
        self.temperature = temperature
        self.top_p = top_p
        self.max_output_tokens = max_output_tokens
        if use_staging:
            project = os.environ['GOOGLE_CLOUD_PROJECT_STAGING']
        else:
            project = os.environ['GOOGLE_CLOUD_PROJECT']
        self.client = genai.Client(vertexai=True, project=project, location=self.COMPUTE_LOCATION)
        self.config = types.GenerateContentConfig(temperature=self.temperature, top_p=self.top_p, max_output_tokens=max_output_tokens, response_modalities=['TEXT'], safety_settings=[types.SafetySetting(category='HARM_CATEGORY_HATE_SPEECH', threshold='OFF'), types.SafetySetting(category='HARM_CATEGORY_DANGEROUS_CONTENT', threshold='OFF'), types.SafetySetting(category='HARM_CATEGORY_SEXUALLY_EXPLICIT', threshold='OFF'), types.SafetySetting(category='HARM_CATEGORY_HARASSMENT', threshold='OFF')])

    def prompt(self, text):
        """
        Generate a response from the Gemini model for the given input text.

        Args:
            text (str): The input text/prompt to send to the model.

        Returns:
            str: The generated response text from the model.

        Note:
            The response is streamed and concatenated before being returned.
        """
        contents = [types.Content(role='user', parts=[types.Part.from_text(text=text)])]
        chunk_generator = self.client.models.generate_content_stream(model=self.model, contents=contents, config=self.config)
        return ''.join([c.text for c in chunk_generator]).strip()