class Gemini:
    COMPUTE_LOCATION = "us-central1"

    def __init__(
        self,
        model="gemini-2.0-flash-001",
        temperature=0.01,
        top_p=0.95,
        max_output_tokens=8192,
    ):
        import os
        from google import genai
        from google.genai import types

        self.model = model
        self.temperature = temperature
        self.top_p = top_p
        self.max_output_tokens = max_output_tokens

        self.client = genai.Client(
            vertexai=True,
            project=os.environ["GOOGLE_CLOUD_PROJECT"],
            location=self.COMPUTE_LOCATION,
        )

        self.config = types.GenerateContentConfig(
            temperature=self.temperature,
            top_p=self.top_p,
            max_output_tokens=max_output_tokens,
            response_modalities=["TEXT"],
            safety_settings=[
                types.SafetySetting(
                    category="HARM_CATEGORY_HATE_SPEECH", threshold="OFF"
                ),
                types.SafetySetting(
                    category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="OFF"
                ),
                types.SafetySetting(
                    category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold="OFF"
                ),
                types.SafetySetting(
                    category="HARM_CATEGORY_HARASSMENT", threshold="OFF"
                ),
            ],
        )

    def prompt(self, text):
        from google.genai import types

        contents = [
            types.Content(role="user", parts=[types.Part.from_text(text=text)])
        ]

        chunk_generator = self.client.models.generate_content_stream(
            model=self.model,
            contents=contents,
            config=self.config,
        )
        return ("".join([c.text for c in chunk_generator])).strip()
