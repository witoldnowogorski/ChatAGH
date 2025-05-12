import os
import ast
import random
from dotenv import load_dotenv

from google import genai

from models.utils import retry_on_exception

ENV_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), ".env")

load_dotenv(dotenv_path=ENV_PATH)


class BaseGoogleModel:
    """
    A base class for interacting with Google's GenAI models.

    This class provides a common interface for generating content using a specified model
    and prompt template. It serves as a foundation for specialized models for query augmentation,
    search enhancement, and answer generation.

    Attributes:
        client (genai.Client): The Google API client initialized using the API key from .env file.
        model (str): The name of the model used for content generation.
        prompt_template (str): A template used to format prompts for the model.

    Methods:
        _inference(contents):
            Sends a list of prompt contents to the model and returns the generated text.
        generate(query: str, **kwargs):
            Formats the prompt using the provided query and context, and generates a response.
    """

    def __init__(self, prompt_template, model_name="gemini-2.0-flash-001"):
        api_keys_str = os.getenv("API_KEYS", "[]")
        self.api_keys = ast.literal_eval(api_keys_str)
        if not self.api_keys:
            raise ValueError("API_KEYS environment variable is not set or empty")

        self.model = model_name
        self.prompt_template = prompt_template

    @retry_on_exception(attempts=3, delay=2, backoff=5)
    def _inference(self, contents, logger=None):
        self.client = genai.Client(api_key=random.choice(self.api_keys))

        if logger:
            logger.debug(f"Inferring {self.__class__.__name__} with prompt:'{contents}'")

        return self.client.models.generate_content(
            model=self.model,
            contents=contents,
        ).text

    def generate(self, **kwargs):
        prompt = self.prompt_template.format(**kwargs)
        print(prompt)
        return self._inference([prompt])
