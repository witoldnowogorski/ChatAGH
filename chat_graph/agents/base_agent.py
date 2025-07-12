import os
import json
import random
from abc import abstractmethod

from langchain_google_genai import ChatGoogleGenerativeAI


class AgentError(Exception):
    pass


class BaseAgent:
    def __init__(
        self,
        prompt_template: str,
        model_name: str = "gemini-2.0-flash-001",
        **kwargs
    ):
        self.prompt_template: str = prompt_template
        self.model_name: str = model_name
        self.path: str | None = os.environ["GEMINI_API_KEYS_PATH"]
        if self.path is None:
            raise AgentError("GEMINI_API_KEYS_PATH env variable is not set, unable to load api keys")
        else:
            with open(self.path, "r") as f:
                self.api_keys = json.load(f)

        self.llm: ChatGoogleGenerativeAI = ChatGoogleGenerativeAI(model=model_name, api_key=self.api_keys[0])

    @abstractmethod
    def inference(self, *args, **kwargs):
        pass

    def run(self, **kwargs):
        api_key = random.choice(self.api_keys)
        self.llm = ChatGoogleGenerativeAI(
            model=self.model_name,
            api_key=api_key,
        )
        return self.inference(**kwargs)
