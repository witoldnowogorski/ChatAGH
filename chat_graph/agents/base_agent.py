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

        self.api_keys = json.loads(os.getenv("GEMINI_API_KEYS", "[]"))

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
