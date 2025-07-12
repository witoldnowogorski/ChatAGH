from abc import abstractmethod

from langchain_google_genai import ChatGoogleGenerativeAI

from chat_graph.utils import log_execution_time


class BaseAgent:
    def __init__(
        self,
        prompt_template,
        model_name="gemini-2.0-flash-001",
        **kwargs
    ):
        self.llm = ChatGoogleGenerativeAI(model=model_name, **kwargs)
        self.prompt_template = prompt_template

    @abstractmethod
    def inference(self, *args, **kwargs):
        pass
