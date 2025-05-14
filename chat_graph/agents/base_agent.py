from abc import abstractmethod

from langchain_google_genai import ChatGoogleGenerativeAI

from chat_graph.inference.utils import retry_on_exception


class BaseAgent:
    def __init__(
        self,
        prompt_template,
        model_name="gemini-2.0-flash-001",
        **kwargs
    ):
        self.llm = ChatGoogleGenerativeAI(model=model_name, api_key="AIzaSyCv5Pwbk1Jg-UsiMQ6yd76FaJlJzdvDdVs", **kwargs)
        self.prompt_template = prompt_template

    @abstractmethod
    @retry_on_exception(delay=1, backoff=3)
    def inference(self, *args, **kwargs):
        pass