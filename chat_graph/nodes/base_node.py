from abc import abstractmethod

from chat_graph.agents.base_agent import BaseAgent
from chat_graph.state import ChatState


class BaseNode:
    @abstractmethod
    def __call__(self, state: ChatState) -> ChatState:
        pass