from abc import abstractmethod

from chat_graph.states import ChatState


class BaseNode:
    @abstractmethod
    def __call__(self, state: ChatState) -> ChatState:
        pass