from abc import abstractmethod

from chat_graph.states import ChatState
from chat_graph.utils import log_execution_time


class BaseNode:
    @abstractmethod
    def __call__(self, state: ChatState) -> ChatState:
        pass

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if '__call__' in cls.__dict__:
            cls.__call__ = log_execution_time(cls.__call__)