from langchain_core.messages import AIMessage

from chat_graph.agents import AnswerGenerationAgent
from chat_graph.nodes.base_node import BaseNode
from chat_graph.states import ChatState


class AnswerGenerationNode(BaseNode):
    def __init__(self):
        self.agent = AnswerGenerationAgent()

    def __call__(self, state: ChatState) -> ChatState:
        retrieved_context = state["processed_retrieved_context"]
        chat_history = state["chat_history"]

        answer = self.agent.inference(
            chat_history=chat_history,
            retrieved_context=retrieved_context
        )
        state["chat_history"] += AIMessage(answer)

        return state