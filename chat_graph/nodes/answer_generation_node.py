from langgraph.config import get_stream_writer
from langchain_core.messages import AIMessage

from chat_graph.agents import AnswerGenerationAgent
from chat_graph.nodes.base_node import BaseNode
from chat_graph.states import ChatState
from chat_graph.utils import logger


class AnswerGenerationNode(BaseNode):
    def __init__(self):
        self.agent = AnswerGenerationAgent()

    def __call__(self, state: ChatState) -> ChatState:
        logger.info("Generating final answer ...")
        writer = get_stream_writer()

        retrieved_context = state["processed_retrieved_context"]
        chat_history = state["chat_history"]

        args = {
            "chat_history": chat_history,
            "retrieved_context": retrieved_context
        }
        answer = ""
        for response_chunk in self.agent.stream_response(**args):
            writer(response_chunk)
            answer += response_chunk.content

        state["chat_history"].append(AIMessage(answer))

        logger.info("Answer generation completed.")

        return state