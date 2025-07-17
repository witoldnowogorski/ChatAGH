from chat_graph.agents import RAGDecisionAgent
from chat_graph.states import ChatState
from chat_graph.nodes.base_node import BaseNode
from chat_graph.utils import retry_on_exception, logger


class RAGDecisionNode(BaseNode):
    def __init__(self):
        self.agent = RAGDecisionAgent()

    @retry_on_exception(delay=1, backoff=1, attempts=3)
    def __call__(self, state: ChatState) -> ChatState:
        logger.info("Performing RAG Decision ...")

        chat_history = state['chat_history']
        processed_retrieved_context = state['processed_retrieved_context'] or ""
        rag_decision = self.agent.run(
            chat_history=chat_history,
            processed_retrieved_context=processed_retrieved_context
        )

        logger.info("RAG Decision: {}".format(rag_decision))

        state["rag_decision"] = rag_decision

        return state
