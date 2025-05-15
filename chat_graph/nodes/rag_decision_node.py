from chat_graph.agents import RAGDecisionAgent
from chat_graph.state import ChatState
from chat_graph.nodes.base_node import BaseNode
from chat_graph.utils import retry_on_exception

class RAGDecisionNode(BaseNode):
    def __init__(self):
        self.agent = RAGDecisionAgent()

    @retry_on_exception(delay=1, backoff=1, attempts=3)
    def __call__(self, state: ChatState) -> ChatState:
        chat_history = state['messages']
        decision = self.agent.inference(chat_history=chat_history)
        return ChatState(
            messages=state["messages"],
            use_rag=decision,
            questions=state["questions"],
            retrieved_docs=state["retrieved_docs"],
            summaries=state['summaries'],
            web_search_flags=state['web_search_flags'],
        )