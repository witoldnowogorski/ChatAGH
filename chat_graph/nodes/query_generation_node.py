from chat_graph.agents.query_generation_agent import QueryGenerationAgent
from chat_graph.nodes.base_node import BaseNode
from chat_graph.states import ChatState
from chat_graph.utils import retry_on_exception


class QueryGenerationNode(BaseNode):
    def __init__(self):
        self.agent = QueryGenerationAgent()

    @retry_on_exception(delay=1, attempts=3)
    def __call__(self, state: ChatState) -> ChatState:
        chat_history = state["chat_history"]
        processed_retrieved_context = state["processed_retrieved_context"]
        query = self.agent.run(
            chat_history=chat_history,
            processed_retrieved_context=processed_retrieved_context
        )
        state["search_query"] = query
        return state
