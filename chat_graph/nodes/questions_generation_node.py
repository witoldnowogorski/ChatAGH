
from chat_graph.agents.questions_generation_agent import QuestionsGenerationAgent
from chat_graph.nodes.base_node import BaseNode
from chat_graph.state import ChatState
from chat_graph.utils import retry_on_exception


class QuestionsGenerationNode(BaseNode):
    def __init__(self):
        self.agent = QuestionsGenerationAgent()

    @retry_on_exception(delay=1, attempts=3)
    def __call__(self, state: ChatState) -> ChatState:
        chat_history = state["messages"]
        questions = self.agent.inference(chat_history=chat_history)
        return ChatState(
            messages=state["messages"],
            use_rag=state["use_rag"],
            questions=questions,
            retrieved_docs=state["retrieved_docs"],
            summaries=state['summaries'],
            web_search_flags=state['web_search_flags'],
        )
