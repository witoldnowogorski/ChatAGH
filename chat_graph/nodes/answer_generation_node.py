
from chat_graph.agents import AnswerGenerationAgent
from chat_graph.nodes.base_node import BaseNode
from chat_graph.state import ChatState


class AnswerGenerationNode(BaseNode):
    def __init__(self):
        self.agent = AnswerGenerationAgent()

    def __call__(self, state: ChatState) -> ChatState:
        summaries = state['summaries']

        answer = self.agent.inference(
            chat_history=state["messages"],
            documents=summaries,
        )

        return ChatState(
            messages=state['messages'] + [answer],
            use_rag=False,
            questions=[],
            retrieved_docs=state['retrieved_docs'],
            summaries=state['summaries'],
            web_search_flags=state['web_search_flags'],
        )
