
from chat_graph.agents import AnswerGenerationAgent
from chat_graph.nodes.base_node import BaseNode
from chat_graph.state import ChatState


class AnswerGenerationNode(BaseNode):
    def __init__(self, chat_history):
        self.agent = AnswerGenerationAgent()
        self.chat_history = chat_history

    def __call__(self, state: ChatState) -> ChatState:
        chat_history = state['messages']

        summaries = state['summaries']

        answer = self.agent.inference(
            chat_history=chat_history,
            documents=summaries,
        )
        chat_history.append(answer)

        return ChatState(
            messages=state['messages'] + [answer],
            use_rag=False,
            questions=[],
            retrieved_docs=state['retrieved_docs'],
            summaries=state['summaries'],
            web_search_flags=state['web_search_flags'],
        )