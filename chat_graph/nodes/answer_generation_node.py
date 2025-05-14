
from chat_graph.agents import AnswerGenerationAgent
from chat_graph.nodes.base_node import BaseNode
from chat_graph.state import ChatState


class AnswerGenerationNode(BaseNode):
    def __init__(self):
        super().__init__(agent=AnswerGenerationAgent())

    def __call__(self, state: ChatState) -> ChatState:
        chat_history = state['messages']
        answer = self.agent.inference(chat_history=chat_history)
        return {
            "messages": state['messages'] + [answer],
            "use_rag": False
        }