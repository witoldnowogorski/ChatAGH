from chat_graph.agents import RAGDecisionAgent
from chat_graph.state import ChatState
from chat_graph.nodes.base_node import BaseNode


class RAGDecisionNode(BaseNode):
    def __init__(self):
        super().__init__(agent=RAGDecisionAgent())

    def __call__(self, state: ChatState):
        chat_history = state['messages']
        decision = self.agent.inference(chat_history=chat_history)
        return {
            "messages": state["messages"],
            "use_rag": decision
        }