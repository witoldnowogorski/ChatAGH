from typing import List, TypedDict

from langchain_core.messages import HumanMessage, AIMessage
from langgraph.graph.state import StateGraph, END, START

from chat_graph.nodes import RAGDecisionNode, AnswerGenerationNode
from chat_graph.state import ChatState


class ChatGraph:
    def __init__(self):

        self.workflow = StateGraph(ChatState)

        self.history = []

        self.workflow.add_node("rag_decision", RAGDecisionNode())
        self.workflow.add_node("answer_generation", AnswerGenerationNode())
        self.workflow.add_node("rag_generation", lambda state: {"messages": [], "use_rag": False})

        self.workflow.add_edge(START, "rag_decision")
        self.workflow.add_conditional_edges(
            "rag_decision",
            lambda state: "rag_generation" if state["use_rag"] else "answer_generation"
        )

        self.workflow.add_edge("answer_generation", END)
        self.workflow.add_edge("rag_generation", END)

        self.graph = self.workflow.compile()

    def invoke(self, message):
        self.history.append(HumanMessage(message))
        return self.graph.invoke({
            "messages": self.history
        })

if __name__ == "__main__":
    chat = ChatGraph()
    print(chat.invoke("Kto jest rektorek AGH?"))
    # print(chat.invoke("Who is 2 + 2?"))
    # print(chat.invoke("What was my previous question?"))
