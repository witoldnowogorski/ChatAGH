from typing import List, TypedDict

from langchain_core.messages import HumanMessage, AIMessage
from langgraph.graph.state import StateGraph, END, START

from models.google_genai.chat_model import ChatModel

class ChatState(TypedDict):
    messages: List[dict]


class ChatGraph:
    def __init__(self):
        self.model = ChatModel()

        self.workflow = StateGraph(ChatState)
        self.workflow.add_edge(START, "answer_generation")
        self.workflow.add_node("answer_generation", self._generate_answer)

        self.history = []
        self.graph = self.workflow.compile()

    def _generate_answer(self, state):
        response = self.model.generate(chat_history=state["messages"])
        answer = AIMessage(response)
        self.history.append(answer)
        return {
            "messages": state["messages"]
        }

    def invoke(self, message):
        self.history.append(HumanMessage(message))
        return self.graph.invoke({
            "messages": self.history
        })

chat = ChatGraph()
print(chat.invoke("HI"))
print(chat.invoke("Who is 2 + 2?"))
print(chat.invoke("What was my previous question?"))
