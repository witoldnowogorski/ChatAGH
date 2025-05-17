import os

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langgraph.graph.state import StateGraph, END, START

from chat_graph.nodes import (
    RAGDecisionNode,
    AnswerGenerationNode,
    QuestionsGenerationNode,
    RetrieverNode,
    DocsAnalyzerNode,
    WebSearchNode
)
from chat_graph.state import ChatState


class ChatGraph:
    def __init__(self):

        self.workflow = StateGraph(ChatState)

        self.history = []

        self.workflow.add_node("rag_decision", RAGDecisionNode())
        self.workflow.add_node("answer_generation", AnswerGenerationNode())
        self.workflow.add_node("questions_generation", QuestionsGenerationNode())
        self.workflow.add_node("retriever", RetrieverNode())
        self.workflow.add_node("docs_analyzer", DocsAnalyzerNode())
        self.workflow.add_node("web_search", WebSearchNode())
        self.workflow.add_node("retrieval_summary", lambda x: x)


        self.workflow.add_edge(START, "rag_decision")
        self.workflow.add_conditional_edges(
            "rag_decision",
            lambda state: "questions_generation" if state["use_rag"] else "answer_generation"
        )

        self.workflow.add_edge("questions_generation", "retriever")
        self.workflow.add_edge("retriever", "docs_analyzer")

        self.workflow.add_conditional_edges(
            "docs_analyzer",
            lambda state: "web_search" if any(state["web_search_flags"]) else "retrieval_summary"
        )
        self.workflow.add_edge("web_search", "retrieval_summary")

        self.workflow.add_edge("retrieval_summary", "answer_generation")
        self.workflow.add_edge("answer_generation", END)
        self.graph = self.workflow.compile()

    def invoke(self, message):
        result = self.graph.invoke({
            "messages": self.history + [HumanMessage(message)],
            "questions": [],
            "use_rag": False,
            "retrieved_docs": [],
            "summaries": [],
            "web_search_flags": [],
        })
        self.history = result["messages"]

        return result


if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(current_dir))
    dotenv_path = os.path.join(project_root, '.env')
    load_dotenv(dotenv_path=dotenv_path)

    chat = ChatGraph()
    # temp_state = chat.invoke("Hej")
    # final_state = chat.invoke("Jak zostac studentem AGH?")

    # print(chat.invoke("Who is 2 + 2?"))
    # print(chat.invoke("What was my previous question?"))
