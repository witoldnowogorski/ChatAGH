import os

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langgraph.graph.state import StateGraph, END, START

from chat_graph.nodes import (
    RAGDecisionNode,
    AnswerGenerationNode,
    QueryGenerationNode,
    DocsAnalyzerNode,
    SearchNode,
    GraphAugmentationNode,
)
from chat_graph.states import ChatState


class ChatGraph:
    def __init__(self):
        self.workflow = StateGraph(ChatState)

        self.workflow.add_node("search_decision", RAGDecisionNode())
        self.workflow.add_node("answer_generation", AnswerGenerationNode())
        self.workflow.add_node("query_generation", QueryGenerationNode())
        self.workflow.add_node("search", SearchNode())
        self.workflow.add_node("docs_analyzer", DocsAnalyzerNode())
        self.workflow.add_node("graph_augmentation", GraphAugmentationNode())

        self.workflow.add_edge(START, "search_decision")
        self.workflow.add_edge("search_decision", END)
        self.workflow.add_conditional_edges(
            "search_decision",
            lambda state: "query_generation" if state["rag_decision"] else "answer_generation"
        )
        self.workflow.add_edge("query_generation", "search")
        self.workflow.add_edge("search", "docs_analyzer")
        self.workflow.add_edge("docs_analyzer", "graph_augmentation")
        self.workflow.add_edge("graph_augmentation", "answer_generation")

        self.graph = self.workflow.compile()

        self.state = ChatState(
            chat_history=[],
            processed_retrieved_context=None,
            rag_decision=False,
            search_query=None,
            retrieved_chunks=None,
            analyzed_context=None
        )

    def invoke(self, message: str):
        self.state["chat_history"].append(HumanMessage(message))
        self.state = self.graph.invoke(self.state)
        return self.state

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(current_dir))
    dotenv_path = os.path.join(project_root, '.env')
    load_dotenv(dotenv_path=dotenv_path)

    chat = ChatGraph()
    temp_state = chat.invoke("Jak wygląda proces rekrutacji na AGH?")
    print(temp_state["chat_history"][-1].content)
