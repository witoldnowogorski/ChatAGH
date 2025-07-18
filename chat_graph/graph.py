import os

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langgraph.graph.state import StateGraph, START, END

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
    """
    ChatGraph defines a conversational workflow for a Retrieval-Augmented Generation (RAG) system
    using a stateful graph architecture.

    The workflow includes:
    - A decision node to determine whether document retrieval is needed.
    - External knowledge base query generation.
    - A search node to retrieve relevant documents.
    - A document analysis and augmentation phase to enrich the context.
    - A final answer generation node.

    Parameters (as keyword arguments):
    - initial_retrieved_chunks (int): Number of initial documents to retrieve during the search phase.
    - window_size (int): Size of the window of chunks to retrieve.
    - num_augmentation_chunks (int): Number of chunks retrieved in context augmentation phase.
    """
    def __init__(self, **kwargs):
        self.workflow = StateGraph(ChatState)

        self.workflow.add_node("search_decision", RAGDecisionNode())
        self.workflow.add_node("answer_generation", AnswerGenerationNode())
        self.workflow.add_node("query_generation", QueryGenerationNode())
        self.workflow.add_node("search", SearchNode(
            initial_retrieved_chunks=kwargs.get("initial_retrieved_chunks", 10),
            window_size=kwargs.get("window_size", 2)
        ))
        self.workflow.add_node("docs_analyzer", DocsAnalyzerNode())
        self.workflow.add_node("graph_augmentation", GraphAugmentationNode(
            num_related_chunks_for_doc=kwargs.get("num_augmentation_chunks", 5)
        ))

        self.workflow.add_edge(START, "search_decision")
        self.workflow.add_conditional_edges(
            "search_decision",
            lambda state: "query_generation" if state["rag_decision"] else "answer_generation"
        )
        self.workflow.add_edge("query_generation", "search")
        self.workflow.add_edge("search", "docs_analyzer")
        self.workflow.add_edge("docs_analyzer", "graph_augmentation")
        self.workflow.add_edge("graph_augmentation", "answer_generation")
        self.workflow.add_edge("answer_generation", END)

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
    temp_state = chat.invoke("Ilu studentów studiuje na AGH?")
    print(temp_state["chat_history"][-1].content)
