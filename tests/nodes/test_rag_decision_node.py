from langchain_core.messages import HumanMessage

from chat_graph.nodes.rag_decision_node import RAGDecisionNode
from chat_graph.state import ChatState


class TestRagDecisionNode:
    def test_rag_decision_node_negative(self):
        node = RAGDecisionNode()
        state = ChatState(
            messages=[HumanMessage("Cześć")],
            questions=[],
            use_rag=False,
            retrieved_docs=[],
            web_search_flags=[],
            summaries=[]
        )
        final_state = node(state)
        assert final_state["use_rag"] is False

    def test_rag_decision_node_positive(self):
        node = RAGDecisionNode()
        state = ChatState(
            messages=[HumanMessage("Jakie domy studenckie znajdują się na miasteczku AGH?")],
            questions=[],
            use_rag=False,
            retrieved_docs=[],
            web_search_flags=[],
            summaries=[]
        )
        final_state = node(state)
        assert final_state["use_rag"] is True