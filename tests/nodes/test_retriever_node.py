from langchain_core.messages import HumanMessage

from chat_graph.nodes.retriever_node import RetrieverNode
from chat_graph.state import ChatState


class TestRetrieverNode:
    def test_retriever_node(self):
        num_retrieved_chunks = 10
        questions = [
            "Jak wyglądają zasady rekrutacji na AGH?",
            "Jak przejśc proces rekrutacji na agh?"
        ]

        node = RetrieverNode(num_chunks=num_retrieved_chunks)
        state = ChatState(
            messages=[HumanMessage("Jak zostać studentem AGH?")],
            questions=questions,
            use_rag=True,
            retrieved_docs=[],
            summaries=[],
            web_search_flags=[]
        )

        result_state = node(state)
        assert len(result_state["retrieved_docs"]) == len(questions)
        for rdocs in result_state["retrieved_docs"]:
            assert len(rdocs) == num_retrieved_chunks