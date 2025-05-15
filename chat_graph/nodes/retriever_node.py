from chat_graph.nodes.base_node import BaseNode
from chat_graph.state import ChatState
from datastores.vector_store.milvus_hybrid_search import MilvusHybridSearch

MILVUS_COLLECTION_NAME = "chatagh"


class RetrieverNode(BaseNode):
    def __init__(self, num_chunks=10):
        self.vector_store = MilvusHybridSearch(MILVUS_COLLECTION_NAME)
        self.num_chunks = num_chunks

    def __call__(self, state: ChatState) -> ChatState:
        questions = state["questions"]
        retrieved_docs = []
        for question in questions:
            res = self.vector_store.search(question, self.num_chunks)
            retrieved_docs.append(res)

        return ChatState(
            messages=state["messages"],
            use_rag=state["use_rag"],
            retrieved_docs=retrieved_docs,
            questions=questions,
            summaries=state['summaries'],
            web_search_flags=state['web_search_flags'],
        )