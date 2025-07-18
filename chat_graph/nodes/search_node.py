import os
from pymongo import MongoClient

from chat_graph.nodes.base_node import BaseNode
from chat_graph.states import ChatState
from chat_graph.utils import log_execution_time, logger
from datastores.vector_store.mongo_atlas_vector_store import MongoDBVectorStore

class SearchNode(BaseNode):
    def __init__(
        self,
        initial_retrieved_chunks: int = 8,
        window_size: int = 3
    ):
        uri = os.environ.get("MONGODB_ATLAS_URI")
        self.initial_retrieved_chunks = initial_retrieved_chunks
        self.window_size = window_size

        self.vector_store = MongoDBVectorStore("chat_agh", "chunks")
        self.mongo_client = MongoClient(uri, tlsAllowInvalidCertificates=True)

    def __call__(self, state: ChatState) -> ChatState:
        logger.info("Performing similarity search on vector store ...")

        query = state["search_query"]
        retrieved_chunks = self.vector_store.search(query, k=self.initial_retrieved_chunks)

        aggregated_docs = self.aggregate_by_document(retrieved_chunks)

        logger.info(
            "Retrieved {} documents, source urls: {}".format(len(retrieved_chunks), aggregated_docs.keys())
        )

        chunks_windows = self.get_chunks_windows(aggregated_docs)

        state["retrieved_chunks"] = chunks_windows

        return state

    @staticmethod
    def aggregate_by_document(retrieved_chunks):
        """Group retrieved chunks by source url's"""
        urls = {}
        for doc in retrieved_chunks:
            if (url := doc.metadata["url"]) in urls:
                urls[url].append(doc)
            else:
                urls[url] = [doc]
        return urls

    def retrieve_chunks_window(self, document):
        """returns window of chunks for given chunk"""
        db = self.mongo_client["chat_agh"]
        collection = db["chunks"]

        collection.create_index([("metadata.url", 1), ("metadata.sequence_number", 1)])

        query = {
            "metadata.url": document.metadata["url"],
            "metadata.sequence_number": {
                "$gte": document.metadata["sequence_number"] - self.window_size,
                "$lte": document.metadata["sequence_number"] + self.window_size
            }
        }

        results = list(collection.find(query))
        return results

    @log_execution_time
    def get_chunks_windows(self, urls):
        """Returns chunks for specific sequence_numbers per URL (batched and deduplicated)."""
        retrieved_docs = {}

        for url, docs in urls.items():
            seq_numbers = set()
            for doc in docs:
                seq = doc.metadata["sequence_number"]
                window_range = range(seq - self.window_size, seq + self.window_size + 1)
                seq_numbers.update(window_range)

            db = self.mongo_client["chat_agh"]
            collection = db["chunks"]

            query = {
                "metadata.url": url,
                "metadata.sequence_number": {"$in": list(seq_numbers)}
            }
            results = collection.find(query)

            seen = set()
            unique_docs = []
            for d in results:
                key = (d["metadata"]["url"], d["metadata"]["sequence_number"])
                if key not in seen:
                    seen.add(key)
                    unique_docs.append(d)

            retrieved_docs[url] = sorted(unique_docs, key=lambda d: d["metadata"]["sequence_number"])

        return retrieved_docs
