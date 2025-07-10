import os
from pymongo import MongoClient

from chat_graph.nodes.base_node import BaseNode
from chat_graph.states import ChatState

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
        query = state["search_query"]
        retrieved_chunks = self.vector_store.search(query, k=self.initial_retrieved_chunks)
        aggregated_docs = self.aggregate_by_document(retrieved_chunks)
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

        query = {
            "metadata.url": document.metadata["url"],
            "metadata.sequence_number": {
                "$gte": document.metadata["sequence_number"] - self.window_size,
                "$lte": document.metadata["sequence_number"] + self.window_size
            }
        }

        pipeline = [
            {"$match": query},
            {"$group": {
                "_id": {
                    "url": "$metadata.url",
                    "sequence_number": "$metadata.sequence_number"
                },
                "doc": {"$first": "$$ROOT"}
            }},
            {"$replaceRoot": {"newRoot": "$doc"}}
        ]

        results = list(collection.aggregate(pipeline))
        return results

    def get_chunks_windows(self, urls):
        """returns windows of chunks for each of the aggregated url's"""
        retrieved_docs = {}
        for url in urls.keys():

            url_docs = []
            seen = set()
            for doc in urls[url]:
                docs_window = self.retrieve_chunks_window(doc)
                for d in docs_window:
                    if (key := (d["metadata"]["url"], d["metadata"]["sequence_number"])) not in seen:
                        url_docs.append(d)
                        seen.add(key)
                    else:
                        continue

            retrieved_docs[url] = sorted(url_docs, key=lambda d: d["metadata"]["sequence_number"])

        return retrieved_docs