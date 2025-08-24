from typing import List
from langchain_core.documents import Document
from pymongo import MongoClient, TEXT
import torch

from chat_graph.utils import log_execution_time
from chat_graph.utils import embedding_model


class MongoDBVectorStore:
    def __init__(self, mongo_client: MongoClient, db_name: str, collection_name: str):
        self.collection = mongo_client[db_name][collection_name]
        self.dense_model = embedding_model
        self._ensure_indexes()

    def _ensure_indexes(self):
        self.collection.create_index([("text", TEXT)])

    def indexing(self, documents: List[Document], batch_size: int = 500):
        results = []
        total = len(documents)

        for i in range(0, total, batch_size):
            batch = documents[i:i+batch_size]
            texts = [doc.page_content for doc in batch]
            embeddings = self.dense_model.encode(texts).tolist()

            records = [
                {
                    "text": doc.page_content,
                    "metadata": doc.metadata,
                    "embedding": emb
                } for doc, emb in zip(batch, embeddings)
            ]

            res = self.collection.insert_many(records)
            results.append(res)
            print(f"Inserted batch {i//batch_size+1}: {len(records)} documents")

        return results

    @log_execution_time
    def search(self, query: str, k: int = 5) -> List[Document]:
        query_vector = self.dense_model.encode(query).tolist()

        pipeline = [
            {
                "$vectorSearch": {
                    "index": "vector_search_index",
                    "path": "embedding",
                    "queryVector": query_vector,
                    "numCandidates": 100,
                    "limit": k
                }
            },
            {
                "$project": {
                    "text": 1,
                    "metadata": 1,
                    "score": {"$meta": "vectorSearchScore"}
                }
            }
        ]

        results = self.collection.aggregate(pipeline)
        return [
            Document(
                page_content=res["text"],
                metadata=res.get("metadata", {}) | {"id": res.get("_id")}
            )
            for res in results
        ]

