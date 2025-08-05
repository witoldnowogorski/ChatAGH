import os
from typing import List

import torch
from langchain_core.documents import Document
from pymongo import MongoClient, TEXT
from sentence_transformers import SentenceTransformer

from chat_graph.utils import log_execution_time


class MongoDBVectorStore:
    def __init__(self, db_name: str, collection_name: str):
        uri = os.environ["MONGODB_ATLAS_URI"]
        self.client = MongoClient(uri)
        self.db = self.client[db_name]
        self.collection = self.db[collection_name]

        if torch.cuda.is_available():
            self.dense_model = SentenceTransformer("intfloat/multilingual-e5-large", device="cuda")
        else:
            self.dense_model = SentenceTransformer("intfloat/multilingual-e5-large")

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

