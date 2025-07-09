from pymongo import MongoClient

from typing import List, Dict
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from datastores.vector_store.utils import bm25_similarity
import numpy as np
from chat_graph.states import ChatState

uri = "mongodb+srv://nowogorskiwitold:XdienEZNuuqLygzX@chatagh.ilvf5bc.mongodb.net/?retryWrites=true&w=majority&appName=ChatAGH"

DOC_TEMPLATE = " -> Document (url: {}): \n{}\n\n"

class GraphAugmentationNode:
    def __init__(self, num_related_chunks_for_doc: int = 5):
        self.client = MongoClient(uri, tlsAllowInvalidCertificates=True)
        self.embedding_model = SentenceTransformer("intfloat/multilingual-e5-large")
        self.num_related_chunks_for_doc = num_related_chunks_for_doc

    def __call__(self, state: ChatState) -> ChatState:
        analyzed_context = state["analyzed_context"]

        context = []
        for url, summary in analyzed_context.items():
            context.append(self.process_single_url(url, summary))

        state["processed_retrieved_context"] = "\n\n\n".join(context)

        return state

    def process_single_url(self, url, summary) -> str:
        related_urls = self.find_related_urls(url)

        chunks = []
        for url in related_urls:
            chunks.extend(self.get_chunks_for_url(url))

        if chunks:
            summary_embedding = self.embedding_model.encode([summary]).tolist()

            top_related_chunks = self.combined_similarity(
                summary=summary,
                summary_embedding=summary_embedding,
                chunks=chunks,
                bm25_similarity_func=bm25_similarity,
                bm25_weight=0,
                top_n=self.num_related_chunks_for_doc,
            )

            source_document = DOC_TEMPLATE.format(url, summary)

            formated_chunks = []
            for c in top_related_chunks:
                formated_chunks.append(
                    DOC_TEMPLATE.format(
                        c["chunk"]["metadata"]["url"],
                        c["chunk"]["text"]
                    )
                )

            formated_context = "\n\n".join(formated_chunks + [source_document])
        else:
            formated_context = DOC_TEMPLATE.format(url, summary)

        return formated_context

    def find_related_urls(self, node: str):
        """Get list of all related urls for a given url."""
        collection = self.client["chat_agh"]["edges"]

        results = collection.find({
            "$or": [
                {"source": node},
                {"target": node}
            ]
        })

        related_nodes = set()
        for doc in results:
            if doc["source"] == node:
                related_nodes.add(doc["target"])
            elif doc["target"] == node:
                related_nodes.add(doc["source"])

        return list(related_nodes)

    def get_chunks_for_url(self, url: str):
        """Returns all chunks for a given url."""
        collection = self.client["chat_agh"]["chunks"]
        print(collection)
        return list(collection.find({"metadata.url": url}))

    def combined_similarity(
        self,
        summary: str,
        summary_embedding: List[float],
        chunks: List[Dict],
        bm25_similarity_func,
        bm25_weight: float = 0.5,
        top_n: int = 5
    ) -> List[Dict]:
        """
        Combines BM25 and embedding-based similarity to rank chunks.

        Args:
            summary: The query or summary text.
            summary_embedding: Precomputed embedding of the summary.
            chunks: List of chunks, each with 'text' and 'embedding' fields.
            bm25_similarity_func: A function that takes two strings and returns BM25 similarity.
            bm25_weight: Weight for BM25 similarity in the final score (0 ≤ bm25_weight ≤ 1).
            top_n: Number of top chunks to return.

        Returns:
            List of top_n chunks with their combined similarity score.
        """
        summary_embedding = np.array(summary_embedding).reshape(1, -1)
        chunk_embeddings = np.array([chunk["embedding"] for chunk in chunks])

        embedding_similarities = cosine_similarity(summary_embedding, chunk_embeddings)[0]

        combined_scores = []
        for idx, chunk in enumerate(chunks):
            bm25_score = bm25_similarity_func(summary, chunk["text"])
            embedding_score = embedding_similarities[idx]
            combined_score = bm25_weight * bm25_score + (1 - bm25_weight) * embedding_score
            combined_scores.append((idx, combined_score))

        top_indices = sorted(combined_scores, key=lambda x: x[1], reverse=True)[:top_n]

        return [
            {
                "chunk": chunks[i],
                "combined_score": score,
                "bm25_score": bm25_similarity_func(summary, chunks[i]["text"]),
                "embedding_score": embedding_similarities[i],
            }
            for i, score in top_indices
        ]