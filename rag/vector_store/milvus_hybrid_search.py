import os
from typing import List

from langchain_core.documents import Document
from sentence_transformers import SentenceTransformer
from pymilvus import (
    MilvusClient,
    utility,
    DataType,
    AnnSearchRequest,
    RRFRanker,
    Function,
    FunctionType,
    connections
)


class MilvusHybridSearch:
    def __init__(self, collection_name: str):
        uri = os.environ["MILVUS_URI"]
        token = os.environ["MILVUS_TOKEN"]
        connections.connect(
            alias="default",
            uri=uri,
            token=token
        )
        self.collection_name = collection_name
        self.client = MilvusClient(uri=uri, token=token)
        self.dense_embedding_model = SentenceTransformer("intfloat/multilingual-e5-large")

        if not utility.has_collection(self.collection_name):
            self._create_collection()

    def _create_collection(self):
        schema = MilvusClient.create_schema(
            auto_id=False,
            enable_dynamic_field=True,
        )

        schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True, auto_id=True)

        schema.add_field(
            field_name="text",
            datatype=DataType.VARCHAR,
            max_length=5000,
            enable_analyzer=True,
            analyzer_types=["bm25"]
        )
        schema.add_field(
            field_name="metadata",
            datatype=DataType.JSON,
            max_length=2000,
        )

        schema.add_field(field_name="sparse", datatype=DataType.SPARSE_FLOAT_VECTOR)
        schema.add_field(field_name="dense", datatype=DataType.FLOAT_VECTOR, dim=self.dense_embedding_model.get_sentence_embedding_dimension())

        bm25_function = Function(
            name="text_bm25_emb",
            input_field_names=["text"],
            output_field_names=["sparse"],
            function_type=FunctionType.BM25,
        )

        schema.add_function(bm25_function)

        index_params = self.client.prepare_index_params()

        index_params.add_index(
            field_name="dense",
            index_name="dense_index",
            index_type="IVF_FLAT",
            metric_type="IP",
            params={"nlist": 128},
            dimension=768,
        )

        index_params.add_index(
            field_name="sparse",
            index_name="sparse_index",
            index_type="SPARSE_INVERTED_INDEX",
            metric_type="BM25",
            params={"inverted_index_algo": "DAAT_MAXSCORE"},
        )

        self.client.create_collection(
            collection_name=self.collection_name,
            schema=schema,
            index_params=index_params
        )

    def indexing(self, documents: List[Document], batch_size: int = 500):
        """
        Index documents in batches to improve performance and memory management.

        Args:
            documents: List of Document objects to index
            batch_size: Number of documents to process in each batch

        Returns:
            List of results from all batch insertions
        """
        results = []
        total_docs = len(documents)

        for i in range(0, total_docs, batch_size):
            batch_docs = documents[i:i + batch_size]
            batch_contents = [doc.page_content for doc in batch_docs]

            batch_embeddings = self.dense_embedding_model.encode(batch_contents).tolist()
            print(f"Batch {i // batch_size + 1} embedding finished: {len(batch_embeddings)} vectors")

            batch_data = [
                {"text": doc.page_content, "dense": emb, "metadata": doc.metadata}
                for doc, emb in zip(batch_docs, batch_embeddings)
            ]

            batch_result = self.client.insert(
                collection_name=self.collection_name,
                data=batch_data
            )

            results.append(batch_result)
            print(
                f"Inserted batch {i // batch_size + 1}/{(total_docs + batch_size - 1) // batch_size}: {len(batch_docs)} documents")

        print(f"Indexing complete: {total_docs} total documents processed in {len(results)} batches")
        return results

    def search(self, query: str, k: int = 5) -> List[Document]:
        query_embedding = self.dense_embedding_model.encode(query).tolist()

        reqs = [
            AnnSearchRequest(
                data=[query],
                anns_field="sparse",
                param={
                    "metric_type": "BM25"
                },
                limit=k * 2
            ),
            AnnSearchRequest(
                data=[query_embedding],
                anns_field="dense",
                param={
                    "metric_type": "IP",
                    "params": {"nprobe": 10}
                },
                limit=k * 2
            )
        ]

        ranker = RRFRanker(100)

        res = self.client.hybrid_search(
            collection_name=self.collection_name,
            reqs=reqs,
            ranker=ranker,
            limit=k,
            output_fields=["*"]
        )

        retrieved_chunks = [
            Document(page_content=r["entity"]["text"], metadata=r["entity"]["metadata"]) for r in res[0]
        ]

        return retrieved_chunks