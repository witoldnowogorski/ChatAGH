import os
from typing import List

from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from rag.embeddings.base_embeddings import BaseEmbeddings
from rag.utils.utils import retry_on_exception


class GoogleEmbeddings(BaseEmbeddings):
    """
    A class that provides embeddings using Google's Generative AI for documents and queries.

    This class leverages the GoogleGenerativeAIEmbeddings to compute vector representations
    for text data. It supports generating embeddings for both documents and individual queries,
    with a retry mechanism in place for the query embedding function.

    Attributes:
        embeddings (GoogleGenerativeAIEmbeddings): An instance used to generate embeddings for text.
        model (str): The name of the embedding model being used.

    Methods:
        embed_documents(documents: List[Document] | List[str], batch_size: Optional):
            Generates embeddings for a list of Document objects or text strings.
        embed_query(text: str) -> List[float]:
            Generates an embedding for a query string with automatic retries on failure.
    """
    def __init__(self, model_name: str = "text-embedding-004"):
        self.embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        self.model = model_name

    def embed_documents(self, documents: List[Document] | List[str], batch_size=None) -> List[List[float]]:
        if isinstance(documents[0], str):
            texts = documents
        else:
            texts = [doc.page_content for doc in documents]

        return self.embeddings.embed_documents(texts, batch_size=batch_size)

    @retry_on_exception(attempts=3)
    def embed_query(self, text: str) -> List[float]:
        return self.embeddings.embed_query(text)
