from llama_index.core.base.embeddings.base import BaseEmbedding

Embedding = list[float]

class GoogleEmbedding(BaseEmbedding):
    def _get_query_embedding(self, query: str) -> Embedding:
        """
        Embed the input query synchronously.
        """

    async def _aget_query_embedding(self, query: str) -> Embedding:
        """
        Embed the input query asynchronously.
        """
    def _get_text_embedding(self, text: str) -> Embedding:
        """
        Embed the input text synchronously.
        """
