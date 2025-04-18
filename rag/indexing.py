import ast
import os
import random

from dotenv import load_dotenv
from langchain_core.documents.base import Document as LangChainDocument
from llama_index.core.node_parser import SemanticSplitterNodeParser
from llama_index.core.schema import Document
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding

from rag.utils.utils import load_json_data
from rag.vector_store.milvus_hybrid_search import MilvusHybridSearch

ENV_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env")
DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data/")


def indexing(
    data_path,
    collection_name,
    chunk_size=1000,
    chunk_overlap=100,
    max_vectors=None,
):
    """
    Index documents from a single data path into a specific vector store collection

    Args:
        data_path (str): Path to the data file
        collection_name (str): Name of the collection in vector store
        chunk_size (int): Size of chunks for document splitting
        chunk_overlap (int): Overlap between chunks

    Returns:
        tuple: (collection_name, number of chunks)
    """
    load_dotenv(dotenv_path=ENV_PATH)
    data = load_json_data(data_path)

    api_keys_str = os.getenv("API_KEYS", "[]")
    api_keys = ast.literal_eval(api_keys_str)

    embed_model = GoogleGenAIEmbedding(
        model_name="text-embedding-004",
        api_key=random.choice(api_keys),
    )

    splitter = SemanticSplitterNodeParser(
        buffer_size=1,
        breakpoint_percentile_threshold=95,
        embed_model=embed_model,
    )

    documents = [
        Document(
            id_=str(idx),
            text=getattr(doc, "page_content", ""),
            extra_info=getattr(doc, "metadata", {}),
        )
        for idx, doc in enumerate(data)
    ]

    nodes = splitter.get_nodes_from_documents(documents)

    if max_vectors:
        nodes = nodes[:max_vectors]

    documents = [
        LangChainDocument(page_content=node.text, metadata=node.metadata)
        for node in nodes
    ]

    print(f"Generated {len(nodes)} nodes from {data_path}")

    vector_store = MilvusHybridSearch(collection_name)

    # Czemu zaczynamy tu od 25500?
    vector_store.indexing(documents[25500:])

    print(f"Indexed to collection: {collection_name}")
    return [(collection_name, len(documents))]


if __name__ == "__main__":
    i = 5
    result = indexing(
        DATA_PATH,
        "chatagh",
        chunk_size=1500,
        chunk_overlap=0,
    )

    print("\nIndexing Summary:")
    for collection_name, count in result:
        print(f"Collection '{collection_name}': {count} chunks")
