import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
from rag.utils.utils import load_json_data
from rag.chunkers.langchain_chunker import LangChainChunker
from rag.vector_store.milvus_hybrid_search import MilvusHybridSearch

ENV_PATH = ".env"
DATA_PATH = ""


def indexing(data_path, collection_name, chunk_size=1000, chunk_overlap=100, max_vectors=None):
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

    chunker = LangChainChunker(chunk_size, chunk_overlap, remove_duplicates=True)
    chunks = chunker.chunk(data)

    if max_vectors:
        chunks = chunks[:max_vectors]

    print(f"Generated {len(chunks)} chunks from {data_path}")

    vector_store = MilvusHybridSearch(collection_name)
    vector_store.indexing(chunks)

    print(f"Indexed to collection: {collection_name}")
    return (collection_name, len(chunks))


if __name__ == "__main__":
    i = 5
    result = indexing(
        "./data",
        "chatagh",
        chunk_size=1500,
        chunk_overlap=0,
    )

    print("\nIndexing Summary:")
    for collection_name, count in result:
        print(f"Collection '{collection_name}': {count} chunks")
