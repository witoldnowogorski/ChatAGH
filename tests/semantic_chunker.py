import ast
import os
import random

import dotenv
from llama_index.core.node_parser import SemanticSplitterNodeParser
from llama_index.core.schema import Document
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding

from rag.utils.utils import load_json_data

dotenv.load_dotenv()
api_keys_str = os.getenv("API_KEYS", "[]")
api_keys = ast.literal_eval(api_keys_str)

embed_model = GoogleGenAIEmbedding(
    model_name="text-embedding-004",
    api_key=random.choice(api_keys),
)

splitter = SemanticSplitterNodeParser(
    buffer_size=1, breakpoint_percentile_threshold=95, embed_model=embed_model
)

data = load_json_data(
    os.path.join(os.path.dirname(os.path.dirname(__file__)), "data/")
)

documents = [
    Document(
        id_=str(idx),
        text=getattr(doc, "page_content", ""),
        extra_info=getattr(doc, "metadata", {}),
    )
    for idx, doc in enumerate(data[:10])
]

nodes = splitter.get_nodes_from_documents(documents)
print(nodes)
