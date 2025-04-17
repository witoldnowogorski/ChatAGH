import os
import ast
import random
from llama_index.core.schema import Document

import dotenv
from llama_index.core.node_parser import SemanticSplitterNodeParser
from llama_index.embeddings.openai import OpenAIEmbedding
import os

from rag.utils.utils import load_json_data
from rag.embeddings.google_embedding import GoogleEmbedding

dotenv.load_dotenv()

embed_model = GoogleEmbedding()
splitter = SemanticSplitterNodeParser(
    buffer_size=1, breakpoint_percentile_threshold=95, embed_model=embed_model
)

os.chdir("../")
data = load_json_data('data/')
documents = [
    Document(
        id_=str(idx),
        text=getattr(doc, "page_content", ""),
        extra_info=getattr(doc, "metadata", {})
    )
    for idx, doc in enumerate(data[:10])
]

nodes = splitter.get_nodes_from_documents(documents)
