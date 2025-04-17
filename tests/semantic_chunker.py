import os
import ast
import random

import dotenv
from llama_index.core.node_parser import SemanticSplitterNodeParser
from llama_index.embeddings.openai import OpenAIEmbedding
import os

from rag.utils.utils import load_json_data

dotenv.load_dotenv()

embed_model = OpenAIEmbedding()
splitter = SemanticSplitterNodeParser(
    buffer_size=1, breakpoint_percentile_threshold=95, embed_model=embed_model
)

os.chdir("../")
data = load_json_data('data/')
documents = data[:10]

# tu wywala jakiś błąd związany z tym, że nasze dokumenty nie mają pola "id_"
nodes = splitter.get_nodes_from_documents(documents)
