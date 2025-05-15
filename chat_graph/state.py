from typing import List, TypedDict

from langchain_core.messages import BaseMessage
from langchain_core.documents import Document


class ChatState(TypedDict):
    messages: List[BaseMessage]
    questions: List[str]
    use_rag: bool
    retrieved_docs: List[List[Document]]
    web_search_flags: List[bool]
    summaries: List[str]