from typing import TypedDict

from langchain_core.messages import BaseMessage


class ChatState(TypedDict):
    chat_history: list[BaseMessage]
    processed_retrieved_context: str | None
    rag_decision: bool
    search_query: str | None
    analyzed_context: dict | None
    retrieved_chunks: dict | None
