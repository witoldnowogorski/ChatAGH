from typing import List, TypedDict


class GraphState(TypedDict):
    question: str,
    generation: str,
    web_search: bool,
    Dosuments