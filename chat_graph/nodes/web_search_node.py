from langchain_community.tools import DuckDuckGoSearchRun

from chat_graph.nodes.base_node import BaseNode
from chat_graph.state import ChatState


class WebSearchNode(BaseNode):
    def __init__(self):
        self.search_engine = DuckDuckGoSearchRun()

    def __call__(self, state: ChatState) -> ChatState:
        web_search_flags = state["web_search_flags"]
        questions = state["questions"]
        summaries = state["summaries"]

        processed_summaries = []
        for summary, web_search_flag, question in zip(summaries, web_search_flags, questions):
            if web_search_flag:
                search_result = self.search_engine.invoke(question)
                processed_summaries.append(search_result)
            else:
                processed_summaries.append(summary)

        return ChatState(
            summaries=processed_summaries,
            questions=questions,
            web_search_flags=web_search_flags,
            retrieved_docs=state["retrieved_docs"],
            use_rag=state["use_rag"],
            messages=state["messages"],
        )

if __name__ == "__main__":
    agent = WebSearchNode()
    print(agent(ChatState(
        summaries=[""],
        questions=["Kto jest rektorem AGH?"],
        web_search_flags=[True],
        retrieved_docs=[],
        use_rag=True,
        messages=[],
    )))

