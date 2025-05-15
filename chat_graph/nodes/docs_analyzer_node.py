from chat_graph.nodes.base_node import BaseNode
from chat_graph.state import ChatState
from chat_graph.agents.docs_analyzer_agent import DocsAnalyzerAgent


class DocsAnalyzerNode(BaseNode):
    def __init__(self):
        self.agent = DocsAnalyzerAgent()

    def __call__(self, state: ChatState) -> ChatState:
        from concurrent.futures import ThreadPoolExecutor

        retrieved_docs = state["retrieved_docs"]
        questions = state["questions"]

        def query_agent(pair):
            question, docs = pair
            result = self.agent.inference(question, docs)
            return result.summary, result.web_search

        with ThreadPoolExecutor() as executor:
            results = list(executor.map(query_agent, zip(questions, retrieved_docs)))

        summaries, web_search = zip(*results)
        summaries = list(summaries)
        web_search = list(web_search)

        return ChatState(
            messages=state["messages"],
            use_rag=state["use_rag"],
            retrieved_docs=retrieved_docs,
            questions=questions,
            web_search_flags=web_search,
            summaries=summaries,
        )






