from chat_graph.agents.docs_analyzer_agent import DocsAnalyzerAgent
from chat_graph.nodes.base_node import BaseNode
from chat_graph.states import ChatState


class DocsAnalyzerNode(BaseNode):
    def __init__(self):
        self.agent = DocsAnalyzerAgent()

    def __call__(self, state: ChatState) -> ChatState:
        query = state["search_query"]
        retrieved_chunks = state["retrieved_chunks"]

        analyzed_context = {}
        for url, docs in retrieved_chunks.items():
            formated_docs = "\n\n\n".join([d["text"] for d in docs])
            analyzed = self.agent.inference(question=query, retrieved_docs=formated_docs)

            if analyzed.relevant:
                analyzed_context[url] = analyzed.summary

        state["analyzed_context"] = analyzed_context

        return state