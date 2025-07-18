from concurrent.futures import ThreadPoolExecutor, as_completed

from chat_graph.agents.docs_analyzer_agent import DocsAnalyzerAgent
from chat_graph.nodes.base_node import BaseNode
from chat_graph.states import ChatState
from chat_graph.utils import logger


class DocsAnalyzerNode(BaseNode):
    def __init__(self):
        self.agent = DocsAnalyzerAgent()

    def __call__(self, state: ChatState) -> ChatState:
        logger.info("Analyzing retrieved context ...")

        query = state["search_query"]
        retrieved_chunks = state["retrieved_chunks"]

        def run_inference(url, docs):
            formated_docs = "\n\n\n".join([d["text"] for d in docs])
            analyzed = self.agent.run(question=query, retrieved_docs=formated_docs)

            logger.info("{}, relevant: {}".format(url, analyzed.relevant))

            if analyzed.relevant:
                return url, analyzed.summary

            return None

        analyzed_context = {}
        with ThreadPoolExecutor() as executor:
            futures = {
                executor.submit(run_inference, url, docs): url
                for url, docs in retrieved_chunks.items()
            }

            for future in as_completed(futures):
                result = future.result()
                if result:
                    url, summary = result
                    analyzed_context[url] = summary

        state["analyzed_context"] = analyzed_context

        return state

