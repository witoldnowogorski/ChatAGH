from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.documents import Document

from chat_graph.nodes.answer_generation_node import AnswerGenerationNode
from chat_graph.state import ChatState


class TestAnswerGenerationNode:
    def test_answer_generation_node(self):
        node = AnswerGenerationNode()

        state = ChatState(
            messages=[
                HumanMessage("Czesc!"),
                AIMessage("Hej, Jak mogę ci pomóc?"),
                HumanMessage("Jak zostać studentem AGH?"),
            ],
            use_rag=True,
            questions=[],
            retrieved_docs=[
                [Document(page_content="Aby zostać studentem trzeba przejść rekrutację.")],
            ],
            summaries=["Aby zostać studentem trzeba przejść rekrutację."],
            web_search_flags=[False]
        )
        result_state = node(state)
        messages = result_state["messages"]
        assert len(messages) == 4
        assert type(messages[-1]) == AIMessage