from langchain_core.messages import HumanMessage

from chat_graph.nodes import QuestionsGenerationNode
from chat_graph.state import ChatState


class TestQuestionsGenerationNode:
    def test_questions_generation_node(self):
        node = QuestionsGenerationNode()
        state = ChatState(
            messages=[HumanMessage("Jak zostać studentem AGH?")],
            retrieved_docs=[],
            summaries=[],
            questions=[],
            use_rag=True,
            web_search_flags=[]
        )
        final_state = node(state)

        assert 0 < len(final_state["questions"]) < 5