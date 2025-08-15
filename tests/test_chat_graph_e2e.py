from langchain_core.messages import AIMessage, HumanMessage

from chat_graph.graph import ChatGraph


class TestChatGraphE2E:
    def test_chat_graph_e2e(self):
        chat = ChatGraph(initial_retrieved_chunks=5, window_size=1, num_augmentation_chunks=3)
        chat_state = chat.invoke("Hej")

        assert not chat_state["rag_decision"]

        chat_state = chat.invoke("Jak zostać studentem?")

        assert chat_state["rag_decision"]
        assert isinstance(chat_state["search_query"], str)
        assert isinstance(chat_state["retrieved_chunks"], dict)
        assert isinstance(chat_state["analyzed_context"], dict)
        assert isinstance(chat_state["processed_retrieved_context"], str)

        chat_history = chat_state["chat_history"]
        assert isinstance(chat_history, list)
        for i in range(len(chat_history)):
            expexcted_message = AIMessage if i % 2 else HumanMessage
            assert isinstance(chat_history[i], expexcted_message)
