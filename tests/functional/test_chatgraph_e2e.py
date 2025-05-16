from pathlib import Path

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage

from chat_graph.graph import ChatGraph

root_path = Path(__file__).parent.parent
dotenv_path = root_path / ".env"
load_dotenv(dotenv_path=dotenv_path)

class TestChatgraphE2E:
    def test_chatgraph_e2e(self):
        chat_graph = ChatGraph()

        chat_graph.invoke("Hej")
        chat_graph.invoke("Jak zostać studentem AGH?")
        final_state = chat_graph.invoke("Dzięki")

        messages = final_state["messages"]
        for msg in messages:
            print(msg)
        assert len(messages) == 6
        for i in range(0, len(messages), 2):
            assert type(messages[i]) == HumanMessage
            assert type(messages[i + 1]) == AIMessage

