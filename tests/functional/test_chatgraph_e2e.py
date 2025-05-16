from pathlib import Path

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage

from chat_graph.graph import ChatGraph

root_path = Path(__file__).parent.parent
dotenv_path = root_path / ".env"
load_dotenv(dotenv_path=dotenv_path)

class TestChatgraphE2E2:
    def test_chatgraph_e2e(self):
        chat_graph = ChatGraph()

        chat_graph.invoke("Hej")
        chat_graph.invoke("Jak zostać studentem AGH?")
        final_state = chat_graph.invoke("Dzięki")

        messages = final_state["messages"]
        assert len(messages) == 4
        assert type(messages[0]) == HumanMessage
        assert type(messages[1]) == AIMessage
        assert type(messages[2]) == HumanMessage
        assert type(messages[3]) == AIMessage

