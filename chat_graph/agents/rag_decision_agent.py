from typing import List

from langchain_core.messages import BaseMessage
from langchain_core.messages.utils import get_buffer_string
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import Runnable
from pydantic import BaseModel, Field

from chat_graph.agents.base_agent import BaseAgent


RAG_DECISION_PROMPT = """
Context:
{processed_retrieved_context}

You are an AI system responsible for determining whether to use Retrieval-Augmented Generation (RAG)
to help answer a user's question. Consider the entire chat history and provided context and decide if external knowledge is needed.

Your goal is to return a JSON object with the format: {{"use_rag": true}} or {{"use_rag": false}}

Use RAG if:
- The user asks about recent, factual, or document-based information.
- The user refers to something that likely exists in external or domain-specific data.
- The assistant previously expressed uncertainty or gave a vague response.
- The user is asking for a document, link, citation, or source.

Avoid RAG if:
- The query is general knowledge the model likely knows (e.g., definitions, basic facts).
- The user is asking follow-up questions that depend on prior AI responses.
- The response can be confidently generated from prior conversation alone.

Chat History:
{chat_history}

Respond ONLY with a valid JSON object with one boolean field: use_rag
"""

class RAGDecisionOutput(BaseModel):
    use_rag: bool = Field(..., description="Whether to use RAG based on the chat history")


class RAGDecisionAgent(BaseAgent):
    def __init__(self, **kwargs):
        super().__init__(prompt_template=RAG_DECISION_PROMPT, **kwargs)
        self.output_parser = PydanticOutputParser(pydantic_object=RAGDecisionOutput)

        self.prompt = PromptTemplate(
            input_variables=["chat_history", "processed_retrieved_context"],
            template=self.prompt_template
        )

        self.chain: Runnable = self.prompt | self.llm | self.output_parser

    def inference(self, chat_history: List[BaseMessage], processed_retrieved_context: str, **kwargs) -> bool:
        result = self.chain.invoke({
            "chat_history": get_buffer_string(chat_history),
            "processed_retrieved_context": processed_retrieved_context,
        })
        return result.use_rag


if __name__ == "__main__":
    from langchain_core.messages import HumanMessage, AIMessage
    history = [
        HumanMessage("Hi"),
        AIMessage("Hello, How can I help you?"),
        HumanMessage("What is the capital of France?"),
    ]

    decision_agent = RAGDecisionAgent()
    print(decision_agent.run(chat_history=history))

    history.extend([
        HumanMessage("How many faculties are there at AGH?"),
    ])
    print(decision_agent.run(chat_history=history))
