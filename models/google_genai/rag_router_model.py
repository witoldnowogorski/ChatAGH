from typing import List

from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda, RunnableMap
from pydantic import BaseModel

from models.google_genai import ChatModel


class RAGRouterOutput(BaseModel):
    use_rag: bool


class RAGRouterModel:
    def __init__(self):
        self.parser = PydanticOutputParser(pydantic_object=RAGRouterOutput)

        self.prompt = PromptTemplate.from_template(
            """
            You are an AI assistant helping decide if a user message requires external knowledge to answer.
            
            Conversation history:
            {chat_history}
            
            Based on this history, should the system use Retrieval-Augmented Generation (RAG) to answer the next message?
            
            Respond ONLY with a JSON object in the form:
            {{ "use_rag": true }} or {{ "use_rag": false }}
            """.strip()
        )

        self.model = ChatModel()

        # Chain: Format prompt → Call model → Parse output
        self.chain = (
            RunnableMap({"chat_history": lambda x: self._format_chat_history(x["messages"])})
            | self.prompt
            | RunnableLambda(lambda prompt: self.model.generate(query=prompt, context=""))
            | self.parser
        )

    def _format_chat_history(self, messages: List[dict]) -> str:
        return "\n".join(f"{m['type'].upper()}: {m['content']}" for m in messages)

    def generate(self, chat_history: List[dict]) -> bool:
        result = self.chain.invoke({"messages": chat_history})
        return result.use_rag
