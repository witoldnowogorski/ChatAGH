from typing import List

from langchain.prompts import PromptTemplate
from langchain_core.messages.utils import get_buffer_string
from langchain.output_parsers import PydanticOutputParser
from langchain_core.runnables import Runnable
from langchain_core.messages import BaseMessage
from pydantic import BaseModel, Field

from chat_graph.agents.base_agent import BaseAgent


QUERY_GENERATION_PROMPT = """
Retrieved context:
{processed_retrieved_context}

You are an AI assistant designed to generate a set of questions to query an external
knowledge base in order to accurately answer the user's question.

CHAT HISTORY:
{chat_history}

TASK:
Your task is to analyze the full conversation history as well as retrieved context and formulate well-phrased, long,
comprehensive question to external knowledge base for which answers will be helpful in answering LAST user question.

RESPONSE FORMAT:
Return the question in a valid JSON format as follows:

{{"question": "YOUR QUESTION HERE"}}}}
Do not include explanations, commentary, or formatting outside of this strict JSON structure.
"""


class QueryGenerationOutput(BaseModel):
    question: str = Field(..., description="List of generated questions")


class QueryGenerationAgent(BaseAgent):
    def __init__(self, max_questions: int = 3, **kwargs):
        super().__init__(prompt_template=QUERY_GENERATION_PROMPT, **kwargs)
        self.max_questions = max_questions
        self.output_parser = PydanticOutputParser(pydantic_object=QueryGenerationOutput)

        self.prompt = PromptTemplate(
            input_variables=["chat_history", "processed_retrieved_context"],
            template=self.prompt_template
        )

        self.chain: Runnable = self.prompt | self.llm | self.output_parser

    def inference(self, chat_history: List[BaseMessage], processed_retrieved_context: list[str]) -> str:
        result = self.chain.invoke({
            "chat_history": get_buffer_string(chat_history),
            "processed_retrieved_context": processed_retrieved_context,
        })
        return result.question
