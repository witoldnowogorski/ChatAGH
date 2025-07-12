from typing import List

from langchain_core.documents import Document
from langchain_core.messages import BaseMessage
from langchain_core.messages.utils import get_buffer_string
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import Runnable
from pydantic import BaseModel, Field

from chat_graph.agents.base_agent import BaseAgent
from chat_graph.utils import retry_on_exception


DOCS_ANALYZER_PROMPT = """
EXTERNAL KNOWLEDGE:
{retrieved_docs}

SYSTEM INSTRUCTION:
You are an AI assistant designed to evaluate the relevance of external knowledge in response to a user query. Above you were provided with a retrieved documents (text containing potentially relevant information).
Your task is to:

Determine whether the external knowledge contains any important and relevant information that can help answer the user's query.
If relevant, provide a comprehensive summary of all such information found in the external knowledge.
If not relevant, do not return a summary.
Respond in strict JSON format with the following structure:

{{
  "relevant": <true | false>,
  "summary": <string | null>
}}
Rules:

An external knowledge is considered relevant only if it contains any non-trivial information that is both important and useful for addressing the user's query.
Set "summary" to null if the knowledge is not relevant.
Be comprehensive, factual, contain all information that might be related to user query and avoid speculation.

User query:
{question}
"""

class DocsAnalyzerOutput(BaseModel):
    relevant: bool = Field(..., description="Wheather the document is relevant")
    summary: str | None = Field(..., description="Summary of the retrieved documents")


class DocsAnalyzerAgent(BaseAgent):
    def __init__(self):
        super().__init__(prompt_template=DOCS_ANALYZER_PROMPT)
        self.output_parser = PydanticOutputParser(pydantic_object=DocsAnalyzerOutput)

        self.prompt = PromptTemplate(
            input_variables=["question", "retrieved_docs"],
            template=self.prompt_template
        )

        self.chain: Runnable = self.prompt | self.llm | self.output_parser

    @retry_on_exception(delay=1, attempts=4, backoff=5)
    def inference(self, question: str, retrieved_docs: str, **kwargs):
        result = self.chain.invoke({
            "question": question,
            "retrieved_docs": retrieved_docs
        })
        return result
