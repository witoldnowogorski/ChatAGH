from typing import List

from langchain.prompts import PromptTemplate
from langchain_core.documents import Document
from langchain.output_parsers import PydanticOutputParser
from langchain_core.runnables import Runnable
from pydantic import BaseModel, Field

from chat_graph.agents.base_agent import BaseAgent


DOCS_ANALYZER_PROMPT = """
You are an AI assistant designed to analyze documents retrieved from knowledge base in the context of a specific question.

Your task is to:
1. Carefully read the documents and understand their content.
2. Determine whether they provide an answer to the user's question.
3. If the documents contain relevant and useful information, summarize it.
4. If they **do not contain enough relevant information**, set the `web_search` flag to true.

Question:
{question}

Retrieved Documents:
{retrieved_docs}

Output ONLY a JSON object with the following structure:
{{
  "summary": "<your summary here>",
  "web_search": <true or false>
}}

- Do NOT include explanations, markdown, or extra text.
- Ensure the JSON is valid and matches the required structure exactly.
- The summary should directly relate to the user's question and be grounded in the retrieved documents.
"""


class DocsAnalyzerOutput(BaseModel):
    summary: str = Field(..., description="Summary of the retrieved documents")
    web_search: bool = Field(..., description="Wheather to use additional web search")


class DocsAnalyzerAgent(BaseAgent):
    def __init__(self):
        super().__init__(prompt_template=DOCS_ANALYZER_PROMPT)
        self.output_parser = PydanticOutputParser(pydantic_object=DocsAnalyzerOutput)

        self.prompt = PromptTemplate(
            input_variables=["question", "retrieved_docs"],
            template=self.prompt_template
        )

        self.chain: Runnable = self.prompt | self.llm | self.output_parser

    def inference(self, question: str, retrieved_docs: List[Document]):
        result = self.chain.invoke({
            "question": question,
            "retrieved_docs": retrieved_docs
        })
        return result
