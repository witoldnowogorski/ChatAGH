from typing import List

from langchain.prompts import PromptTemplate
from langchain_core.messages.utils import get_buffer_string
from langchain.output_parsers import PydanticOutputParser
from langchain_core.runnables import Runnable
from langchain_core.messages import BaseMessage
from pydantic import BaseModel, Field

from chat_graph.agents.base_agent import BaseAgent


QUESTIONS_GENERATION_PROMPT = """
You are an AI assistant designed to generate a set of questions to query an external
knowledge base in order to accurately answer the user's question.

CHAT HISTORY:
{chat_history}

TASK:
Your task is to analyze the full conversation history and formulate a set of
well-phrased questions to external knowledge base for which answers will be helpful in answering LAST user question.

Number of questions should be between 1 and 5, and should match the complexity of the user's question.

RESPONSE FORMAT:
Return the questions in a valid JSON format as follows:

{{"questions": ["question1", "question2", "question3", ...]}}
Do not include explanations, commentary, or formatting outside of this strict JSON structure.
"""


class QuestionsOutput(BaseModel):
    questions: List[str] = Field(..., description="List of generated questions")


class QuestionsGenerationAgent(BaseAgent):
    def __init__(self, max_questions: int = 3, **kwargs):
        super().__init__(prompt_template=QUESTIONS_GENERATION_PROMPT, **kwargs)
        self.max_questions = max_questions
        self.output_parser = PydanticOutputParser(pydantic_object=QuestionsOutput)

        self.prompt = PromptTemplate(
            input_variables=["chat_history"],
            template=self.prompt_template
        )

        self.chain: Runnable = self.prompt | self.llm | self.output_parser

    def inference(self, chat_history: List[BaseMessage]) -> List[str]:
        result = self.chain.invoke({
            "chat_history": get_buffer_string(chat_history),
        })
        return result.questions[:self.max_questions]
