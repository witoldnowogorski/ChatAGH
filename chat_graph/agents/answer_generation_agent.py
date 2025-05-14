from langchain_core.prompts import PromptTemplate
from langchain_core.messages.utils import get_buffer_string
from langchain_core.runnables import Runnable

from chat_graph.agents.base_agent import BaseAgent


ANSWER_GENERATION_PROMPT = """
You are an intelligent, polite, and helpful AI assistant engaging in a multi-turn conversation with a user.

Your goal is to continue the dialogue naturally, providing accurate and helpful responses while respecting the context of prior messages.

Follow these rules:
- Stay on topic and be concise.
- Use professional tone and clear language.
- If the user asks a factual or knowledge-based question, respond based on the context of the conversation.
- Do NOT make up facts. If you don’t know the answer, say so.
- Do not repeat the user's message in your answer.
- Do not refer to yourself as a language model.

Here is the full chat history so far:
{chat_history}

Now generate the assistant’s next response.
Respond with only the content of your reply. Do not include system messages or metadata.
"""


class AnswerGenerationAgent(BaseAgent):
    def __init__(self, **kwargs):
        super().__init__(prompt_template=ANSWER_GENERATION_PROMPT, **kwargs)

        self.prompt = PromptTemplate(
            input_variables=["chat_history"],
            template=self.prompt_template
        )

        self.chain: Runnable = self.prompt | self.llm

    def inference(self, chat_history):
        chat_history = get_buffer_string(chat_history)
        return self.chain.invoke(chat_history)
