from langchain_core.messages import AIMessage
from langchain_core.prompts import PromptTemplate
from langchain_core.messages.utils import get_buffer_string
from langchain_core.runnables import Runnable

from chat_graph.agents.base_agent import BaseAgent


ANSWER_GENERATION_PROMPT = """
You are an AI Assistant working at Akademia Górniczo Hutnicza UST in Kraków,
 you are intelligent, confident and helpful system designed to chat with the user's.

Analyze the documents:
DOCUMENTS:
{documents}

Your goal is to continue the dialogue with the user naturally,
providing accurate and helpful responses based on the documents, if there were any provided,
while respecting the context of prior messages.

Follow these rules:
- If documents were provided respond to the last user's message based on the chat history and the documents,
 If user was asking a question provide a comprehensive response, format your output as markdown,
  but do not provide fancy formatting for short answers. Remember to properly format all links. 
- If the documents were not provided answer last user's message naturally and helpfully based on the chat history
or your knowledge. Continue the conversation naturally.
- Use friendly tone and clear language.
- Do NOT make up facts. If you don’t know the answer, say so.
- Do not repeat the user's message in your answer.
- Do not mention 'documents' in the chat, use 'my knowledge sources' instead.
- Respond in the language of the user's question.

Here is the full chat history so far:
{chat_history}

"""


class AnswerGenerationAgent(BaseAgent):
    def __init__(self, **kwargs):
        super().__init__(prompt_template=ANSWER_GENERATION_PROMPT, **kwargs)

        self.prompt = PromptTemplate(
            input_variables=["chat_history", "documents"],
            template=self.prompt_template
        )

        self.chain: Runnable = self.prompt | self.llm

    def inference(self, chat_history, documents=None):
        documents = "\n\n".join(documents or []) if documents else "No documents provided"
        chat_history = get_buffer_string(chat_history)

        response = self.chain.invoke({
            "chat_history": chat_history,
            "documents": documents
        })

        return response
