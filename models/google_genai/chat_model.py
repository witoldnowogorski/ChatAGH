from models.google_genai.base_model import BaseGoogleModel

CHAT_PROMPT_TEMPLATE = f"""
Your are an helpful AI assistant. Your answer users messages in polite way.
Provide next message in the chat.

CHAT:
 {{chat_history}}
 
AI Assistant:
"""

MESSAGE_TYPE_MAPPING = {
    "ai": "AI Assistant",
    "human": "User"
}


class ChatModel(BaseGoogleModel):
    def __init__(self):
        super().__init__(prompt_template=CHAT_PROMPT_TEMPLATE)

    def generate(self, **kwargs):
        formtted_history = "\n\n".join([
            f"{MESSAGE_TYPE_MAPPING[message.type]}: \n {message.content}"
            for message in kwargs.get("chat_history")
        ])
        prompt = self.prompt_template.format(chat_history=formtted_history)

        return self._inference([prompt])
