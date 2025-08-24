from __future__ import annotations
from django.conf import settings
from django.db import models
from langchain_core.messages import AIMessage, HumanMessage


class Conversation(models.Model):
    """
    A user's conversation thread.

    Title is auto-filled from the first user message if blank.
    """
    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name="conversations")
    title = models.CharField(max_length=200, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ["-updated_at"]

    def __str__(self) -> str:
        return self.title or f"Conversation {self.pk}"

class Message(models.Model):
    """
    A single chat message within a conversation.

    role: 'user' or 'assistant'
    """
    ROLE_USER = "user"
    ROLE_ASSISTANT = "assistant"
    ROLES = [(ROLE_USER, "User"), (ROLE_ASSISTANT, "Assistant")]

    conversation = models.ForeignKey(Conversation, on_delete=models.CASCADE, related_name="messages")
    role = models.CharField(max_length=10, choices=ROLES)
    content = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["created_at"]

    def __str__(self) -> str:
        return f"{self.role}: {self.content[:40]}"

    def to_langchain_message(self):
        return AIMessage(str(self.content)) if self.role == Message.ROLE_ASSISTANT else HumanMessage(str(self.content))
