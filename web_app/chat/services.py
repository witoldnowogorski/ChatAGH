from __future__ import annotations
from django.db import transaction
from .models import Conversation, Message

def create_conversation_for(user) -> Conversation:
    """Create a new conversation for the user."""
    return Conversation.objects.create(user=user, title="")

def add_user_message(conv: Conversation, text: str) -> Message:
    """Append a user message and auto-title the conversation if empty."""
    with transaction.atomic():
        msg = Message.objects.create(conversation=conv, role=Message.ROLE_USER, content=text)
        if not conv.title:
            conv.title = text.strip()[:80] or "New Conversation"
            conv.save(update_fields=["title", "updated_at"])
    return msg

def add_assistant_message(conv: Conversation, text: str) -> Message:
    """Append an assistant message."""
    return Message.objects.create(conversation=conv, role=Message.ROLE_ASSISTANT, content=text)
