from __future__ import annotations
from typing import Iterable
from django.db.models import QuerySet
from .models import Conversation, Message

def user_conversations(user) -> QuerySet[Conversation]:
    """Return conversations owned by the user."""
    return Conversation.objects.filter(user=user).only("id", "title", "updated_at").order_by("-updated_at")

def conversation_messages(conv: Conversation) -> QuerySet[Message]:
    """Return messages in chronological order for a conversation."""
    return conv.messages.select_related(None).only("id", "role", "content", "created_at").order_by("created_at")

def as_history_tuples(messages: Iterable[Message]) -> list[tuple[str, str]]:
    """
    Convert Message objects to (role, content) tuples for backends that accept history.
    """
    return [(m.role, m.content) for m in messages]
