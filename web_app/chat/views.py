from __future__ import annotations
from typing import Iterator
from django.contrib.auth.decorators import login_required
from django.http import (
    HttpRequest, HttpResponse, JsonResponse, StreamingHttpResponse, Http404
)
from django.shortcuts import get_object_or_404, redirect, render
from django.views.decorators.http import require_http_methods
from django.utils.encoding import iri_to_uri

from chat_graph.graph import ChatGraph
from .models import Conversation, Message
from . import selectors, services


@login_required
def index(request: HttpRequest) -> HttpResponse:
    """
    Default landing: if user has conversations, show the most recent,
    else create a new one.
    """
    conv = request.user.conversations.order_by("-updated_at").first()
    if conv:
        return redirect("chat:conversation", conversation_id=conv.id)
    conv = services.create_conversation_for(request.user)
    return redirect("chat:conversation", conversation_id=conv.id)

@login_required
def conversation_view(request: HttpRequest, conversation_id: int) -> HttpResponse:
    """
    Render a conversation with its messages and a minimal chat UI.
    """
    conv = _user_conversation_or_404(request, conversation_id)
    context = {
        "conversation": conv,
        "messages": selectors.conversation_messages(conv),
        "conversations": selectors.user_conversations(request.user),
    }
    return render(request, "chat/conversation.html", context)

@login_required
@require_http_methods(["POST"])
def new_conversation(request: HttpRequest) -> HttpResponse:
    """Create a new conversation and redirect to it."""
    conv = services.create_conversation_for(request.user)
    return redirect("chat:conversation", conversation_id=conv.id)

@login_required
@require_http_methods(["POST"])
def send_message(request: HttpRequest, conversation_id: int) -> JsonResponse:
    """
    Receive a user message, store it, and return an SSE stream URL
    that the client should open to stream the assistant reply.
    """
    conv = _user_conversation_or_404(request, conversation_id)
    text = (request.POST.get("text") or "").strip()
    if not text:
        return JsonResponse({"ok": False, "error": "Empty message."}, status=400)

    user_msg = services.add_user_message(conv, text)
    stream_url = iri_to_uri(f"/chat/{conv.id}/stream?message_id={user_msg.id}")
    return JsonResponse({"ok": True, "stream_url": stream_url})

@login_required
@require_http_methods(["GET"])
def stream_reply(request: HttpRequest, conversation_id: int) -> StreamingHttpResponse:
    """
    SSE endpoint: streams the assistant's reply token by token.

    Query params:
        message_id: int (required) – the user Message.id to reply to.

    The server buffers tokens and, after streaming completes, stores the
    assistant Message in the database.
    """
    conv = _user_conversation_or_404(request, conversation_id)
    try:
        message_id = int(request.GET.get("message_id", ""))
    except ValueError:
        return StreamingHttpResponse(_sse_error("Invalid message_id"), content_type="text/event-stream")
    user_msg = get_object_or_404(Message, id=message_id, conversation=conv, role=Message.ROLE_USER)
    messages = list(selectors.conversation_messages(conv))

    lc_messages = [m.to_langchain_message() for m in messages]
    graph = ChatGraph(chat_history=lc_messages)

    def event_stream() -> Iterator[bytes]:
        """Generator that yields SSE 'data:' lines as bytes and saves the final message."""
        assistant_text_parts: list[str] = []
        try:
            for token in graph.invoke_stream(user_msg.content):
                assistant_text_parts.append(token)
                yield f"data: {token}\n\n".encode("utf-8")
            # Signal completion
            yield b"event: done\ndata: end\n\n"
        except Exception as exc:  # capture backend errors
            msg = f"error: {type(exc).__name__}: {exc}"
            yield f"event: error\ndata: {msg}\n\n".encode("utf-8")
        else:
            # Persist the final assistant message only if streaming succeeded
            final = "".join(assistant_text_parts)
            services.add_assistant_message(conv, final)

    resp = StreamingHttpResponse(event_stream(), content_type="text/event-stream")
    # Helpful for proxies: disable buffering for real-time streaming
    resp["Cache-Control"] = "no-cache"
    resp["X-Accel-Buffering"] = "no"
    return resp

def _user_conversation_or_404(request: HttpRequest, conversation_id: int) -> Conversation:
    """Fetch a conversation owned by the current user or 404."""
    conv = get_object_or_404(Conversation, id=conversation_id)
    if conv.user_id != request.user.id:
        raise Http404("Conversation not found.")
    return conv

def _sse_error(message: str) -> Iterator[bytes]:
    """Small helper to emit a single SSE error event."""
    yield f"event: error\ndata: {message}\n\n".encode("utf-8")
