from django.urls import path
from . import views

app_name = "chat"

urlpatterns = [
    path("", views.index, name="index"),
    path("new", views.new_conversation, name="new"),
    path("<int:conversation_id>/", views.conversation_view, name="conversation"),
    path("<int:conversation_id>/send", views.send_message, name="send"),
    path("<int:conversation_id>/stream", views.stream_reply, name="stream"),
]
