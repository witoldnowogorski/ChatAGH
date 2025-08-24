// Chat App JavaScript Utilities

class ChatApp {
  constructor() {
    this.isStreaming = false;
    this.eventSource = null;
    this.messageForm = null;
    this.messageInput = null;
    this.sendBtn = null;
    this.messagesContainer = null;

    this.init();
  }

  init() {
    this.messageForm = document.getElementById("message-form");
    this.messageInput = document.getElementById("message-input");
    this.sendBtn = document.getElementById("send-btn");
    this.messagesContainer = document.getElementById("messages-container");

    if (
      this.messageForm &&
      this.messageInput &&
      this.sendBtn &&
      this.messagesContainer
    ) {
      this.bindEvents();
      this.messageInput.focus();
      this.scrollToBottom();
    }
  }

  bindEvents() {
    // Auto-resize textarea
    this.messageInput.addEventListener("input", (e) =>
      this.autoResize(e.target),
    );

    // Handle Enter key
    this.messageInput.addEventListener("keydown", (e) => this.handleKeyDown(e));

    // Handle form submission
    this.messageForm.addEventListener("submit", (e) => this.handleSubmit(e));

    // Handle page visibility changes
    document.addEventListener("visibilitychange", () =>
      this.handleVisibilityChange(),
    );
  }

  autoResize(textarea) {
    textarea.style.height = "auto";
    textarea.style.height = `${textarea.scrollHeight}px`;
  }

  handleKeyDown(e) {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      if (!this.isStreaming && e.target.value.trim()) {
        this.messageForm.dispatchEvent(new Event("submit"));
      }
    }
  }

  async handleSubmit(e) {
    e.preventDefault();

    if (this.isStreaming) return;

    const text = this.messageInput.value.trim();
    if (!text) return;

    this.setStreamingState(true);
    this.addUserMessage(text);
    this.clearInput();
    this.removeEmptyState();

    try {
      const response = await this.sendMessage(text);
      if (response.ok) {
        this.startStreaming(response.stream_url);
      } else {
        this.showError(response.error || "Failed to send message");
        this.setStreamingState(false);
      }
    } catch (error) {
      console.error("Error sending message:", error);
      this.showError("Network error occurred");
      this.setStreamingState(false);
    }
  }

  async sendMessage(text) {
    const formData = new FormData();
    formData.append("text", text);
    formData.append("csrfmiddlewaretoken", this.getCSRFToken());

    const response = await fetch(this.getSendURL(), {
      method: "POST",
      body: formData,
      headers: {
        "X-CSRFToken": this.getCSRFToken(),
      },
    });

    return await response.json();
  }

  startStreaming(streamUrl) {
    const typingIndicator = this.addTypingIndicator();
    let assistantMessage = null;

    this.eventSource = new EventSource(streamUrl);

    this.eventSource.onmessage = (event) => {
      if (!assistantMessage) {
        typingIndicator.remove();
        assistantMessage = this.createAssistantMessage();
      }
      this.appendToMessage(assistantMessage, event.data);
      this.scrollToBottom();
    };

    this.eventSource.addEventListener("done", () => {
      this.eventSource.close();
      this.finalizeStreamingMessage(assistantMessage);
      this.setStreamingState(false);
    });

    this.eventSource.addEventListener("error", (event) => {
      console.error("Streaming error:", event);
      this.cleanupStreaming(typingIndicator);
      this.showError("Error receiving response");
    });

    this.eventSource.onerror = () => {
      console.error("EventSource connection failed");
      this.cleanupStreaming(typingIndicator);
    };
  }

  cleanupStreaming(typingIndicator) {
    if (this.eventSource) {
      this.eventSource.close();
      this.eventSource = null;
    }
    if (typingIndicator && typingIndicator.parentNode) {
      typingIndicator.remove();
    }
    this.setStreamingState(false);
  }

  addUserMessage(content) {
    const messageEl = this.createMessageElement(content, "user");
    this.messagesContainer.appendChild(messageEl);
    this.scrollToBottom();
  }

  addTypingIndicator() {
    const typingEl = document.createElement("div");
    typingEl.className = "typing-indicator";
    typingEl.id = "typing-indicator";

    const bubble = document.createElement("div");
    bubble.className = "typing-bubble";

    typingEl.appendChild(bubble);
    this.messagesContainer.appendChild(typingEl);
    this.scrollToBottom();

    return typingEl;
  }

  createAssistantMessage() {
    const messageEl = document.createElement("div");
    messageEl.className = "message assistant";

    const bubble = document.createElement("div");
    bubble.className = "message-bubble";

    const markdownContent = document.createElement("div");
    markdownContent.className = "markdown-content";
    markdownContent.setAttribute("data-streaming", "true");

    bubble.appendChild(markdownContent);
    messageEl.appendChild(bubble);
    this.messagesContainer.appendChild(messageEl);

    return markdownContent;
  }

  createMessageElement(content, role) {
    const messageEl = document.createElement("div");
    messageEl.className = `message ${role}`;

    const bubble = document.createElement("div");
    bubble.className = "message-bubble";
    bubble.textContent = content;

    messageEl.appendChild(bubble);
    return messageEl;
  }

  appendToMessage(messageElement, token) {
    // If this is a streaming markdown content element
    if (
      messageElement.classList.contains("markdown-content") &&
      messageElement.getAttribute("data-streaming") === "true"
    ) {
      // Store accumulated text in a data attribute
      const currentText =
        messageElement.getAttribute("data-accumulated-text") || "";
      const newText = currentText + token;
      messageElement.setAttribute("data-accumulated-text", newText);

      // Just show raw text during streaming, don't parse markdown yet
      messageElement.textContent = newText;
    } else {
      // Fallback for non-markdown elements
      messageElement.textContent += token;
    }
  }

  setStreamingState(streaming) {
    this.isStreaming = streaming;
    this.sendBtn.disabled = streaming;
    this.sendBtn.textContent = streaming ? "Sending..." : "Send";

    if (!streaming) {
      this.messageInput.focus();
    }
  }

  clearInput() {
    this.messageInput.value = "";
    this.messageInput.style.height = "auto";
  }

  removeEmptyState() {
    const emptyState = this.messagesContainer.querySelector(".empty-state");
    if (emptyState) {
      emptyState.remove();
    }
  }

  showError(message) {
    const errorEl = document.createElement("div");
    errorEl.className = "alert alert-error";
    errorEl.textContent = message;
    errorEl.style.margin = "1rem";

    this.messagesContainer.appendChild(errorEl);
    this.scrollToBottom();

    // Auto-remove after 5 seconds
    setTimeout(() => {
      if (errorEl.parentNode) {
        errorEl.remove();
      }
    }, 5000);
  }

  scrollToBottom() {
    this.messagesContainer.scrollTop = this.messagesContainer.scrollHeight;
  }

  handleVisibilityChange() {
    if (document.hidden && this.eventSource) {
      // Pause streaming when tab becomes hidden
      console.log("Tab hidden, maintaining connection...");
    } else if (!document.hidden && this.isStreaming) {
      // Resume streaming when tab becomes visible
      console.log("Tab visible, streaming continues...");
    }
  }

  getCSRFToken() {
    return document.querySelector("[name=csrfmiddlewaretoken]").value;
  }

  getSendURL() {
    const conversationId = this.extractConversationId();
    return `/chat/${conversationId}/send`;
  }

  extractConversationId() {
    // Extract conversation ID from current URL
    const pathParts = window.location.pathname.split("/");
    const chatIndex = pathParts.indexOf("chat");
    return pathParts[chatIndex + 1];
  }

  finalizeStreamingMessage(messageElement) {
    if (
      messageElement &&
      messageElement.classList.contains("markdown-content")
    ) {
      // Remove streaming indicator
      messageElement.removeAttribute("data-streaming");

      // Parse the final accumulated text as markdown
      const finalText = messageElement.getAttribute("data-accumulated-text");
      if (finalText && typeof marked !== "undefined") {
        try {
          messageElement.innerHTML = marked.parse(finalText);
        } catch (e) {
          console.error("Error parsing final markdown:", e);
          messageElement.textContent = finalText;
        }
      } else if (finalText) {
        // Fallback to plain text if marked is not available
        messageElement.textContent = finalText;
      }
    }
  }
}

// Utility functions
const ChatUtils = {
  // Format timestamps
  formatTime(date) {
    return new Intl.DateTimeFormat("en-US", {
      hour: "numeric",
      minute: "2-digit",
      hour12: true,
    }).format(date);
  },

  // Escape HTML to prevent XSS
  escapeHtml(text) {
    const div = document.createElement("div");
    div.textContent = text;
    return div.innerHTML;
  },

  // Copy text to clipboard
  async copyToClipboard(text) {
    try {
      await navigator.clipboard.writeText(text);
      return true;
    } catch (err) {
      console.error("Failed to copy text:", err);
      return false;
    }
  },

  // Show temporary notification
  showNotification(message, type = "info") {
    const notification = document.createElement("div");
    notification.className = `notification notification-${type}`;
    notification.textContent = message;
    notification.style.cssText = `
            position: fixed;
            top: 20px;
            right: 20px;
            padding: 1rem;
            background: ${type === "error" ? "#f8d7da" : "#d1ecf1"};
            border: 1px solid ${type === "error" ? "#f5c6cb" : "#bee5eb"};
            border-radius: 4px;
            z-index: 1000;
            max-width: 300px;
        `;

    document.body.appendChild(notification);

    setTimeout(() => {
      notification.remove();
    }, 3000);
  },
};

// Initialize chat app when DOM is ready
document.addEventListener("DOMContentLoaded", () => {
  new ChatApp();
});

// Export for potential module use
if (typeof module !== "undefined" && module.exports) {
  module.exports = { ChatApp, ChatUtils };
}
