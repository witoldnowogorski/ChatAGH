import markdown
import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage
from chat_graph.graph import ChatGraph

# --- Page Config ---
st.set_page_config(
    page_title="ChatAGH development",
    layout="wide",
)

# --- CSS Styling ---
st.markdown("""
<style>
    :root {
        --background-color: #0E1117;
        --secondary-background-color: #262730;
        --text-color: #FAFAFA;
        --primary-color: #FF4B4B;
        --secondary-color: #7E57C2;
        --accent-color: #4CAF50;
    }

    .main {
        background-color: var(--background-color);
        color: var(--text-color);
    }

    .chat-container {
        padding: 10px;
        border-radius: 10px;
        margin-bottom: 10px;
        position: relative;
        max-width: 80%;
    }

    .user-message {
        background-color: #4E5D8C;
        float: right;
        margin-left: 20%;
        border-radius: 15px 15px 0 15px;
    }

    .ai-message {
        background-color: #383C4A;
        float: left;
        margin-right: 20%;
        border-radius: 15px 15px 15px 0;
    }

    .message-content {
        padding: 10px;
        color: white;
    }

    .clear {
        clear: both;
    }

    .chat-input {
        background-color: var(--secondary-background-color);
        padding: 15px;
        border-radius: 10px;
        margin-top: 20px;
    }

    .app-header {
        text-align: center;
        margin-bottom: 20px;
        padding: 10px;
        background-color: var(--secondary-background-color);
        border-radius: 10px;
    }

    .timestamp {
        font-size: 0.7em;
        opacity: 0.7;
        text-align: right;
        padding-top: 5px;
    }

    .avatar {
        width: 35px;
        height: 35px;
        border-radius: 50%;
        display: inline-block;
        margin-right: 10px;
        vertical-align: middle;
    }

    .username {
        font-weight: bold;
        margin-bottom: 5px;
    }

    .message-row {
        display: flex;
        margin-bottom: 15px;
        align-items: flex-start;
    }

    .typing-indicator {
        display: flex;
        padding: 10px;
    }

    .typing-indicator span {
        height: 8px;
        width: 8px;
        background-color: #9E9EA1;
        border-radius: 50%;
        display: block;
        margin: 0 2px;
        opacity: 0.4;
    }

    .typing-indicator span:nth-child(1) { animation: pulse 1s infinite; }
    .typing-indicator span:nth-child(2) { animation: pulse 1s infinite 0.2s; }
    .typing-indicator span:nth-child(3) { animation: pulse 1s infinite 0.4s; }

    @keyframes pulse {
        0% { opacity: 0.4; transform: scale(1); }
        50% { opacity: 1; transform: scale(1.2); }
        100% { opacity: 0.4; transform: scale(1); }
    }
</style>
""", unsafe_allow_html=True)

# --- Session State ---
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

if 'chat_graph' not in st.session_state:
    st.session_state.chat_graph = ChatGraph()

# --- Header ---
st.markdown("<div class='app-header'><h1>ChatAGH development</h1></div>", unsafe_allow_html=True)

# --- Layout Columns ---
col1, col2 = st.columns([3, 1])

with col2:
    st.markdown("### Chat Settings")
    show_rag_info = st.toggle("Show RAG Information", value=False)
    clear_btn = st.button("Clear Chat History")

# --- Chat Container ---
with col1:
    chat_container = st.container()

    with chat_container:
        if not st.session_state.chat_history:
            st.markdown("<div style='text-align: center; padding: 30px;'><p>How can I help you?</p></div>",
                        unsafe_allow_html=True)
        else:
            for message in st.session_state.chat_history:
                if isinstance(message, HumanMessage):
                    st.markdown(f"""
                        <div class='message-row'>
                            <div style='flex-grow: 1; display: flex; justify-content: flex-end;'>
                                <div class='chat-container user-message'>
                                    <div class='username'>You</div>
                                    <div class='message-content'>{message.content}</div>
                                </div>
                            </div>
                        </div>
                        <div class='clear'></div>
                    """, unsafe_allow_html=True)
                elif isinstance(message, AIMessage):
                    content = markdown.markdown(message.content)
                    st.markdown(f"""
                        <div class='message-row'>
                            <div style='margin-right: 10px;'>
                                <div style='background-color: #7E57C2; width: 35px; height: 35px; border-radius: 50%; display: flex; align-items: center; justify-content: center; color: white; font-weight: bold;'>
                                    AI
                                </div>
                            </div>
                            <div class='chat-container ai-message'>
                                <div class='username'>Assistant</div>
                                <div class='message-content'>{content}</div>
                            </div>
                        </div>
                        <div class='clear'></div>
                    """, unsafe_allow_html=True)

    if st.session_state.get("pending_user_message"):
        user_input = st.session_state.pending_user_message
        with chat_container:
            typing_indicator = st.empty()
            typing_indicator.markdown("""
            <div class='message-row'>
                <div style='margin-right: 10px;'>
                    <div style='background-color: #7E57C2; width: 35px; height: 35px; border-radius: 50%; display: flex; align-items: center; justify-content: center; color: white; font-weight: bold;'>
                        AI
                    </div>
                </div>
                <div class='chat-container ai-message'>
                    <div class='username'>RAG Assistant</div>
                    <div class='typing-indicator'>
                        <span></span>
                        <span></span>
                        <span></span>
                    </div>
                </div>
            </div>
            <div class='clear'></div>
            """, unsafe_allow_html=True)

        try:
            response = st.session_state.chat_graph.invoke(user_input)
            ai_messages = [msg for msg in response.get("messages", []) if isinstance(msg, AIMessage)]
            ai_response = ai_messages[-1] if ai_messages else AIMessage("RAG NOT IMPLEMENTED")

            st.session_state.chat_history.append(ai_response)

            if show_rag_info and "use_rag" in response:
                rag_status = "Used RAG" if response["use_rag"] else "Did not use RAG"
                st.session_state.chat_history.append(
                    AIMessage(content=f"_System info: {rag_status} for this query._"))

        except Exception as e:
            st.session_state.chat_history.append(AIMessage(content=f"Error: {str(e)}"))

        st.session_state.pending_user_message = None
        typing_indicator.empty()
        st.rerun()

    def handle_chat_input():
        user_input = st.session_state.user_input
        if user_input:
            st.session_state.chat_history.append(HumanMessage(content=user_input))
            st.session_state.user_input = ""
            st.session_state.should_rerun = True
            st.session_state.pending_user_message = user_input

    st.text_input("", key="user_input", on_change=handle_chat_input,
                  placeholder="Type your message here...")

# --- Clear Chat ---
if clear_btn:
    st.session_state.chat_history = []
    st.session_state.chat_graph = ChatGraph()
    st.rerun()

if st.session_state.get("should_rerun", False):
    st.session_state.should_rerun = False  # reset flag
    st.rerun()
