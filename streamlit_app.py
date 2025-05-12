import streamlit as st
from chat_graph.graph import ChatGraph


st.set_page_config(page_title="Chat Interface", page_icon="💬")
st.title("AI Chat Interface")

if "chat" not in st.session_state:
    st.session_state.chat = ChatGraph()

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

if prompt := st.chat_input("What would you like to ask?"):
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.write(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = st.session_state.chat.invoke(prompt)
            ai_message = st.session_state.chat.history[-1]
            ai_content = ai_message.content if hasattr(ai_message, 'content') else str(ai_message)

            st.write(ai_content)

            st.session_state.messages.append({"role": "assistant", "content": ai_content})

