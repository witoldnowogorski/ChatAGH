import streamlit as st
import tempfile
import time
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag.utils.utils import load_env
from rag.indexing import indexing
from rag.inference import inference
from rag.utils.logger import LOG_FILE


def read_logs():
    """Read logs from the log file."""
    try:
        with open(LOG_FILE, "r", encoding="utf-8") as file:
            return file.readlines()
    except FileNotFoundError:
        return ["Log file not found. Please check if the log file exists."]
    except Exception as e:
        return [f"Error reading logs: {str(e)}"]


def auto_refresh_logs():
    """Auto refresh logs if auto-refresh is enabled."""
    if st.session_state.get("auto_refresh", False):
        st.experimental_rerun()


def main():
    st.title("Chat AGH development")

    load_env()

    tab1, tab2 = st.tabs(["Inference", "Logs"])

    with tab1:
        st.header("Inference")
        query = st.text_area("Enter your query", height=100)
        if st.button("Run Inference"):
            if query:
                with st.spinner("Running inference..."):
                    try:
                        response, source_docs = inference(query)
                        st.subheader("Model Response")
                        st.write(response)
                        st.subheader("Retrieved Documents")
                        for i, doc in enumerate(source_docs):
                            with st.expander(f"Document {i + 1}"):
                                st.write(doc)
                    except Exception as e:
                        st.error(f"Error during inference: {str(e)}")
            else:
                st.error("Please enter a query")

    with tab2:
        st.header("Logs")

        log_lines = read_logs()

        filter_options = ["All Logs", "INFO", "DEBUG", "WARNING", "ERROR", "CRITICAL"]
        selected_filter = st.selectbox("Filter logs by level:", filter_options)

        search_term = st.text_input("Search logs:", "")

        filtered_logs = []
        for line in log_lines:
            if selected_filter != "All Logs" and selected_filter not in line:
                continue
            if search_term and search_term.lower() not in line.lower():
                continue
            filtered_logs.append(line)

        st.text_area("Log Output", value="".join(filtered_logs), height=400, disabled=True)

        st.download_button(
            label="Download Logs",
            data="".join(filtered_logs),
            file_name=f"ChatAGH_logs_{time.strftime('%Y-%m-%d_%H-%M-%S')}.log",
            mime="text/plain"
        )

if __name__ == "__main__":
    main()