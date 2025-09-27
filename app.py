# app.py
import streamlit as st
from repo_analyzer import clone_repo

st.set_page_config(layout="wide")
st.title("Code Insights Assistant 🧠")

st.header("Analyze any public GitHub repository")
github_url = st.text_input("Enter the GitHub repository URL (e.g., https://github.com/streamlit/streamlit)")

if st.button("Analyze Repository"):
    if github_url:
        with st.spinner(f"Cloning repository from {github_url}..."):
            try:
                # Clone the repo and store the path in session state
                repo_path = clone_repo(github_url)
                st.session_state.repo_path = repo_path
                st.success(f"Repository successfully cloned to: {repo_path}")
                st.info("Next step: Process the code and ask your questions below.")
            except Exception as e:
                st.error(f"Error: {e}")
    else:
        st.warning("Please enter a GitHub repository URL.")