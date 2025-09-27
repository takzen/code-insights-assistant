# app.py
import streamlit as st
from repo_analyzer import clone_repo, process_and_embed_repo, generate_answer

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

if st.button("Analyze Repository"):
    if github_url:
        with st.spinner(f"Cloning repository..."):
            try:
                repo_path = clone_repo(github_url)
                st.session_state.repo_path = repo_path
                st.success(f"Repository cloned.")
            except Exception as e:
                st.error(f"Error cloning: {e}")
        
        # This part runs after cloning is successful
        if 'repo_path' in st.session_state:
            with st.spinner("Processing code and creating vector database... This may take a while for large repos."):
                try:
                    # Create vector store and save it in session state
                    vector_store = process_and_embed_repo(st.session_state.repo_path)
                    st.session_state.vector_store = vector_store
                    st.success("Repository processed and indexed successfully!")
                except Exception as e:
                    st.error(f"Error processing repository: {e}")
    else:
        st.warning("Please enter a GitHub repository URL.")

# Add the Q&A section, which appears only after the repo is processed
if 'vector_store' in st.session_state:
    st.header("Ask Questions About the Codebase")
    user_query = st.text_input("e.g., 'How is the `st.button` function implemented?'")

    if st.button("Get Answer"):
        if user_query:
            with st.spinner("Searching the codebase and generating an answer..."):
                vector_store = st.session_state.vector_store
                answer = generate_answer(vector_store, user_query)
                st.markdown(answer)
        else:
            st.warning("Please enter a question.")