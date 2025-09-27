# Code Insights Assistant (RAG with Gemini Pro & Local Embeddings)

### An intelligent assistant that analyzes GitHub repositories using RAG with local Sentence Transformers, Google's Gemini 2.5 Pro, and a vector database to answer questions about the codebase.

![Python](https://img.shields.io/badge/Python-3.9+-blue?logo=python) ![Streamlit](https://img.shields.io/badge/Streamlit-1.50.0-orange?logo=streamlit) ![LangChain](https://img.shields.io/badge/LangChain-0.3.27-green) ![ChromaDB](https://img.shields.io/badge/ChromaDB-1.1.0-purple) ![Google Gemini](https://img.shields.io/badge/Google_Gemini-2.5_Pro-blue?logo=google-gemini)

## 🚀 Overview

This project is an advanced, senior-level portfolio piece demonstrating a practical implementation of the **Retrieval-Augmented Generation (RAG)** pattern. The application allows users to have an intelligent conversation with a GitHub repository. It clones the codebase, processes the source files, and uses a vector database to provide relevant context to Google's Gemini Pro model, enabling it to answer specific questions about the code's implementation and structure.

A key feature of this project is its **hybrid AI approach**: it uses a powerful, local open-source model from **Hugging Face** for the high-volume task of creating embeddings (avoiding API rate limits), and a state-of-the-art large model (**Gemini Pro**) for the final, nuanced task of generating answers.

## ✨ Key Features & Techniques

*   **Hybrid AI Strategy:** Utilizes a local **Sentence Transformer** model (`all-MiniLM-L6-v2`) for efficient, offline embedding generation, combined with a powerful cloud-based LLM (**Google Gemini Pro**) for reasoning and response generation.
*   **Retrieval-Augmented Generation (RAG):** The core of the project. The LLM's knowledge is "grounded" in the specific context of a Git repository, preventing hallucinations and providing accurate, context-aware answers.
*   **Vector Database for Code:** Source code files are chunked, converted into vector embeddings, and stored in an in-memory **ChromaDB** instance for efficient similarity search.
*   **Orchestration with LangChain:** Leverages `langchain` for robust document loading and text splitting, demonstrating knowledge of standard GenAI orchestration frameworks.
*   **End-to-End System Design:** Showcases the ability to design and build a complete system: from data ingestion (Git cloning) and processing (embedding) to a final, user-facing application.

## 🛠️ How to Run

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/takzen/code-insights-assistant.git
    cd code-insights-assistant
    ```

2.  **Create a virtual environment and install dependencies:**
    *   This project requires Python 3.9+. Create a virtual environment (`uv venv`).
    *   Install the required packages using this command:
        ```bash
        uv pip install streamlit python-dotenv GitPython langchain langchain-community langchain-google-genai google-generativeai chromadb sentence-transformers
        ```
    *   *(Note: The first time you run the app, the `sentence-transformers` library will download the embedding model, which may take a minute.)*

3.  **Set up your Google AI API Key:**
    *   Create a file named `.env` in the root of the project.
    *   Add your Google AI API key (only used for the final answer generation): `GOOGLE_API_KEY="YOUR_API_KEY_HERE"`

4.  **Run the Streamlit application:**
    ```bash
    streamlit run app.py
    ```

## 🖼️ Showcase

| 1. Repository Input                                     | 2. Q&A with the Codebase                                  |
| :------------------------------------------------------ | :-------------------------------------------------------- |
| ![Repo Input](images/01_repo_input.png)                 | ![Q&A Showcase](images/02_qa_showcase.png)                |
| *The user provides a URL to a public GitHub repository.* | *After processing, the user can ask specific questions about the code, and Gemini provides context-aware answers.* |