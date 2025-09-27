# repo_analyzer.py
import os
import shutil
import stat
from git import Repo
from dotenv import load_dotenv
import google.generativeai as genai
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings # Import for local embeddings

# Load environment variables from .env file
load_dotenv()

# --- Error Handling Function for Windows ---
def on_rm_error(func, path, exc_info):
    """
    Error handler for shutil.rmtree.
    If the error is for readonly files, it attempts to change the file permissions and retry.
    """
    if not os.access(path, os.W_OK):
        os.chmod(path, stat.S_IWUSR)
        func(path)
    else:
        raise

# --- Core Logic Functions ---

def clone_repo(github_url: str) -> str:
    """
    Clones a GitHub repository to a local temporary path.
    Returns the path to the cloned repository.
    """
    temp_dir = "temp_repo"
    
    # If the directory already exists, remove it robustly
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir, onerror=on_rm_error)
    
    try:
        Repo.clone_from(github_url, temp_dir)
        return temp_dir
    except Exception as e:
        raise Exception(f"Failed to clone repository: {e}")

def process_and_embed_repo(repo_path: str) -> Chroma:
    """
    Loads Python files from the repo, splits them into chunks, creates local embeddings,
    and stores them in a Chroma vector database in-memory.
    """
    # Load only Python (.py) files from the cloned directory, showing progress
    loader = DirectoryLoader(repo_path, glob="**/*.py", loader_cls=TextLoader, show_progress=True)
    documents = loader.load()

    if not documents:
        raise ValueError("No Python files found in the repository. This tool only analyzes .py files.")

    # Split the loaded documents into smaller chunks for better context
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=250)
    texts = text_splitter.split_documents(documents)

    # --- KEY CHANGE: Use a local, open-source model for embeddings ---
    # This model will be downloaded once and then run locally, avoiding API rate limits.
    # "all-MiniLM-L6-v2" is a small but powerful model, great for this task.
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    # Create an in-memory Chroma vector store from the document chunks
    vector_store = Chroma.from_documents(texts, embeddings)
    
    return vector_store

def generate_answer(vector_store: Chroma, query: str) -> str:
    """
    Performs a similarity search, builds a prompt with the retrieved context,
    and generates an answer using the Gemini Pro model.
    """
    # Configure the generative model API key
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        return "Error: GOOGLE_API_KEY not found in .env file. Please set it up."
    genai.configure(api_key=api_key)
    
    model = genai.GenerativeModel('gemini-2.5-pro')

    # Create a retriever to find relevant documents (k=5 means top 5 results)
    retriever = vector_store.as_retriever(search_kwargs={"k": 5})
    relevant_docs = retriever.invoke(query)

    # Prepare the context by joining the content of relevant documents
    context = "\n\n".join([doc.page_content for doc in relevant_docs])

    # Create a prompt template that instructs the LLM
    prompt_template = f"""
    You are an expert code assistant. Your task is to answer questions based ONLY on the provided code context from a GitHub repository. 
    If the context does not contain the answer, state that clearly and do not make up information.

    CONTEXT:
    ---
    {context}
    ---

    QUESTION:
    {query}

    ANSWER:
    """
    
    try:
        response = model.generate_content(prompt_template)
        return response.text
    except Exception as e:
        return f"An error occurred while generating the answer with the Gemini API: {e}"