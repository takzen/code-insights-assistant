# repo_analyzer.py
import os
import shutil
from git import Repo

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings

def clone_repo(github_url: str) -> str:
    """
    Clones a GitHub repository to a temporary local path.
    Returns the path to the cloned repository.
    """
    # Generate a temporary path for the repository
    temp_dir = os.path.join("/tmp", "repo_clone")
    
    # If the directory already exists, remove it
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
    
    try:
        # Clone the repository
        Repo.clone_from(github_url, temp_dir)
        return temp_dir
    except Exception as e:
        # Return an error message if cloning fails
        raise Exception(f"Failed to clone repository: {e}")
    

def process_and_embed_repo(repo_path: str) -> Chroma:
    """
    Loads Python files, splits them into chunks, creates embeddings,
    and stores them in a Chroma vector database.
    """
    # Load only Python files from the directory
    loader = DirectoryLoader(repo_path, glob="**/*.py", loader_cls=TextLoader)
    documents = loader.load()

    # Split documents into smaller chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_documents(documents)

    # Initialize Google's embedding model
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    # Create a Chroma vector store from the documents
    # It will be created in-memory
    vector_store = Chroma.from_documents(texts, embeddings)
    
    return vector_store