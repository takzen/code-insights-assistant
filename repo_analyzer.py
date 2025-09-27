# repo_analyzer.py
import os
import shutil
from git import Repo

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