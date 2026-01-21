import os
from config import get_config

def create_session_upload_path(session_uid, file_name, category=None):
    """
    Creates and returns a dynamic path for a specific file in a session.
    Structure: sessions/<session_uid>/<category>/<file_base_name>/
    Categories: docs, txt, img, other
    """
    config = get_config()
    project_root = config["paths"]["project_root"]
    
    # 1. Base sessions directory
    sessions_dir = os.path.join(project_root, "sessions")
    
    # 2. Determine file category (if not provided)
    if not category:
        ext = os.path.splitext(file_name)[1].lower().strip(".")
        if ext in ["pdf", "docx", "doc"]:
            category = "docs"
        elif ext in ["jpg", "jpeg", "png", "gif", "bmp", "webp"]:
            category = "img"
        elif ext in ["txt", "md", "csv"]:
            category = "txt"
        else:
            category = "other"
    
    # 3. Handle folder name for the file (remove extension for the folder name)
    file_base_name = os.path.splitext(file_name)[0]
    
    # 4. Construct the full dynamic path
    # Path: sessions / user_123 / pdf / my_document /
    target_path = os.path.join(sessions_dir, session_uid, category, file_base_name)
    
    # 5. Create directories if they don't exist
    os.makedirs(target_path, exist_ok=True)
    
    return target_path

def get_file_artifacts_paths(session_uid, file_name, category=None):
    """
    Returns a dictionary of paths for artifacts extracted from a file.
    All artifacts are stored within the dedicated folder for that file.
    """
    # Get the base folder for this file
    base_folder = create_session_upload_path(session_uid, file_name, category=category)
    
    return {
        "base_folder": base_folder,
        "extracted_text": os.path.join(base_folder, "extracted_text.txt"),
        "extracted_images": os.path.join(base_folder, "images"), # Directory for images extracted from PDF/Video
        "extracted_audio": os.path.join(base_folder, "audio"),   # Directory for audio clips
        "embeddings_pkl": os.path.join(base_folder, "embeddings.pkl"), # The final vector store
        "metadata_json": os.path.join(base_folder, "metadata.json")
    }

def get_session_base_path(session_uid):
    """Returns the base path for a given session UID."""
    config = get_config()
    return os.path.join(config["paths"]["project_root"], "sessions", session_uid)

def get_session_overall_artifacts_paths(session_uid):
    """
    Returns paths for session-level combined artifacts (the 'overall' context).
    Stored in sessions/<session_uid>/_overall/
    """
    session_base = get_session_base_path(session_uid)
    overall_folder = os.path.join(session_base, "_overall")
    
    os.makedirs(overall_folder, exist_ok=True)
    
    return {
        "base_folder": overall_folder,
        "extracted_text": os.path.join(overall_folder, "overall_extracted_text.txt"),
        "embeddings_pkl": os.path.join(overall_folder, "overall_embeddings.pkl"),
        "metadata_json": os.path.join(overall_folder, "overall_metadata.json")
    }

if __name__ == "__main__":
    # Example usage/test
    test_session = "user_abc_789"
    test_file = "research_paper.pdf"
    
    # 1. Create/Get path for a file
    path = create_session_upload_path(test_session, test_file)
    print(f"Base folder for file created: {path}")
    
    # 2. Get all artifact paths for that file
    artifacts = get_file_artifacts_paths(test_session, test_file)
    print(f"\nPlanned artifact paths for {test_file}:")
    for key, val in artifacts.items():
        print(f" - {key}: {val}")
