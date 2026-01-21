import os
import sys
from huggingface_hub import snapshot_download
from config import get_config

def download_e5_model():
    # Get configuration from config.py
    config = get_config()
    model_id = config["model"]["model_id"]
    target_dir = config["model"]["local_path"]
    
    print(f"--- Model Downloader ---")
    print(f"Model ID: {model_id}")
    print(f"Local Destination: {target_dir}")
    print(f"------------------------")
    
    # Ensure directory exists
    os.makedirs(target_dir, exist_ok=True)
    
    try:
        print("Starting download... (Tqdm progress bar will appear if installed)")
        # snapshot_download automatically uses tqdm if installed
        local_path = snapshot_download(
            repo_id=model_id,
            local_dir=target_dir,
            local_dir_use_symlinks=False,
            revision="main",
            resume_download=True  # Helpful for large models
        )
        print(f"\n[SUCCESS] Model successfully downloaded to: {local_path}")
        
    except Exception as e:
        print(f"\n[ERROR] An error occurred: {e}")
        sys.exit(1)

if __name__ == "__main__":
    download_e5_model()
