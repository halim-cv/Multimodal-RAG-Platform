import os
import sys
import numpy as np
import pickle

# Project Imports
from config import get_config, apply_cpu_limit
from handle_session import (
    get_file_artifacts_paths, 
    create_session_upload_path,
    get_session_overall_artifacts_paths
)
from embeddings_utils import (
    get_local_model, 
    process_session_file, 
    generate_session_overall_artifacts,
    load_embeddings, 
    build_faiss_index, 
    embed_query, 
    search
)
import shutil
import tkinter as tk
from tkinter import filedialog
import datetime

# --------------------
# Session Helpers
# --------------------
def select_file_via_dialog():
    """
    Opens a native file selection dialog.
    """
    root = tk.Tk()
    root.withdraw()  # Hide the main tkinter window
    root.attributes("-topmost", True) # Bring to front
    
    file_path = filedialog.askopenfilename(
        title="Select Document for RAG",
        filetypes=[
            ("All supported", "*.pdf *.docx *.txt *.md"),
            ("PDF files", "*.pdf"),
            ("Word files", "*.docx"),
            ("Text files", "*.txt;*.md"),
            ("All files", "*.*")
        ]
    )
    root.destroy()
    return file_path

def list_sessions(sessions_dir):
    if not os.path.exists(sessions_dir):
        return []
    return [d for d in os.listdir(sessions_dir) if os.path.isdir(os.path.join(sessions_dir, d))]

def list_files_in_session(session_path):
    files = []
    # Structure: sessions/<uid>/<category>/<file_base_name>/
    # We look for folders that have an 'extracted_text.txt' or represent a file
    for category in ["docs", "txt", "pdf"]: # Check common categories
        cat_path = os.path.join(session_path, category)
        if os.path.exists(cat_path):
            for file_folder in os.listdir(cat_path):
                folder_path = os.path.join(cat_path, file_folder)
                if os.path.isdir(folder_path):
                    files.append({
                        "name": file_folder,
                        "category": category,
                        "folder_path": folder_path
                    })
    return files

def import_file_to_session(session_uid, source_path):
    """
    Copies a file to the correct session folder and triggers processing.
    """
    if not os.path.exists(source_path):
        print(f"Error: Source file {source_path} does not exist.")
        return None

    file_name = os.path.basename(source_path)
    # create_session_upload_path handles folder creation and category determination
    target_folder = create_session_upload_path(session_uid, file_name)
    target_path = os.path.join(target_folder, file_name)

    print(f"Copying {file_name} to session...")
    shutil.copy2(source_path, target_path)
    
    print("Processing file (extraction + embedding)...")
    try:
        process_session_file(session_uid, target_path)
        return file_name
    except Exception as e:
        print(f"Error processing imported file: {e}")
        return None

def main():
    print("=== Multi-modal RAG Platform CLI ===")
    config = get_config()
    apply_cpu_limit()
    
    # 1. Load Model Pre-emptively
    print("Loading model...")
    try:
        model = get_local_model()
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # 2. Session Selection
    sessions_dir = os.path.join(config["paths"]["project_root"], "sessions")
    
    while True:
        sessions = list_sessions(sessions_dir)
        print("\nAvailable Sessions:")
        for i, s in enumerate(sessions):
            print(f"[{i}] {s}")
        
        print(f"[{len(sessions)}] + New Chat (Select New File)")
        print(f"[{len(sessions)+1}] Exit")

        choice = input("\nSelect an option: ")
        try:
            val = int(choice)
            if val == len(sessions):
                # New Chat Flow
                print("\nOpening file picker...")
                source_path = select_file_via_dialog()
                if not source_path:
                    print("No file selected.")
                    continue
                
                # Create a new session with timestamp
                session_uid = f"session_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
                print(f"Creating new session: {session_uid}")
                
                # Import first file
                import_file_to_session(session_uid, source_path)
                
                # Sequential addition loop
                while True:
                    add_more = input("\nWould you like to add another file to this session? (y/n): ").lower()
                    if add_more == 'y':
                        print("Opening file picker...")
                        extra_path = select_file_via_dialog()
                        if extra_path:
                            import_file_to_session(session_uid, extra_path)
                        else:
                            print("No file selected.")
                    else:
                        break
                break
            elif val == len(sessions)+1:
                return
            else:
                session_uid = sessions[val]
                break
        except Exception as e:
            print(f"Invalid selection: {e}")
            continue

    # 3. File Selection
    session_path = os.path.join(sessions_dir, session_uid)
    files = list_files_in_session(session_path)

    if not files:
        print(f"No files found in session {session_uid}.")
        return

    print(f"\nFiles in {session_uid}:")
    for i, f in enumerate(files):
        # Check if embedding exists using the known category
        artifact_paths = get_file_artifacts_paths(session_uid, f['name'], category=f['category'])
        has_emb = os.path.exists(artifact_paths["embeddings_pkl"])
        status = "[Embedded]" if has_emb else "[Pending]"
        print(f"[{i}] {f['name']} ({f['category']}) {status}")
    
    # Add Overall Option
    overall_artifact_paths = get_session_overall_artifacts_paths(session_uid)
    has_overall_emb = os.path.exists(overall_artifact_paths["embeddings_pkl"])
    overall_status = "[Available]" if has_overall_emb else "[Ready to Generate]"
    print(f"[{len(files)}] *** OVERALL SESSION CONTEXT *** {overall_status}")
    print(f"[{len(files)+1}] + Add New File to this Session")
    
    f_choice = input("\nSelect file index (Enter to select all embedded files): ").strip()
    selected_files = []
    
    try:
        if f_choice == "":
            # Select all embedded files
            for f in files:
                a_paths = get_file_artifacts_paths(session_uid, f['name'], category=f['category'])
                if os.path.exists(a_paths["embeddings_pkl"]):
                    selected_files.append(f)
            if not selected_files:
                print("No embedded files found in this session.")
                return
            print(f"Selected {len(selected_files)} files for combined chat.")
        else:
            idx = int(f_choice)
            if idx == len(files):
                # Overall Session Context Flow
                overall_paths = get_session_overall_artifacts_paths(session_uid)
                if not os.path.exists(overall_paths["embeddings_pkl"]):
                    print("\nGenerating Session-wide context and embeddings...")
                    try:
                        generate_session_overall_artifacts(session_uid)
                    except Exception as e:
                        print(f"Error generating overall context: {e}")
                        return
                
                selected_files = [{"name": "_overall", "category": "_overall", "is_overall": True}]
            elif idx == len(files) + 1:
                # Add New File logic via Dialog
                while True:
                    print("\nOpening file picker...")
                    source_path = select_file_via_dialog()
                    if not source_path:
                        if idx == len(files): # First time
                            print("No file selected.")
                        break
                    
                    new_file_name = import_file_to_session(session_uid, source_path)
                    
                    add_more = input("\nWould you like to add another file? (y/n): ").lower()
                    if add_more != 'y':
                        break
                
                # Refresh file list
                files = list_files_in_session(session_path)
                if not files:
                    return
                
                print("\nUpdated Files in Session:")
                for i, f in enumerate(files):
                    artifact_paths = get_file_artifacts_paths(session_uid, f['name'], category=f['category'])
                    has_emb = os.path.exists(artifact_paths["embeddings_pkl"])
                    status = "[Embedded]" if has_emb else "[Pending]"
                    print(f"[{i}] {f['name']} ({f['category']}) {status}")
                
                f_choice = input("\nSelect file index to chat (Enter to select all embedded): ").strip()
                if f_choice == "":
                    for f in files:
                        a_paths = get_file_artifacts_paths(session_uid, f['name'], category=f['category'])
                        if os.path.exists(a_paths["embeddings_pkl"]):
                            selected_files.append(f)
                else:
                    selected_files = [files[int(f_choice)]]
            else:
                selected_files = [files[idx]]
    except Exception as e:
        print(f"Selection error: {e}")
        return

    if not selected_files:
        print("No files selected.")
        return

    # 4. Check/Create/Combine Embeddings
    all_embeddings = []
    all_texts = []
    all_metadata = []

    for selected_file in selected_files:
        if selected_file.get("is_overall"):
            overall_paths = get_session_overall_artifacts_paths(session_uid)
            emb_path = overall_paths["embeddings_pkl"]
        else:
            artifact_paths = get_file_artifacts_paths(session_uid, selected_file['name'], category=selected_file['category'])
            emb_path = artifact_paths["embeddings_pkl"]
        
        should_embed = False
        if not os.path.exists(emb_path):
            if selected_file.get("is_overall"):
                print("\nOverall embeddings missing. Generating...")
                generate_session_overall_artifacts(session_uid)
            else:
                print(f"\nEmbeddings not found for {selected_file['name']}. Generating now...")
                should_embed = True
        
        if should_embed:
            # ... (original logic for single file embedding)
            artifact_paths = get_file_artifacts_paths(session_uid, selected_file['name'], category=selected_file['category']) # re-eval if needed
            text_path = artifact_paths["extracted_text"]
            source_dir = os.path.dirname(text_path)
            potential_files = [f for f in os.listdir(source_dir) if f not in ["extracted_text.txt", "embeddings.pkl", "metadata.json"]]
            
            if potential_files:
                source_file_path = os.path.join(source_dir, potential_files[0])
                print(f"Found source file: {potential_files[0]}")
                process_session_file(session_uid, source_file_path)
            elif os.path.exists(text_path):
                print("Generating embeddings from existing extracted text...")
                process_session_file(session_uid, text_path)
            else:
                 print(f"Source file not found for {selected_file['name']}. Skipping.")
                 continue

        # Load and append
        try:
            print(f"Loading embeddings for {selected_file['name']}...")
            embs, txs, meta = load_embeddings(emb_path)
            all_embeddings.append(embs)
            all_texts.extend(txs)
            all_metadata.extend(meta)
        except Exception as e:
            print(f"Error loading embeddings for {selected_file['name']}: {e}")

    if not all_embeddings:
        print("No embeddings could be loaded.")
        return

    # 5. Build Index
    print(f"\nBuilding combined FAISS index with {len(all_texts)} total chunks...")
    try:
        combined_embeddings = np.vstack(all_embeddings)
        index = build_faiss_index(combined_embeddings, use_cosine=True)
        print(f"Index built successfully.")
    except Exception as e:
        print(f"Error building combined index: {e}")
        return

    # 6. Chat Loop
    print("\n--- Chat Mode Active ---")
    print("Type 'exit' to quit.")
    while True:
        query = input("\nUser: ")
        if query.lower() in ['exit', 'quit']:
            break
        
        if not query.strip():
            continue

        # Search
        q_vec = embed_query(model, query, normalize=True)
        results = search(index, q_vec, k=3, texts=all_texts, metadata=all_metadata)

        print("\nAssistant (Top Matches):")
        for res in results:
            source_info = ""
            if res.get("metadata") and "file_name" in res["metadata"]:
                source_info = f" [Source: {res['metadata']['file_name']}]"
            
            print(f"- [Score: {res['score']:.4f}]{source_info} {res['text'][:200]}...")

if __name__ == "__main__":
    main()
