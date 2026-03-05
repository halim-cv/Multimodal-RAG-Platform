import os
import torch

# --- Dynamic Path Configuration ---
# Get the directory where this script is located
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Project root is two levels up from model/code/
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, "..", ".."))

# --- Device Auto-Detection ---
def _detect_device() -> str:
    """Return the best available device: cuda > mps > cpu."""
    if torch.cuda.is_available():
        print(f"[Config] GPU detected: {torch.cuda.get_device_name(0)} — using CUDA")
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        print("[Config] Apple Silicon GPU detected — using MPS")
        return "mps"
    print("[Config] No GPU detected — using CPU")
    return "cpu"

DEVICE = _detect_device()

# --- Model Configuration ---
MODEL_SETTINGS = {
    "model_id": "intfloat/e5-small-v2",
    "model_name": "E5-Small-v2",
    "local_path": os.path.join(PROJECT_ROOT, "model", "modeldownload"),
    "max_length": 512,
    "embedding_dim": 384,
    "device": DEVICE,
    "max_threads": 3,  # CPU thread cap (respected even on GPU for CPU-bound ops)
}

# --- Paths ---
DIR_PATHS = {
    "project_root": PROJECT_ROOT,
    "model_dir": MODEL_SETTINGS["local_path"],
}

def apply_cpu_limit():
    """Applies the CPU thread limit to supported libraries."""
    threads = MODEL_SETTINGS.get("max_threads", 3)
    torch.set_num_threads(threads)
    os.environ["OMP_NUM_THREADS"] = str(threads)
    os.environ["MKL_NUM_THREADS"] = str(threads)
    print(f"CPU limit applied: {threads} core(s)/thread(s)")

def get_config():
    """Returns the full configuration dictionary."""
    return {
        "model": MODEL_SETTINGS,
        "paths": DIR_PATHS
    }

if __name__ == "__main__":
    # Test print to verify paths
    config = get_config()
    print(f"Project Root: {config['paths']['project_root']}")
    print(f"Model Local Path: {config['model']['local_path']}")
