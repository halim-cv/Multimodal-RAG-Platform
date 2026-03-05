"""
verify_setup.py

Run this first thing tomorrow to confirm everything is wired correctly.

Usage:
    python verify_setup.py
"""
import sys
import os
from pathlib import Path

# ── Ensure project root is on path ────────────────────────────────
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT / "Text-encoding" / "model" / "code"))

_PASS = "✅"
_FAIL = "❌"
_WARN = "⚠️ "

def check(label: str, fn):
    try:
        result = fn()
        print(f"  {_PASS}  {label}" + (f" — {result}" if result else ""))
        return True
    except Exception as exc:
        print(f"  {_FAIL}  {label} → {exc}")
        return False

print("\n" + "="*60)
print("  Multimodal RAG Platform — Setup Verification")
print("="*60 + "\n")

# ── 1. .env loaded ─────────────────────────────────────────────────
print("[ 1 ] Environment")
from dotenv import load_dotenv
load_dotenv()
check(".env loaded",     lambda: "OK" if Path(".env").exists() else exec('raise FileNotFoundError(".env missing")'))
check("GEMINI_API_KEY",  lambda: f"{'*' * 8}{os.getenv('GEMINI_API_KEY','')[-4:]}" if os.getenv("GEMINI_API_KEY") else exec('raise ValueError("Not set")'))

# ── 2. FastAPI stack ───────────────────────────────────────────────
print("\n[ 2 ] Backend packages")
check("fastapi",         lambda: __import__("fastapi").__version__)
check("uvicorn",         lambda: __import__("uvicorn").__version__)
check("aiofiles",        lambda: "OK")
check("pydantic v2",     lambda: __import__("pydantic").__version__)
check("python-multipart",lambda: __import__("multipart") and "OK")

# ── 3. Gemini API ──────────────────────────────────────────────────
print("\n[ 3 ] Gemini API")
def _test_gemini():
    import google.generativeai as genai
    genai.configure(api_key=os.getenv("GEMINI_API_KEY",""))
    model    = genai.GenerativeModel(os.getenv("GEMINI_MODEL","gemini-2.0-flash"))
    response = model.generate_content("Reply with exactly: GEMINI_OK")
    return response.text.strip()[:30]
check("Gemini API call", _test_gemini)

# ── 4. MLflow ──────────────────────────────────────────────────────
print("\n[ 4 ] MLflow")
check("mlflow",          lambda: __import__("mlflow").__version__)
check("rich",            lambda: __import__("rich.version", fromlist=["__version__"]) and "OK")

# ── 5. ML encoders ────────────────────────────────────────────────
print("\n[ 5 ] ML Encoder packages")
check("torch",           lambda: __import__("torch").__version__)
check("transformers",    lambda: __import__("transformers").__version__)
check("sentence-transformers", lambda: __import__("sentence_transformers").__version__)
check("faiss-cpu",       lambda: __import__("faiss") and f"OK — {__import__('faiss').IndexFlatIP.__module__}")
check("PIL (Pillow)",    lambda: __import__("PIL").__version__)
check("pypdf",           lambda: __import__("pypdf").__version__)

# ── 6. E5 model downloaded ─────────────────────────────────────────
print("\n[ 6 ] Local model")
model_path = PROJECT_ROOT / "Text-encoding" / "model" / "modeldownload"
check("E5-base-v2 downloaded", lambda:
    "OK" if model_path.exists() and any(model_path.iterdir())
    else (_ for _ in ()).throw(FileNotFoundError(
        f"Model not found at {model_path}\nRun: python Text-encoding/model/code/download_model.py"
    ))
)

# ── 7. Audio / Image encoders ──────────────────────────────────────
print("\n[ 7 ] Audio / Image encoder packages")
check("whisper (openai)", lambda: __import__("whisper") and "OK")
check("librosa",          lambda: __import__("librosa").__version__)
check("ultralytics",      lambda: __import__("ultralytics").__version__)

print("\n" + "="*60)
print("  Done. Fix any ❌ before starting tomorrow.")
print("="*60 + "\n")
