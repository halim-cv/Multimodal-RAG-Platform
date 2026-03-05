"""
verify_setup.py — Run first thing tomorrow to confirm everything works.
"""
import sys, os
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT / "Text-encoding" / "model" / "code"))

_PASS = "✅"
_FAIL = "❌"

def check(label, fn):
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

# ── 1. Environment ─────────────────────────────────────────────
print("[ 1 ] Environment")
from dotenv import load_dotenv
load_dotenv()
check(".env loaded",     lambda: "OK" if Path(".env").exists() else None)
check("GEMINI_API_KEY",  lambda: f"****{os.getenv('GEMINI_API_KEY','')[-4:]}")
check("GEMINI_MODEL",    lambda: os.getenv("GEMINI_MODEL", "NOT SET"))

# ── 2. Backend packages ────────────────────────────────────────
print("\n[ 2 ] Backend packages")
check("fastapi",          lambda: __import__("fastapi").__version__)
check("uvicorn",          lambda: __import__("uvicorn").__version__)
check("aiofiles",         lambda: "OK")
check("pydantic v2",      lambda: __import__("pydantic").__version__)
check("python-multipart", lambda: __import__("multipart") and "OK")

# ── 3. Gemini API (new google-genai SDK) ────────────────────────
print("\n[ 3 ] Gemini API (google-genai)")
def _test_gemini():
    from google import genai
    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY", ""))
    model  = os.getenv("GEMINI_MODEL", "gemini-3-flash-preview")
    r = client.models.generate_content(model=model, contents="Reply: ONLINE")
    return r.text.strip()[:40]
check("google-genai SDK", lambda: __import__("google.genai", fromlist=["genai"]) and "OK")
check("Live API call",    _test_gemini)

# ── 4. MLflow ──────────────────────────────────────────────────
print("\n[ 4 ] MLflow")
check("mlflow",           lambda: __import__("mlflow").__version__)

# ── 5. ML encoder packages ────────────────────────────────────
print("\n[ 5 ] ML Encoder packages")
check("torch",            lambda: __import__("torch").__version__)
check("transformers",     lambda: __import__("transformers").__version__)
check("sentence-transformers", lambda: __import__("sentence_transformers").__version__)
check("faiss-cpu",        lambda: __import__("faiss") and "OK")
check("PIL (Pillow)",     lambda: __import__("PIL").__version__)
check("pypdf",            lambda: __import__("pypdf").__version__)

# ── 6. E5 model ───────────────────────────────────────────────
print("\n[ 6 ] Local E5 model")
model_path = PROJECT_ROOT / "Text-encoding" / "model" / "modeldownload"
check("E5-base-v2 downloaded", lambda:
    "OK" if model_path.exists() and any(model_path.iterdir()) else None
)

# ── 7. Audio / Image encoders ─────────────────────────────────
print("\n[ 7 ] Audio / Image encoder packages")
check("whisper (openai)", lambda: __import__("whisper") and "OK")
check("librosa",          lambda: __import__("librosa").__version__)
check("ultralytics",      lambda: __import__("ultralytics").__version__)

print("\n" + "="*60)
print("  Done. Fix any ❌ before starting.")
print("="*60 + "\n")
