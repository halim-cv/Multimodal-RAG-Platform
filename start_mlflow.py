"""
start_mlflow.py

Convenience launcher for the MLflow tracking server.

Usage:
    python start_mlflow.py             # starts on http://localhost:5000
    python start_mlflow.py --port 5001 # custom port

The server stores all runs in ./mlruns/ (local SQLite — no extra DB setup needed).
Open the UI at: http://localhost:5000

Then set in your .env:
    MLFLOW_TRACKING_URI=http://localhost:5000
"""

import subprocess
import sys
import argparse
from pathlib import Path

_ROOT = Path(__file__).parent


def main():
    parser = argparse.ArgumentParser(description="Start the MLflow tracking server")
    parser.add_argument("--port",    type=int, default=5000,         help="Port to listen on (default: 5000)")
    parser.add_argument("--host",    type=str, default="127.0.0.1",  help="Host to bind  (default: 127.0.0.1)")
    parser.add_argument("--backend", type=str, default=f"sqlite:///{_ROOT}/mlruns/mlflow.db",
                        help="Backend store URI  (default: local SQLite)")
    parser.add_argument("--artifacts", type=str, default=str(_ROOT / "mlruns" / "artifacts"),
                        help="Artifact root     (default: ./mlruns/artifacts)")
    args = parser.parse_args()

    cmd = [
        sys.executable, "-m", "mlflow", "server",
        "--host",                  args.host,
        "--port",                  str(args.port),
        "--backend-store-uri",     args.backend,
        "--default-artifact-root", args.artifacts,
    ]

    print("=" * 60)
    print("  MLflow Tracking Server")
    print("=" * 60)
    print(f"  UI  →  http://{args.host}:{args.port}")
    print(f"  DB  →  {args.backend}")
    print(f"  Art →  {args.artifacts}")
    print("=" * 60)
    print("  Add to .env:")
    print(f"  MLFLOW_TRACKING_URI=http://{args.host}:{args.port}")
    print("=" * 60)
    print("  Press Ctrl+C to stop.\n")

    subprocess.run(cmd)


if __name__ == "__main__":
    main()
