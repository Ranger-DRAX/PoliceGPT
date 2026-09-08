# -*- coding: utf-8 -*-
"""
PoliceGPT - FastAPI Server Runner
==================================
Launch the PoliceGPT Legal RAG API server using Uvicorn.

Usage:
    python scripts/run_api.py
    python scripts/run_api.py --port 8000 --reload
"""

import os
os.environ.setdefault("PYTHONIOENCODING", "utf-8")
os.environ["PYTHONUTF8"] = "1"

import sys
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except AttributeError:
        pass

import argparse
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT))

from rich.console import Console
from rich.panel import Panel
import uvicorn

console = Console(highlight=False)


def parse_args():
    parser = argparse.ArgumentParser(
        description="PoliceGPT — FastAPI Server Runner",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--host",
        type=str,
        default=os.environ.get("APP_HOST", "0.0.0.0"),
        help="Host interface to bind the server (default: 0.0.0.0).",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.environ.get("APP_PORT", 8000)),
        help="Port to run the server on (default: 8000).",
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        default=False,
        help="Enable auto-reload for local code development.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of worker processes (default: 1).",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    console.print(Panel.fit(
        f"[bold cyan]PoliceGPT[/bold cyan] — Legal RAG API Server\n"
        f"[green]• Address:[/green] http://{args.host}:{args.port}\n"
        f"[green]• Swagger Docs:[/green] http://localhost:{args.port}/docs\n"
        f"[green]• ReDoc:[/green] http://localhost:{args.port}/redoc\n"
        f"[green]• Health Check:[/green] http://localhost:{args.port}/api/v1/health\n"
        f"[dim]Dense FAISS (BGE-M3) + Sparse RRF + Google Gemini 3.5 Flash + Legal Guardrails[/dim]",
        border_style="cyan",
    ))

    uvicorn.run(
        "api.main:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        workers=args.workers if not args.reload else 1,
    )


if __name__ == "__main__":
    main()
