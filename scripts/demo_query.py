"""
CLI Smoke Test & Interactive Query Interface for PoliceGPT RAG.
"""

import argparse
import sys
from pathlib import Path
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

# Ensure src is in pythonpath
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from policegpt_rag.pipeline import PoliceGPTRAGPipeline

console = Console()


def run_demo(query_text: str, language: str = "bn"):
    console.print(f"[bold cyan]Initializing PoliceGPT RAG Pipeline...[/bold cyan]")
    pipeline = PoliceGPTRAGPipeline()

    console.print(f"\n[bold yellow]Query:[/bold yellow] {query_text}")
    resp = pipeline.query(query_text=query_text, language=language)

    # Print Answer
    console.print(
        Panel(
            resp.answer,
            title="[bold green]PoliceGPT Cited Answer[/bold green]",
            subtitle=f"Confidence: {resp.confidence_score:.2f} | Grounded: {resp.is_grounded}",
            border_style="green",
        )
    )

    # Print Citations & Sources Table
    if resp.retrieved_sources:
        table = Table(title="Retrieved Legal Sources (Top Contexts)")
        table.add_column("Rank", justify="center", style="cyan")
        table.add_column("Act / Law", style="magenta")
        table.add_column("Section / Dhara", style="yellow")
        table.add_column("Snippet Preview", style="white")

        for idx, src in enumerate(resp.retrieved_sources[:3]):
            act = src.get("act_name_bn") or src.get("act_name_en") or "N/A"
            sec = src.get("section_number") or "N/A"
            content = src.get("content", "")[:120] + "..."
            table.add_row(str(idx + 1), act, sec, content)

        console.print(table)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PoliceGPT Interactive Demo Query")
    parser.add_argument(
        "--query",
        default="পেনাল কোড অনুযায়ী চুরির শাস্তি কী?",
        help="Legal query in Bengali or English",
    )
    parser.add_argument("--lang", default="bn", choices=["bn", "en"], help="Response language")
    args = parser.parse_args()

    run_demo(args.query, args.lang)
