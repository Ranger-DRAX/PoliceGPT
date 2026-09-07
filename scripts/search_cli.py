# -*- coding: utf-8 -*-
"""
PoliceGPT - Interactive Hybrid Legal Search CLI
================================================
Query Bangladesh legal statutes using Hybrid (FAISS Dense + Sparse Lexical RRF) search,
with optional Cross-Encoder precision reranking and latency telemetry.

Usage:
    python scripts/search_cli.py --query "চুরির শাস্তি কি?"
    python scripts/search_cli.py --query "police powers to arrest without warrant" --no-reranker
    python scripts/search_cli.py --interactive
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
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT))

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from rich.rule import Rule
from rich.prompt import Prompt
from rich import box

from policegpt_rag.retrieval.hybrid_search import HybridRetriever, RetrievedChunk

console = Console(highlight=False)


def parse_args():
    parser = argparse.ArgumentParser(
        description="PoliceGPT — Hybrid Legal Search CLI (Bengali & English)",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--query",
        "-q",
        type=str,
        default=None,
        help="Legal query string to search (in Bengali or English).",
    )
    parser.add_argument(
        "--top-k",
        "-k",
        type=int,
        default=5,
        help="Number of legal chunks to retrieve (default: 5).",
    )
    parser.add_argument(
        "--interactive",
        "-i",
        action="store_true",
        default=False,
        help="Launch interactive terminal query session.",
    )
    parser.add_argument(
        "--index-dir",
        type=str,
        default="data/processed/indexes",
        help="Path to compiled index directory.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/retrieval.yaml",
        help="Path to retrieval configuration file.",
    )
    parser.add_argument(
        "--reranker",
        dest="use_reranker",
        action="store_true",
        default=None,
        help="Force enable cross-encoder reranker.",
    )
    parser.add_argument(
        "--no-reranker",
        dest="use_reranker",
        action="store_false",
        help="Disable cross-encoder reranker (pure RRF mode).",
    )
    return parser.parse_args()


def display_results(query: str, results: list, elapsed_sec: float, profile: dict = None):
    console.print(Rule(f"[bold green]Query: \"{query}\"[/bold green] [dim]({len(results)} matches in {elapsed_sec*1000:.1f}ms)[/dim]"))

    if not results:
        console.print("[bold yellow]No relevant legal sections found.[/bold yellow]\n")
        return

    for idx, res in enumerate(results, 1):
        doc_label = res.act_name_en or res.doc_id
        sec_label = f"Section {res.section_number}" if res.section_number else "General Provision"
        title_label = f" — {res.section_title}" if res.section_title else ""

        header_text = f"[bold cyan]#{idx} | {doc_label}[/bold cyan] [bold yellow]({sec_label}{title_label})[/bold yellow]"

        rerank_badge = ""
        if res.rerank_score is not None:
            rerank_badge = f" | [dim]Rerank Score:[/dim] [bold magenta]{res.rerank_score:.4f}[/bold magenta]"

        score_details = (
            f"[dim]RRF Score:[/dim] [bold green]{res.rrf_score:.5f}[/bold green]{rerank_badge} | "
            f"[dim]Dense Cosine:[/dim] {res.dense_score:.4f} (Rank #{res.dense_rank or '—'}) | "
            f"[dim]Sparse Lexical:[/dim] {res.sparse_score or 0:.2f} (Rank #{res.sparse_rank or '—'})"
        )

        content_preview = res.content.strip()
        if len(content_preview) > 400:
            content_preview = content_preview[:400] + " …"

        pages_str = f"Pages: {', '.join(map(str, res.page_numbers))}" if res.page_numbers else ""

        body = f"{score_details}\n\n{content_preview}\n\n[dim]{pages_str} | Chunk: {res.chunk_id}[/dim]"

        console.print(Panel(
            body,
            title=header_text,
            title_align="left",
            border_style="cyan" if idx == 1 else "blue",
            box=box.ROUNDED,
        ))

    if profile:
        console.print(
            f"[dim]Latency Breakdown: Encode: {profile.get('t_encode_ms', 0)}ms | "
            f"FAISS: {profile.get('t_dense_ms', 0)}ms | "
            f"Sparse: {profile.get('t_sparse_ms', 0)}ms | "
            f"RRF: {profile.get('t_rrf_ms', 0)}ms | "
            f"Rerank: {profile.get('t_rerank_ms', 0)}ms | "
            f"Total: {profile.get('t_total_ms', 0)}ms[/dim]\n"
        )
    else:
        console.print()


def run_search_session(retriever: HybridRetriever, top_k: int):
    console.print("[bold green]Interactive Legal Search Ready.[/bold green] Type 'exit' or 'quit' to finish.\n")
    while True:
        try:
            query = Prompt.ask("[bold yellow]Enter Legal Query[/bold yellow]").strip()
            if not query:
                continue
            if query.lower() in ("exit", "quit", "q"):
                console.print("[dim]Exiting search session.[/dim]")
                break

            t0 = time.perf_counter()
            results = retriever.retrieve(query, top_k=top_k)
            t_elapsed = time.perf_counter() - t0

            display_results(query, results, t_elapsed, retriever.last_profile)
        except (KeyboardInterrupt, EOFError):
            console.print("\n[dim]Search terminated.[/dim]")
            break


def main():
    args = parse_args()

    console.print(Panel.fit(
        "[bold cyan]PoliceGPT[/bold cyan] - Hybrid Legal Retrieval CLI\n"
        "[dim]Dense FAISS (BGE-M3 Cosine) + Sparse Lexical Inverted Index (RRF Fusion) + BGE Reranker[/dim]",
        border_style="cyan",
    ))

    index_dir = Path(args.index_dir)
    if not index_dir.is_absolute():
        index_dir = REPO_ROOT / index_dir

    if not (index_dir / "dense_index.faiss").is_file():
        console.print(f"[bold red]Index files not found in:[/bold red] {index_dir}")
        console.print("Please compile indexes first: [bold green]python scripts/build_index.py[/bold green]")
        sys.exit(1)

    console.print(f"Loading indexes from [cyan]{index_dir.name}/[/cyan]...")
    t_load = time.perf_counter()

    retriever = HybridRetriever(config_path=REPO_ROOT / args.config)
    if args.use_reranker is not None:
        retriever.use_reranker = args.use_reranker

    retriever.load_indexes(index_dir)

    reranker_status = "Disabled"
    if retriever.use_reranker and retriever.reranker:
        st = retriever.reranker.get_status()
        if st.get("is_enabled"):
            reranker_status = f"[bold green]Active ({st.get('device')})[/bold green]"
        else:
            reranker_status = f"[bold yellow]Auto-Disabled (No CUDA / Low VRAM)[/bold yellow]"
    else:
        reranker_status = "[dim]Disabled[/dim]"

    console.print(
        f"[green]✓[/green] Loaded [bold]{retriever.dense_index.total_vectors}[/bold] legal chunks "
        f"across [bold]{len(retriever.sparse_index.inverted_index)}[/bold] lexical terms in {time.perf_counter() - t_load:.2f}s | "
        f"Reranker: {reranker_status}\n"
    )

    if args.query:
        t0 = time.perf_counter()
        results = retriever.retrieve(args.query, top_k=args.top_k)
        t_elapsed = time.perf_counter() - t0
        display_results(args.query, results, t_elapsed, retriever.last_profile)
    else:
        run_search_session(retriever, top_k=args.top_k)


if __name__ == "__main__":
    main()
