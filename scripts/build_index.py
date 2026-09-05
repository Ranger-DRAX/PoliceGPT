# -*- coding: utf-8 -*-
"""
PoliceGPT - Hybrid Index Builder CLI
=====================================
Pipeline: Parquet Embeddings -> FAISS IndexFlatIP + Sparse Lexical Inverted Index

Usage:
    python scripts/build_index.py
    python scripts/build_index.py --config configs/retrieval.yaml
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

# Ensure package root is in sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT))

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.rule import Rule
from rich import box
import yaml

from policegpt_rag.indexing.faiss_index import FAISSVectorIndex
from policegpt_rag.indexing.sparse_index import SparseLexicalIndex

console = Console(highlight=False)


def parse_args():
    parser = argparse.ArgumentParser(
        description="PoliceGPT — Compile Hybrid Search Indexes from Embeddings",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/retrieval.yaml",
        help="Path to retrieval configuration file.",
    )
    parser.add_argument(
        "--embeddings-dir",
        type=str,
        default=None,
        help="Override path to Parquet embeddings directory.",
    )
    parser.add_argument(
        "--index-dir",
        type=str,
        default=None,
        help="Override output directory for compiled indexes.",
    )
    return parser.parse_args()


def format_size(bytes_num: int) -> str:
    """Format bytes into readable MB or KB."""
    if bytes_num >= 1024 * 1024:
        return f"{bytes_num / (1024 * 1024):.2f} MB"
    elif bytes_num >= 1024:
        return f"{bytes_num / 1024:.1f} KB"
    return f"{bytes_num} B"


def main():
    args = parse_args()

    console.print(Panel.fit(
        "[bold cyan]PoliceGPT[/bold cyan] - Hybrid Index Compiler\n"
        "[dim]Parquet Embeddings -> FAISS IndexFlatIP (Dense) + Inverted Index (Sparse)[/dim]",
        border_style="cyan",
    ))

    # Load configuration
    cfg_path = REPO_ROOT / args.config
    cfg = {}
    if cfg_path.is_file():
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}

    idx_cfg = cfg.get("indexing", {})
    paths_cfg = idx_cfg.get("paths", {})
    embeddings_dir = Path(args.embeddings_dir or paths_cfg.get("embeddings_dir", "data/processed/embeddings"))
    if not embeddings_dir.is_absolute():
        embeddings_dir = REPO_ROOT / embeddings_dir

    index_dir = Path(args.index_dir or paths_cfg.get("index_dir", "data/processed/indexes"))
    if not index_dir.is_absolute():
        index_dir = REPO_ROOT / index_dir

    chunks_dir = REPO_ROOT / "data" / "processed"

    # Discover Parquet files
    parquet_files = sorted(list(embeddings_dir.glob("*_embeddings.parquet")))
    if not parquet_files:
        console.print(f"[bold red]No Parquet embedding files found in:[/bold red] {embeddings_dir}")
        console.print("Run embedding generation first: [cyan]python scripts/build_embeddings.py --all[/cyan]")
        sys.exit(1)

    console.print(f"Found [bold green]{len(parquet_files)}[/bold green] Parquet embedding file(s) in {embeddings_dir.name}/")
    for pf in parquet_files:
        console.print(f"  • [cyan]{pf.name}[/cyan] ({format_size(pf.stat().st_size)})")

    t_start = time.perf_counter()

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        # Step 1: Build FAISS dense index
        task1 = progress.add_task("[bold cyan]Building FAISS dense index (IndexFlatIP)...[/bold cyan]", total=None)
        dense_index = FAISSVectorIndex(dimension=1024)
        dense_index.build_from_parquet_files(parquet_files, chunks_json_dir=chunks_dir)
        dense_index.save(index_dir)
        progress.update(task1, completed=1)

        # Step 2: Build Sparse lexical inverted index
        task2 = progress.add_task("[bold yellow]Building Sparse lexical inverted index...[/bold yellow]", total=None)
        sparse_index = SparseLexicalIndex()
        sparse_index.build_from_parquet_files(parquet_files)
        sparse_index.save(index_dir)
        progress.update(task2, completed=1)

    elapsed = time.perf_counter() - t_start

    # Print Summary Table
    faiss_file = index_dir / "dense_index.faiss"
    meta_file = index_dir / "chunks_metadata.json"
    sparse_file = index_dir / "sparse_index.json"

    console.print(Rule("[bold green]== Hybrid Index Compilation Summary ==[/bold green]"))
    table = Table(box=box.ROUNDED, show_lines=True, header_style="bold white on dark_blue")
    table.add_column("Artifact", style="cyan")
    table.add_column("Type", justify="center")
    table.add_column("Records / Terms", justify="right")
    table.add_column("Disk Size", justify="right")
    table.add_column("Status", justify="center")

    table.add_row(
        faiss_file.name,
        "FAISS IndexFlatIP (Dense)",
        f"{dense_index.total_vectors:,} vectors (1024-d)",
        format_size(faiss_file.stat().st_size) if faiss_file.is_file() else "—",
        "[green]READY[/green]",
    )
    table.add_row(
        sparse_file.name,
        "Inverted Lexical (Sparse)",
        f"{len(sparse_index.inverted_index):,} terms ({sparse_index.total_docs} docs)",
        format_size(sparse_file.stat().st_size) if sparse_file.is_file() else "—",
        "[green]READY[/green]",
    )
    table.add_row(
        meta_file.name,
        "Metadata & Text Registry",
        f"{len(dense_index.metadata_registry):,} chunks",
        format_size(meta_file.stat().st_size) if meta_file.is_file() else "—",
        "[green]READY[/green]",
    )

    console.print(table)
    console.print(
        f"\n[bold green]✓ Hybrid indexes compiled successfully in {elapsed:.2f}s![/bold green]"
    )
    console.print(f"Artifacts directory: [cyan]{index_dir}[/cyan]")
    console.print("You can now search using: [bold yellow]python scripts/search_cli.py --query \"চুরির শাস্তি কি?\"[/bold yellow]\n")


if __name__ == "__main__":
    main()
