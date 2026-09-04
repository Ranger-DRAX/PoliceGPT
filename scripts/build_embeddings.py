# -*- coding: utf-8 -*-
"""
PoliceGPT - BGE-M3 Embedding Generation CLI
============================================
Pipeline: Chunks JSON -> BGEM3Embedder -> Resume Check -> PyArrow Parquet

Usage:
    python scripts/build_embeddings.py --doc police_act_1861
    python scripts/build_embeddings.py --all
    python scripts/build_embeddings.py --all --batch-size 2
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
from typing import List, Dict, Any

# Ensure src is in sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT))

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn
from rich.rule import Rule
from rich import box
import yaml

from policegpt_rag.embedding.embed import BGEM3Embedder
from policegpt_rag.embedding.batch_runner import EmbeddingBatchRunner

console = Console(highlight=False)


def parse_args():
    parser = argparse.ArgumentParser(
        description="PoliceGPT — BGE-M3 Chunk Embedding CLI",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--doc",
        type=str,
        help="Process a single document by doc_id (e.g., 'police_act_1861').",
    )
    group.add_argument(
        "--all",
        action="store_true",
        default=False,
        help="Process all available chunk files in data/processed/ (default).",
    )
    group.add_argument(
        "--check-gpu",
        action="store_true",
        default=False,
        help="Check CUDA GPU availability, VRAM, and driver status without running embeddings.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override batch size (default: 4 from config, use 2 for ultra-low VRAM).",
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=["cuda", "cpu"],
        default=None,
        help="Force device ('cuda' or 'cpu').",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/embedding.yaml",
        help="Path to embedding configuration file.",
    )
    return parser.parse_args()


def print_gpu_diagnostic(console: Console) -> bool:
    """Print detailed PyTorch and CUDA hardware status for GTX 1050 Ti."""
    try:
        import torch
    except ImportError:
        console.print(Panel(
            "[bold red]PyTorch is not installed in the active environment![/bold red]\n\n"
            "To install PyTorch with CUDA for your GTX 1050 Ti (4GB):\n"
            "  [bold green]pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118[/bold green]",
            title="[bold red]GPU Check: PyTorch Missing[/bold red]",
            border_style="red",
        ))
        return False

    cuda_available = torch.cuda.is_available()
    torch_version = torch.__version__
    cuda_version = getattr(torch.version, "cuda", "N/A")

    if cuda_available:
        device_name = torch.cuda.get_device_name(0)
        props = torch.cuda.get_device_properties(0)
        total_vram_gb = props.total_memory / (1024**3)
        try:
            free_mem, total_mem = torch.cuda.mem_get_info()
            free_vram_gb = free_mem / (1024**3)
        except Exception:
            free_vram_gb = total_vram_gb

        info_lines = [
            f"[bold green]✓ CUDA is AVAILABLE and READY[/bold green]",
            f"• [cyan]GPU Model:[/cyan]    [bold]{device_name}[/bold]",
            f"• [cyan]Total VRAM:[/cyan]   {total_vram_gb:.2f} GB ({int(total_vram_gb * 1024)} MB)",
            f"• [cyan]Free VRAM:[/cyan]    {free_vram_gb:.2f} GB ({int(free_vram_gb * 1024)} MB)",
            f"• [cyan]PyTorch:[/cyan]      {torch_version} (CUDA {cuda_version})",
            f"• [cyan]Optimizations:[/cyan] FP16 active | Dynamic allocator enabled | Small batches (<=4)",
        ]
        console.print(Panel(
            "\n".join(info_lines),
            title="[bold green]Hardware Accelerator: NVIDIA GPU Active[/bold green]",
            border_style="green",
        ))
        return True
    else:
        info_lines = [
            f"[bold yellow]CUDA is NOT active in this Python environment.[/bold yellow]",
            f"• [cyan]PyTorch installed:[/cyan] {torch_version} (CPU build, no CUDA runtime)",
            "",
            f"[bold]To activate CUDA for your NVIDIA GTX 1050 Ti (4GB VRAM):[/bold]",
            f"Run this command inside your activated virtual environment:",
            f"  [bold green]pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118[/bold green]",
            f"  [bold green]pip install FlagEmbedding[/bold green]",
            "",
            f"[dim]Note: Embedding will automatically run on CPU until CUDA PyTorch is installed.[/dim]",
        ]
        console.print(Panel(
            "\n".join(info_lines),
            title="[bold yellow]Hardware Accelerator: CPU Mode (CUDA Inactive)[/bold yellow]",
            border_style="yellow",
        ))
        return False


def discover_chunk_files(processed_dir: Path) -> List[Path]:
    """Find all *_chunks.json files in data/processed/."""
    if not processed_dir.is_dir():
        return []
    return sorted(list(processed_dir.glob("*_chunks.json")))


def print_summary_table(results: List[Dict[str, Any]], total_elapsed: float):
    console.print(Rule("[bold green]== Embedding Pipeline Summary ==[/bold green]"))
    table = Table(box=box.ROUNDED, show_lines=True, header_style="bold white on dark_green")
    table.add_column("Document", style="cyan")
    table.add_column("Total Chunks", justify="right")
    table.add_column("Newly Embedded", justify="right")
    table.add_column("Skipped (Cached)", justify="right")
    table.add_column("Dim", justify="center")
    table.add_column("OOM Events", justify="center")
    table.add_column("Final Batch", justify="center")
    table.add_column("Time", justify="right")

    total_chunks = 0
    total_embedded = 0
    total_skipped = 0

    for r in results:
        total_chunks += r["total_chunks"]
        total_embedded += r["newly_embedded"]
        total_skipped += r["skipped_chunks"]

        oom_str = (
            f"[red]{r.get('oom_count', 0)}[/red]"
            if r.get("oom_count", 0) > 0
            else "[dim]0[/dim]"
        )

        table.add_row(
            r["source_file"],
            str(r["total_chunks"]),
            f"[bold green]{r['newly_embedded']}[/bold green]",
            f"[dim]{r['skipped_chunks']}[/dim]",
            str(r.get("embedding_dim", 1024)),
            oom_str,
            str(r.get("final_batch_size", 4)),
            f"{r.get('elapsed_sec', 0):.1f}s",
        )

    console.print(table)
    console.print(
        f"\n[bold]Total:[/bold] {len(results)} doc(s), "
        f"{total_chunks:,} chunks ({total_embedded:,} embedded, {total_skipped:,} cached) "
        f"in [bold green]{total_elapsed:.1f}s[/bold green].\n"
    )


def main():
    args = parse_args()

    console.print(Panel.fit(
        "[bold cyan]PoliceGPT[/bold cyan] - BGE-M3 Multilingual Embedding Layer\n"
        "[dim]Chunks JSON -> BGEM3FlagModel (fp16/cpu fallback) -> PyArrow Parquet[/dim]",
        border_style="cyan",
    ))

    # Display GPU hardware diagnostics
    gpu_ready = print_gpu_diagnostic(console)

    if args.check_gpu:
        sys.exit(0 if gpu_ready else 1)

    # Load configuration
    cfg_path = REPO_ROOT / args.config
    cfg = {}
    if cfg_path.is_file():
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}

    # Override CLI flags
    model_name = cfg.get("model_name", "BAAI/bge-m3")
    device = args.device or cfg.get("device", "cuda")
    batch_size = args.batch_size or cfg.get("batch_size", 4)
    use_fp16 = cfg.get("use_fp16", True)
    max_length = cfg.get("max_length", 512)
    return_dense = cfg.get("return_dense", True)
    return_sparse = cfg.get("return_sparse", True)
    return_colbert_vecs = cfg.get("return_colbert_vecs", False)
    normalize_embeddings = cfg.get("normalize_embeddings", True)

    paths_cfg = cfg.get("paths", {})
    input_dir = REPO_ROOT / paths_cfg.get("input_dir", "data/processed")
    output_dir = REPO_ROOT / paths_cfg.get("output_dir", "data/processed/embeddings")

    # Determine files to process
    if args.doc:
        candidate = input_dir / f"{args.doc}_chunks.json"
        if not candidate.is_file():
            # Try exact name if provided
            candidate = input_dir / f"{args.doc}.json"
        if not candidate.is_file():
            console.print(f"[bold red]ERROR:[/bold red] Chunks file for '{args.doc}' not found at: {candidate}")
            sys.exit(1)
        chunk_files = [candidate]
    else:
        # Default or --all: discover all chunk files
        chunk_files = discover_chunk_files(input_dir)
        if not chunk_files:
            console.print(f"[bold yellow]No chunk JSON files found in:[/bold yellow] {input_dir}")
            console.print("Run ingestion first: [cyan]python scripts/run_ingestion.py --all --save-chunks[/cyan]")
            sys.exit(0)

    console.print(f"Discovered [bold green]{len(chunk_files)}[/bold green] document(s) to process.")
    console.print(f"Device: [cyan]{device}[/cyan] | Batch size: [cyan]{batch_size}[/cyan] | Output: [cyan]{output_dir}[/cyan]\n")

    # Initialize Embedder and Runner
    embedder = BGEM3Embedder(
        model_name=model_name,
        device=device,
        use_fp16=use_fp16,
        batch_size=batch_size,
        max_length=max_length,
        return_dense=return_dense,
        return_sparse=return_sparse,
        return_colbert_vecs=return_colbert_vecs,
        normalize_embeddings=normalize_embeddings,
    )

    runner = EmbeddingBatchRunner(
        embedder=embedder,
        output_dir=output_dir,
    )

    results: List[Dict[str, Any]] = []
    overall_start = time.perf_counter()

    for idx, chunk_file in enumerate(chunk_files, 1):
        console.print(Rule(f"[bold cyan][{idx}/{len(chunk_files)}] Processing: {chunk_file.name}[/bold cyan]"))
        doc_start = time.perf_counter()

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("{task.completed}/{task.total} chunks"),
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("Embedding chunks...", total=100)

            def update_progress(completed: int, total: int):
                progress.update(task, completed=completed, total=total)

            try:
                report = runner.process_document(
                    chunk_file,
                    progress_callback=update_progress,
                )
                report["elapsed_sec"] = time.perf_counter() - doc_start
                results.append(report)
            except Exception as e:
                console.print(f"[bold red]Failed processing {chunk_file.name}:[/bold red] {e}")

    overall_elapsed = time.perf_counter() - overall_start
    if results:
        print_summary_table(results, overall_elapsed)


if __name__ == "__main__":
    main()
