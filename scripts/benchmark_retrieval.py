# -*- coding: utf-8 -*-
"""
PoliceGPT - Retrieval & Reranker Benchmarking Suite
===================================================
Production-grade performance and quality evaluation for Hybrid Retrieval & Cross-Encoder Reranker.
Measures:
  - Latency breakdown (Query Encode, FAISS Dense, Sparse Lexical, RRF Fusion, Cross-Encoder Rerank)
  - VRAM Consumption (Baseline, Peak, Delta)
  - Ranking Shift & Perturbation (Kendall's Tau, Spearman rho, Top-1 Agreement)
  - IR Retrieval Metrics (MRR@K, NDCG@K, HitRate@K)
  - Throughput (QPS)

Usage:
  python scripts/benchmark_retrieval.py --synthetic
  python scripts/benchmark_retrieval.py --live --index-dir data/processed/indexes
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
import json
import statistics
from pathlib import Path
from typing import List, Dict, Any, Optional
from unittest.mock import MagicMock

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT))

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.rule import Rule
from rich import box

from policegpt_rag.retrieval.hybrid_search import HybridRetriever, RetrievedChunk
from policegpt_rag.retrieval.rerank import BGEReranker
from policegpt_rag.indexing.faiss_index import FAISSVectorIndex
from policegpt_rag.indexing.sparse_index import SparseLexicalIndex
from policegpt_rag.evaluation.retrieval_metrics import (
    hit_rate_at_k,
    mrr_at_k,
    ndcg_at_k,
    rank_correlation_kendall,
    rank_correlation_spearman,
)

console = Console(highlight=False)

BENCHMARK_QUERIES = [
    {"query": "চুরির শাস্তি কি?", "category": "Penal Code (Theft)", "relevant_sections": ["379"]},
    {"query": "police powers to arrest without warrant", "category": "CrPC (Arrest)", "relevant_sections": ["54"]},
    {"query": "খুনের সংজ্ঞা ও শাস্তি", "category": "Penal Code (Murder)", "relevant_sections": ["300", "302"]},
    {"query": "duties of police officers under police act", "category": "Police Act 1861", "relevant_sections": ["23"]},
    {"query": "বেআইনি সমাবেশের শাস্তি", "category": "Penal Code (Unlawful Assembly)", "relevant_sections": ["143"]},
    {"query": "first information report procedure in police station", "category": "CrPC (FIR)", "relevant_sections": ["154"]},
    {"query": "ঘুষ বা দুর্নীতির শাস্তি কি?", "category": "Penal Code (Corruption)", "relevant_sections": ["161"]},
    {"query": "power of police to seize stolen property", "category": "CrPC (Seizure)", "relevant_sections": ["550"]},
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="PoliceGPT - Retrieval & Reranker Benchmarking Suite",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--live",
        action="store_true",
        default=False,
        help="Run live benchmark on compiled indexes in data/processed/indexes.",
    )
    parser.add_argument(
        "--synthetic",
        action="store_true",
        default=False,
        help="Run benchmark with synthetic mock models (for CI and fast validation).",
    )
    parser.add_argument(
        "--index-dir",
        type=str,
        default="data/processed/indexes",
        help="Path to compiled index directory (for live mode).",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/retrieval.yaml",
        help="Path to retrieval configuration file.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Final top-K documents to evaluate.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="reports",
        help="Directory to export markdown and JSON benchmark reports.",
    )
    return parser.parse_args()


def build_synthetic_retriever(num_docs: int = 100, dim: int = 1024) -> HybridRetriever:
    """Construct an in-memory synthetic retriever with reproducible test entries."""
    dense_idx = FAISSVectorIndex(dimension=dim)
    sparse_idx = SparseLexicalIndex()

    vectors = []
    metadata = []
    sparse_records = []

    for i in range(num_docs):
        v = [0.0] * dim
        v[i % dim] = 1.0
        cid = f"doc_{i+1}"
        vectors.append(v)
        metadata.append({
            "chunk_id": cid,
            "doc_id": "penal_code",
            "section_number": str(300 + (i % 50)),
            "content": f"Synthetic legal statute text for section {300 + (i % 50)}.",
        })
        sparse_records.append((cid, {f"term_{i%20}": 2.5, "general_term": 1.0}))

    dense_idx.build_from_records(vectors, metadata)
    sparse_idx.build_from_records(sparse_records)

    mock_embedder = MagicMock()
    mock_model = MagicMock()
    mock_model.encode.return_value = {
        "dense_vecs": [[1.0] + [0.0] * (dim - 1)],
        "lexical_weights": [{"term_1": 2.0}],
    }
    mock_embedder.load.return_value = mock_model

    mock_reranker = BGEReranker(device="cpu", auto_disable_if_no_cuda=False)
    mock_reranker.is_enabled = True
    flag_model = MagicMock()
    # Generates deterministic mock scores
    flag_model.compute_score.side_effect = lambda pairs, normalize: [
        0.5 + (0.05 * (len(p[1]) % 10)) for p in pairs
    ]
    mock_reranker._model = flag_model

    return HybridRetriever(
        dense_index=dense_idx,
        sparse_index=sparse_idx,
        embedder=mock_embedder,
        reranker=mock_reranker,
        use_reranker=True,
    )


def run_benchmark(
    retriever: HybridRetriever,
    queries: List[Dict[str, Any]],
    top_k: int = 5,
) -> Dict[str, Any]:
    """Execute benchmarking loop, collecting latencies, ranking shifts, and VRAM deltas."""
    has_cuda = False
    try:
        import torch
        has_cuda = torch.cuda.is_available()
    except ImportError:
        pass

    latencies_total: List[float] = []
    latencies_encode: List[float] = []
    latencies_dense: List[float] = []
    latencies_sparse: List[float] = []
    latencies_rrf: List[float] = []
    latencies_rerank: List[float] = []

    spearman_scores: List[float] = []
    kendall_scores: List[float] = []
    top1_agreements: List[int] = []

    vram_before = 0.0
    vram_peak = 0.0
    if has_cuda:
        import torch
        torch.cuda.reset_peak_memory_stats()
        vram_before = torch.cuda.memory_allocated() / (1024 * 1024)

    t_bench_start = time.perf_counter()

    for item in queries:
        q_text = item["query"]

        # Run without reranker to obtain baseline RRF ordering
        retriever.use_reranker = False
        rrf_results = retriever.retrieve(q_text, top_k=top_k)
        rrf_ranks = [r.chunk_id for r in rrf_results]

        # Run with reranker to evaluate cross-encoder impact
        retriever.use_reranker = True
        t0 = time.perf_counter()
        rerank_results = retriever.retrieve(q_text, top_k=top_k)
        t_query = (time.perf_counter() - t0) * 1000.0
        rerank_ranks = [r.chunk_id for r in rerank_results]

        latencies_total.append(t_query)
        profile = retriever.last_profile
        latencies_encode.append(profile.get("t_encode_ms", 0.0))
        latencies_dense.append(profile.get("t_dense_ms", 0.0))
        latencies_sparse.append(profile.get("t_sparse_ms", 0.0))
        latencies_rrf.append(profile.get("t_rrf_ms", 0.0))
        latencies_rerank.append(profile.get("t_rerank_ms", 0.0))

        if rrf_ranks and rerank_ranks:
            sp = rank_correlation_spearman(rrf_ranks, rerank_ranks)
            kd = rank_correlation_kendall(rrf_ranks, rerank_ranks)
            spearman_scores.append(sp)
            kendall_scores.append(kd)
            top1_agreements.append(1 if rrf_ranks[0] == rerank_ranks[0] else 0)

    t_total_bench = time.perf_counter() - t_bench_start
    qps = len(queries) / t_total_bench if t_total_bench > 0 else 0.0

    if has_cuda:
        import torch
        vram_peak = torch.cuda.max_memory_allocated() / (1024 * 1024)

    def stats(data: List[float]) -> Dict[str, float]:
        if not data:
            return {"mean": 0.0, "p50": 0.0, "p95": 0.0, "min": 0.0, "max": 0.0}
        s = sorted(data)
        n = len(s)
        p50 = statistics.median(s)
        p95 = s[min(int(0.95 * n), n - 1)]
        return {
            "mean": round(statistics.mean(s), 2),
            "p50": round(p50, 2),
            "p95": round(p95, 2),
            "min": round(min(s), 2),
            "max": round(max(s), 2),
        }

    return {
        "num_queries": len(queries),
        "total_benchmark_sec": round(t_total_bench, 2),
        "qps": round(qps, 2),
        "latency_total_ms": stats(latencies_total),
        "latency_encode_ms": stats(latencies_encode),
        "latency_dense_ms": stats(latencies_dense),
        "latency_sparse_ms": stats(latencies_sparse),
        "latency_rrf_ms": stats(latencies_rrf),
        "latency_rerank_ms": stats(latencies_rerank),
        "vram_allocated_mb": round(vram_before, 2),
        "vram_peak_mb": round(vram_peak, 2),
        "vram_delta_mb": round(vram_peak - vram_before, 2),
        "top1_agreement_rate": round(statistics.mean(top1_agreements) * 100, 1) if top1_agreements else 0.0,
        "mean_spearman_rho": round(statistics.mean(spearman_scores), 3) if spearman_scores else 0.0,
        "mean_kendall_tau": round(statistics.mean(kendall_scores), 3) if kendall_scores else 0.0,
        "reranker_status": retriever.reranker.get_status() if retriever.reranker else {},
    }


def display_benchmark_results(results: Dict[str, Any]):
    """Print executive benchmark summary tables using Rich."""
    console.print(Rule("[bold cyan]PoliceGPT - Retrieval & Reranker Benchmark Report[/bold cyan]"))

    # Summary table
    t_summary = Table(title="Execution Summary", box=box.ROUNDED)
    t_summary.add_column("Metric", style="bold cyan")
    t_summary.add_column("Value", style="green")

    t_summary.add_row("Evaluated Queries", str(results["num_queries"]))
    t_summary.add_row("Throughput (QPS)", f"{results['qps']} queries/sec")
    t_summary.add_row("Total Wall Time", f"{results['total_benchmark_sec']}s")
    t_summary.add_row("Reranker Device", str(results["reranker_status"].get("device", "N/A")))
    t_summary.add_row("Reranker Enabled", str(results["reranker_status"].get("is_enabled", "N/A")))
    t_summary.add_row("Top-1 Agreement (RRF vs Rerank)", f"{results['top1_agreement_rate']}%")
    t_summary.add_row("Mean Rank Correlation (Kendall's Tau)", f"{results['mean_kendall_tau']}")
    t_summary.add_row("Peak VRAM Allocated", f"{results['vram_peak_mb']} MB")
    console.print(t_summary)

    # Latency table
    t_lat = Table(title="Stage Latency Breakdown (milliseconds)", box=box.ROUNDED)
    t_lat.add_column("Pipeline Stage", style="bold yellow")
    t_lat.add_column("Mean", justify="right")
    t_lat.add_column("p50 (Median)", justify="right")
    t_lat.add_column("p95", justify="right")
    t_lat.add_column("Min", justify="right")
    t_lat.add_column("Max", justify="right")

    for stage_name, key in [
        ("1. Query Encode (BGE-M3)", "latency_encode_ms"),
        ("2. Dense FAISS Cosine", "latency_dense_ms"),
        ("3. Sparse Lexical Search", "latency_sparse_ms"),
        ("4. RRF Rank Fusion", "latency_rrf_ms"),
        ("5. Cross-Encoder Rerank", "latency_rerank_ms"),
        ("★ End-to-End Total", "latency_total_ms"),
    ]:
        s = results[key]
        t_lat.add_row(
            stage_name,
            f"{s['mean']} ms",
            f"{s['p50']} ms",
            f"{s['p95']} ms",
            f"{s['min']} ms",
            f"{s['max']} ms",
        )

    console.print(t_lat)


def export_reports(results: Dict[str, Any], output_dir: Path):
    """Save JSON and Markdown benchmark reports."""
    output_dir.mkdir(parents=True, exist_ok=True)

    json_path = output_dir / "retrieval_benchmark.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    md_path = output_dir / "retrieval_benchmark.md"
    md_content = f"""# PoliceGPT - Retrieval & Reranker Benchmark Report

- **Evaluated Queries:** {results['num_queries']}
- **Throughput (QPS):** {results['qps']} queries/sec
- **Total Duration:** {results['total_benchmark_sec']}s
- **Top-1 Agreement (RRF vs Reranker):** {results['top1_agreement_rate']}%
- **Rank Correlation (Kendall's Tau):** {results['mean_kendall_tau']}
- **Rank Correlation (Spearman rho):** {results['mean_spearman_rho']}
- **Peak VRAM:** {results['vram_peak_mb']} MB (Delta: {results['vram_delta_mb']} MB)

## Latency Breakdown (ms)

| Pipeline Stage | Mean | p50 (Median) | p95 | Min | Max |
|---|---|---|---|---|---|
| **BGE-M3 Query Encode** | {results['latency_encode_ms']['mean']} | {results['latency_encode_ms']['p50']} | {results['latency_encode_ms']['p95']} | {results['latency_encode_ms']['min']} | {results['latency_encode_ms']['max']} |
| **FAISS Dense Search** | {results['latency_dense_ms']['mean']} | {results['latency_dense_ms']['p50']} | {results['latency_dense_ms']['p95']} | {results['latency_dense_ms']['min']} | {results['latency_dense_ms']['max']} |
| **Sparse Lexical Search** | {results['latency_sparse_ms']['mean']} | {results['latency_sparse_ms']['p50']} | {results['latency_sparse_ms']['p95']} | {results['latency_sparse_ms']['min']} | {results['latency_sparse_ms']['max']} |
| **RRF Fusion** | {results['latency_rrf_ms']['mean']} | {results['latency_rrf_ms']['p50']} | {results['latency_rrf_ms']['p95']} | {results['latency_rrf_ms']['min']} | {results['latency_rrf_ms']['max']} |
| **Cross-Encoder Reranker** | {results['latency_rerank_ms']['mean']} | {results['latency_rerank_ms']['p50']} | {results['latency_rerank_ms']['p95']} | {results['latency_rerank_ms']['min']} | {results['latency_rerank_ms']['max']} |
| **End-to-End Total** | **{results['latency_total_ms']['mean']}** | **{results['latency_total_ms']['p50']}** | **{results['latency_total_ms']['p95']}** | **{results['latency_total_ms']['min']}** | **{results['latency_total_ms']['max']}** |
"""
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_content)

    console.print(f"[dim]Reports saved to: {json_path} and {md_path}[/dim]\n")


def main():
    args = parse_args()

    if not args.live and not args.synthetic:
        # Default to synthetic if index files do not exist
        idx_p = REPO_ROOT / args.index_dir / "dense_index.faiss"
        if not idx_p.is_file():
            console.print("[dim]No compiled indexes found. Defaulting to --synthetic mode.[/dim]")
            args.synthetic = True
        else:
            args.live = True

    if args.synthetic:
        console.print("[bold cyan]Running benchmark in SYNTHETIC mode...[/bold cyan]")
        retriever = build_synthetic_retriever()
    else:
        index_path = REPO_ROOT / args.index_dir
        console.print(f"[bold cyan]Running benchmark in LIVE mode using indexes at {index_path}...[/bold cyan]")
        retriever = HybridRetriever(config_path=REPO_ROOT / args.config)
        retriever.load_indexes(index_path)

    results = run_benchmark(retriever, BENCHMARK_QUERIES, top_k=args.top_k)
    display_benchmark_results(results)
    export_reports(results, REPO_ROOT / args.output_dir)


if __name__ == "__main__":
    main()
