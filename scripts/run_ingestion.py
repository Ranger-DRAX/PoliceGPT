# -*- coding: utf-8 -*-
"""
PoliceGPT - Document Processing CLI
=====================================
Pipeline: PyMuPDF -> Quality Detection -> OCR Fallback -> Clean -> Chunk -> Report

Usage:
    python scripts/run_ingestion.py                        # process both default PDFs
    python scripts/run_ingestion.py --doc policeAct
    python scripts/run_ingestion.py --all --report
    python scripts/run_ingestion.py --inspect 5           # print first 5 chunks
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

import json
import argparse
import time
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import List, Optional

# ── make sure the package is importable from the repo root ──────────────────
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn
from rich.rule import Rule
from rich import box

from policegpt_rag.ingestion.parse_pdf import PDFParser
from policegpt_rag.ingestion.quality_check import DocumentQualityChecker, QualityCheckResult
from policegpt_rag.ingestion.ocr_fallback import OCRFallbackEngine
from policegpt_rag.preprocessing.normalize import UnicodeNormalizer
from policegpt_rag.preprocessing.boilerplate import BoilerplateCleaner
from policegpt_rag.preprocessing.chunker import LegalSectionChunker, LegalChunk

# Force UTF-8 safe output on Windows terminals
console = Console(highlight=False)

# ── Document metadata registry ───────────────────────────────────────────────
DOCUMENTS = {
    "policeAct": {
        "path": REPO_ROOT / "data" / "Police_Law" / "policeAct.pdf",
        "doc_id": "police_act_1861",
        "act_name_bn": None,
        "act_name_en": "The Police Act, 1861",
        "act_year": 1861,
        "is_expected_bangla": False,   # English-language statute
    },
    "policeRegulations": {
        "path": REPO_ROOT / "data" / "Police_Law" / "policeRegulations.pdf",
        "doc_id": "police_regulations",
        "act_name_bn": None,
        "act_name_en": "Police Regulations of Bengal",
        "act_year": None,
        "is_expected_bangla": False,  # English-language regulations
    },
    "penalCode": {
        "path": REPO_ROOT / "data" / "Police_Law" / "Bangladesh_The_Penal_Code_1860.pdf",
        "doc_id": "penal_code_1860",
        "act_name_bn": None,
        "act_name_en": "The Penal Code, 1860",
        "act_year": 1860,
        "is_expected_bangla": False,  # English-language code
    },
}


# ── Per-page quality record ──────────────────────────────────────────────────
@dataclass
class PageReport:
    page: int
    text_length: int
    bangla_ratio: float
    valid_char_ratio: float
    passed: bool
    needs_ocr: bool
    reason: str
    ocr_triggered: bool = False
    ocr_chars_recovered: int = 0


@dataclass
class DocumentReport:
    doc_id: str
    source_file: str
    total_pages: int
    pages_passed: int
    pages_ocr_triggered: int
    total_chunks: int
    section_chunks: int
    sliding_window_chunks: int
    elapsed_sec: float
    page_reports: List[PageReport] = field(default_factory=list)
    sample_chunks: List[dict] = field(default_factory=list)


# ── Core pipeline ────────────────────────────────────────────────────────────

def run_pipeline(
    doc_key: str,
    inspect_n: int = 5,
    save_chunks_json: bool = False,
) -> DocumentReport:
    meta = DOCUMENTS[doc_key]
    pdf_path: Path = meta["path"]

    console.print(Rule(f"[bold cyan]>> Processing: {pdf_path.name}[/bold cyan]"))

    quality_checker = DocumentQualityChecker(
        min_text_length_per_page=50,
        min_bangla_unicode_ratio=0.15,
        min_valid_char_ratio=0.80,
    )
    ocr_engine = OCRFallbackEngine(languages=["bn", "en"], dpi=300)
    normalizer = UnicodeNormalizer()
    boilerplate_cleaner = BoilerplateCleaner()
    chunker = LegalSectionChunker(target_chunk_size=512, chunk_overlap=64, min_chunk_size=100)

    t_start = time.perf_counter()
    page_reports: List[PageReport] = []

    # ── Step 1: Extract + quality check ────────────────────────────────────
    console.print("\n[bold yellow][1] PyMuPDF Extraction + Quality Detection[/bold yellow]")

    try:
        import pymupdf as fitz
    except ImportError:
        import fitz

    doc = fitz.open(str(pdf_path))
    total_pages = len(doc)
    parsed_pages = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total} pages"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Extracting + quality check...", total=total_pages)

        for page_idx, page in enumerate(doc):
            page_num = page_idx + 1
            raw_text = page.get_text("text")

            quality: QualityCheckResult = quality_checker.evaluate_text(
                raw_text, is_expected_bangla=meta["is_expected_bangla"]
            )

            final_text = raw_text
            ocr_triggered = False
            ocr_chars_recovered = 0

            if quality.needs_ocr:
                try:
                    pix = page.get_pixmap(dpi=ocr_engine.dpi)
                    img_bytes = pix.tobytes("png")
                    ocr_text = ocr_engine.ocr_image_or_page(img_bytes)
                    if ocr_text.strip():
                        final_text = ocr_text
                        ocr_chars_recovered = len(ocr_text.strip())
                    ocr_triggered = True
                except Exception:
                    pass  # OCR not installed — graceful skip

            page_reports.append(PageReport(
                page=page_num,
                text_length=len(raw_text.strip()),
                bangla_ratio=quality.bangla_ratio,
                valid_char_ratio=quality.valid_char_ratio,
                passed=quality.passed,
                needs_ocr=quality.needs_ocr,
                reason=quality.reason,
                ocr_triggered=ocr_triggered,
                ocr_chars_recovered=ocr_chars_recovered,
            ))
            parsed_pages.append((page_num, final_text))
            progress.advance(task)

    doc.close()

    pages_passed = sum(1 for r in page_reports if r.passed)
    pages_ocr = sum(1 for r in page_reports if r.ocr_triggered)

    _print_quality_table(page_reports, max_rows=20)

    # ── Step 2: Normalize + Boilerplate clean ──────────────────────────────
    console.print("\n[bold yellow][2] Unicode Normalization + Boilerplate Removal[/bold yellow]")
    full_text = ""
    source_pages = []
    for page_num, text in parsed_pages:
        normalized = normalizer.normalize(text)
        cleaned = boilerplate_cleaner.clean(normalized)
        full_text += "\n" + cleaned
        source_pages.append(page_num)

    console.print(
        f"  [green]✓[/green] Combined text: [bold]{len(full_text):,}[/bold] chars across {total_pages} pages"
    )

    # ── Step 3: Legal section chunking ────────────────────────────────────
    console.print("\n[bold yellow][3] Legal Section-Aware Chunking[/bold yellow]")
    chunks: List[LegalChunk] = chunker.chunk_document(
        text=full_text,
        doc_id=meta["doc_id"],
        act_name_bn=meta["act_name_bn"],
        act_name_en=meta["act_name_en"],
        act_year=meta["act_year"],
        source_pages=source_pages,
    )

    section_chunks = sum(1 for c in chunks if c.section_number is not None)
    sliding_chunks = len(chunks) - section_chunks

    console.print(f"  [green]✓[/green] Total chunks produced  : [bold]{len(chunks)}[/bold]")
    console.print(f"  [cyan]↳[/cyan] Section-boundary chunks : [bold]{section_chunks}[/bold]")
    console.print(f"  [cyan]↳[/cyan] Sliding-window fallback : [bold]{sliding_chunks}[/bold]")

    _print_chunk_table(chunks, inspect_n=inspect_n)

    elapsed = time.perf_counter() - t_start

    sample_dicts = [c.model_dump() for c in chunks[:inspect_n]]
    if save_chunks_json:
        out_dir = REPO_ROOT / "data" / "processed"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{meta['doc_id']}_chunks.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump([c.model_dump() for c in chunks], f, ensure_ascii=False, indent=2)
        console.print(f"\n  [bold green]Chunks saved ->[/bold green] {out_path}")

    return DocumentReport(
        doc_id=meta["doc_id"],
        source_file=pdf_path.name,
        total_pages=total_pages,
        pages_passed=pages_passed,
        pages_ocr_triggered=pages_ocr,
        total_chunks=len(chunks),
        section_chunks=section_chunks,
        sliding_window_chunks=sliding_chunks,
        elapsed_sec=elapsed,
        page_reports=page_reports,
        sample_chunks=sample_dicts,
    )


# ── Pretty Printers ──────────────────────────────────────────────────────────

def _print_quality_table(page_reports: List[PageReport], max_rows: int = 20):
    table = Table(
        title="[bold]Per-Page Quality Report[/bold]",
        box=box.ROUNDED,
        show_lines=True,
        header_style="bold magenta",
    )
    table.add_column("Page",  justify="right", style="dim", width=6)
    table.add_column("Chars", justify="right")
    table.add_column("BN%",   justify="right")
    table.add_column("Valid%",justify="right")
    table.add_column("Status",justify="center")
    table.add_column("OCR",   justify="center")
    table.add_column("Reason", overflow="fold")

    shown = page_reports[:max_rows]
    for r in shown:
        status = "[green]PASS[/green]" if r.passed else "[red]FAIL[/red]"
        ocr = "[yellow]✓[/yellow]" if r.ocr_triggered else "—"
        table.add_row(
            str(r.page),
            f"{r.text_length:,}",
            f"{r.bangla_ratio:.1%}",
            f"{r.valid_char_ratio:.1%}",
            status,
            ocr,
            r.reason[:80],
        )

    if len(page_reports) > max_rows:
        table.add_row(
            "…", "…", "…", "…",
            f"[dim]{len(page_reports)-max_rows} more rows[/dim]", "…", ""
        )

    console.print(table)


def _print_chunk_table(chunks: List[LegalChunk], inspect_n: int = 5):
    if not chunks:
        console.print("  [red]No chunks produced.[/red]")
        return

    table = Table(
        title=f"[bold]First {min(inspect_n, len(chunks))} Chunks[/bold]",
        box=box.SIMPLE_HEAVY,
        show_lines=True,
        header_style="bold blue",
    )
    table.add_column("chunk_id",  style="dim", overflow="fold")
    table.add_column("Section #", justify="center")
    table.add_column("Title",     overflow="fold")
    table.add_column("Content preview", overflow="fold")
    table.add_column("Chars",     justify="right")

    for c in chunks[:inspect_n]:
        preview = c.content.replace("\n", " ")[:120]
        if len(c.content) > 120:
            preview += "…"
        table.add_row(
            c.chunk_id,
            c.section_number or "—",
            (c.section_title or "")[:40],
            preview,
            str(len(c.content)),
        )

    console.print(table)


def _print_summary(reports: List[DocumentReport]):
    console.print(Rule("[bold green]== Full Pipeline Summary ==[/bold green]"))
    table = Table(box=box.ROUNDED, show_lines=True, header_style="bold white on dark_blue")
    table.add_column("Document")
    table.add_column("Pages",   justify="right")
    table.add_column("Passed",  justify="right")
    table.add_column("OCR",     justify="right")
    table.add_column("Chunks",  justify="right")
    table.add_column("Section", justify="right")
    table.add_column("Sliding", justify="right")
    table.add_column("Time",    justify="right")

    for r in reports:
        pct = r.pages_passed / r.total_pages * 100 if r.total_pages else 0
        table.add_row(
            r.source_file,
            str(r.total_pages),
            f"{r.pages_passed} ({pct:.0f}%)",
            str(r.pages_ocr_triggered),
            f"[bold]{r.total_chunks}[/bold]",
            str(r.section_chunks),
            str(r.sliding_window_chunks),
            f"{r.elapsed_sec:.1f}s",
        )

    console.print(table)


def _save_report_json(reports: List[DocumentReport], out_path: Path):
    data = [asdict(r) for r in reports]
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    console.print(f"\n[bold green]JSON report ->[/bold green] {out_path}")


# ── CLI entry point ──────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="PoliceGPT — Document Processing Pipeline",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--pdf",  type=str, help="Path to a single PDF file.")
    group.add_argument(
        "--doc",
        choices=list(DOCUMENTS.keys()),
        help="Process one registered document by key.",
    )
    group.add_argument(
        "--all",
        action="store_true",
        default=False,
        help="Process ALL registered documents (default when no flag given).",
    )
    parser.add_argument("--inspect",     type=int, default=5, metavar="N",
                        help="Print first N chunks per doc (default: 5).")
    parser.add_argument("--report",      action="store_true",
                        help="Save JSON quality report to data/processed/.")
    parser.add_argument("--save-chunks", action="store_true",
                        help="Dump all chunks to data/processed/<doc_id>_chunks.json.")
    return parser.parse_args()


def main():
    args = parse_args()

    console.print(Panel.fit(
        "[bold cyan]PoliceGPT[/bold cyan] - Document Processing Layer\n"
        "[dim]PyMuPDF -> Quality Check -> OCR Fallback -> Normalize -> Chunk[/dim]",
        border_style="cyan",
    ))

    if args.pdf:
        custom_path = Path(args.pdf)
        if not custom_path.is_absolute():
            custom_path = REPO_ROOT / custom_path
        DOCUMENTS["_custom"] = {
            "path": custom_path,
            "doc_id": custom_path.stem,
            "act_name_bn": None,
            "act_name_en": custom_path.stem,
            "act_year": None,
            "is_expected_bangla": True,
        }
        doc_keys = ["_custom"]
    elif args.doc:
        doc_keys = [args.doc]
    else:
        # default: process all
        doc_keys = list(DOCUMENTS.keys())

    reports: List[DocumentReport] = []
    for key in doc_keys:
        try:
            report = run_pipeline(
                doc_key=key,
                inspect_n=args.inspect,
                save_chunks_json=args.save_chunks,
            )
            reports.append(report)
        except FileNotFoundError as e:
            console.print(f"[bold red]ERROR:[/bold red] {e}")

    if reports:
        _print_summary(reports)

    if args.report and reports:
        out_dir = REPO_ROOT / "data" / "processed"
        out_dir.mkdir(parents=True, exist_ok=True)
        _save_report_json(reports, out_dir / "quality_report.json")

    console.print(f"\n[dim]Done. Processed {len(reports)} document(s).[/dim]")


if __name__ == "__main__":
    main()
