"""
PoliceGPT - Legal Document Ingestion & Chunking Pipeline.
Connects: PyMuPDF Extraction -> Quality Check -> OCR Fallback -> Unicode Normalizer -> Boilerplate Cleaner -> Legal Section Chunker.
"""

from typing import List, Dict, Any, Optional
from pathlib import Path
from pydantic import BaseModel, Field
from loguru import logger

from .ingestion.parse_pdf import PDFParser, ParsedPage
from .ingestion.quality_check import DocumentQualityChecker, QualityCheckResult
from .ingestion.ocr_fallback import OCRFallbackEngine
from .preprocessing.normalize import UnicodeNormalizer
from .preprocessing.boilerplate import BoilerplateCleaner
from .preprocessing.chunker import LegalSectionChunker, LegalChunk


class DocumentProcessingResult(BaseModel):
    """Result of processing an entire document through the chunking pipeline."""
    doc_id: str
    source_file: str
    total_pages: int
    passed_pages: int
    ocr_triggered_pages: int
    total_chunks: int
    section_chunks: int
    sliding_chunks: int
    chunks: List[LegalChunk]
    page_quality: List[Dict[str, Any]] = Field(default_factory=list)


class PoliceGPTChunkingPipeline:
    """
    Core Document Ingestion & Legal Section Chunking Pipeline.
    """
    def __init__(
        self,
        target_chunk_size: int = 512,
        chunk_overlap: int = 64,
        min_chunk_size: int = 100,
        min_text_length_per_page: int = 50,
        min_bangla_unicode_ratio: float = 0.15,
        min_valid_char_ratio: float = 0.80,
        ocr_engine: str = "easyocr",
        use_gpu: bool = False,
    ):
        logger.info("Initializing PoliceGPT Document Chunking Pipeline...")

        # 1. Ingestion & Quality Gates
        self.quality_checker = DocumentQualityChecker(
            min_text_length_per_page=min_text_length_per_page,
            min_bangla_unicode_ratio=min_bangla_unicode_ratio,
            min_valid_char_ratio=min_valid_char_ratio,
        )
        self.ocr_engine = OCRFallbackEngine(engine=ocr_engine, use_gpu=use_gpu)
        self.pdf_parser = PDFParser(
            quality_checker=self.quality_checker,
            ocr_engine=self.ocr_engine,
        )

        # 2. Text Normalization & Cleaning
        self.normalizer = UnicodeNormalizer()
        self.boilerplate_cleaner = BoilerplateCleaner()

        # 3. Structure-Aware Legal Chunker
        self.chunker = LegalSectionChunker(
            target_chunk_size=target_chunk_size,
            chunk_overlap=chunk_overlap,
            min_chunk_size=min_chunk_size,
        )

    def process_pdf(
        self,
        pdf_path: str | Path,
        doc_id: Optional[str] = None,
        act_name_bn: Optional[str] = None,
        act_name_en: Optional[str] = None,
        act_year: Optional[int] = None,
        is_expected_bangla: bool = True,
    ) -> DocumentProcessingResult:
        """
        End-to-end PDF processing:
        PyMuPDF extract -> Quality detection -> OCR fallback -> Normalize -> Clean -> Legal chunk
        """
        path = Path(pdf_path)
        if not path.exists():
            raise FileNotFoundError(f"PDF file not found at: {path}")

        effective_doc_id = doc_id or path.stem
        logger.info(f"Processing PDF document: {path.name} (doc_id={effective_doc_id})")

        # Step 1: Extract page text & evaluate quality (with OCR fallback)
        parsed_pages: List[ParsedPage] = self.pdf_parser.parse_pdf(
            file_path=path,
            is_expected_bangla=is_expected_bangla,
        )

        # Step 2: Normalize and strip boilerplate
        full_cleaned_text = ""
        source_pages = []
        page_quality_records = []
        passed_count = 0
        ocr_count = 0

        for page in parsed_pages:
            normalized = self.normalizer.normalize(page.text)
            cleaned = self.boilerplate_cleaner.clean(normalized)
            full_cleaned_text += "\n" + cleaned
            source_pages.append(page.page_number)

            if page.quality.passed:
                passed_count += 1
            if page.quality.needs_ocr:
                ocr_count += 1

            page_quality_records.append(page.to_dict())

        # Step 3: Legal structure-aware chunking
        chunks: List[LegalChunk] = self.chunker.chunk_document(
            text=full_cleaned_text,
            doc_id=effective_doc_id,
            act_name_bn=act_name_bn,
            act_name_en=act_name_en,
            act_year=act_year,
            source_pages=source_pages,
        )

        section_chunks = sum(1 for c in chunks if c.section_number is not None)
        sliding_chunks = len(chunks) - section_chunks

        logger.info(
            f"Completed processing for '{path.name}': {len(chunks)} chunks produced "
            f"({section_chunks} section, {sliding_chunks} sliding window)."
        )

        return DocumentProcessingResult(
            doc_id=effective_doc_id,
            source_file=path.name,
            total_pages=len(parsed_pages),
            passed_pages=passed_count,
            ocr_triggered_pages=ocr_count,
            total_chunks=len(chunks),
            section_chunks=section_chunks,
            sliding_chunks=sliding_chunks,
            chunks=chunks,
            page_quality=page_quality_records,
        )

    def process_text(
        self,
        raw_text: str,
        doc_id: str,
        act_name_bn: Optional[str] = None,
        act_name_en: Optional[str] = None,
        act_year: Optional[int] = None,
        source_pages: Optional[List[int]] = None,
    ) -> List[LegalChunk]:
        """
        Directly process and chunk pre-extracted raw text.
        """
        normalized = self.normalizer.normalize(raw_text)
        cleaned = self.boilerplate_cleaner.clean(normalized)
        return self.chunker.chunk_document(
            text=cleaned,
            doc_id=doc_id,
            act_name_bn=act_name_bn,
            act_name_en=act_name_en,
            act_year=act_year,
            source_pages=source_pages or [],
        )


# Backwards compatibility alias
PoliceGPTRAGPipeline = PoliceGPTChunkingPipeline
