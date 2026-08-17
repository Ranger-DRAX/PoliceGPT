"""
Legal structure-aware chunker for Bangladesh statutes, penal codes, and police regulations.
Detects Section (ধারা), Chapter (অধ্যায়), Rule (বিধি), and Sub-clause boundaries to prevent
cross-section semantic contamination while generating rich metadata per chunk.
"""

from typing import List, Dict, Any, Optional
import re
from pydantic import BaseModel, Field


class LegalChunk(BaseModel):
    chunk_id: str
    doc_id: str
    act_name_bn: Optional[str] = None
    act_name_en: Optional[str] = None
    act_year: Optional[int] = None
    chapter: Optional[str] = None
    section_number: Optional[str] = None
    section_title: Optional[str] = None
    jurisdiction: str = "Bangladesh"
    content: str
    page_numbers: List[int] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class LegalSectionChunker:
    def __init__(
        self,
        target_chunk_size: int = 512,
        chunk_overlap: int = 64,
        min_chunk_size: int = 100,
    ):
        self.target_chunk_size = target_chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size

        # Regex patterns for Bengali & English legal boundaries
        self.section_pattern = re.compile(
            r"(?m)(?:^|\n)(?P<header>(?:ধারা\s+[০-৯0-9]+[A-Za-zক-হ]?|Section\s+\d+[A-Za-z]?|বিধি\s+[০-৯0-9]+|Rule\s+\d+|অনুচ্ছেদ\s+[০-৯0-9]+|Article\s+\d+)\s*[।.\-:]\s*(?P<title>[^\n]+)?)"
        )
        self.chapter_pattern = re.compile(
            r"(?m)(?:^|\n)(?P<chapter>(?:অধ্যায়\s+[০-৯0-9IVXLCDM]+|Chapter\s+[IVXLCDM\d]+)\s*[।.\-:]\s*(?P<title>[^\n]+)?)"
        )

    def chunk_document(
        self,
        text: str,
        doc_id: str,
        act_name_bn: Optional[str] = None,
        act_name_en: Optional[str] = None,
        act_year: Optional[int] = None,
        source_pages: Optional[List[int]] = None,
    ) -> List[LegalChunk]:
        """
        Segment legal text based on Section / Dhara markers. If a section is large,
        sub-chunk recursively while retaining section title & number in metadata.
        """
        if not text.strip():
            return []

        chunks: List[LegalChunk] = []
        matches = list(self.section_pattern.finditer(text))

        if not matches:
            # Fallback: simple sliding window chunking if no explicit legal sections found
            return self._sliding_window_chunk(
                text=text,
                doc_id=doc_id,
                act_name_bn=act_name_bn,
                act_name_en=act_name_en,
                act_year=act_year,
                source_pages=source_pages or [],
            )

        for i, match in enumerate(matches):
            sec_header = match.group("header").strip()
            sec_title = (match.group("title") or "").strip()
            start_pos = match.start()
            end_pos = matches[i + 1].start() if i + 1 < len(matches) else len(text)

            section_body = text[start_pos:end_pos].strip()

            # Extract clean section number from header
            sec_num_match = re.search(r"[০-৯0-9]+[A-Za-zক-হ]?", sec_header)
            sec_number = sec_num_match.group(0) if sec_num_match else None

            # If section body exceeds target chunk size, split with overlap
            if len(section_body) > self.target_chunk_size * 1.5:
                sub_texts = self._split_with_overlap(section_body, self.target_chunk_size, self.chunk_overlap)
                for sub_idx, sub_text in enumerate(sub_texts):
                    chunks.append(
                        LegalChunk(
                            chunk_id=f"{doc_id}_sec_{sec_number or i}_{sub_idx}",
                            doc_id=doc_id,
                            act_name_bn=act_name_bn,
                            act_name_en=act_name_en,
                            act_year=act_year,
                            section_number=sec_number,
                            section_title=sec_title or sec_header,
                            content=sub_text,
                            page_numbers=source_pages or [],
                            metadata={
                                "header": sec_header,
                                "is_subchunk": True,
                                "subchunk_index": sub_idx,
                            },
                        )
                    )
            else:
                chunks.append(
                    LegalChunk(
                        chunk_id=f"{doc_id}_sec_{sec_number or i}",
                        doc_id=doc_id,
                        act_name_bn=act_name_bn,
                        act_name_en=act_name_en,
                        act_year=act_year,
                        section_number=sec_number,
                        section_title=sec_title or sec_header,
                        content=section_body,
                        page_numbers=source_pages or [],
                        metadata={"header": sec_header},
                    )
                )

        return chunks

    def _split_with_overlap(self, text: str, chunk_size: int, overlap: int) -> List[str]:
        words = text.split()
        if len(words) <= chunk_size:
            return [text]

        sub_chunks = []
        step = max(chunk_size - overlap, 1)
        for i in range(0, len(words), step):
            window = words[i : i + chunk_size]
            sub_chunks.append(" ".join(window))
            if i + chunk_size >= len(words):
                break
        return sub_chunks

    def _sliding_window_chunk(
        self,
        text: str,
        doc_id: str,
        act_name_bn: Optional[str],
        act_name_en: Optional[str],
        act_year: Optional[int],
        source_pages: List[int],
    ) -> List[LegalChunk]:
        sub_texts = self._split_with_overlap(text, self.target_chunk_size, self.chunk_overlap)
        return [
            LegalChunk(
                chunk_id=f"{doc_id}_chunk_{idx}",
                doc_id=doc_id,
                act_name_bn=act_name_bn,
                act_name_en=act_name_en,
                act_year=act_year,
                content=chunk_str,
                page_numbers=source_pages,
            )
            for idx, chunk_str in enumerate(sub_texts)
        ]
