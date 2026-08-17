import pytest
from policegpt_rag.preprocessing.chunker import LegalSectionChunker


def test_section_aware_chunking():
    chunker = LegalSectionChunker(target_chunk_size=100, chunk_overlap=20)
    raw_legal_text = """
ধারা ৩৭৮ । চুরি
যে ব্যক্তি অন্যের দখলে থাকা কোনো অস্থাবর সম্পত্তি অসদুপায়ে নেওয়ার উদ্দেশ্যে স্থানান্তর করে, সে চুরি করেছে বলে গণ্য হয়।

ধারা ৩৭৯ । চুরির শাস্তি
যে ব্যক্তি চুরি করে, সে যেকোনো বর্ণনার কারাদণ্ডে—যার মেয়াদ তিন বছর পর্যন্ত হতে পারে, অথবা অর্থদণ্ডে, কিংবা উভয় দণ্ডেই দণ্ডিত হবে।
"""
    chunks = chunker.chunk_document(
        text=raw_legal_text,
        doc_id="penal_code_1860",
        act_name_bn="দণ্ডবিধি, ১৮৬০",
        act_name_en="The Penal Code, 1860",
        act_year=1860,
    )

    assert len(chunks) == 2
    assert chunks[0].section_number == "৩৭৮"
    assert chunks[1].section_number == "৩৭৯"
    assert "চুরি" in chunks[0].content
    assert "শাস্তি" in chunks[1].content


def test_empty_document_chunking():
    chunker = LegalSectionChunker()
    chunks = chunker.chunk_document(text="", doc_id="empty_doc")
    assert len(chunks) == 0
