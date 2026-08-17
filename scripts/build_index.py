"""
CLI Script: Build Vector Index from data/interim -> data/processed -> Qdrant / FAISS
Cleans text, performs structure-aware chunking, generates embeddings, and uploads to index.
"""

from pathlib import Path
import json
import argparse
import sys
from loguru import logger

# Ensure src is in pythonpath
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from policegpt_rag.preprocessing.normalize import UnicodeNormalizer
from policegpt_rag.preprocessing.boilerplate import BoilerplateCleaner
from policegpt_rag.preprocessing.chunker import LegalSectionChunker, LegalChunk
from policegpt_rag.embedding.embed import BGEM3Embedder
from policegpt_rag.embedding.batch_runner import BatchEmbeddingRunner
from policegpt_rag.indexing.qdrant_index import QdrantHybridIndex
from policegpt_rag.indexing.faiss_index import LocalFaissIndex


def build_index(interim_dir: str, processed_dir: str, use_qdrant: bool = True):
    interim_path = Path(interim_dir)
    processed_path = Path(processed_dir)
    processed_path.mkdir(parents=True, exist_ok=True)

    normalizer = UnicodeNormalizer()
    cleaner = BoilerplateCleaner()
    chunker = LegalSectionChunker()
    embedder = BGEM3Embedder()
    batch_runner = BatchEmbeddingRunner(embedder)

    interim_files = list(interim_path.glob("*_interim.json"))
    if not interim_files:
        logger.warning(f"No interim JSON files found in {interim_dir}. Generating demo sample chunk.")
        # Create a sample chunk for index verification
        demo_chunks = [
            LegalChunk(
                chunk_id="penal_code_1860_sec_378",
                doc_id="penal_code_1860",
                act_name_bn="দণ্ডবিধি, ১৮৬০",
                act_name_en="The Penal Code, 1860",
                act_year=1860,
                section_number="৩৭৮",
                section_title="চুরি (Theft)",
                content="যে ব্যক্তি অন্যের দখলে থাকা কোনো অস্থাবর সম্পত্তি সেই ব্যক্তির সম্মতি ছাড়া অসদুপায়ে নেওয়ার উদ্দেশ্যে স্থানান্তর করে, সে চুরি করেছে বলে গণ্য হয়।",
            ),
            LegalChunk(
                chunk_id="penal_code_1860_sec_379",
                doc_id="penal_code_1860",
                act_name_bn="দণ্ডবিধি, ১৮৬০",
                act_name_en="The Penal Code, 1860",
                act_year=1860,
                section_number="৩৭৯",
                section_title="চুরির শাস্তি (Punishment for theft)",
                content="যে ব্যক্তি চুরি করে, সে যেকোনো বর্ণনার কারাদণ্ডে—যার মেয়াদ তিন বছর পর্যন্ত হতে পারে, অথবা অর্থদণ্ডে, কিংবা উভয় দণ্ডেই দণ্ডিত হবে।",
            ),
        ]
        all_chunks = demo_chunks
    else:
        all_chunks = []
        for file in interim_files:
            with open(file, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            full_text = "\n".join([p["text"] for p in data.get("pages", [])])
            normalized = normalizer.normalize(full_text)
            cleaned = cleaner.clean(normalized)

            chunks = chunker.chunk_document(
                text=cleaned,
                doc_id=file.stem.replace("_interim", ""),
            )
            all_chunks.extend(chunks)

    logger.info(f"Total legal chunks to embed and index: {len(all_chunks)}")
    records = batch_runner.process_chunks(all_chunks)

    # Save to parquet
    batch_runner.save_processed_parquet(records, processed_path / "legal_chunks_embedded.parquet")

    if use_qdrant:
        logger.info("Indexing into Qdrant...")
        qdrant = QdrantHybridIndex()
        qdrant.create_collection(recreate=True)
        qdrant.upsert_records(records)
    else:
        logger.info("Indexing into FAISS...")
        faiss_idx = LocalFaissIndex()
        faiss_idx.build_index(records)
        faiss_idx.save("data/processed/faiss_store")

    logger.info("Index build completed successfully!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build Vector Index for PoliceGPT")
    parser.add_argument("--interim-dir", default="data/interim")
    parser.add_argument("--processed-dir", default="data/processed")
    parser.add_argument("--use-qdrant", action="store_true", default=False)
    args = parser.parse_args()

    build_index(args.interim_dir, args.processed_dir, args.use_qdrant)
