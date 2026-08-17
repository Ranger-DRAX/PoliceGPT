"""
PoliceGPT RAG Pipeline Orchestrator.
Connects: Ingestion -> Normalization -> Chunking -> Embeddings -> Index -> Hybrid Retrieval -> Reranking -> LLM -> Cited Answer.
"""

from typing import List, Dict, Any, Optional
from pathlib import Path
from loguru import logger
from pydantic import BaseModel

from .ingestion.parse_pdf import PDFParser
from .ingestion.quality_check import DocumentQualityChecker
from .ingestion.ocr_fallback import OCRFallbackEngine
from .preprocessing.normalize import UnicodeNormalizer
from .preprocessing.boilerplate import BoilerplateCleaner
from .preprocessing.chunker import LegalSectionChunker, LegalChunk
from .embedding.embed import BGEM3Embedder
from .embedding.batch_runner import BatchEmbeddingRunner
from .indexing.qdrant_index import QdrantHybridIndex
from .indexing.faiss_index import LocalFaissIndex
from .retrieval.hybrid_search import HybridSearchRetriever
from .retrieval.rerank import CrossEncoderReranker
from .generation.prompt_templates import LegalPromptBuilder
from .generation.llm_client import LLMClient
from .generation.guardrails import LegalGuardrails, GuardrailValidationResult


class PoliceGPTResponse(BaseModel):
    query: str
    answer: str
    citations: List[str]
    retrieved_sources: List[Dict[str, Any]]
    confidence_score: float
    is_grounded: bool
    guardrail_status: GuardrailValidationResult


class PoliceGPTRAGPipeline:
    def __init__(
        self,
        qdrant_host: str = "localhost",
        qdrant_port: int = 6333,
        collection_name: str = "policegpt_legal_corpus",
        llm_provider: str = "ollama",
        llm_model: str = "qwen2.5:7b",
        embedding_model: str = "BAAI/bge-m3",
        reranker_model: str = "BAAI/bge-reranker-v2-m3",
    ):
        logger.info("Initializing PoliceGPT RAG Pipeline components...")

        # 1. Ingestion & Quality
        self.quality_checker = DocumentQualityChecker()
        self.ocr_engine = OCRFallbackEngine()
        self.pdf_parser = PDFParser(self.quality_checker, self.ocr_engine)

        # 2. Preprocessing
        self.normalizer = UnicodeNormalizer()
        self.boilerplate_cleaner = BoilerplateCleaner()
        self.chunker = LegalSectionChunker()

        # 3. Embedding & Indexing
        self.embedder = BGEM3Embedder(model_name=embedding_model)
        self.batch_runner = BatchEmbeddingRunner(self.embedder)
        self.qdrant_index = QdrantHybridIndex(
            collection_name=collection_name, host=qdrant_host, port=qdrant_port
        )
        self.faiss_index = LocalFaissIndex()

        # 4. Retrieval & Reranking
        self.retriever = HybridSearchRetriever(
            embedder=self.embedder,
            qdrant_index=self.qdrant_index,
            faiss_index=self.faiss_index,
        )
        self.reranker = CrossEncoderReranker(model_name=reranker_model)

        # 5. Generation & Guardrails
        self.prompt_builder = LegalPromptBuilder()
        self.llm_client = LLMClient(provider=llm_provider, model_name=llm_model)
        self.guardrails = LegalGuardrails()

    def process_and_index_document(
        self,
        pdf_path: str | Path,
        doc_id: str,
        act_name_bn: Optional[str] = None,
        act_name_en: Optional[str] = None,
        act_year: Optional[int] = None,
        use_qdrant: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Full ingestion pipeline: Parse PDF -> Clean -> Section Chunk -> Embed -> Index
        """
        logger.info(f"Starting ingestion workflow for {pdf_path}...")
        parsed_pages = self.pdf_parser.parse_pdf(pdf_path)

        full_cleaned_text = ""
        source_pages = []
        for p in parsed_pages:
            normalized = self.normalizer.normalize(p.text)
            cleaned = self.boilerplate_cleaner.clean(normalized)
            full_cleaned_text += "\n" + cleaned
            source_pages.append(p.page_number)

        chunks: List[LegalChunk] = self.chunker.chunk_document(
            text=full_cleaned_text,
            doc_id=doc_id,
            act_name_bn=act_name_bn,
            act_name_en=act_name_en,
            act_year=act_year,
            source_pages=source_pages,
        )

        embedded_records = self.batch_runner.process_chunks(chunks)

        if use_qdrant:
            self.qdrant_index.create_collection(recreate=False)
            self.qdrant_index.upsert_records(embedded_records)
        else:
            self.faiss_index.build_index(embedded_records)

        return embedded_records

    def query(
        self,
        query_text: str,
        language: str = "bn",
        top_k: int = 10,
        rerank_top_n: int = 5,
        filters: Optional[Dict[str, Any]] = None,
    ) -> PoliceGPTResponse:
        """
        Query answering workflow:
        Validate query -> Hybrid Retrieve -> Cross-Encoder Rerank -> Prompt LLM -> Verify Guardrails -> Return
        """
        # Guardrail: Check safety of query
        is_safe, msg = self.guardrails.validate_query(query_text)
        if not is_safe:
            return PoliceGPTResponse(
                query=query_text,
                answer=f"অনুরোধটি বাতিল করা হয়েছে: {msg}",
                citations=[],
                retrieved_sources=[],
                confidence_score=0.0,
                is_grounded=False,
                guardrail_status=GuardrailValidationResult(
                    is_safe=False,
                    has_mandatory_citations=False,
                    detected_citations=[],
                    out_of_scope_flag=True,
                    final_output=msg,
                    violations=[msg],
                ),
            )

        # 1. Retrieval
        candidate_chunks = self.retriever.retrieve(
            query=query_text, top_k=top_k, filters=filters, use_qdrant=True
        )

        # 2. Reranking
        top_chunks = self.reranker.rerank(
            query=query_text, candidate_chunks=candidate_chunks, top_n=rerank_top_n
        )

        # 3. Prompting & LLM Generation
        sys_prompt = self.prompt_builder.build_system_prompt(language=language)
        user_prompt = self.prompt_builder.build_user_prompt(
            query=query_text, retrieved_chunks=top_chunks, language=language
        )

        raw_llm_output = self.llm_client.generate(sys_prompt, user_prompt)

        # 4. Guardrail Validation & Post-processing
        guardrail_result = self.guardrails.validate_and_postprocess(
            raw_answer=raw_llm_output, retrieved_context=top_chunks, language=language
        )

        avg_confidence = (
            sum(c.get("rerank_score", c.get("score", 0.8)) for c in top_chunks) / len(top_chunks)
            if top_chunks
            else 0.0
        )

        return PoliceGPTResponse(
            query=query_text,
            answer=guardrail_result.final_output,
            citations=guardrail_result.detected_citations,
            retrieved_sources=top_chunks,
            confidence_score=float(avg_confidence),
            is_grounded=guardrail_result.has_mandatory_citations,
            guardrail_status=guardrail_result,
        )
