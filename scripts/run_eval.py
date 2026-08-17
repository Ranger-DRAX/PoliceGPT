"""
CLI Script: Run Retrieval & Generation Evaluation against data/eval benchmarks.
"""

from pathlib import Path
import json
import argparse
import sys
from loguru import logger

# Ensure src is in pythonpath
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from policegpt_rag.evaluation.retrieval_metrics import RetrievalEvaluator
from policegpt_rag.evaluation.generation_metrics import GenerationEvaluator
from policegpt_rag.pipeline import PoliceGPTRAGPipeline


def run_evaluation(retrieval_gold_path: str, qa_gold_path: str):
    logger.info("Starting Evaluation on Gold Benchmarks...")
    ret_evaluator = RetrievalEvaluator()
    gen_evaluator = GenerationEvaluator()
    pipeline = PoliceGPTRAGPipeline()

    # 1. Retrieval Benchmark
    ret_file = Path(retrieval_gold_path)
    if ret_file.exists():
        gold_pairs = []
        queries = []
        with open(ret_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    item = json.loads(line)
                    gold_pairs.append(item)
                    queries.append(item["query"])

        retrieved_doc_ids = []
        for q in queries:
            resp = pipeline.query(q, top_k=10)
            doc_ids = [s.get("chunk_id", "") for s in resp.retrieved_sources]
            retrieved_doc_ids.append(doc_ids)

        ret_scores = ret_evaluator.evaluate_retrievals(gold_pairs, retrieved_doc_ids)
        logger.info(f"=== RETRIEVAL BENCHMARK RESULTS ({ret_scores.total_queries} queries) ===")
        logger.info(f"HitRate@1:  {ret_scores.hit_rate_at_1:.4f}")
        logger.info(f"HitRate@3:  {ret_scores.hit_rate_at_3:.4f}")
        logger.info(f"HitRate@5:  {ret_scores.hit_rate_at_5:.4f}")
        logger.info(f"HitRate@10: {ret_scores.hit_rate_at_10:.4f}")
        logger.info(f"MRR@10:     {ret_scores.mrr_at_10:.4f}")

    # 2. QA Generation Benchmark
    qa_file = Path(qa_gold_path)
    if qa_file.exists():
        qa_queries = []
        references = []
        with open(qa_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    item = json.loads(line)
                    qa_queries.append(item["query"])
                    references.append(item["reference_answer"])

        generated_answers = []
        retrieved_contexts = []
        for q in qa_queries:
            resp = pipeline.query(q)
            generated_answers.append(resp.answer)
            retrieved_contexts.append([s.get("content", "") for s in resp.retrieved_sources])

        gen_scores = gen_evaluator.evaluate_qa_dataset(
            qa_queries, generated_answers, retrieved_contexts, references
        )
        logger.info(f"=== GENERATION BENCHMARK RESULTS ({gen_scores.total_evaluated} samples) ===")
        logger.info(f"Faithfulness:        {gen_scores.faithfulness:.4f}")
        logger.info(f"Answer Relevancy:    {gen_scores.answer_relevancy:.4f}")
        logger.info(f"Citation Precision:  {gen_scores.citation_precision:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Evaluation Benchmarks")
    parser.add_argument("--ret-gold", default="data/eval/retrieval_gold.jsonl")
    parser.add_argument("--qa-gold", default="data/eval/qa_gold.jsonl")
    args = parser.parse_args()

    run_evaluation(args.ret_gold, args.qa_gold)
