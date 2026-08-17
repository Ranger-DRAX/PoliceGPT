"""
Prepare Instruction-Tuning Dataset from legal QA gold pairs and structured statutes.
Outputs ShareGPT / Alpaca format JSONL for SFT / QLoRA training.
"""

from typing import List, Dict, Any
from pathlib import Path
import json
from loguru import logger


def format_instruction_sample(
    system_prompt: str, user_query: str, legal_context: str, ground_truth_answer: str
) -> Dict[str, Any]:
    """
    Format sample into conversational format.
    """
    return {
        "conversations": [
            {"from": "system", "value": system_prompt},
            {
                "from": "human",
                "value": f"আইনি প্রসঙ্গ (Context):\n{legal_context}\n\nপ্রশ্ন:\n{user_query}",
            },
            {"from": "gpt", "value": ground_truth_answer},
        ]
    }


def convert_qa_gold_to_sft(
    qa_gold_path: str | Path, output_jsonl_path: str | Path
):
    qa_path = Path(qa_gold_path)
    out_path = Path(output_jsonl_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Converting {qa_path} to SFT format at {out_path}...")
    samples = []
    if qa_path.exists():
        with open(qa_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    item = json.loads(line)
                    sample = format_instruction_sample(
                        system_prompt="আপনি PoliceGPT, বাংলাদেশ পুলিশ ও আইনের একজন নির্ভরযোগ্য সহকারী।",
                        user_query=item.get("query", ""),
                        legal_context="দণ্ডবিধি ১৮৬০ এর সংশ্লিষ্ট ধারা।",
                        ground_truth_answer=item.get("reference_answer", ""),
                    )
                    samples.append(sample)

    with open(out_path, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    logger.info(f"Saved {len(samples)} instruction tuning samples to {out_path}.")


if __name__ == "__main__":
    convert_qa_gold_to_sft("data/eval/qa_gold.jsonl", "data/processed/sft_train.jsonl")
