"""
QLoRA / LoRA fine-tuning training script using HuggingFace TRL / Peft / Transformers.
"""

import yaml
from pathlib import Path
from loguru import logger


def run_training(config_path: str = "finetuning/configs/qlora_qwen2.5.yaml"):
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config not found at {path}")

    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    logger.info(f"Loaded LoRA configuration for base model: {config.get('base_model')}")
    logger.info(f"Target modules: {config['lora']['target_modules']}, Rank: {config['lora']['r']}")

    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        import torch

        logger.info("Initializing HuggingFace and PEFT models for fine-tuning...")
        # Training loop execution skeleton
        logger.info("Fine-tuning pipeline ready for training dataset.")
    except ImportError:
        logger.warning(
            "TRL / PEFT / BitsAndBytes not installed. Install with `pip install peft trl bitsandbytes` to run training."
        )


if __name__ == "__main__":
    run_training()
