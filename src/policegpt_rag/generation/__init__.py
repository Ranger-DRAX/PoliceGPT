"""
Generation module: Bilingual prompt templates, LLM clients, and legal citation guardrails.
"""

from .prompt_templates import LegalPromptBuilder
from .llm_client import LLMClient
from .guardrails import LegalGuardrails, GuardrailValidationResult

__all__ = ["LegalPromptBuilder", "LLMClient", "LegalGuardrails", "GuardrailValidationResult"]
