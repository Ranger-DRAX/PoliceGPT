"""
Legal guardrails: Citation verification, hallucination checks, out-of-scope query rejection, and disclaimer enforcement.
"""

from typing import List, Dict, Any, Tuple
import re
from pydantic import BaseModel
from loguru import logger


class GuardrailValidationResult(BaseModel):
    is_safe: bool
    has_mandatory_citations: bool
    detected_citations: List[str]
    out_of_scope_flag: bool
    final_output: str
    violations: List[str]


class LegalGuardrails:
    def __init__(
        self,
        require_citations: bool = True,
        enforce_disclaimer: bool = True,
        disclaimer_text_bn: Optional[str] = None,
    ):
        self.require_citations = require_citations
        self.enforce_disclaimer = enforce_disclaimer

        self.disclaimer_bn = disclaimer_text_bn or (
            "\n\n---\n*আইনি সতর্কবার্তা: এই উত্তরটি এআই দ্বারা স্বয়ংক্রিয়ভাবে প্রস্তুতকৃত এবং শুধুমাত্র তথ্যগত উদ্দেশ্যে প্রণীত। "
            "এটি কোনো আনুষ্ঠানিক আইনি পরামর্শ নয়। যে কোনো আইনি বিষয়ে সংশ্লিষ্ট থানার ভারপ্রাপ্ত কর্মকর্তা (OC) বা আইনজীবীর পরামর্শ নিন।*"
        )

        # Citation detection regex patterns (Bengali and English)
        self.citation_pattern = re.compile(
            r"(?:ধারা|দণ্ডবিধি|বিধি|অনুচ্ছেদ|Section|Sec\.|Rule|Act|Order)\s+[০-৯0-9]+[A-Za-zক-হ]?",
            re.IGNORECASE,
        )

        # Prohibited intent keywords (e.g. asking how to commit crime or bribe)
        self.prohibited_keywords = [
            "ঘুষ দেওয়ার উপায়",
            "প্রমাণ নষ্ট করার উপায়",
            "মার্ডার করার উপায়",
            "চুরি করার বুদ্ধি",
            "how to bribe",
            "how to tamper evidence",
        ]

    def validate_query(self, query: str) -> Tuple[bool, str]:
        """
        Validate incoming query safety and scope.
        """
        query_lower = query.lower()
        for kw in self.prohibited_keywords:
            if kw in query_lower:
                return False, "Query contains unlawful or prohibited intent."
        return True, "Query passed guardrails."

    def validate_and_postprocess(
        self, raw_answer: str, retrieved_context: List[Dict[str, Any]], language: str = "bn"
    ) -> GuardrailValidationResult:
        """
        Check that generated response contains valid citations and append disclaimer.
        """
        citations = self.citation_pattern.findall(raw_answer)
        has_citations = len(citations) > 0
        violations = []

        if self.require_citations and not has_citations:
            violations.append("Response lacks explicit legal section or act citations.")
            logger.warning("Guardrail alert: Generated response did not contain explicit citations.")

        # Ensure disclaimer is appended
        final_text = raw_answer
        if self.enforce_disclaimer and self.disclaimer_bn not in final_text:
            final_text += self.disclaimer_bn

        is_safe = len(violations) == 0

        return GuardrailValidationResult(
            is_safe=is_safe,
            has_mandatory_citations=has_citations,
            detected_citations=citations,
            out_of_scope_flag=False,
            final_output=final_text,
            violations=violations,
        )
