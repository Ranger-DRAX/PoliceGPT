"""
Boilerplate and artifact cleaner for legal documents.
Strips recurring headers, footers, page numbers, watermarks, and gazette notices.
"""

import re
from typing import List, Optional


class BoilerplateCleaner:
    def __init__(self, custom_patterns: Optional[List[str]] = None):
        # Common Bangladesh Gazette / Ministry / Police header & footer patterns
        self.default_patterns = [
            r"^\s*বাংলাদেশ\s+গেজেট,\s+অতিরিক্ত,\s+[^\n]+",
            r"^\s*THE\s+BANGLADESH\s+GAZETTE,\s+EXTRAORDINARY[^\n]+",
            r"^\s*Page\s+\d+\s+of\s+\d+\s*$",
            r"^\s*পৃষ্ঠা\s+[০-৯0-9]+\s*$",
            r"^\s*\[\s*মূল্য\s*:\s*টাকা[^\n]+\]",
            r"^\s*রেজিস্টার্ড\s+নং\s+[^\n]+",
            r"^\s*Registered\s+No\.\s+[^\n]+",
            r"^\s*—+\s*$",  # Lone horizontal divider dashes
        ]
        if custom_patterns:
            self.default_patterns.extend(custom_patterns)

        self.compiled_regexes = [
            re.compile(pattern, re.IGNORECASE | re.MULTILINE) for pattern in self.default_patterns
        ]

    def clean(self, text: str) -> str:
        if not text:
            return ""

        cleaned = text
        for regex in self.compiled_regexes:
            cleaned = regex.sub("", cleaned)

        # Clean redundant empty lines
        lines = [line.strip() for line in cleaned.splitlines()]
        filtered_lines = [line for line in lines if line]
        return "\n".join(filtered_lines)
