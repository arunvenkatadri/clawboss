"""Privacy shielding — PII detection and redaction for tool call data.

Regex-based pattern matching that intercepts tool call inputs and outputs,
replacing sensitive data with placeholders. Fast, zero dependencies,
catches the common cases.

For HIPAA/GDPR-grade redaction, use a dedicated NER service — this
catches accidental leakage, not adversarial exfiltration.

Usage:
    from clawboss.redact import Redactor

    redactor = Redactor(categories=["email", "phone", "ssn"])
    clean = redactor.redact("Call me at 555-123-4567 or bob@example.com")
    # "Call me at [PHONE] or [EMAIL]"
"""

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

# ---------------------------------------------------------------------------
# Pattern registry
# ---------------------------------------------------------------------------

# Each pattern: (category_name, compiled_regex, placeholder)
_PATTERNS: List[Tuple[str, re.Pattern, str]] = [
    (
        "email",
        re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"),
        "[EMAIL]",
    ),
    (
        "phone",
        re.compile(
            r"(?<!\d)"  # not preceded by digit
            r"(?:"
            r"\+?1[-.\s]?"  # optional country code
            r")?"
            r"(?:\(?\d{3}\)?[-.\s]?)"  # area code
            r"\d{3}[-.\s]?\d{4}"  # number
            r"(?!\d)"  # not followed by digit
        ),
        "[PHONE]",
    ),
    (
        "ssn",
        re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
        "[SSN]",
    ),
    (
        "credit_card",
        re.compile(
            r"\b(?:"
            r"\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}"  # 16 digits
            r"|"
            r"\d{4}[-\s]?\d{6}[-\s]?\d{5}"  # amex 15 digits
            r")\b"
        ),
        "[CREDIT_CARD]",
    ),
    (
        "api_key",
        re.compile(
            r"(?:"
            r"sk-[A-Za-z0-9]{20,}"  # OpenAI-style
            r"|"
            r"sk-ant-[A-Za-z0-9-]{20,}"  # Anthropic-style
            r"|"
            r"AKIA[A-Z0-9]{16}"  # AWS access key
            r"|"
            r"ghp_[A-Za-z0-9]{36}"  # GitHub PAT
            r"|"
            r"xox[boaprs]-[A-Za-z0-9-]{10,}"  # Slack token
            r")"
        ),
        "[API_KEY]",
    ),
    (
        "ip_address",
        re.compile(
            r"\b(?:(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\.){3}"
            r"(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\b"
        ),
        "[IP_ADDR]",
    ),
]

# All known category names
ALL_CATEGORIES: Set[str] = {name for name, _, _ in _PATTERNS}


# ---------------------------------------------------------------------------
# Redactor
# ---------------------------------------------------------------------------


@dataclass
class RedactionResult:
    """Result of a redaction pass."""

    text: str
    redacted_count: int = 0
    categories_found: List[str] = field(default_factory=list)


class Redactor:
    """PII redactor with configurable categories.

    Args:
        categories: Which PII types to detect. Defaults to all.
                   Options: "email", "phone", "ssn", "credit_card",
                   "api_key", "ip_address"
    """

    def __init__(self, categories: Optional[List[str]] = None):
        if categories is None:
            self._patterns = _PATTERNS
        else:
            cats = set(categories)
            self._patterns = [(n, p, r) for n, p, r in _PATTERNS if n in cats]

    @property
    def categories(self) -> List[str]:
        return [name for name, _, _ in self._patterns]

    def redact(self, text: str) -> RedactionResult:
        """Redact PII from a string.

        Returns a RedactionResult with the cleaned text and metadata
        about what was found.
        """
        if not isinstance(text, str):
            return RedactionResult(text=str(text))

        result_text = text
        total_count = 0
        found_categories: List[str] = []

        for name, pattern, placeholder in self._patterns:
            new_text, count = pattern.subn(placeholder, result_text)
            if count > 0:
                total_count += count
                if name not in found_categories:
                    found_categories.append(name)
                result_text = new_text

        return RedactionResult(
            text=result_text,
            redacted_count=total_count,
            categories_found=found_categories,
        )

    def redact_dict(self, d: Dict[str, Any]) -> Tuple[Dict[str, Any], int]:
        """Redact PII from all string values in a dict (shallow).

        Returns (cleaned_dict, total_redaction_count).
        """
        cleaned = {}
        total = 0
        for k, v in d.items():
            if isinstance(v, str):
                result = self.redact(v)
                cleaned[k] = result.text
                total += result.redacted_count
            else:
                cleaned[k] = v
        return cleaned, total
