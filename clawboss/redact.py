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
    # -- International patterns --
    (
        "phone",  # international phone numbers: +XX XXXX XXXXXX etc.
        re.compile(
            r"(?<!\d)"
            r"\+(?:44|49|33|61|81|86|91|7|34|39|55|82|65|852)"  # country codes
            r"[-.\s]?\d{1,5}[-.\s]?\d{3,5}[-.\s]?\d{3,5}"
            r"(?!\d)"
        ),
        "[PHONE]",
    ),
    (
        "national_id",  # UK National Insurance Number
        re.compile(r"\b[A-CEGHJ-PR-TW-Z]{2}\s?\d{2}\s?\d{2}\s?\d{2}\s?[A-D]\b"),
        "[NATIONAL_ID]",
    ),
    (
        "national_id",  # German Tax ID (Steuerliche Identifikationsnummer)
        re.compile(r"\b\d{2}\s?\d{3}\s?\d{3}\s?\d{3}\b"),
        "[NATIONAL_ID]",
    ),
    (
        "iban",  # International Bank Account Number
        re.compile(
            r"\b[A-Z]{2}\d{2}[\s]?[\dA-Z]{4}[\s]?[\dA-Z]{4}[\s]?[\dA-Z]{4}(?:[\s]?[\dA-Z]{1,4}){0,5}\b"
        ),
        "[IBAN]",
    ),
    (
        "passport",  # Common passport number patterns
        re.compile(r"\b[A-Z]{1,2}\d{6,9}\b"),
        "[PASSPORT]",
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
    """PII redactor with configurable categories and optional NLP augmentation.

    Args:
        categories: Which PII types to detect. Defaults to all.
                   Options: "email", "phone", "ssn", "credit_card",
                   "api_key", "ip_address", "national_id", "iban", "passport"
        use_nlp: If True, augment regex with spaCy NER for names, locations,
                 and other context-dependent PII. Requires spacy + a model.
                 Install with: pip install spacy && python -m spacy download en_core_web_sm
    """

    def __init__(self, categories: Optional[List[str]] = None, use_nlp: bool = False):
        if categories is None:
            self._patterns = _PATTERNS
        else:
            cats = set(categories)
            self._patterns = [(n, p, r) for n, p, r in _PATTERNS if n in cats]

        self._nlp = None
        if use_nlp:
            self._init_nlp()

    def _init_nlp(self) -> None:
        """Load spaCy model for NER. Fails silently if not installed."""
        try:
            import spacy  # type: ignore[import-not-found]

            self._nlp = spacy.load("en_core_web_sm")
        except (ImportError, OSError):
            pass  # spaCy not installed or model not downloaded

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

        # NLP augmentation — catch names, locations, orgs that regex misses
        if self._nlp is not None:
            nlp_text, nlp_count, nlp_cats = self._nlp_redact(result_text)
            if nlp_count > 0:
                result_text = nlp_text
                total_count += nlp_count
                for cat in nlp_cats:
                    if cat not in found_categories:
                        found_categories.append(cat)

        return RedactionResult(
            text=result_text,
            redacted_count=total_count,
            categories_found=found_categories,
        )

    # NER entity label → placeholder mapping
    _NER_LABELS = {
        "PERSON": "[PERSON]",
        "GPE": "[LOCATION]",
        "LOC": "[LOCATION]",
        "ORG": "[ORG]",
        "NORP": "[GROUP]",
    }

    def _nlp_redact(self, text: str) -> Tuple[str, int, List[str]]:
        """Apply spaCy NER to catch context-dependent PII."""
        if self._nlp is None:
            return text, 0, []
        doc = self._nlp(text)
        replacements = []
        categories: List[str] = []
        for ent in doc.ents:
            if ent.label_ in self._NER_LABELS:
                replacements.append((ent.start_char, ent.end_char, self._NER_LABELS[ent.label_]))
                cat = self._NER_LABELS[ent.label_].strip("[]").lower()
                if cat not in categories:
                    categories.append(cat)

        if not replacements:
            return text, 0, []

        # Apply replacements in reverse order to preserve offsets
        result = text
        for start, end, placeholder in sorted(replacements, reverse=True):
            result = result[:start] + placeholder + result[end:]

        return result, len(replacements), categories

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
