"""
Structured Data Extraction Module.

Extracts structured fields from document text:
  - Dates (multiple formats: DD.MM.YYYY, YYYY-MM-DD, etc.)
  - Monetary amounts (EUR, USD, with various notations)
  - Email addresses
  - Phone numbers
  - IBAN numbers (common in German documents)
  - Reference/invoice numbers

Uses regex for speed + LLM for complex extraction when available.
"""

import re
import json
import logging
from typing import Optional

logger = logging.getLogger(__name__)


# ── Regex Patterns ──
PATTERNS = {
    "dates": [
        r"\b(\d{1,2}[./]\d{1,2}[./]\d{2,4})\b",          # DD.MM.YYYY or DD/MM/YYYY
        r"\b(\d{4}-\d{2}-\d{2})\b",                        # YYYY-MM-DD (ISO)
        r"\b(\d{1,2}\.\s*(?:Jan(?:uar)?|Feb(?:ruar)?|Mär(?:z)?|Mar(?:ch)?|Apr(?:il)?|Mai|May|Jun(?:i|e)?|Jul(?:i|y)?|Aug(?:ust)?|Sep(?:tember)?|Okt(?:ober)?|Oct(?:ober)?|Nov(?:ember)?|Dez(?:ember)?|Dec(?:ember)?)\s*\.?\s*\d{2,4})\b",  # 15. Januar 2024
    ],
    "amounts": [
        r"(?:EUR|€)\s*([\d.,]+(?:\.\d{2})?)",              # EUR 1.234,56 or €1234.56
        r"([\d.,]+)\s*(?:EUR|€)",                           # 1.234,56 EUR
        r"\$\s*([\d.,]+(?:\.\d{2})?)",                      # $1,234.56
        r"([\d.,]+)\s*(?:USD|\$)",                           # 1,234.56 USD
    ],
    "emails": [
        r"\b([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})\b",
    ],
    "phones": [
        r"(?:\+49|0049|0)\s*[\d\s/.-]{8,15}",              # German phone numbers
        r"\+\d{1,3}\s*[\d\s.-]{6,14}",                     # International
    ],
    "ibans": [
        r"\b([A-Z]{2}\d{2}\s*(?:\d{4}\s*){4,8}\d{0,4})\b", # IBAN
    ],
    "reference_numbers": [
        r"(?:Rechnungs?(?:nummer|nr\.?)|Invoice\s*(?:No\.?|Number|#))\s*[:\s]*([A-Z0-9][\w-]{3,20})",
        r"(?:Bestell(?:nummer|nr\.?)|Order\s*(?:No\.?|Number|#))\s*[:\s]*([A-Z0-9][\w-]{3,20})",
        r"(?:Aktenzeichen|Ref(?:erence)?\.?\s*(?:No\.?|Number|#)?)\s*[:\s]*([A-Z0-9][\w-]{3,20})",
    ],
}


def extract_structured_data(text: str) -> dict:
    """
    Extract all structured fields from document text using regex.

    Args:
        text: Full document text

    Returns:
        dict with extracted fields, each as a list of unique matches
    """
    results = {}

    for field_name, patterns in PATTERNS.items():
        matches = set()
        for pattern in patterns:
            found = re.findall(pattern, text, re.IGNORECASE | re.MULTILINE)
            for match in found:
                cleaned = match.strip() if isinstance(match, str) else match
                if cleaned:
                    matches.add(cleaned)
        results[field_name] = sorted(matches)

    # Add summary stats
    results["summary"] = {
        "total_fields_found": sum(len(v) for k, v in results.items() if k != "summary"),
        "has_financial_data": len(results.get("amounts", [])) > 0,
        "has_contact_info": len(results.get("emails", [])) + len(results.get("phones", [])) > 0,
        "has_references": len(results.get("reference_numbers", [])) > 0,
        "document_language": _detect_language(text),
    }

    return results


def extract_key_value_pairs(text: str) -> list[dict]:
    """
    Extract key-value pairs from structured documents (forms, invoices).
    Looks for patterns like "Key: Value" or "Key    Value" (tabular).
    """
    pairs = []

    # Pattern: "Label: Value" or "Label:  Value"
    kv_pattern = r"^([A-Za-zÄÖÜäöüß\s]{2,30})\s*:\s*(.+)$"
    for match in re.finditer(kv_pattern, text, re.MULTILINE):
        key = match.group(1).strip()
        value = match.group(2).strip()
        if len(value) > 0 and len(value) < 200:
            pairs.append({"key": key, "value": value})

    return pairs


def generate_extraction_summary(text: str) -> dict:
    """
    Generate a complete extraction report combining all methods.
    """
    structured = extract_structured_data(text)
    kv_pairs = extract_key_value_pairs(text)

    return {
        "structured_fields": structured,
        "key_value_pairs": kv_pairs,
        "text_stats": {
            "total_characters": len(text),
            "total_words": len(text.split()),
            "total_lines": text.count("\n") + 1,
        },
    }


def _detect_language(text: str) -> str:
    """Simple heuristic language detection (German vs English)."""
    german_indicators = ["und", "der", "die", "das", "ist", "für", "mit", "von",
                         "Rechnung", "Datum", "Betrag", "Straße", "Vertrag"]
    english_indicators = ["the", "and", "for", "with", "from", "invoice",
                          "date", "amount", "total", "contract"]

    text_lower = text.lower()
    de_count = sum(1 for w in german_indicators if f" {w} " in f" {text_lower} ")
    en_count = sum(1 for w in english_indicators if f" {w} " in f" {text_lower} ")

    if de_count > en_count:
        return "de"
    elif en_count > de_count:
        return "en"
    return "unknown"
