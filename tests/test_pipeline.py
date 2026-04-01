"""
Tests for the Intelligent Document Processor.

Run: pytest tests/ -v
"""

import pytest
from app.extraction import extract_structured_data, extract_key_value_pairs, _detect_language


# ── Structured Extraction Tests ──

class TestStructuredExtraction:

    def test_extracts_german_dates(self):
        text = "Rechnungsdatum: 15.03.2024 und Fälligkeit am 30.04.2024"
        result = extract_structured_data(text)
        assert len(result["dates"]) >= 2
        assert "15.03.2024" in result["dates"]

    def test_extracts_iso_dates(self):
        text = "Created on 2024-01-15 and updated 2024-06-30."
        result = extract_structured_data(text)
        assert "2024-01-15" in result["dates"]
        assert "2024-06-30" in result["dates"]

    def test_extracts_euro_amounts(self):
        text = "Gesamtbetrag: EUR 1.234,56 inkl. MwSt. Nettobetrag: €500,00"
        result = extract_structured_data(text)
        assert len(result["amounts"]) >= 1

    def test_extracts_emails(self):
        text = "Kontakt: info@example.de oder support@company.com"
        result = extract_structured_data(text)
        assert "info@example.de" in result["emails"]
        assert "support@company.com" in result["emails"]

    def test_extracts_iban(self):
        text = "IBAN: DE89 3704 0044 0532 0130 00"
        result = extract_structured_data(text)
        assert len(result["ibans"]) >= 1

    def test_extracts_invoice_numbers(self):
        text = "Rechnungsnummer: INV-2024-001234"
        result = extract_structured_data(text)
        assert len(result["reference_numbers"]) >= 1

    def test_empty_text(self):
        result = extract_structured_data("")
        assert result["summary"]["total_fields_found"] == 0


class TestKeyValueExtraction:

    def test_extracts_key_value_pairs(self):
        text = "Name: Max Mustermann\nAdresse: Musterstraße 1\nDatum: 15.03.2024"
        pairs = extract_key_value_pairs(text)
        keys = [p["key"] for p in pairs]
        assert "Name" in keys
        assert "Adresse" in keys

    def test_empty_text_returns_empty(self):
        pairs = extract_key_value_pairs("")
        assert pairs == []


class TestLanguageDetection:

    def test_detects_german(self):
        text = "Dies ist eine Rechnung für die Lieferung der bestellten Ware."
        assert _detect_language(text) == "de"

    def test_detects_english(self):
        text = "This is an invoice for the delivery of the ordered goods."
        assert _detect_language(text) == "en"
