"""
OCR Module — Extracts text from PDFs (native + scanned) and images.

Pipeline:
  1. Native PDF → extract text directly with pdfplumber (fast, accurate)
  2. Scanned PDF / Images → Tesseract OCR (supports English + German)
  3. Returns clean text ready for chunking
"""

import io
import logging
from pathlib import Path
from typing import Optional

import pdfplumber
from PIL import Image
from pdf2image import convert_from_path, convert_from_bytes
import pytesseract

from app.config import TESSERACT_CMD, OCR_LANGUAGES

logger = logging.getLogger(__name__)
pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD


def extract_text_from_pdf(file_path: str | Path) -> dict:
    """
    Extract text from a PDF file. Tries native extraction first,
    falls back to OCR for scanned pages.

    Returns:
        dict with keys: text, pages, method, page_details
    """
    file_path = Path(file_path)
    all_text = []
    page_details = []
    ocr_pages = 0
    native_pages = 0

    with pdfplumber.open(file_path) as pdf:
        for i, page in enumerate(pdf.pages):
            # Try native text extraction first
            text = page.extract_text() or ""

            if len(text.strip()) < 50:
                # Likely scanned — use OCR
                logger.info(f"Page {i+1}: native text too short ({len(text)} chars), using OCR")
                text = _ocr_pdf_page(file_path, page_number=i)
                ocr_pages += 1
                method = "ocr"
            else:
                native_pages += 1
                method = "native"

            all_text.append(text)
            page_details.append({
                "page": i + 1,
                "method": method,
                "chars": len(text),
                "preview": text[:100] + "..." if len(text) > 100 else text
            })

    combined = "\n\n".join(all_text)
    total_method = "native" if ocr_pages == 0 else "ocr" if native_pages == 0 else "hybrid"

    return {
        "text": combined,
        "pages": len(page_details),
        "method": total_method,
        "native_pages": native_pages,
        "ocr_pages": ocr_pages,
        "total_chars": len(combined),
        "page_details": page_details,
    }


def extract_text_from_image(file_path: str | Path) -> dict:
    """Extract text from an image file using Tesseract OCR."""
    file_path = Path(file_path)
    image = Image.open(file_path)

    # Preprocess for better OCR accuracy
    image = _preprocess_image(image)
    text = pytesseract.image_to_string(image, lang=OCR_LANGUAGES)

    return {
        "text": text,
        "pages": 1,
        "method": "ocr",
        "total_chars": len(text),
    }


def extract_text(file_path: str | Path) -> dict:
    """
    Auto-detect file type and extract text.
    Supports: .pdf, .png, .jpg, .jpeg, .tiff, .bmp
    """
    file_path = Path(file_path)
    suffix = file_path.suffix.lower()

    if suffix == ".pdf":
        return extract_text_from_pdf(file_path)
    elif suffix in {".png", ".jpg", ".jpeg", ".tiff", ".bmp", ".webp"}:
        return extract_text_from_image(file_path)
    else:
        raise ValueError(f"Unsupported file type: {suffix}")


def _ocr_pdf_page(file_path: Path, page_number: int) -> str:
    """Convert a single PDF page to image and OCR it."""
    images = convert_from_path(
        str(file_path),
        first_page=page_number + 1,
        last_page=page_number + 1,
        dpi=300,
    )
    if not images:
        return ""

    image = _preprocess_image(images[0])
    return pytesseract.image_to_string(image, lang=OCR_LANGUAGES)


def _preprocess_image(image: Image.Image) -> Image.Image:
    """
    Preprocess image for better OCR accuracy:
    - Convert to grayscale
    - Increase contrast
    """
    # Convert to grayscale
    image = image.convert("L")

    # Simple thresholding for better contrast
    threshold = 140
    image = image.point(lambda p: 255 if p > threshold else 0)

    return image
