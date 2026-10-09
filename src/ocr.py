# VisionRAG OCR module
# CPU-only OCR using RapidOCR with on-disk caching.
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import fitz  # PyMuPDF
from PIL import Image

from src.utils import ensure_dir, load_jsonl, save_jsonl


class OCREngine:
    """Wrapper around RapidOCR with file-based caching."""

    def __init__(self, cache_dir: Path):
        self.cache_dir = cache_dir
        ensure_dir(cache_dir)
        self._engine = None
        self._cache: Dict[str, str] = {}
        self._load_cache()

    def _load_cache(self) -> None:
        """Load all cached OCR results into memory."""
        cache_file = self.cache_dir / "ocr_cache.jsonl"
        if cache_file.exists():
            for record in load_jsonl(cache_file):
                self._cache[record["page_id"]] = record["text"]

    def _save_cache(self, page_id: str, text: str) -> None:
        """Append a single OCR result to the cache file."""
        cache_file = self.cache_dir / "ocr_cache.jsonl"
        save_jsonl([{"page_id": page_id, "text": text}], cache_file)
        self._cache[page_id] = text

    def _get_engine(self):
        """Lazy-load the RapidOCR engine."""
        if self._engine is None:
            from rapidocr_onnxruntime import RapidOCR
            self._engine = RapidOCR()
        return self._engine

    def ocr_page(self, page_id: str, image_path: Path) -> str:
        """OCR a page image, using cache if available."""
        if page_id in self._cache:
            return self._cache[page_id]

        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        # RapidOCR accepts file path or numpy array
        engine = self._get_engine()
        result, _ = engine(str(image_path))

        # Extract text from OCR result
        # RapidOCR returns list of (bbox, text, confidence)
        texts = [r[1] for r in result] if result else []
        text = "\n".join(texts).strip()

        # Cache and return
        self._save_cache(page_id, text)
        return text


def extract_text_from_pdf_page(pdf_path: Path, page_num: int) -> str:
    """Extract text from a PDF page using PyMuPDF (for pages with text layer)."""
    doc = fitz.open(pdf_path)
    try:
        page = doc[page_num - 1]  # 0-indexed
        return page.get_text("text").strip()
    finally:
        doc.close()


def get_page_text(
    pdf_path: Path,
    page_num: int,
    ocr_engine: Optional[OCREngine],
    page_id: str,
    image_path: Path,
) -> str:
    """Get text for a page: try PyMuPDF first, fall back to OCR if empty."""
    text = extract_text_from_pdf_page(pdf_path, page_num)
    if text:
        return text

    # No text layer - use OCR if engine provided
    if ocr_engine is not None:
        return ocr_engine.ocr_page(page_id, image_path)

    return ""