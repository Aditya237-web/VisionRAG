# Tests for Text Baseline Retriever
from __future__ import annotations

import tempfile
from pathlib import Path

import fitz
import pytest

from src.retriever_text import TextRetriever
from src.utils import load_config


def create_test_pdf(output_path: Path, pages_content: List[str]) -> None:
    """Create a test PDF with given page contents."""
    doc = fitz.open()
    for content in pages_content:
        page = doc.new_page(width=400, height=300)
        if content:
            page.insert_text((50, 50), content, fontsize=12)
    doc.save(str(output_path))
    doc.close()


def test_chunk_to_page_mapping():
    """Test that chunks correctly map to page_ids matching visual retriever format."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        pdf_dir = tmpdir / "pdfs"
        pdf_dir.mkdir()
        index_dir = tmpdir / "index"
        index_dir.mkdir()

        # Create test PDF with 2 pages
        pdf_path = pdf_dir / "test_doc.pdf"
        create_test_pdf(pdf_path, [
            "This is page one content about transformers and attention mechanisms.",
            "This is page two content about neural networks and deep learning."
        ])

        config = load_config()
        retriever = TextRetriever(config)
        retriever.build_index(pdf_dir, index_dir)

        # Load and verify
        from src.utils import load_text_index
        index, chunks, chunk_to_page, page_order = load_text_index(index_dir)

        # Verify page_order format matches visual retriever: "doc_page_0004"
        assert len(page_order) == 2
        assert page_order[0] == "test_doc_page_0001"
        assert page_order[1] == "test_doc_page_0002"

        # Verify each chunk has correct page_id
        for chunk, page_id in zip(chunks, chunk_to_page):
            assert chunk["page_id"] == page_id
            assert page_id in page_order
            assert page_id.startswith("test_doc_page_")


def test_retrieve_output_format():
    """Test that retrieve() returns List[Tuple[str, float]] with correct page_ids."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        pdf_dir = tmpdir / "pdfs"
        pdf_dir.mkdir()
        index_dir = tmpdir / "index"
        index_dir.mkdir()

        pdf_path = pdf_dir / "test_doc.pdf"
        create_test_pdf(pdf_path, [
            "Multi-head attention is a key component of transformers.",
            "The transformer architecture uses self-attention layers."
        ])

        config = load_config()
        retriever = TextRetriever(config)
        retriever.build_index(pdf_dir, index_dir)

        results = retriever.retrieve("What is multi-head attention?", k=2)

        # Verify output format
        assert isinstance(results, list)
        assert len(results) <= 2
        for page_id, score in results:
            assert isinstance(page_id, str)
            assert isinstance(score, float)
            assert page_id.startswith("test_doc_page_")


def test_scanned_page_handling():
    """Test that pages with no extractable text are skipped and counted."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        pdf_dir = tmpdir / "pdfs"
        pdf_dir.mkdir()
        index_dir = tmpdir / "index"
        index_dir.mkdir()

        # Create PDF with one text page and one empty (scanned) page
        pdf_path = pdf_dir / "mixed_doc.pdf"
        doc = fitz.open()
        # Page 1: has text
        page1 = doc.new_page(width=400, height=300)
        page1.insert_text((50, 50), "This page has text content about attention.", fontsize=12)
        # Page 2: empty (simulates scanned page)
        doc.new_page(width=400, height=300)
        doc.save(str(pdf_path))
        doc.close()

        config = load_config()
        retriever = TextRetriever(config)
        retriever.build_index(pdf_dir, index_dir)

        # Verify build stats show scanned page
        import json
        with open(index_dir / "build_stats.json", "r") as f:
            stats = json.load(f)

        assert stats["pages_with_text"] == 1
        assert stats["pages_no_text"] == 1
        # Only one page should be in page_order
        from src.utils import load_text_index
        _, _, _, page_order = load_text_index(index_dir)
        assert len(page_order) == 1
        assert page_order[0] == "mixed_doc_page_0001"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
