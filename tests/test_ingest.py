# Tests for ingestion module
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import fitz  # PyMuPDF
import pytest

from src.ingest import render_pdf_to_images, process_pdf_dir


def create_test_pdf(output_path: Path, num_pages: int = 2) -> None:
    """Create a simple test PDF with given number of pages."""
    doc = fitz.open()
    for i in range(num_pages):
        page = doc.new_page(width=200, height=100)
        page.insert_text((50, 50), f"Test Page {i + 1}", fontsize=12)
    doc.save(str(output_path))
    doc.close()


def test_render_pdf_to_images(tmp_path: Path):
    """Test rendering a PDF to images."""
    # Create test PDF
    pdf_path = tmp_path / "test.pdf"
    create_test_pdf(pdf_path, num_pages=2)
    
    # Render to images
    output_dir = tmp_path / "pages"
    metadata = render_pdf_to_images(pdf_path, output_dir, dpi=100)
    
    # Check metadata
    assert len(metadata) == 2
    assert metadata[0]["page_num"] == 1
    assert metadata[1]["page_num"] == 2
    assert metadata[0]["doc"] == "test.pdf"
    assert metadata[1]["doc"] == "test.pdf"
    
    # Check image files exist
    img1 = output_dir / "page_0001.png"
    img2 = output_dir / "page_0002.png"
    assert img1.exists()
    assert img2.exists()
    
    # Check metadata fields
    for m in metadata:
        assert "image_path" in m
        assert "width" in m
        assert "height" in m
        assert m["width"] > 0
        assert m["height"] > 0


def test_process_pdf_dir(tmp_path: Path):
    """Test processing a directory of PDFs."""
    pdf_dir = tmp_path / "pdfs"
    pdf_dir.mkdir()
    pages_dir = tmp_path / "pages"
    
    # Create two test PDFs
    pdf1 = pdf_dir / "doc1.pdf"
    pdf2 = pdf_dir / "doc2.pdf"
    create_test_pdf(pdf1, num_pages=2)
    create_test_pdf(pdf2, num_pages=3)
    
    # Process
    process_pdf_dir(pdf_dir, pages_dir, dpi=100)
    
    # Check output structure
    assert (pages_dir / "doc1").exists()
    assert (pages_dir / "doc2").exists()
    
    # Check metadata file
    metadata_path = pages_dir / "metadata.jsonl"
    assert metadata_path.exists()
    
    # Load and verify metadata
    with open(metadata_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    
    assert len(lines) == 5  # 2 + 3 pages
    
    metadata = [json.loads(line) for line in lines]
    
    # Verify page numbers and doc names
    doc1_pages = [m for m in metadata if m["doc"] == "doc1.pdf"]
    doc2_pages = [m for m in metadata if m["doc"] == "doc2.pdf"]
    
    assert len(doc1_pages) == 2
    assert len(doc2_pages) == 3
    
    assert [m["page_num"] for m in doc1_pages] == [1, 2]
    assert [m["page_num"] for m in doc2_pages] == [1, 2, 3]
    
    # Verify image paths are relative and exist
    for m in metadata:
        img_path = Path.cwd() / m["image_path"]
        assert img_path.exists(), f"Image not found: {img_path}"


def test_empty_pdf_dir(tmp_path: Path, capsys):
    """Test processing empty directory."""
    pdf_dir = tmp_path / "empty_pdfs"
    pdf_dir.mkdir()
    pages_dir = tmp_path / "pages"
    
    process_pdf_dir(pdf_dir, pages_dir, dpi=100)
    
    captured = capsys.readouterr()
    assert "No PDF files found" in captured.out


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
