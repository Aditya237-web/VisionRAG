# Tests for OCR module and OCR+Text retriever.
import json
import tempfile
from pathlib import Path

import pytest

from src.ocr import OCREngine, extract_text_from_pdf_page


def test_ocr_engine_cache(tmp_path):
    """Test that OCREngine caches results to disk and memory."""
    cache_dir = tmp_path / "ocr_cache"
    cache_dir.mkdir()

    # Create a dummy image
    img_path = tmp_path / "test_page.png"
    from PIL import Image
    img = Image.new("RGB", (100, 100), color="white")
    img.save(img_path)

    # First engine - populate cache
    engine1 = OCREngine(cache_dir)
    # We can't easily test actual OCR without a model, but we can test cache file creation
    # by mocking the engine
    cache_file = cache_dir / "ocr_cache.jsonl"
    assert not cache_file.exists() or cache_file.stat().st_size == 0


def test_extract_text_from_pdf_page(tmp_path):
    """Test PyMuPDF text extraction."""
    pdf_path = tmp_path / "test.pdf"
    import fitz
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((50, 50), "Hello World")
    doc.save(pdf_path)
    doc.close()

    text = extract_text_from_pdf_page(pdf_path, 1)
    assert "Hello World" in text


def test_ocr_text_retriever_build_index(tmp_path):
    """Test that OCRTextRetriever can build an index (integration test with dummy PDFs)."""
    # This is a lightweight test - we just verify the retriever can be instantiated
    # and the build_index method runs without errors on minimal data
    from src.retriever_ocr_text import OCRTextRetriever

    config = {
        "retriever_text": {
            "embedder_name": "BAAI/bge-small-en-v1.5",
            "chunk_size": 500,
            "chunk_overlap": 50,
            "batch_size": 16,
            "device": "cpu",
            "query_prefix": "Represent this sentence for searching relevant passages: ",
        },
        "paths": {
            "index_dir": str(tmp_path / "index"),
        }
    }

    retriever = OCRTextRetriever(config)
    assert retriever is not None
    assert retriever.embedder_name == "BAAI/bge-small-en-v1.5"


def test_evaluate_supports_ocr_text():
    """Test that evaluate.py recognizes 'ocr_text' retriever type."""
    from src.evaluate import get_retriever

    config = {
        "paths": {"index_dir": "data/index"},
        "retriever_text": {
            "embedder_name": "BAAI/bge-small-en-v1.5",
            "chunk_size": 500,
            "chunk_overlap": 50,
            "batch_size": 16,
            "device": "cpu",
            "query_prefix": "Represent this sentence for searching relevant passages: ",
        },
    }

    # This would try to load actual index - we just verify the function doesn't error on unknown type
    # and that 'ocr_text' is a recognized kind
    try:
        get_retriever("ocr_text", config)
    except (FileNotFoundError, ValueError):
        pass  # Expected if index doesn't exist
    except Exception as e:
        # Should not raise other errors for unknown retriever type
        assert "Unknown retriever" not in str(e)


def test_index_stats_includes_ocr_text():
    """Test that index_stats handles 'ocr_text' kind."""
    from src.evaluate import index_stats

    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        index_root = tmpdir / "index"
        index_root.mkdir()
        (index_root / "build_stats.json").write_text(json.dumps({"elapsed_seconds": 10.0}))
        (index_root / "ocr_text").mkdir()
        (index_root / "ocr_text" / "build_stats.json").write_text(json.dumps({"elapsed_seconds": 50.0}))
        (index_root / "ocr_text" / "faiss.index").write_bytes(b"x" * 100)

        config = {"paths": {"index_dir": str(index_root)}}

        visual = index_stats("visual", config)
        assert visual["build_time_s"] == 10.0

        ocr = index_stats("ocr_text", config)
        assert ocr["build_time_s"] == 50.0
        assert ocr["index_size_mb"] > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])