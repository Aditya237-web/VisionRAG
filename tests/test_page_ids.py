# Tests for page id construction/parsing, especially doc names with underscores.
import json
from pathlib import Path

import pytest

from src.evaluate import page_id_for
from src.page_ids import build_page_id, parse_page_id

ROOT = Path(__file__).resolve().parents[1]


def test_build_page_id():
    assert build_page_id("paper1", 4) == "paper1_page_0004"
    assert build_page_id("f_paper01", 1) == "f_paper01_page_0001"
    assert build_page_id("BERT_paper01", 16) == "BERT_paper01_page_0016"
    assert build_page_id("f_paper01", "7") == "f_paper01_page_0007"


def test_parse_page_id_underscore_doc_name():
    assert parse_page_id("f_paper01_page_0001") == ("f_paper01", 1)
    assert parse_page_id("f_paper02_page_0007") == ("f_paper02", 7)
    assert parse_page_id("BERT_paper01_page_0016") == ("BERT_paper01", 16)
    assert parse_page_id("paper1_page_0004") == ("paper1", 4)


def test_parse_page_id_split_on_last_separator():
    # Doc name that itself contains '_page_' must not confuse the parser
    assert parse_page_id("a_page_b_page_0001") == ("a_page_b", 1)


def test_round_trip():
    for doc in ["paper1", "paper2", "f_paper01", "f_paper02", "BERT_paper01", "a_page_b"]:
        for page_num in (1, 16, 51):
            page_id = build_page_id(doc, page_num)
            assert parse_page_id(page_id) == (doc, page_num)


@pytest.mark.parametrize(
    "bad",
    ["", "paper1", "paper1_page_", "_page_0001", "paper1_page_abc", "page_0001"],
)
def test_parse_page_id_rejects_invalid(bad):
    with pytest.raises(ValueError, match="Invalid page_id"):
        parse_page_id(bad)


def test_evaluate_page_id_for_round_trip():
    question = {"doc": "f_paper01", "gold_page": 1, "type": "text"}
    page_id = page_id_for(question)
    assert page_id == "f_paper01_page_0001"
    assert parse_page_id(page_id) == ("f_paper01", 1)


def test_pipeline_image_path_for_underscore_doc():
    # Mirrors the pipeline's page_id -> image path resolution
    doc_name, page_num = parse_page_id("f_paper01_page_0001")
    img_path = ROOT / "data" / "pages" / doc_name / f"page_{page_num:04d}.png"
    if not (ROOT / "data" / "pages" / "metadata.jsonl").exists():
        pytest.skip("pages not ingested")
    assert img_path.exists(), f"expected {img_path}"


def test_indexes_contain_underscore_page_ids():
    visual_order = ROOT / "data" / "index" / "page_order.json"
    text_order = ROOT / "data" / "index" / "text" / "page_order.json"
    if not visual_order.exists():
        pytest.skip("visual index not built")

    with open(visual_order, encoding="utf-8") as f:
        visual_pages = json.load(f)
    assert "f_paper01_page_0001" in visual_pages
    assert "BERT_paper01_page_0016" in visual_pages
    assert len(visual_pages) == 51

    if text_order.exists():
        with open(text_order, encoding="utf-8") as f:
            text_pages = json.load(f)
        assert "f_paper01_page_0001" in text_pages
        # scanned pages have no text, so they must not be in the text index
        assert not any(p.startswith("BERT_paper01") for p in text_pages)
        assert len(text_pages) == 35
