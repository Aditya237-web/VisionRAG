# Tests for the evaluation harness metrics using a fake retriever.
# Expected values are hand-computed, not produced by the implementation.
import json
from pathlib import Path

import pytest

from src.evaluate import (
    aggregate,
    evaluate_retriever,
    format_markdown,
    index_stats,
    load_questions,
    page_id_for,
    reciprocal_rank,
    topk_hit,
)

SRC_EVALUATE = Path(__file__).resolve().parents[1] / "src" / "evaluate.py"


class FakeRetriever:
    """Returns a pre-programmed ranking regardless of the query text."""

    def __init__(self, ranking):
        self.ranking = ranking

    def retrieve(self, query, k=5):
        page_ids = self.ranking[query][:k]
        return [(pid, float(len(page_ids) - i)) for i, pid in enumerate(page_ids)]


def _q(i, qtype, split="test"):
    return {
        "id": f"q{i:03d}",
        "question": f"question {i}",
        "doc": "paper1",
        "gold_page": i,
        "type": qtype,
        "split": split,
    }


QUESTIONS = [
    _q(1, "text"),
    _q(2, "text"),
    _q(3, "table"),
    _q(4, "chart"),
    _q(5, "table"),
]

# Gold ranks: q1 -> 1, q2 -> 2, q3 -> 3, q4 -> MISS, q5 -> 5
RANKING = {
    "question 1": [
        "paper1_page_0001", "paper1_page_0007", "paper1_page_0008",
        "paper1_page_0009", "paper1_page_0010",
    ],
    "question 2": [
        "paper1_page_0007", "paper1_page_0002", "paper1_page_0008",
        "paper1_page_0009", "paper1_page_0010",
    ],
    "question 3": [
        "paper1_page_0007", "paper1_page_0008", "paper1_page_0003",
        "paper1_page_0009", "paper1_page_0010",
    ],
    "question 4": [
        "paper1_page_0007", "paper1_page_0008", "paper1_page_0009",
        "paper1_page_0010", "paper1_page_0011",
    ],
    "question 5": [
        "paper1_page_0007", "paper1_page_0008", "paper1_page_0009",
        "paper1_page_0010", "paper1_page_0005",
    ],
}


def test_topk_hit_hand_computed():
    ranked = ["a", "b", "c", "d", "e"]
    assert topk_hit(ranked, "a", 1) is True
    assert topk_hit(ranked, "c", 1) is False
    assert topk_hit(ranked, "c", 3) is True
    assert topk_hit(ranked, "e", 3) is False
    assert topk_hit(ranked, "e", 5) is True
    assert topk_hit(ranked, "z", 5) is False
    assert topk_hit([], "a", 5) is False


def test_reciprocal_rank_hand_computed():
    ranked = ["a", "b", "c", "d", "e"]
    assert reciprocal_rank(ranked, "a") == pytest.approx(1.0)
    assert reciprocal_rank(ranked, "b") == pytest.approx(0.5)
    assert reciprocal_rank(ranked, "c") == pytest.approx(1.0 / 3.0)
    assert reciprocal_rank(ranked, "e") == pytest.approx(0.2)
    assert reciprocal_rank(ranked, "z") == 0.0


def test_overall_metrics_hand_computed():
    # ranks: 1, 2, 3, MISS, 5  over 5 questions
    # top1 = 1/5, top3 = 3/5, top5 = 4/5
    # MRR = (1 + 1/2 + 1/3 + 0 + 1/5) / 5 = (61/30)/5 = 61/150
    result = evaluate_retriever(FakeRetriever(RANKING), QUESTIONS, k_max=5)
    overall = result["overall"]
    assert overall["n"] == 5
    assert overall["top1"] == pytest.approx(1 / 5)
    assert overall["top3"] == pytest.approx(3 / 5)
    assert overall["top5"] == pytest.approx(4 / 5)
    assert overall["mrr"] == pytest.approx(61 / 150)
    assert overall["avg_latency_s"] >= 0.0


def test_by_type_breakdown_hand_computed():
    # text: ranks 1, 2    -> top1=1/2, top3=1, top5=1, MRR=(1+1/2)/2=3/4
    # table: ranks 3, 5   -> top1=0, top3=1/2, top5=1, MRR=(1/3+1/5)/2=4/15
    # chart: MISS         -> all 0
    result = evaluate_retriever(FakeRetriever(RANKING), QUESTIONS, k_max=5)
    by_type = result["by_type"]

    assert set(by_type) == {"text", "table", "chart"}
    assert "scanned" not in by_type

    text = by_type["text"]
    assert text["n"] == 2
    assert text["top1"] == pytest.approx(1 / 2)
    assert text["top3"] == pytest.approx(1.0)
    assert text["top5"] == pytest.approx(1.0)
    assert text["mrr"] == pytest.approx(3 / 4)

    table = by_type["table"]
    assert table["n"] == 2
    assert table["top1"] == pytest.approx(0.0)
    assert table["top3"] == pytest.approx(1 / 2)
    assert table["top5"] == pytest.approx(1.0)
    assert table["mrr"] == pytest.approx(4 / 15)

    chart = by_type["chart"]
    assert chart["n"] == 1
    assert chart["top1"] == 0.0
    assert chart["top3"] == 0.0
    assert chart["top5"] == 0.0
    assert chart["mrr"] == 0.0


def test_per_question_records():
    result = evaluate_retriever(FakeRetriever(RANKING), QUESTIONS, k_max=5)
    records = result["per_question"]
    assert [r["id"] for r in records] == ["q001", "q002", "q003", "q004", "q005"]
    assert [r["rank"] for r in records] == [1, 2, 3, None, 5]
    assert records[3]["rr"] == 0.0
    assert records[0]["gold"] == "paper1_page_0001"
    assert all(r["latency_s"] >= 0.0 for r in records)


def test_k_max_must_cover_top5():
    with pytest.raises(ValueError, match="k_max"):
        evaluate_retriever(FakeRetriever(RANKING), QUESTIONS, k_max=3)


def test_aggregate_empty():
    assert aggregate([]) == {"n": 0, "top1": 0.0, "top3": 0.0, "top5": 0.0, "mrr": 0.0}


def test_assign_split_removed_from_entry_logic():
    # Splits are no longer assigned while writing questions; they are assigned
    # afterwards by scripts/assign_splits.py (covered in test_assign_splits.py).
    import src.evaluate as evaluate_module

    assert not hasattr(evaluate_module, "assign_split")


def test_page_id_for():
    assert page_id_for({"doc": "paper1", "gold_page": 4}) == "paper1_page_0004"
    assert page_id_for({"doc": "paper2", "gold_page": "12"}) == "paper2_page_0012"


def test_load_questions_valid(tmp_path):
    path = tmp_path / "questions.json"
    path.write_text(json.dumps(QUESTIONS), encoding="utf-8")
    loaded = load_questions(path)
    assert len(loaded) == 5
    assert loaded[0]["id"] == "q001"


def test_load_questions_rejects_bad_records(tmp_path):
    path = tmp_path / "questions.json"

    missing_field = [_q(1, "text")]
    del missing_field[0]["gold_page"]
    path.write_text(json.dumps(missing_field), encoding="utf-8")
    with pytest.raises(ValueError, match="missing fields"):
        load_questions(path)

    bad_type = [_q(1, "diagram")]
    path.write_text(json.dumps(bad_type), encoding="utf-8")
    with pytest.raises(ValueError, match="invalid type"):
        load_questions(path)

    bad_split = [_q(1, "text", split="holdout")]
    path.write_text(json.dumps(bad_split), encoding="utf-8")
    with pytest.raises(ValueError, match="invalid split"):
        load_questions(path)

    duplicates = [_q(1, "text"), _q(1, "table")]
    path.write_text(json.dumps(duplicates), encoding="utf-8")
    with pytest.raises(ValueError, match="Duplicate"):
        load_questions(path)

    path.write_text(json.dumps({"not": "a list"}), encoding="utf-8")
    with pytest.raises(ValueError, match="JSON array"):
        load_questions(path)

    with pytest.raises(FileNotFoundError):
        load_questions(tmp_path / "nope.json")


def test_load_questions_requires_split_by_default(tmp_path):
    path = tmp_path / "questions.json"
    unassigned = [{k: v for k, v in _q(1, "text").items() if k != "split"}]
    path.write_text(json.dumps(unassigned), encoding="utf-8")

    with pytest.raises(ValueError, match="assign_splits"):
        load_questions(path)

    loaded = load_questions(path, require_split=False)
    assert len(loaded) == 1
    assert "split" not in loaded[0]


def test_index_stats(tmp_path):
    index_dir = tmp_path / "index"
    (index_dir / "text").mkdir(parents=True)
    (index_dir / "build_stats.json").write_text(
        json.dumps({"elapsed_seconds": 12.5}), encoding="utf-8"
    )
    (index_dir / "visual_embeddings.pt").write_bytes(b"x" * 1000)
    (index_dir / "text" / "build_stats.json").write_text(
        json.dumps({"elapsed_seconds": 3.0}), encoding="utf-8"
    )
    (index_dir / "text" / "faiss.index").write_bytes(b"y" * 500)
    config = {"paths": {"index_dir": str(index_dir)}}

    visual = index_stats("visual", config)
    assert visual["build_time_s"] == 12.5
    expected_visual = (
        (index_dir / "build_stats.json").stat().st_size
        + (index_dir / "visual_embeddings.pt").stat().st_size
    )
    assert visual["index_size_mb"] == pytest.approx(expected_visual / 1e6, abs=1e-6)

    text = index_stats("text", config)
    assert text["build_time_s"] == 3.0
    expected_text = (
        (index_dir / "text" / "build_stats.json").stat().st_size
        + (index_dir / "text" / "faiss.index").stat().st_size
    )
    assert text["index_size_mb"] == pytest.approx(expected_text / 1e6, abs=1e-6)


def test_format_markdown_renders_tables():
    report = {
        "questions_file": "data/eval/questions.json",
        "split": "test",
        "n_eval": 2,
        "n_dev": 1,
        "n_test": 2,
        "k_max": 5,
        "retriever_order": ["visual", "text"],
        "retrievers": {
            "visual": {
                "overall": {
                    "n": 2, "top1": 0.5, "top3": 1.0, "top5": 1.0, "mrr": 0.75,
                    "avg_latency_s": 1.234, "build_time_s": 332.5,
                    "load_time_s": 0.5, "index_size_mb": 12.84,
                    "top1_ci": (0.4, 0.6), "top3_ci": (0.9, 1.1), "top5_ci": (0.9, 1.1), "mrr_ci": (0.6, 0.9),
                },
                "by_type": {"text": {"n": 2, "top1": 0.5, "top3": 1.0, "top5": 1.0, "mrr": 0.75}},
                "per_question": [
                    {"id": "q002", "type": "text", "gold": "paper1_page_0005", "rank": 1}
                ],
            },
            "text": {
                "overall": {
                    "n": 2, "top1": 0.0, "top3": 0.5, "top5": 0.5, "mrr": 0.25,
                    "avg_latency_s": 0.01, "build_time_s": 18.5,
                    "load_time_s": 0.1, "index_size_mb": 0.24,
                    "top1_ci": (0.0, 0.1), "top3_ci": (0.4, 0.6), "top5_ci": (0.4, 0.6), "mrr_ci": (0.1, 0.4),
                },
                "by_type": {"text": {"n": 2, "top1": 0.0, "top3": 0.5, "top5": 0.5, "mrr": 0.25}},
                "per_question": [
                    {"id": "q002", "type": "text", "gold": "paper1_page_0005", "rank": None}
                ],
            },
        },
    }
    md = format_markdown(report)
    assert "# VisionRAG Retrieval Evaluation" in md
    assert "## Overall (test split)" in md
    assert "## By type (test split) — with 95% Wilson score CIs" in md
    assert "## Per-question gold rank (test split)" in md
    assert "| Retriever | N | Top-1 (95% CI) | Top-3 (95% CI) | Top-5 (95% CI) | MRR (95% CI) |" in md
    assert "| visual | 2 | 0.500 (0.400, 0.600) | 1.000 (0.900, 1.100) | 1.000 (0.900, 1.100) | 0.750 (0.600, 0.900) | 1.234 | 0.5 | 332.5 | 12.84 |" in md
    assert "| text | 2 | 0.000 (0.000, 0.100) | 0.500 (0.400, 0.600) | 0.500 (0.400, 0.600) | 0.250 (0.100, 0.400) | 0.010 | 0.1 | 18.5 | 0.24 |" in md


def test_evaluate_never_calls_ollama():
    source = SRC_EVALUATE.read_text(encoding="utf-8")
    for forbidden in ("import ollama", "src.generator", "src.pipeline", "api/generate", "11434"):
        assert forbidden not in source, f"evaluate.py must not reference {forbidden!r}"
