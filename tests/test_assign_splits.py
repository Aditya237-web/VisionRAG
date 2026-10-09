# Tests for stratified dev/test split assignment (scripts/assign_splits.py).
import pytest

from scripts.assign_splits import assign_stratified_splits, target_dev_count


def _questions(types):
    return [
        {
            "id": f"q{i:03d}",
            "question": f"question {i}",
            "doc": "paper1",
            "gold_page": i,
            "type": t,
        }
        for i, t in enumerate(types, 1)
    ]


def test_target_dev_count():
    assert target_dev_count(0) == 0
    assert target_dev_count(1) == 0
    assert target_dev_count(2) == 1
    assert target_dev_count(5) == 1
    assert target_dev_count(6) == 1
    assert target_dev_count(8) == 2
    assert target_dev_count(10) == 2
    assert target_dev_count(27) == 5


def test_assigns_split_to_every_question():
    questions = _questions(["text", "text"])
    assert "split" not in questions[0]
    assign_stratified_splits(questions, seed=1)
    assert all(q["split"] in ("dev", "test") for q in questions)


def test_empty_list_is_a_no_op():
    assert assign_stratified_splits([]) == []


def test_seed_set_20_80_and_stratified():
    # 5 questions: table, text, text, chart, table
    questions = _questions(["table", "text", "text", "chart", "table"])
    assign_stratified_splits(questions, dev_fraction=0.2, seed=42)

    dev = [q for q in questions if q["split"] == "dev"]
    test = [q for q in questions if q["split"] == "test"]
    assert len(dev) == 1  # round(0.2 * 5) = 1 -> 20% dev / 80% test
    assert len(test) == 4
    # Largest remainder puts the single dev slot in the table type
    assert dev[0]["type"] == "table"


def test_stratified_counts_on_larger_set():
    types = ["text"] * 12 + ["table"] * 8 + ["chart"] * 5 + ["scanned"] * 2
    questions = _questions(types)
    assign_stratified_splits(questions, dev_fraction=0.2, seed=42)

    # 27 questions -> floor(0.2*27 + 0.5) = 5 dev
    assert target_dev_count(27) == 5
    dev = [q for q in questions if q["split"] == "dev"]
    assert len(dev) == 5

    # Proportional per type: text 12*5/27=2.2, table 8*5/27=1.5,
    # chart 5*5/27=0.9, scanned 2*5/27=0.4; largest remainder gives the
    # leftover slots to chart, then table -> text 2, table 2, chart 1, scanned 0
    dev_by_type = {}
    for q in dev:
        dev_by_type[q["type"]] = dev_by_type.get(q["type"], 0) + 1
    assert dev_by_type == {"text": 2, "table": 2, "chart": 1}


def test_same_seed_is_deterministic():
    types = ["text"] * 6 + ["table"] * 6
    first = assign_stratified_splits(_questions(types), seed=7)
    second = assign_stratified_splits(_questions(types), seed=7)
    assert [q["split"] for q in first] == [q["split"] for q in second]


def test_reassignment_is_idempotent():
    questions = _questions(["text"] * 6 + ["table"] * 6)
    assign_stratified_splits(questions, seed=42)
    once = [q["split"] for q in questions]
    assign_stratified_splits(questions, seed=42)
    twice = [q["split"] for q in questions]
    assert once == twice


def test_question_order_is_preserved():
    questions = _questions(["text", "table", "chart"])
    ids_before = [q["id"] for q in questions]
    assign_stratified_splits(questions, seed=42)
    assert [q["id"] for q in questions] == ids_before


def test_single_question_goes_to_test():
    questions = _questions(["text"])
    assign_stratified_splits(questions, seed=42)
    assert questions[0]["split"] == "test"
