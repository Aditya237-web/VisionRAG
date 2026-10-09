# Tests for CSV question importer (scripts/import_questions.py).
import csv
import json
import tempfile
from pathlib import Path

import pytest

from scripts.import_questions import (
    get_pdf_page_count,
    next_question_id,
    question_exists,
    validate_row,
)


def test_next_question_id():
    questions = []
    assert next_question_id(questions) == "q001"
    questions.append({"id": "q001"})
    assert next_question_id(questions) == "q002"
    questions.append({"id": "q005"})
    assert next_question_id(questions) == "q006"


def test_question_exists_case_insensitive():
    questions = [{"question": "What is X?"}, {"question": "Another question"}]
    assert question_exists(questions, "what is x?") is True
    assert question_exists(questions, "WHAT IS X?") is True
    assert question_exists(questions, "What is Y?") is False


def test_validate_row_success(tmp_path):
    # Create a dummy PDF with 10 pages
    pdf_dir = tmp_path / "pdfs"
    pdf_dir.mkdir()
    pdf_path = pdf_dir / "paper1.pdf"
    import fitz
    doc = fitz.open()
    for _ in range(10):
        doc.new_page()
    doc.save(pdf_path)
    doc.close()

    existing = []
    row = {
        "question": "What is the capital of France?",
        "doc": "paper1",
        "gold_page": "5",
        "type": "text",
    }
    record = validate_row(row, pdf_dir, existing)
    assert record is not None
    assert record["question"] == "What is the capital of France?"
    assert record["doc"] == "paper1"
    assert record["gold_page"] == 5
    assert record["type"] == "text"
    assert "id" not in record
    assert "split" not in record


def test_validate_row_missing_columns():
    existing = []
    row = {"question": "Q?", "doc": "paper1", "gold_page": "1"}  # missing type
    assert validate_row(row, Path("."), existing) is None


def test_validate_row_invalid_type():
    existing = []
    row = {
        "question": "Q?",
        "doc": "paper1",
        "gold_page": "1",
        "type": "diagram",
    }
    assert validate_row(row, Path("."), existing) is None


def test_validate_row_missing_pdf(tmp_path):
    existing = []
    row = {
        "question": "Q?",
        "doc": "nonexistent",
        "gold_page": "1",
        "type": "text",
    }
    assert validate_row(row, tmp_path / "pdfs", existing) is None


def test_validate_row_gold_page_out_of_range(tmp_path):
    pdf_dir = tmp_path / "pdfs"
    pdf_dir.mkdir()
    pdf_path = pdf_dir / "paper1.pdf"
    import fitz
    doc = fitz.open()
    for _ in range(5):
        doc.new_page()
    doc.save(pdf_path)
    doc.close()

    existing = []
    row = {
        "question": "Q?",
        "doc": "paper1",
        "gold_page": "10",  # > 5
        "type": "text",
    }
    assert validate_row(row, pdf_dir, existing) is None


def test_validate_row_duplicate_question(tmp_path):
    pdf_dir = tmp_path / "pdfs"
    pdf_dir.mkdir()
    pdf_path = pdf_dir / "paper1.pdf"
    import fitz
    doc = fitz.open()
    for _ in range(5):
        doc.new_page()
    doc.save(pdf_path)
    doc.close()

    existing = [{"question": "What is the answer?"}]
    row = {
        "question": "what is the answer?",  # case-insensitive duplicate
        "doc": "paper1",
        "gold_page": "1",
        "type": "text",
    }
    assert validate_row(row, pdf_dir, existing) is None


def test_validate_row_gold_page_not_integer(tmp_path):
    pdf_dir = tmp_path / "pdfs"
    pdf_dir.mkdir()
    pdf_path = pdf_dir / "paper1.pdf"
    import fitz
    doc = fitz.open()
    doc.new_page()
    doc.save(pdf_path)
    doc.close()

    existing = []
    row = {
        "question": "Q?",
        "doc": "paper1",
        "gold_page": "notanint",
        "type": "text",
    }
    assert validate_row(row, pdf_dir, existing) is None


def test_get_pdf_page_count(tmp_path):
    pdf_dir = tmp_path / "pdfs"
    pdf_dir.mkdir()
    pdf_path = pdf_dir / "test.pdf"
    import fitz
    doc = fitz.open()
    for _ in range(7):
        doc.new_page()
    doc.save(pdf_path)
    doc.close()
    assert get_pdf_page_count(pdf_path) == 7


def test_full_import(tmp_path):
    # Setup: create dummy PDFs
    pdf_dir = tmp_path / "pdfs"
    pdf_dir.mkdir()
    for name, pages in [("paper1", 10), ("paper2", 5)]:
        pdf_path = pdf_dir / f"{name}.pdf"
        import fitz
        doc = fitz.open()
        for _ in range(pages):
            doc.new_page()
        doc.save(pdf_path)
        doc.close()

    # Create questions.json with existing entries
    questions_path = tmp_path / "questions.json"
    questions_path.write_text(
        json.dumps([
            {"id": "q001", "question": "Existing Q1", "doc": "paper1", "gold_page": 1, "type": "text", "split": "test"},
            {"id": "q002", "question": "Existing Q2", "doc": "paper2", "gold_page": 2, "type": "table", "split": "dev"},
        ]), encoding="utf-8"
    )

    # Create CSV with valid and invalid rows
    csv_path = tmp_path / "import.csv"
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["question", "doc", "gold_page", "type"])
        writer.writeheader()
        writer.writerow({"question": "New Q1", "doc": "paper1", "gold_page": "3", "type": "text"})
        writer.writerow({"question": "New Q2", "doc": "paper2", "gold_page": "4", "type": "table"})
        writer.writerow({"question": "Existing Q1", "doc": "paper1", "gold_page": "1", "type": "text"})  # duplicate
        writer.writerow({"question": "Bad type", "doc": "paper1", "gold_page": "1", "type": "diagram"})  # invalid type
        writer.writerow({"question": "Missing doc", "doc": "nonexistent", "gold_page": "1", "type": "text"})  # no pdf
        writer.writerow({"question": "Bad page", "doc": "paper1", "gold_page": "99", "type": "text"})  # out of range

    # Import via the script's main function logic
    from scripts.import_questions import load_existing_questions, save_questions, next_question_id, validate_row, question_exists

    questions = load_existing_questions(questions_path)
    imported = 0
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            record = validate_row(row, pdf_dir, questions)
            if record is None:
                continue
            record["id"] = next_question_id(questions)
            questions.append(record)
            imported += 1

    save_questions(questions, questions_path)

    assert imported == 2  # only the first two valid rows
    assert len(questions) == 4  # 2 existing + 2 new
    assert questions[2]["id"] == "q003"
    assert questions[3]["id"] == "q004"
    assert questions[2]["question"] == "New Q1"
    assert questions[3]["question"] == "New Q2"
    assert "split" not in questions[2]
    assert "split" not in questions[3]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])