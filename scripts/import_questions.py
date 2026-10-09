# VisionRAG CSV question importer
# Reads a CSV with columns: question, doc, gold_page, type
# Appends to data/eval/questions.json (no split field, same id scheme as build_questions.py)
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import fitz  # PyMuPDF

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluate import VALID_TYPES, load_questions  # noqa: E402


VALID_TYPES_LIST = list(VALID_TYPES)


def get_pdf_page_count(pdf_path: Path) -> int:
    """Return the number of pages in a PDF."""
    doc = fitz.open(pdf_path)
    try:
        return len(doc)
    finally:
        doc.close()


def next_question_id(questions: List[Dict[str, Any]]) -> str:
    """Next sequential id: q001, q002, ... based on existing ids."""
    max_n = 0
    for q in questions:
        suffix = str(q.get("id", "")).lstrip("qQ")
        if suffix.isdigit():
            max_n = max(max_n, int(suffix))
    return f"q{max_n + 1:03d}"


def save_questions(questions: List[Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(questions, f, indent=2, ensure_ascii=False)
        f.write("\n")


def load_existing_questions(path: Path) -> List[Dict[str, Any]]:
    """Load existing questions, tolerating missing split field."""
    if not path.exists():
        return []
    return load_questions(path, require_split=False)


def question_exists(questions: List[Dict[str, Any]], question_text: str) -> bool:
    """Check if a question with the same text already exists (case-insensitive)."""
    normalized = question_text.strip().lower()
    for q in questions:
        if q.get("question", "").strip().lower() == normalized:
            return True
    return False


def validate_row(
    row: Dict[str, str],
    pdf_dir: Path,
    existing_questions: List[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    """Validate a CSV row and return the record to append, or None if invalid/duplicate."""
    # Required columns
    for col in ("question", "doc", "gold_page", "type"):
        if col not in row or not row[col].strip():
            print(f"  Skip: missing or empty '{col}' in row: {row}")
            return None

    question_text = row["question"].strip()
    doc = row["doc"].strip()
    type_ = row["type"].strip().lower()
    gold_page_str = row["gold_page"].strip()

    # Validate type
    if type_ not in VALID_TYPES_LIST:
        print(f"  Skip: invalid type '{type_}' (must be one of {VALID_TYPES_LIST}): {question_text[:60]}")
        return None

    # Validate doc exists in data/pdfs
    pdf_path = pdf_dir / f"{doc}.pdf"
    if not pdf_path.exists():
        print(f"  Skip: doc '{doc}' not found at {pdf_path}: {question_text[:60]}")
        return None

    # Validate gold_page is integer
    try:
        gold_page = int(gold_page_str)
    except ValueError:
        print(f"  Skip: gold_page must be integer, got '{gold_page_str}': {question_text[:60]}")
        return None

    # Validate gold_page within PDF page count
    page_count = get_pdf_page_count(pdf_path)
    if not (1 <= gold_page <= page_count):
        print(f"  Skip: gold_page {gold_page} out of range (1..{page_count}) for {doc}: {question_text[:60]}")
        return None

    # Check duplicate question text
    if question_exists(existing_questions, question_text):
        print(f"  Skip: duplicate question (already exists): {question_text[:80]}")
        return None

    # Build record (no split field)
    return {
        "question": question_text,
        "doc": doc,
        "gold_page": gold_page,
        "type": type_,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Import questions from CSV to questions.json (no split assigned)"
    )
    parser.add_argument("csv_path", help="Path to CSV file with columns: question,doc,gold_page,type")
    parser.add_argument("--questions", default=None, help="Path to questions.json")
    parser.add_argument("--pdf_dir", default=None, help="Directory containing PDFs")
    args = parser.parse_args()

    csv_path = Path(args.csv_path)
    if not csv_path.exists():
        raise SystemExit(f"CSV file not found: {csv_path}")

    questions_path = (
        Path(args.questions) if args.questions else ROOT / "data" / "eval" / "questions.json"
    )
    pdf_dir = Path(args.pdf_dir) if args.pdf_dir else ROOT / "data" / "pdfs"

    if not pdf_dir.exists():
        raise SystemExit(f"PDF directory not found: {pdf_dir}")

    # Load existing questions
    questions = load_existing_questions(questions_path)
    print(f"Loaded {len(questions)} existing questions from {questions_path}")

    # Read CSV
    imported = 0
    skipped = 0
    type_counts: Dict[str, int] = {}
    doc_counts: Dict[str, int] = {}

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            record = validate_row(row, pdf_dir, questions)
            if record is None:
                skipped += 1
                continue

            # Assign id and append
            record["id"] = next_question_id(questions)
            questions.append(record)

            # Update counts
            type_counts[record["type"]] = type_counts.get(record["type"], 0) + 1
            doc_counts[record["doc"]] = doc_counts.get(record["doc"], 0) + 1
            imported += 1

    # Save
    save_questions(questions, questions_path)
    print(f"Saved {len(questions)} total questions to {questions_path}")

    # Summary
    print("\n=== Import Summary ===")
    print(f"Imported: {imported}")
    print(f"Skipped:  {skipped}")
    if type_counts:
        print("\nBy type:")
        for t in VALID_TYPES_LIST:
            if t in type_counts:
                print(f"  {t:8s}: {type_counts[t]}")
    if doc_counts:
        print("\nBy doc:")
        for d, c in sorted(doc_counts.items()):
            print(f"  {d}: {c}")


if __name__ == "__main__":
    main()