# VisionRAG interactive question labeling helper
# Opens a page image, prompts for question + type, appends to questions.json.
# Splits are NOT assigned here; run scripts/assign_splits.py when done writing.
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluate import VALID_TYPES, load_questions  # noqa: E402


def load_pages(pages_dir: Path) -> List[Dict[str, Any]]:
    """Read metadata.jsonl into a list of page records."""
    metadata_path = pages_dir / "metadata.jsonl"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata not found: {metadata_path}")

    pages: List[Dict[str, Any]] = []
    with open(metadata_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            meta = json.loads(line)
            pages.append(
                {
                    "page_id": f"{meta['doc_name']}_page_{meta['page_num']:04d}",
                    "doc": meta["doc_name"],
                    "page_num": meta["page_num"],
                    "image_path": ROOT / meta["image_path"],
                }
            )
    return pages


def next_question_id(questions: List[Dict[str, Any]]) -> str:
    """Next sequential id: q001, q002, ... based on existing ids."""
    max_n = 0
    for q in questions:
        suffix = str(q.get("id", "")).lstrip("qQ")
        if suffix.isdigit():
            max_n = max(max_n, int(suffix))
    return f"q{max_n + 1:03d}"


def open_image(path: Path) -> None:
    """Open the page image with the OS default viewer."""
    if not path.exists():
        print(f"  Image not found: {path}")
        return
    try:
        if os.name == "nt":
            os.startfile(str(path))  # type: ignore[attr-defined]
        elif sys.platform == "darwin":
            subprocess.Popen(["open", str(path)])
        else:
            subprocess.Popen(["xdg-open", str(path)])
        print(f"  Opened: {path}")
    except OSError as exc:
        print(f"  Could not open image: {exc}")
        print(f"  Open it manually: {path}")


def prompt_type() -> Optional[str]:
    """Ask for the question type; empty input aborts."""
    allowed = "/".join(VALID_TYPES)
    while True:
        value = input(f"Type [{allowed}] (Enter=abort): ").strip().lower()
        if not value:
            return None
        if value in VALID_TYPES:
            return value
        print(f"  Invalid, choose one of: {allowed}")


def save_questions(questions: List[Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(questions, f, indent=2, ensure_ascii=False)
        f.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Label VisionRAG eval questions while viewing page images"
    )
    parser.add_argument("--questions", default=None, help="Path to questions.json")
    parser.add_argument("--pages_dir", default=str(ROOT / "data" / "pages"))
    parser.add_argument("--start", type=int, default=1, help="1-based page to start at")
    args = parser.parse_args()

    pages_dir = Path(args.pages_dir)
    questions_path = (
        Path(args.questions) if args.questions else ROOT / "data" / "eval" / "questions.json"
    )

    pages = load_pages(pages_dir)
    if not pages:
        raise SystemExit(f"No pages found in {pages_dir}")
    questions = (
        load_questions(questions_path, require_split=False)
        if questions_path.exists()
        else []
    )
    n_assigned = sum(1 for q in questions if "split" in q)
    idx = max(0, min(args.start - 1, len(pages) - 1))

    print(f"{len(pages)} pages | {len(questions)} questions in {questions_path}")
    if questions and n_assigned != len(questions):
        print(f"  ({n_assigned}/{len(questions)} have splits; "
              "run scripts/assign_splits.py after writing)")
    print(
        "Commands: Enter=label this page, n=next, p=prev, g <n>=goto page, q=quit"
    )

    while True:
        page = pages[idx]
        print(f"\n[{idx + 1}/{len(pages)}] {page['page_id']}  {page['image_path'].name}")
        try:
            cmd = input("> ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            break

        if cmd == "q":
            print("Bye.")
            break
        if cmd == "n":
            idx = min(idx + 1, len(pages) - 1)
            continue
        if cmd == "p":
            idx = max(idx - 1, 0)
            continue
        if cmd.startswith("g"):
            parts = cmd.split()
            try:
                target = int(parts[1])
            except (IndexError, ValueError):
                print("  Usage: g <page-number>")
                continue
            if not 1 <= target <= len(pages):
                print(f"  Page must be 1..{len(pages)}")
                continue
            idx = target - 1
            continue
        if cmd != "":
            print("  Unknown command")
            continue

        # Enter: show image, then ask for question + type
        open_image(page["image_path"])
        question = input("Question (Enter=abort): ").strip()
        if not question:
            print("  Aborted.")
            continue
        qtype = prompt_type()
        if qtype is None:
            print("  Aborted.")
            continue

        record = {
            "id": next_question_id(questions),
            "question": question,
            "doc": page["doc"],
            "gold_page": page["page_num"],
            "type": qtype,
        }
        questions.append(record)
        save_questions(questions, questions_path)
        print(
            f"  Saved {record['id']} gold={page['page_id']} type={record['type']} "
            "(no split yet, run scripts/assign_splits.py when done)"
        )


if __name__ == "__main__":
    main()
