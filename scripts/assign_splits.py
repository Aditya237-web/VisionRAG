# VisionRAG split assignment
# Assigns dev/test splits in one pass AFTER all questions are written.
# Stratified by question type, ~20% dev / 80% test, fixed random seed.
from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluate import VALID_TYPES, load_questions  # noqa: E402

DEFAULT_SEED = 42
DEFAULT_DEV_FRACTION = 0.2


def target_dev_count(n: int, dev_fraction: float = DEFAULT_DEV_FRACTION) -> int:
    """How many dev questions a set of n should get (~dev_fraction, at least 1 if n>=2)."""
    if n <= 1:
        return 0
    return max(1, math.floor(dev_fraction * n + 0.5))


def assign_stratified_splits(
    questions: List[Dict[str, Any]],
    dev_fraction: float = DEFAULT_DEV_FRACTION,
    seed: int = DEFAULT_SEED,
) -> List[Dict[str, Any]]:
    """Set 'split' on every question: stratified by type, ~dev_fraction dev.

    Total dev count is split across types proportionally (largest remainder),
    then each type's questions are shuffled with the fixed seed and the first
    dev_quota of them become 'dev'. Fully deterministic for a given seed and
    input order; question records are modified in place.
    """
    n = len(questions)
    if n == 0:
        return questions

    target = target_dev_count(n, dev_fraction)

    by_type: Dict[str, List[int]] = {}
    for idx, q in enumerate(questions):
        by_type.setdefault(q["type"], []).append(idx)

    # Proportional quota per type (largest remainder method)
    quota = {t: target * len(idxs) / n for t, idxs in by_type.items()}
    dev_quota = {t: math.floor(q) for t, q in quota.items()}
    leftover = target - sum(dev_quota.values())
    if leftover > 0:
        ranked = sorted(by_type, key=lambda t: (-(quota[t] - dev_quota[t]), t))
        for t in ranked[:leftover]:
            dev_quota[t] += 1

    # Seeded shuffle within each type; first dev_quota indices become dev
    rng = random.Random(seed)
    for t in sorted(by_type):
        idxs = list(by_type[t])
        rng.shuffle(idxs)
        dev_ids = set(idxs[: dev_quota[t]])
        for idx in by_type[t]:
            questions[idx]["split"] = "dev" if idx in dev_ids else "test"

    return questions


def save_questions(questions: List[Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(questions, f, indent=2, ensure_ascii=False)
        f.write("\n")


def print_summary(questions: List[Dict[str, Any]]) -> None:
    n = len(questions)
    dev = [q for q in questions if q["split"] == "dev"]
    test = [q for q in questions if q["split"] == "test"]
    print(f"Total: {n}  dev: {len(dev)} ({len(dev) / n:.0%})  "
          f"test: {len(test)} ({len(test) / n:.0%})")
    for eval_type in VALID_TYPES:
        typed = [q for q in questions if q["type"] == eval_type]
        if not typed:
            continue
        t_dev = sum(1 for q in typed if q["split"] == "dev")
        print(f"  {eval_type:8s} {len(typed):3d} -> dev {t_dev}, test {len(typed) - t_dev}")
    if dev:
        print(f"dev ids: {', '.join(q['id'] for q in dev)}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Assign dev/test splits to questions.json "
                    "(stratified by type, fixed seed)"
    )
    parser.add_argument("--questions", default=None, help="Path to questions.json")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED,
                        help=f"Random seed (default: {DEFAULT_SEED})")
    parser.add_argument("--dev-fraction", type=float, default=DEFAULT_DEV_FRACTION,
                        help=f"Target dev fraction (default: {DEFAULT_DEV_FRACTION})")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print the summary without writing the file")
    args = parser.parse_args()

    questions_path = (
        Path(args.questions) if args.questions else ROOT / "data" / "eval" / "questions.json"
    )

    questions = load_questions(questions_path, require_split=False)
    if not questions:
        raise SystemExit(f"No questions in {questions_path}")

    assign_stratified_splits(questions, dev_fraction=args.dev_fraction, seed=args.seed)

    print(f"seed={args.seed} dev_fraction={args.dev_fraction}")
    print_summary(questions)

    if args.dry_run:
        print(f"Dry run, not written: {questions_path}")
    else:
        save_questions(questions, questions_path)
        print(f"Wrote {len(questions)} questions to {questions_path}")


if __name__ == "__main__":
    main()
