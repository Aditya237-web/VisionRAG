# VisionRAG evaluation harness (retrieval only)
# Never imports the generator and never calls the Ollama API.
from __future__ import annotations

import argparse
import gc
import json
import math
import random
import time
import logging
import torch
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from src.page_ids import build_page_id

VALID_TYPES: Tuple[str, ...] = ("text", "table", "chart", "scanned")
VALID_SPLITS: Tuple[str, ...] = ("dev", "test")
REQUIRED_FIELDS: Tuple[str, ...] = ("id", "question", "doc", "gold_page", "type")


def page_id_for(question: Dict[str, Any]) -> str:
    """Build the retriever page_id from doc + gold_page fields."""
    return build_page_id(question["doc"], int(question["gold_page"]))


def load_questions(path: str | Path, require_split: bool = True) -> List[Dict[str, Any]]:
    """Load and validate the eval question set.

    Splits are assigned in one pass after all questions are written
    (scripts/assign_splits.py), so helper tools call this with
    require_split=False before that has happened.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Questions file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        questions = json.load(f)
    if not isinstance(questions, list):
        raise ValueError(f"{path} must contain a JSON array of questions")

    seen_ids = set()
    for i, q in enumerate(questions):
        if not isinstance(q, dict):
            raise ValueError(f"Question #{i} is not an object")
        missing = [k for k in REQUIRED_FIELDS if k not in q]
        if missing:
            raise ValueError(f"Question #{i} ({q.get('id', '?')}): missing fields {missing}")
        if q["type"] not in VALID_TYPES:
            raise ValueError(
                f"Question {q['id']}: invalid type {q['type']!r}, expected one of {VALID_TYPES}"
            )
        if require_split and "split" not in q:
            raise ValueError(
                f"Question {q['id']}: no split assigned yet; "
                "run scripts/assign_splits.py first"
            )
        if "split" in q and q["split"] not in VALID_SPLITS:
            raise ValueError(
                f"Question {q['id']}: invalid split {q['split']!r}, expected one of {VALID_SPLITS}"
            )
        if q["id"] in seen_ids:
            raise ValueError(f"Duplicate question id: {q['id']}")
        seen_ids.add(q["id"])
    return questions


def topk_hit(ranked_page_ids: Sequence[str], gold_page_id: str, k: int) -> bool:
    """True if the gold page appears in the top-k ranked page ids."""
    return gold_page_id in list(ranked_page_ids)[:k]


def reciprocal_rank(ranked_page_ids: Sequence[str], gold_page_id: str) -> float:
    """1/rank of the gold page, 0.0 if it was not retrieved."""
    for i, page_id in enumerate(ranked_page_ids):
        if page_id == gold_page_id:
            return 1.0 / (i + 1)
    return 0.0


def wilson_ci(p: float, n: int, z: float = 1.96) -> Tuple[float, float]:
    """Wilson score interval for a binomial proportion.

    Args:
        p: Observed proportion (0 <= p <= 1)
        n: Number of trials (n > 0)
        z: Z-score for confidence level (default 1.96 for 95%)

    Returns:
        Tuple of (lower_bound, upper_bound)
    """
    if n == 0:
        return (0.0, 0.0)
    denom = 1 + z * z / n
    centre = p + z * z / (2 * n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    lower = (centre - half) / denom
    upper = (centre + half) / denom
    return (max(0.0, lower), min(1.0, upper))


def aggregate(records: Sequence[Dict[str, Any]]) -> Dict[str, float]:
    """Aggregate per-question records into top-1/3/5 accuracy and MRR."""
    n = len(records)
    if n == 0:
        return {"n": 0, "top1": 0.0, "top3": 0.0, "top5": 0.0, "mrr": 0.0}
    return {
        "n": n,
        "top1": sum(1 for r in records if topk_hit(r["retrieved"], r["gold"], 1)) / n,
        "top3": sum(1 for r in records if topk_hit(r["retrieved"], r["gold"], 3)) / n,
        "top5": sum(1 for r in records if topk_hit(r["retrieved"], r["gold"], 5)) / n,
        "mrr": sum(r["rr"] for r in records) / n,
    }


def bootstrap_ci(records: Sequence[Dict[str, Any]], metric: str, n_resamples: int = 1000, confidence: float = 0.95) -> Tuple[float, float]:
    """Compute 95% bootstrap confidence interval for a metric.

    Args:
        records: Per-question records from evaluate_retriever
        metric: One of "top1", "top3", "mrr"
        n_resamples: Number of bootstrap resamples (default 1000)
        confidence: Confidence level (default 0.95)

    Returns:
        Tuple of (lower_bound, upper_bound)
    """
    n = len(records)
    if n == 0:
        return (0.0, 0.0)

    # Extract per-question binary outcomes for the metric
    if metric == "top1":
        outcomes = [1.0 if topk_hit(r["retrieved"], r["gold"], 1) else 0.0 for r in records]
    elif metric == "top3":
        outcomes = [1.0 if topk_hit(r["retrieved"], r["gold"], 3) else 0.0 for r in records]
    elif metric == "top5":
        outcomes = [1.0 if topk_hit(r["retrieved"], r["gold"], 5) else 0.0 for r in records]
    elif metric == "mrr":
        outcomes = [r["rr"] for r in records]
    else:
        raise ValueError(f"Unknown metric: {metric}")

    point_estimate = sum(outcomes) / n
    bootstrap_estimates = []

    rng = random.Random(42)  # Fixed seed for reproducibility
    for _ in range(n_resamples):
        # Resample with replacement
        sample = rng.choices(outcomes, k=n)
        bootstrap_estimates.append(sum(sample) / n)

    alpha = (1 - confidence) / 2
    lower = sorted(bootstrap_estimates)[int(alpha * n_resamples)]
    upper = sorted(bootstrap_estimates)[int((1 - alpha) * n_resamples)]

    return (lower, upper)


def paired_comparison(records_a: Sequence[Dict[str, Any]], records_b: Sequence[Dict[str, Any]], k: int = 3) -> Dict[str, Any]:
    """Paired comparison of two retrievers on the same questions.

    Win/loss/tie is defined by whether the gold page is in the top-k results:
    - A wins: A has gold in top-k, B does not
    - B wins: B has gold in top-k, A does not
    - Tie: Both have gold in top-k, or both don't

    Also computes a sign-test p-value for the null hypothesis that both retrievers
    are equally likely to win (binomial test, two-sided).

    Args:
        records_a: Per-question records for retriever A
        records_b: Per-question records for retriever B
        k: Cutoff for "top-k" (default 3)

    Returns:
        Dict with keys: "win", "loss", "tie", "p_value", "n_discordant"
        From perspective of A vs B.
    """
    # Build maps for quick lookup
    map_b = {r["id"]: r for r in records_b}

    win = loss = tie = 0
    for r_a in records_a:
        r_b = map_b.get(r_a["id"])
        if r_b is None:
            continue
        # Check if gold is in top-k for each retriever
        a_topk = r_a.get("rank") is not None and r_a["rank"] <= k
        b_topk = r_b.get("rank") is not None and r_b["rank"] <= k

        if a_topk and not b_topk:
            win += 1
        elif b_topk and not a_topk:
            loss += 1
        else:
            tie += 1

    n_discordant = win + loss
    # Sign test: under H0 (equal performance), wins ~ Binomial(n_discordant, 0.5)
    if n_discordant > 0:
        from scipy.stats import binomtest
        # Two-sided exact binomial test
        p_value = binomtest(win, n_discordant, 0.5, alternative="two-sided").pvalue
    else:
        p_value = 1.0

    return {
        "win": win,
        "loss": loss,
        "tie": tie,
        "p_value": p_value,
        "n_discordant": n_discordant,
    }


def evaluate_retriever(
    retriever: Any,
    questions: Sequence[Dict[str, Any]],
    k_max: int = 5,
    verbose: bool = False,
) -> Dict[str, Any]:
    """Run a retriever over questions and compute metrics.

    The retriever only needs a retrieve(query, k) -> [(page_id, score)] method.
    Measures per-query retrieval latency; index/model load time is excluded.
    """
    if k_max < 5:
        raise ValueError("k_max must be >= 5 to report top-5 accuracy")

    records: List[Dict[str, Any]] = []
    for i, q in enumerate(questions):
        gold = page_id_for(q)
        start = time.perf_counter()
        hits = retriever.retrieve(q["question"], k_max)
        elapsed = time.perf_counter() - start

        ranked = [page_id for page_id, _score in hits]
        rank: Optional[int] = None
        for j, page_id in enumerate(ranked):
            if page_id == gold:
                rank = j + 1
                break

        record = {
            "id": q["id"],
            "question": q["question"],
            "type": q["type"],
            "gold": gold,
            "retrieved": ranked,
            "rank": rank,
            "rr": reciprocal_rank(ranked, gold),
            "latency_s": elapsed,
        }
        records.append(record)
        if verbose:
            status = f"rank={rank}" if rank is not None else "MISS"
            print(f"  [{i + 1}/{len(questions)}] {q['id']} {status} ({elapsed:.2f}s)")

    overall = aggregate(records)
    overall["avg_latency_s"] = (
        sum(r["latency_s"] for r in records) / len(records) if records else 0.0
    )

    # Add bootstrap CIs for key metrics
    overall["top1_ci"] = bootstrap_ci(records, "top1")
    overall["top3_ci"] = bootstrap_ci(records, "top3")
    overall["top5_ci"] = bootstrap_ci(records, "top5")
    overall["mrr_ci"] = bootstrap_ci(records, "mrr")

    by_type: Dict[str, Dict[str, float]] = {}
    for eval_type in VALID_TYPES:
        subset = [r for r in records if r["type"] == eval_type]
        if subset:
            by_type[eval_type] = aggregate(subset)

    return {"overall": overall, "by_type": by_type, "per_question": records}


def index_stats(kind: str, config: Dict[str, Any]) -> Dict[str, Any]:
    """Index build time (from build_stats.json) and size on disk in MB."""
    index_root = Path(config["paths"]["index_dir"])
    if kind == "visual":
        index_dir = index_root
        files = [p for p in index_dir.iterdir() if p.is_file()]
    elif kind in ("visual_int8", "visual_mean_pool_8", "visual_mean_pool_16", "visual_fp16_local", "visual_mean_pool_64", "visual_mean_pool_256"):
        index_dir = index_root / kind
        files = [p for p in index_dir.rglob("*") if p.is_file()]
    elif kind == "ocr_text":
        index_dir = index_root / "ocr_text"
        files = [p for p in index_dir.rglob("*") if p.is_file()]
    elif kind == "hybrid":
        # Hybrid uses visual + ocr_text indexes
        index_dir_visual = index_root
        files_visual = [p for p in index_dir_visual.iterdir() if p.is_file()]
        index_dir_ocr = index_root / "ocr_text"
        files_ocr = [p for p in index_dir_ocr.rglob("*") if p.is_file()]
        files = files_visual + files_ocr

        # For file listing, use visual index dir
        index_dir = index_dir_visual

        # For build stats, sum visual + ocr_text build times
        build_time_visual = None
        stats_path_visual = index_dir_visual / "build_stats.json"
        if stats_path_visual.exists():
            with open(stats_path_visual, "r", encoding="utf-8") as f:
                build_time_visual = json.load(f).get("elapsed_seconds")
        
        build_time_ocr = None
        stats_path_ocr = index_dir_ocr / "build_stats.json"
        if stats_path_ocr.exists():
            with open(stats_path_ocr, "r", encoding="utf-8") as f:
                build_time_ocr = json.load(f).get("elapsed_seconds")
        
        build_time = (build_time_visual or 0) + (build_time_ocr or 0)

        return {
            "build_time_s": build_time,
            "index_size_mb": sum(p.stat().st_size for p in files) / 1e6,
            "n_files": len(files),
            "build_time_visual_s": build_time_visual,
            "build_time_ocr_s": build_time_ocr,
        }
    else:
        index_dir = index_root / "text"
        files = [p for p in index_dir.rglob("*") if p.is_file()]

    stats_path = index_dir / "build_stats.json"
    build_time = None
    if stats_path.exists():
        with open(stats_path, "r", encoding="utf-8") as f:
            build_time = json.load(f).get("elapsed_seconds")

    return {
        "build_time_s": build_time,
        "index_size_mb": sum(p.stat().st_size for p in files) / 1e6,
        "n_files": len(files),
    }


class SharedVisualModel:
    """Singleton holder for shared ColSmol model and processor."""

    _instance: Optional["SharedVisualModel"] = None
    _model: Any = None
    _processor: Any = None
    _device: str = "cpu"
    _torch_dtype: Any = torch.float32

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if self._model is not None:
            return

    def load(self, model_name: str, device: str, torch_dtype: Any) -> None:
        """Load model and processor once."""
        if self._model is not None:
            return
        print(f"Loading shared ColSmol model: {model_name} on {device}")
        from colpali_engine.models import ColIdefics3, ColIdefics3Processor
        self._processor = ColIdefics3Processor.from_pretrained(model_name)
        self._model = ColIdefics3.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
        ).to(device)
        self._model.eval()
        self._device = device
        self._torch_dtype = torch_dtype
        print("Shared model loaded successfully")

    def get_model(self):
        return self._model

    def get_processor(self):
        return self._processor

    def get_device(self):
        return self._device

    def get_dtype(self):
        return self._torch_dtype

    def free(self) -> None:
        if self._model is not None:
            del self._model
        if self._processor is not None:
            del self._processor
        self._model = None
        self._processor = None
        gc.collect()
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def get_retriever(kind: str, config: Dict[str, Any], shared_model: Optional[SharedVisualModel] = None) -> Any:
    """Construct and load a retriever by name (heavy imports kept lazy)."""
    if kind == "visual":
        from src.retriever_visual import VisualRetriever

        retriever = VisualRetriever(config)
        retriever.load_index(Path(config["paths"]["index_dir"]))
        # Use shared model
        if shared_model:
            retriever.model = shared_model.get_model()
            retriever.processor = shared_model.get_processor()
            retriever.device = shared_model.get_device()
            retriever.torch_dtype = shared_model.get_dtype()
        else:
            retriever.load_model()
        return retriever

    if kind == "visual_int8":
        from src.build_compressed_visual import CompressedVisualRetriever

        retriever = CompressedVisualRetriever(config, compression_mode="int8")
        retriever.load_index(Path(config["paths"]["index_dir"]) / "visual_int8")
        # Pre-dequantize all pages at load time
        retriever._dequantize_all()
        return retriever

    if kind == "visual_mean_pool_64":
        from src.build_compressed_visual import CompressedVisualRetriever

        retriever = CompressedVisualRetriever(config, compression_mode="mean_pool_64")
        retriever.load_index(Path(config["paths"]["index_dir"]) / "visual_mean_pool_64")
        return retriever

    if kind == "visual_mean_pool_256":
        from src.build_compressed_visual import CompressedVisualRetriever

        retriever = CompressedVisualRetriever(config, compression_mode="mean_pool_256")
        retriever.load_index(Path(config["paths"]["index_dir"]) / "visual_mean_pool_256")
        return retriever

    if kind == "visual_mean_pool_8":
        from src.build_compressed_visual import CompressedVisualRetriever

        retriever = CompressedVisualRetriever(config, compression_mode="mean_pool_8")
        retriever.load_index(Path(config["paths"]["index_dir"]) / "visual_mean_pool_8")
        return retriever

    if kind == "visual_mean_pool_16":
        from src.build_compressed_visual import CompressedVisualRetriever

        retriever = CompressedVisualRetriever(config, compression_mode="mean_pool_16")
        retriever.load_index(Path(config["paths"]["index_dir"]) / "visual_mean_pool_16")
        return retriever

    if kind == "visual_fp16_local":
        from src.retriever_visual import VisualRetriever

        local_config = config.copy()
        if "retriever_visual" not in local_config:
            local_config["retriever_visual"] = {}
        local_config["retriever_visual"]["use_local_patches_only"] = True
        retriever = VisualRetriever(local_config)
        retriever.load_index(Path(config["paths"]["index_dir"]) / "visual_fp16_local")
        if shared_model:
            retriever.model = shared_model.get_model()
            retriever.processor = shared_model.get_processor()
            retriever.device = shared_model.get_device()
            retriever.torch_dtype = shared_model.get_dtype()
        else:
            retriever.load_model()
        return retriever

    if kind == "text":
        from src.retriever_text import TextRetriever

        retriever = TextRetriever(config)
        retriever.load_index(Path(config["paths"]["index_dir"]) / "text")
        return retriever

    if kind == "ocr_text":
        from src.retriever_ocr_text import OCRTextRetriever

        retriever = OCRTextRetriever(config)
        retriever.load_index(Path(config["paths"]["index_dir"]) / "ocr_text")
        return retriever

    if kind == "hybrid":
        from src.retriever_hybrid import HybridRetriever

        return HybridRetriever(config)

    raise ValueError(
        f"Unknown retriever {kind!r}, expected 'visual', 'visual_int8', "
        "'visual_mean_pool_64', 'visual_mean_pool_256', 'visual_fp16_local', 'text', 'ocr_text', or 'hybrid'"
    )


def _num(value: Any, fmt: str = ".3f") -> str:
    if isinstance(value, (int, float)):
        return format(value, fmt)
    return "n/a"


def _pct(hits: int, n: int) -> str:
    """Percentage string for hits/n, e.g. 19/32 -> '59.4%', 32/32 -> '100%'."""
    pct = hits / n * 100
    if hits == n:
        return "100%"
    return format(pct, ".1f") + "%"


def _fmt_p(p: float) -> str:
    """Format p-value: <0.001 for values below 0.001, else 3 decimal places."""
    if p < 0.001:
        return "<0.001"
    return format(p, ".3f")


def format_markdown(report: Dict[str, Any]) -> str:
    """Render the evaluation report as a markdown table string."""
    order: List[str] = report.get("retriever_order") or list(report.get("retrievers", {}))
    retrievers = report.get("retrievers", {})

    lines: List[str] = ["# VisionRAG Retrieval Evaluation", ""]
    lines.append(f"- Questions: `{report['questions_file']}`")
    lines.append(
        f"- Split: **{report['split']}** ({report['n_eval']} questions evaluated)"
    )
    lines.append(
        f"- Splits in file: {report['n_dev']} dev / {report['n_test']} test"
    )
    lines.append(
        f"- Metrics: top-1/3/5 page accuracy, MRR@{report['k_max']} "
        "(never calls Ollama)"
    )

    lines.append("")
    lines.append(f"## Overall ({report['split']} split)")
    lines.append("")
    lines.append(
        "| Retriever | N | Top-1 (95% CI) | Top-3 (95% CI) | Top-5 (95% CI) | MRR (95% CI) | "
        "Avg query latency (s) | Load time (s) | Index build time (s) | Index size (MB) |"
    )
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for kind in order:
        overall = retrievers.get(kind, {}).get("overall", {})
        top1 = overall.get("top1", 0)
        top1_ci = overall.get("top1_ci", (0, 0))
        top3 = overall.get("top3", 0)
        top3_ci = overall.get("top3_ci", (0, 0))
        top5 = overall.get("top5", 0)
        top5_ci = overall.get("top5_ci", (0, 0))
        mrr = overall.get("mrr", 0)
        mrr_ci = overall.get("mrr_ci", (0, 0))
        lines.append(
            "| {} | {} | {:.3f} ({:.3f}, {:.3f}) | {:.3f} ({:.3f}, {:.3f}) | {:.3f} ({:.3f}, {:.3f}) | {:.3f} ({:.3f}, {:.3f}) | {} | {} | {} | {} |".format(
                kind,
                overall.get("n", 0),
                top1, top1_ci[0], top1_ci[1],
                top3, top3_ci[0], top3_ci[1],
                top5, top5_ci[0], top5_ci[1],
                mrr, mrr_ci[0], mrr_ci[1],
                _num(overall.get("avg_latency_s")),
                _num(overall.get("load_time_s"), ".1f"),
                _num(overall.get("build_time_s"), ".1f"),
                _num(overall.get("index_size_mb"), ".2f"),
            )
        )

    lines.append("")
    hybrid_overall = retrievers.get("hybrid", {}).get("overall", {})
    bt_visual = hybrid_overall.get("build_time_visual_s")
    bt_ocr = hybrid_overall.get("build_time_ocr_s")
    if bt_visual is not None and bt_ocr is not None:
        bt_sum = bt_visual + bt_ocr
        lines.append(
            f"*Overall CIs: bootstrap (1000 resamples, seed 42). By-type CIs: Wilson score interval. Hybrid build time = visual ({bt_visual:.1f} s) + OCR+Text ({bt_ocr:.1f} s) = {bt_sum:.1f} s (computed from unrounded values); derived compressed indexes (int8, MeanPool, local) exclude the fp16 build time.*"
        )
    else:
        lines.append("*Overall CIs: bootstrap (1000 resamples, seed 42). By-type CIs: Wilson score interval. Hybrid build time = visual + OCR+Text; derived compressed indexes (int8, MeanPool, local) exclude the fp16 build time.*")

    types_present = [
        t for t in VALID_TYPES
        if any(t in retrievers.get(k, {}).get("by_type", {}) for k in order)
    ]
    if types_present:
        lines.append("")
        lines.append(f"## By type ({report['split']} split) — with 95% Wilson score CIs")
        lines.append("")
        lines.append("| Retriever | Type | N | Top-1 (95% CI) | Top-3 (95% CI) | Top-5 (95% CI) | MRR |")
        lines.append("| --- | --- | ---: | ---: | ---: | ---: | ---: |")
        for kind in order:
            by_type = retrievers.get(kind, {}).get("by_type", {})
            for eval_type in types_present:
                stats = by_type.get(eval_type)
                if not stats:
                    continue
                n = stats.get("n", 0)
                top1 = stats.get("top1", 0)
                top3 = stats.get("top3", 0)
                top5 = stats.get("top5", 0)
                mrr = stats.get("mrr", 0)
                # Compute Wilson score CIs for this type
                if n > 0:
                    top1_ci = wilson_ci(top1, n)
                    top3_ci = wilson_ci(top3, n)
                    top5_ci = wilson_ci(top5, n)
                else:
                    top1_ci = top3_ci = top5_ci = (0, 0)
                lines.append(
                    "| {} | {} | {} | {:.3f} ({:.3f}, {:.3f}) | {:.3f} ({:.3f}, {:.3f}) | {:.3f} ({:.3f}, {:.3f}) | {:.3f} |".format(
                        kind,
                        eval_type,
                        n,
                        top1, top1_ci[0], top1_ci[1],
                        top3, top3_ci[0], top3_ci[1],
                        top5, top5_ci[0], top5_ci[1],
                        mrr,
                    )
                )

    # Non-scanned questions (test split minus scanned)
    non_scanned = report.get("non_scanned")
    if non_scanned and non_scanned.get("n"):
        ns_n = non_scanned["n"]
        lines.append("")
        lines.append(f"## Non-scanned questions (n={ns_n})")
        lines.append("")
        lines.append("| Retriever | Top-1 | Top-3 | Top-5 |")
        lines.append("| --- | ---: | ---: | ---: |")
        for kind in order:
            stats = non_scanned.get("retrievers", {}).get(kind)
            if not stats:
                continue
            lines.append(
                "| {} | {}/{} ({}) | {}/{} ({}) | {}/{} ({}) |".format(
                    kind,
                    stats["top1"], ns_n, _pct(stats["top1"], ns_n),
                    stats["top3"], ns_n, _pct(stats["top3"], ns_n),
                    stats["top5"], ns_n, _pct(stats["top5"], ns_n),
                )
            )

    # Paired comparisons vs visual (top-1, top-3, top-5)
    if "visual" in retrievers:
        for k in [1, 3, 5]:
            lines.append("")
            lines.append(f"## Paired Comparisons vs Visual (Top-{k} Rank)")
            lines.append("")
            lines.append("| Retriever | Visual Win | Visual Loss | Tie | Sign-test p |")
            lines.append("| --- | ---: | ---: | ---: | ---: |")
            for kind in order:
                if kind == "visual":
                    continue
                vs_visual = retrievers.get(kind, {}).get(f"vs_visual_top{k}")
                if vs_visual:
                    p = vs_visual.get("p_value", 1.0)
                    lines.append(
                        "| {} | {} | {} | {} | {} |".format(
                            kind, vs_visual.get("win", 0), vs_visual.get("loss", 0), vs_visual.get("tie", 0), _fmt_p(p)
                        )
                    )

        # By-type paired comparisons for scanned questions
        lines.append("")
        lines.append("## Paired Comparisons vs Visual on Scanned Questions (n=16)")
        lines.append("")
        for k in [1, 3, 5]:
            lines.append("")
            lines.append(f"### Top-{k}")
            lines.append("")
            lines.append("| Challenger | Visual Win | Visual Loss | Tie | Sign-test p |")
            lines.append("| --- | ---: | ---: | ---: | ---: |")
            for kind in ["ocr_text", "text"]:
                vs_visual = retrievers.get(kind, {}).get(f"vs_visual_scanned_top{k}")
                if vs_visual:
                    p = vs_visual.get("p_value", 1.0)
                    lines.append(
                        "| {} | {} | {} | {} | {} |".format(
                            kind, vs_visual.get("win", 0), vs_visual.get("loss", 0), vs_visual.get("tie", 0), _fmt_p(p)
                        )
                    )

    per_question = retrievers.get(order[0], {}).get("per_question", []) if order else []
    if per_question:
        lines.append("")
        lines.append(f"## Per-question gold rank ({report['split']} split)")
        lines.append("")
        lines.append("| ID | Type | Gold | " + " | ".join(order) + " |")
        lines.append("| --- | --- | --- | " + " | ".join(["---:"] * len(order)) + " |")
        maps = {
            kind: {r["id"]: r for r in retrievers.get(kind, {}).get("per_question", [])}
            for kind in order
        }
        for record in per_question:
            cells = []
            for kind in order:
                found = maps[kind].get(record["id"])
                rank = found.get("rank") if found else None
                cells.append(str(rank) if rank is not None else "MISS")
            lines.append(
                f"| {record['id']} | {record['type']} | `{record['gold']}` | "
                + " | ".join(cells)
                + " |"
            )

    lines.append("")
    return "\n".join(lines)


def _write_log(log_path: Path, message: str) -> None:
    """Append a message to the log file."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(message + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="VisionRAG retrieval evaluation")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--questions", default=None, help="Path to questions.json")
    parser.add_argument("--results", default=None, help="Path to results.json")
    parser.add_argument("--markdown", default=None, help="Path to results.md")
    parser.add_argument("--log", default=None, help="Path to log file")
    parser.add_argument(
        "--retrievers", default="visual,text,ocr_text,hybrid",
        help="Comma-separated retrievers to evaluate (visual,visual_int8,visual_mean_pool_64,visual_mean_pool_256,text,ocr_text,hybrid)",
    )
    parser.add_argument("--k", type=int, default=5, help="Cutoff for retrieval")
    parser.add_argument(
        "--split", default="test", choices=["test", "dev", "all"],
        help="Which split to report on (default: test)",
    )
    args = parser.parse_args()

    from src.utils import load_config

    config = load_config(args.config)
    eval_dir = Path(config["paths"]["eval_dir"])
    questions_path = (
        Path(args.questions)
        if args.questions
        else eval_dir / config["eval"]["questions_file"]
    )
    results_path = (
        Path(args.results) if args.results else eval_dir / config["eval"]["results_file"]
    )
    markdown_path = Path(args.markdown) if args.markdown else eval_dir / "results.md"
    log_path = Path(args.log) if args.log else eval_dir / "evaluation.log"

    # Clear log file
    if log_path.exists():
        log_path.unlink()

    _write_log(log_path, f"=== VisionRAG Evaluation Started at {datetime.now(timezone.utc).isoformat()} ===")
    _write_log(log_path, f"Config: {args.config}")
    _write_log(log_path, f"Questions: {questions_path}")
    _write_log(log_path, f"Split: {args.split}")
    _write_log(log_path, f"Retrievers: {args.retrievers}")

    config = load_config(args.config)
    eval_dir = Path(config["paths"]["eval_dir"])
    questions_path = (
        Path(args.questions)
        if args.questions
        else eval_dir / config["eval"]["questions_file"]
    )
    results_path = (
        Path(args.results) if args.results else eval_dir / config["eval"]["results_file"]
    )
    markdown_path = Path(args.markdown) if args.markdown else eval_dir / "results.md"

    questions = load_questions(questions_path)
    n_dev = sum(1 for q in questions if q["split"] == "dev")
    n_test = sum(1 for q in questions if q["split"] == "test")

    if args.split == "all":
        selected = list(questions)
    else:
        selected = [q for q in questions if q["split"] == args.split]
    if not selected:
        raise SystemExit(f"No questions in split {args.split!r} ({questions_path})")

    _write_log(log_path, f"Loaded {len(questions)} questions ({n_dev} dev, {n_test} test)")
    _write_log(log_path, f"Evaluating {len(selected)} questions from '{args.split}' split")

    report: Dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "questions_file": str(questions_path),
        "split": args.split,
        "n_dev": n_dev,
        "n_test": n_test,
        "n_eval": len(selected),
        "k_max": args.k,
        "retriever_order": [],
        "retrievers": {},
    }

    names = [n.strip() for n in args.retrievers.split(",") if n.strip()]

    # Initialize shared model for visual-family retrievers
    visual_kinds = [k for k in names if k.startswith("visual")]
    shared_model = None
    if visual_kinds:
        vr_cfg = config.get("retriever_visual", {})
        model_name = vr_cfg.get("model_name", "vidore/colSmol-256M")
        device = vr_cfg.get("device", "auto")
        device = "cuda" if (device == "auto" and torch.cuda.is_available()) else "cpu"
        torch_dtype = torch.float32 if device == "cpu" else torch.bfloat16

        shared_model = SharedVisualModel()
        shared_model.load("vidore/colSmol-256M", device, torch.float32 if device == "cpu" else torch.bfloat16)

    report: Dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "questions_file": str(questions_path),
        "split": args.split,
        "n_dev": n_dev,
        "n_test": n_test,
        "n_eval": len(selected),
        "k_max": args.k,
        "retriever_order": [],
        "retrievers": {},
    }

    names = [n.strip() for n in args.retrievers.split(",") if n.strip()]
    for kind in names:
        _write_log(log_path, f"\n=== Evaluating retriever: {kind} ({len(selected)} questions) ===")
        print(f"\n=== Evaluating retriever: {kind} ({len(selected)} questions) ===")
        load_start = time.perf_counter()
        retriever = get_retriever(kind, config, shared_model)
        load_time = time.perf_counter() - load_start

        evaluation = evaluate_retriever(retriever, selected, k_max=args.k, verbose=True)
        stats = index_stats(kind, config)
        evaluation["overall"]["load_time_s"] = round(load_time, 3)
        evaluation["overall"]["build_time_s"] = stats["build_time_s"]
        evaluation["overall"]["index_size_mb"] = stats["index_size_mb"]
        if kind == "hybrid":
            evaluation["overall"]["build_time_visual_s"] = stats.get("build_time_visual_s")
            evaluation["overall"]["build_time_ocr_s"] = stats.get("build_time_ocr_s")

        report["retrievers"][kind] = evaluation
        report["retriever_order"].append(kind)

        overall = evaluation["overall"]
        msg = (
            f"  top1={overall['top1']:.3f} top3={overall['top3']:.3f} "
            f"top5={overall['top5']:.3f} mrr={overall['mrr']:.3f} "
            f"avg_latency={overall['avg_latency_s']:.3f}s "
            f"load={load_time:.1f}s"
        )
        print(msg)
        _write_log(log_path, msg)

        del retriever
        gc.collect()

    # Free shared model
    if shared_model:
        shared_model.free()

    # Paired comparisons between visual and other retrievers (top-1, top-3, top-5)
    visual_records = report["retrievers"].get("visual", {}).get("per_question", [])
    if visual_records:
        for kind in names:
            if kind == "visual":
                continue
            other_records = report["retrievers"].get(kind, {}).get("per_question", [])
            if other_records:
                for k in [1, 3, 5]:
                    comparison = paired_comparison(visual_records, other_records, k)
                    report["retrievers"][kind][f"vs_visual_top{k}"] = comparison
                    report["retrievers"]["visual"][f"vs_{kind}_top{k}"] = {
                        "win": comparison["loss"],
                        "loss": comparison["win"],
                        "tie": comparison["tie"],
                        "p_value": comparison["p_value"],
                    }

        # By-type paired comparisons for scanned questions
        scanned_ids = {r["id"] for r in visual_records if r["type"] == "scanned"}
        visual_scanned = [r for r in visual_records if r["id"] in scanned_ids]
        for kind in ["ocr_text", "text"]:
            other_records = report["retrievers"].get(kind, {}).get("per_question", [])
            other_scanned = [r for r in other_records if r["id"] in scanned_ids]
            if other_scanned:
                for k in [1, 3, 5]:
                    comparison = paired_comparison(visual_scanned, other_scanned, k)
                    report["retrievers"][kind][f"vs_visual_scanned_top{k}"] = comparison
                    report["retrievers"]["visual"][f"vs_{kind}_scanned_top{k}"] = {
                        "win": comparison["loss"],
                        "loss": comparison["win"],
                        "tie": comparison["tie"],
                        "p_value": comparison["p_value"],
                    }

    # Non-scanned questions (all evaluated questions except type == "scanned")
    ns_kinds = [k for k in ("visual", "text", "ocr_text", "hybrid") if k in report["retrievers"]]
    if ns_kinds:
        ns_n = sum(
            1
            for r in report["retrievers"][ns_kinds[0]]["per_question"]
            if r["type"] != "scanned"
        )
        if ns_n:
            ns_report: Dict[str, Any] = {"n": ns_n, "retrievers": {}}
            for kind in ns_kinds:
                records = [
                    r
                    for r in report["retrievers"][kind]["per_question"]
                    if r["type"] != "scanned"
                ]
                ns_report["retrievers"][kind] = {
                    "top1": sum(1 for r in records if topk_hit(r["retrieved"], r["gold"], 1)),
                    "top3": sum(1 for r in records if topk_hit(r["retrieved"], r["gold"], 3)),
                    "top5": sum(1 for r in records if topk_hit(r["retrieved"], r["gold"], 5)),
                }
            report["non_scanned"] = ns_report

    markdown = format_markdown(report)

    results_path.parent.mkdir(parents=True, exist_ok=True)
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    with open(markdown_path, "w", encoding="utf-8") as f:
        f.write(markdown)

    print("\n" + markdown)
    print(f"Saved: {results_path}")
    print(f"Saved: {markdown_path}")
    _write_log(log_path, f"Saved: {results_path}")
    _write_log(log_path, f"Saved: {markdown_path}")
    _write_log(log_path, f"=== Evaluation Complete at {datetime.now(timezone.utc).isoformat()} ===")


if __name__ == "__main__":
    main()