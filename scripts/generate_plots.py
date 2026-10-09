# VisionRAG Plot Generator
# Generates evaluation plots and updates results.md
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def load_results(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def plot_accuracy_by_type(results: Dict[str, Any], output_dir: Path) -> None:
    """Bar chart: top-1/3/5 accuracy by question type for each retriever."""
    retrievers = results["retriever_order"]
    types = [t for t in ["text", "table", "chart", "scanned"] 
             if any(t in results["retrievers"][r].get("by_type", {}) for r in retrievers)]
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    metrics = ["top1", "top3", "top5"]
    colors = {"visual": "#1f77b4", "visual_int8": "#aec7e8", "text": "#ff7f0e", 
              "ocr_text": "#2ca02c", "hybrid": "#d62728"}
    
    for idx, qtype in enumerate(types):
        ax = axes[idx]
        x = np.arange(len(retrievers))
        width = 0.25
        
        for m_idx, metric in enumerate(metrics):
            values = []
            for r in retrievers:
                stats = results["retrievers"][r]["by_type"].get(qtype, {})
                values.append(stats.get(metric, 0))
            
            offset = (m_idx - 1) * width
            bars = ax.bar(x + offset, values, width, label=metric.upper(), 
                         color=[colors.get(r, "#7f7f7f") for r in retrievers], alpha=0.8)
            
            for bar, val in zip(bars, values):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                       f"{val:.2f}", ha="center", va="bottom", fontsize=8)
        
        ax.set_xticks(x)
        ax.set_xticklabels(retrievers, rotation=15, ha="right")
        ax.set_ylabel("Accuracy")
        ax.set_title(f"{qtype.capitalize()} Questions (n={results['retrievers'][retrievers[0]]['by_type'][qtype]['n']})")
        ax.set_ylim(0, 1.1)
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.3)
    
    plt.suptitle("Retrieval Accuracy by Question Type (Test Split)", fontsize=14)
    plt.tight_layout()
    plt.savefig(output_dir / "accuracy_by_type.png", dpi=150, bbox_inches="tight")
    plt.close()


def plot_storage_vs_accuracy(results: Dict[str, Any], output_dir: Path) -> None:
    """Scatter plot: Index size (MB) vs Top-3 accuracy."""
    retrievers = results["retriever_order"]
    
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = {"visual": "#1f77b4", "visual_int8": "#aec7e8", "text": "#ff7f0e", 
              "ocr_text": "#2ca02c", "hybrid": "#d62728"}
    markers = {"visual": "o", "visual_int8": "s", "text": "^", "ocr_text": "D", "hybrid": "*"}
    
    for r in retrievers:
        overall = results["retrievers"][r]["overall"]
        size = overall.get("index_size_mb", 0)
        top3 = overall.get("top3", 0)
        
        if size > 0:  # Skip retrievers without size info
            ax.scatter(size, top3, s=150, label=r, 
                      color=colors.get(r, "#7f7f7f"), marker=markers.get(r, "o"),
                      edgecolor="black", linewidth=0.5, alpha=0.8, zorder=3)
            ax.annotate(f"  {r}\n  Top-3: {top3:.2f}", (size, top3),
                       xytext=(5, 5), textcoords="offset points", fontsize=8)
    
    ax.set_xlabel("Index Size (MB)")
    ax.set_ylabel("Top-3 Accuracy")
    ax.set_title("Storage vs Top-3 Accuracy (Test Split)")
    ax.set_xscale("log")
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_ylim(0, 1.05)
    
    plt.tight_layout()
    plt.savefig(output_dir / "storage_vs_accuracy.png", dpi=150, bbox_inches="tight")
    plt.close()


def plot_latency(results: Dict[str, Any], output_dir: Path) -> None:
    """Bar chart: Average query latency by retriever."""
    retrievers = results["retriever_order"]
    
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = {"visual": "#1f77b4", "visual_int8": "#aec7e8", "text": "#ff7f0e", 
              "ocr_text": "#2ca02c", "hybrid": "#d62728"}
    
    latencies = [results["retrievers"][r]["overall"].get("avg_latency_s", 0) 
                 for r in retrievers]
    colors_list = [colors.get(r, "#7f7f7f") for r in retrievers]
    
    bars = ax.bar(retrievers, latencies, color=colors_list, alpha=0.8, edgecolor="black", linewidth=0.5)
    
    for bar, val in zip(bars, latencies):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
               f"{val:.3f}s", ha="center", va="bottom", fontsize=10, fontweight="bold")
    
    ax.set_ylabel("Average Query Latency (s)")
    ax.set_title("Average Query Latency by Retriever (Test Split)")
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(0, max(latencies) * 1.2)
    
    plt.tight_layout()
    plt.savefig(output_dir / "latency.png", dpi=150, bbox_inches="tight")
    plt.close()


def plot_by_type_heatmap(results: Dict[str, Any], output_dir: Path) -> None:
    """Heatmap: Top-1 accuracy by retriever and question type."""
    retrievers = results["retriever_order"]
    types = [t for t in ["text", "table", "chart", "scanned"] 
             if any(t in results["retrievers"][r].get("by_type", {}) for r in retrievers)]
    
    data = []
    for r in retrievers:
        row = []
        for qtype in types:
            stats = results["retrievers"][r]["by_type"].get(qtype, {})
            row.append(stats.get("top1", 0))
        data.append(row)
    
    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(data, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    
    ax.set_xticks(range(len(types)))
    ax.set_xticklabels([t.capitalize() for t in types])
    ax.set_yticks(range(len(retrievers)))
    ax.set_yticklabels(retrievers)
    ax.set_title("Top-1 Accuracy by Retriever and Question Type")
    
    for i in range(len(retrievers)):
        for j in range(len(types)):
            text = ax.text(j, i, f"{data[i][j]:.2f}", ha="center", va="center",
                          color="white" if data[i][j] < 0.5 else "black", fontweight="bold")
    
    plt.colorbar(im, ax=ax, label="Top-1 Accuracy")
    plt.tight_layout()
    plt.savefig(output_dir / "heatmap_top1.png", dpi=150, bbox_inches="tight")
    plt.close()


def update_results_md(results: Dict[str, Any], output_path: Path) -> None:
    """Update results.md with tables and plot references."""
    retrievers = results["retriever_order"]
    
    lines = ["# VisionRAG Evaluation Results\n"]
    lines.append(f"- Questions: `{results['questions_file']}`")
    lines.append(f"- Split: **{results['split']}** ({results['n_eval']} questions evaluated)")
    lines.append(f"- Splits in file: {results['n_dev']} dev / {results['n_test']} test")
    lines.append(f"- Metrics: top-1/3/5 page accuracy, MRR@{results['k_max']} (never calls Ollama)\n")
    
    # Overall table
    lines.append("## Overall (Test Split)\n")
    lines.append("| Retriever | N | Top-1 | Top-3 | Top-5 | MRR | Avg Latency (s) | Build Time (s) | Index Size (MB) |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    
    for r in retrievers:
        o = results["retrievers"][r]["overall"]
        lines.append(
            f"| {r} | {o.get('n', 0)} | {o.get('top1', 0):.3f} | {o.get('top3', 0):.3f} | "
            f"{o.get('top5', 0):.3f} | {o.get('mrr', 0):.3f} | {o.get('avg_latency_s', 0):.3f} | "
            f"{o.get('build_time_s', 'n/a')} | {o.get('index_size_mb', 0):.2f} |"
        )
    
    # By type table
    types_present = [t for t in ["text", "table", "chart", "scanned"]
                     if any(t in results["retrievers"][r].get("by_type", {}) for r in results["retriever_order"])]
    
    if types_present:
        lines.append("\n## By Type (Test Split)\n")
        lines.append("| Retriever | Type | N | Top-1 | Top-3 | Top-5 | MRR |")
        lines.append("| --- | --- | ---: | ---: | ---: | ---: | ---: |")
        for r in retrievers:
            by_type = results["retrievers"][r].get("by_type", {})
            for qtype in types_present:
                stats = by_type.get(qtype)
                if not stats:
                    continue
                lines.append(
                    f"| {r} | {qtype} | {stats.get('n', 0)} | {stats.get('top1', 0):.3f} | "
                    f"{stats.get('top3', 0):.3f} | {stats.get('top5', 0):.3f} | {stats.get('mrr', 0):.3f} |"
                )
    
    # Plots
    lines.append("\n## Plots\n")
    lines.append("### Accuracy by Question Type")
    lines.append("![Accuracy by Type](accuracy_by_type.png)\n")
    
    lines.append("### Storage vs Top-3 Accuracy")
    lines.append("![Storage vs Accuracy](storage_vs_accuracy.png)\n")
    
    lines.append("### Query Latency")
    lines.append("![Latency](latency.png)\n")
    
    lines.append("### Top-1 Accuracy Heatmap")
    lines.append("![Heatmap](heatmap_top1.png)\n")
    
    # Per-question table
    lines.append("## Per-Question Gold Rank (Test Split)")
    lines.append("| ID | Type | Gold | " + " | ".join(retrievers) + " |")
    lines.append("| --- | --- | --- | " + " | ".join(["---:"] * len(retrievers)) + " |")
    
    first_r = retrievers[0]
    for record in results["retrievers"][first_r].get("per_question", []):
        cells = []
        for r in retrievers:
            found = results["retrievers"][r]["per_question"]
            m = {rec["id"]: rec for rec in found}.get(record["id"])
            rank = m.get("rank") if m else None
            cells.append(str(rank) if rank is not None else "MISS")
        lines.append(
            f"| {record['id']} | {record['type']} | `{record['gold']}` | "
            + " | ".join(cells) + " |"
        )
    
    lines.append("")
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Generate evaluation plots and update results.md")
    parser.add_argument("--results", default=None, help="Path to results.json")
    parser.add_argument("--output-dir", default=None, help="Output directory for plots")
    parser.add_argument("--results-md", default=None, help="Path to results.md")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    results_path = Path(args.results) if args.results else root / "data" / "eval" / "results.json"
    output_dir = Path(args.output_dir) if args.output_dir else root / "data" / "eval" / "plots"
    md_path = Path(args.results_md) if args.results_md else root / "data" / "eval" / "results.md"

    if not results_path.exists():
        raise SystemExit(f"Results file not found: {results_path}")

    results = load_results(results_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Generating plots...")
    plot_accuracy_by_type(results, output_dir)
    plot_storage_vs_accuracy(results, output_dir)
    plot_latency(results, output_dir)
    plot_by_type_heatmap(results, output_dir)

    print("Updating results.md...")
    update_results_md(results, md_path)

    print(f"Plots saved to: {output_dir}")
    print(f"Markdown updated: {md_path}")


if __name__ == "__main__":
    main()