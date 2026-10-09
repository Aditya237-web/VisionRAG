# VisionRAG Hybrid Retriever
# Reciprocal Rank Fusion (RRF) of visual and text-side retrievers.
from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from src.utils import load_config


class HybridRetriever:
    """Hybrid retriever using Reciprocal Rank Fusion (RRF).

    Combines ranked results from two retrievers using RRF:
    score = sum(1 / (k + rank_i)) for each retriever i
    """

    def __init__(
        self,
        config: Dict[str, Any],
        retriever_a_name: str = "visual",
        retriever_b_name: str = "ocr_text",
        rrf_k: int = 60,
    ):
        self.config = config
        self.retriever_a_name = retriever_a_name
        self.retriever_b_name = retriever_b_name
        self.rrf_k = rrf_k

        self.retriever_a = None
        self.retriever_b = None

    def _get_retriever(self, name: str):
        """Lazy-load a retriever by name."""
        if name == "visual":
            from src.retriever_visual import VisualRetriever
            retriever = VisualRetriever(self.config)
            retriever.load_index(Path(self.config["paths"]["index_dir"]))
            retriever.load_model()
            return retriever
        elif name == "text":
            from src.retriever_text import TextRetriever
            retriever = TextRetriever(self.config)
            retriever.load_index(Path(self.config["paths"]["index_dir"]) / "text")
            return retriever
        elif name == "ocr_text":
            from src.retriever_ocr_text import OCRTextRetriever
            retriever = OCRTextRetriever(self.config)
            retriever.load_index(Path(self.config["paths"]["index_dir"]) / "ocr_text")
            return retriever
        else:
            raise ValueError(f"Unknown retriever: {name}")

    def _ensure_loaded(self):
        if self.retriever_a is None:
            self.retriever_a = self._get_retriever(self.retriever_a_name)
        if self.retriever_b is None:
            self.retriever_b = self._get_retriever(self.retriever_b_name)

    def retrieve(self, query: str, k: int = 3) -> List[Tuple[str, float]]:
        """Retrieve top-k pages using RRF.

        Retrieves k*3 from each retriever to have enough candidates for fusion.
        """
        self._ensure_loaded()

        # Retrieve more candidates from each for better fusion
        search_k = max(k * 3, 60)
        results_a = self.retriever_a.retrieve(query, search_k)
        results_b = self.retriever_b.retrieve(query, search_k)

        # RRF scoring: score = sum(1 / (rrf_k + rank))
        rrf_scores: Dict[str, float] = {}

        for rank, (page_id, _) in enumerate(results_a, 1):
            rrf_scores[page_id] = rrf_scores.get(page_id, 0.0) + 1.0 / (self.rrf_k + rank)

        for rank, (page_id, _) in enumerate(results_b, 1):
            rrf_scores[page_id] = rrf_scores.get(page_id, 0.0) + 1.0 / (self.rrf_k + rank)

        # Sort by RRF score descending
        sorted_pages = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_pages[:k]

    def free_memory(self):
        """Free memory from both retrievers."""
        for retriever in [self.retriever_a, self.retriever_b]:
            if retriever is not None:
                if hasattr(retriever, "model") and retriever.model is not None:
                    del retriever.model
                if hasattr(retriever, "processor") and retriever.processor is not None:
                    del retriever.processor
                if hasattr(retriever, "embedder") and retriever.embedder is not None:
                    del retriever.embedder
                retriever._embeddings = None
                retriever._page_order = None
                retriever._metadata = None
                retriever.index = None
                retriever.chunks = None
                retriever.chunk_to_page = None
                retriever.page_order = None

        import gc
        gc.collect()
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Hybrid Retriever (RRF) for VisionRAG")
    parser.add_argument("--query", type=str, help="Query string to search")
    parser.add_argument("--k", type=int, default=3, help="Number of results to return")
    parser.add_argument("--rrf_k", type=int, default=60, help="RRF parameter k")
    parser.add_argument("--retriever_a", type=str, default="visual", help="First retriever")
    parser.add_argument("--retriever_b", type=str, default="ocr_text", help="Second retriever")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    hybrid = HybridRetriever(
        config,
        retriever_a_name=args.retriever_a,
        retriever_b_name=args.retriever_b,
        rrf_k=args.rrf_k,
    )

    if args.query:
        results = hybrid.retrieve(args.query, args.k)
        print(f"\nTop-{args.k} RRF results for: {args.query}")
        for i, (page_id, score) in enumerate(results, 1):
            print(f"  {i}. {page_id}  score={score:.4f}")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()