# VisionRAG Pipeline
# Ties retriever + generator together, manages model memory
from __future__ import annotations

import gc
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from src.utils import load_config
from src.page_ids import parse_page_id
from src.retriever_visual import VisualRetriever
from src.retriever_text import TextRetriever
from src.generator import Generator, GeneratorError


class Pipeline:
    """Main pipeline: retrieve -> free retriever memory -> generate."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.pipeline_cfg = config.get("pipeline", {})
        self.default_retriever = self.pipeline_cfg.get("default_retriever", "visual")
        self.top_k = self.pipeline_cfg.get("top_k", 3)

        self._visual_retriever: Optional[VisualRetriever] = None
        self._text_retriever: Optional[TextRetriever] = None
        self._generator: Optional[Generator] = None

    def _get_visual_retriever(self) -> VisualRetriever:
        if self._visual_retriever is None:
            self._visual_retriever = VisualRetriever(self.config)
        return self._visual_retriever

    def _get_text_retriever(self) -> TextRetriever:
        if self._text_retriever is None:
            self._text_retriever = TextRetriever(self.config)
        return self._text_retriever

    def _get_generator(self) -> Generator:
        if self._generator is None:
            self._generator = Generator(self.config)
        return self._generator

    def _free_visual_retriever(self):
        """Free visual retriever model from memory."""
        if self._visual_retriever is not None:
            if hasattr(self._visual_retriever, "model") and self._visual_retriever.model is not None:
                del self._visual_retriever.model
            if hasattr(self._visual_retriever, "processor") and self._visual_retriever.processor is not None:
                del self._visual_retriever.processor
            self._visual_retriever._embeddings = None
            self._visual_retriever._page_order = None
            self._visual_retriever._metadata = None
            gc.collect()
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def _free_text_retriever(self):
        """Free text retriever model from memory."""
        if self._text_retriever is not None:
            if hasattr(self._text_retriever, "embedder") and self._text_retriever.embedder is not None:
                del self._text_retriever.embedder
            if hasattr(self._text_retriever, "index") and self._text_retriever.index is not None:
                del self._text_retriever.index
            self._text_retriever.chunks = []
            self._text_retriever.chunk_to_page = []
            self._text_retriever.page_order = []
            gc.collect()

    def ask(
        self,
        question: str,
        retriever: str = "visual",
        k: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Ask a question using the specified retriever.

        Args:
            question: The question to answer
            retriever: "visual" or "text"
            k: Number of pages to retrieve (default from config)

        Returns:
            Dict with keys: answer, sources (list of page_ids), scores, retriever, latency
        """
        if k is None:
            k = self.top_k

        if retriever not in ("visual", "text"):
            raise ValueError(f"Unknown retriever: {retriever}. Use 'visual' or 'text'.")

        start_time = time.time()

        # Retrieve
        if retriever == "visual":
            vr = self._get_visual_retriever()
            results = vr.retrieve(question, k)
            self._free_visual_retriever()
        else:
            tr = self._get_text_retriever()
            results = tr.retrieve(question, k)
            self._free_text_retriever()

        if not results:
            return {
                "answer": "not found in the provided pages",
                "sources": [],
                "scores": [],
                "retriever": retriever,
                "latency": time.time() - start_time,
            }

        # Get page IDs and scores
        page_ids = [pid for pid, _ in results]
        scores = [score for _, score in results]

        # Build image paths for generator
        pages_dir = Path(self.config["paths"]["pages_dir"])
        image_paths = []
        valid_page_ids = []
        valid_scores = []

        for page_id, score in zip(page_ids, scores):
            # page_id format: doc_name_page_XXXX (doc_name may contain underscores)
            # image path: data/pages/doc_name/page_XXXX.png
            try:
                doc_name, page_num = parse_page_id(page_id)
            except ValueError:
                continue
            img_path = pages_dir / doc_name / f"page_{page_num:04d}.png"
            if img_path.exists():
                image_paths.append(img_path)
                valid_page_ids.append(page_id)
                valid_scores.append(score)

        if not image_paths:
            return {
                "answer": "not found in the provided pages",
                "sources": [],
                "scores": [],
                "retriever": retriever,
                "latency": time.time() - start_time,
            }

        # Generate answer
        gen = self._get_generator()
        try:
            gen_start = time.time()
            answer, _ = gen.generate(question, image_paths, valid_page_ids)
            gen_time = time.time() - gen_start
        except GeneratorError as e:
            return {
                "answer": f"Error: {e}",
                "sources": valid_page_ids,
                "scores": valid_scores,
                "retriever": retriever,
                "latency": time.time() - start_time,
                "generator_error": str(e),
            }

        total_latency = time.time() - start_time

        return {
            "answer": answer,
            "sources": valid_page_ids,
            "scores": valid_scores,
            "retriever": retriever,
            "latency": total_latency,
            "generation_latency": gen_time,
        }

    def close(self):
        """Close all components."""
        if self._generator:
            self._generator.close()
        self._free_visual_retriever()
        self._free_text_retriever()


def main():
    import argparse

    parser = argparse.ArgumentParser(description="VisionRAG Pipeline")
    parser.add_argument(
        "--question", type=str, required=True, help="Question to answer"
    )
    parser.add_argument(
        "--retriever",
        type=str,
        choices=["visual", "text"],
        default="visual",
        help="Retriever to use (visual or text)"
    )
    parser.add_argument(
        "--k", type=int, default=3, help="Number of pages to retrieve"
    )
    parser.add_argument(
        "--config", type=str, default="config.yaml", help="Path to config.yaml"
    )
    args = parser.parse_args()

    config = load_config(args.config)
    pipeline = Pipeline(config)

    try:
        result = pipeline.ask(args.question, args.retriever, args.k)
        print(f"Answer: {result['answer']}")
        print(f"Retriever: {result['retriever']}")
        print(f"Latency: {result['latency']:.2f}s")
        if "generation_latency" in result:
            print(f"Generation latency: {result['generation_latency']:.2f}s")
        print(f"Sources: {list(zip(result['sources'], result['scores']))}")
    finally:
        pipeline.close()


if __name__ == "__main__":
    main()
