# VisionRAG Visual Retriever
# ColPali/ColSmol late-interaction retrieval using colpali-engine
from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from PIL import Image
from tqdm import tqdm

from src.utils import load_config, ensure_dir, load_jsonl, save_jsonl
from src.page_ids import build_page_id


class VisualRetriever:
    """Visual retriever using ColPali/ColSmol multi-vector embeddings."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        vr_cfg = config.get("retriever_visual", {})

        self.model_name = vr_cfg.get("model_name", "vidore/colSmol-256M")
        self.batch_size = vr_cfg.get("batch_size", 1)
        self.device_str = vr_cfg.get("device", "auto")
        self.max_length = vr_cfg.get("max_length", 512)
        self.use_local_patches_only = vr_cfg.get("use_local_patches_only", False)

        self.device = self._get_device()
        self.torch_dtype = torch.float32 if self.device == "cpu" else torch.bfloat16

        self.processor = None
        self.model = None
        self._embeddings: Optional[Dict[str, torch.Tensor]] = None
        self._page_order: Optional[List[str]] = None
        self._metadata: Optional[List[Dict]] = None

    def _get_device(self) -> str:
        if self.device_str != "auto":
            return self.device_str
        return "cuda" if torch.cuda.is_available() else "cpu"

    def load_model(self) -> None:
        """Load the ColIdefics3 model and processor."""
        if self.model is not None:
            return

        print(f"Loading model: {self.model_name} on {self.device}")
        from colpali_engine.models import ColIdefics3, ColIdefics3Processor

        self.processor = ColIdefics3Processor.from_pretrained(self.model_name)
        self.model = ColIdefics3.from_pretrained(
            self.model_name,
            torch_dtype=self.torch_dtype,
        ).to(self.device)
        self.model.eval()
        print("Model loaded successfully")

    def _get_page_id(self, doc_name: str, page_num: int) -> str:
        """Generate consistent page_id matching metadata.jsonl."""
        return build_page_id(doc_name, page_num)

    def _extract_page_embeddings(
        self,
        outputs: torch.Tensor,
        batch_doc: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """Extract page embeddings using attention_mask (all non-padding tokens)."""
        # outputs: [1, seq_len, 128]
        # batch_doc['attention_mask']: [1, seq_len] - 1 for non-padding tokens
        attention_mask = batch_doc["attention_mask"][0]  # [seq_len]

        if self.use_local_patches_only:
            # Use local patches only (excludes global patch)
            mask = self.processor.get_local_image_mask(batch_doc)[0]
        else:
            # Use all non-padding tokens (default)
            mask = attention_mask.bool()

        # Extract embeddings for this page
        page_embeddings = outputs[0][mask]  # [n_tokens, 128]
        return page_embeddings.to(torch.float32).cpu()

    def embed_page(self, image: Image.Image) -> torch.Tensor:
        """Embed a single page image, return [n_tokens, 128] float32 tensor."""
        self.load_model()

        images = [image.convert("RGB")]
        batch_doc = self.processor.process_images(images)
        batch_doc = {k: v.to(self.device) for k, v in batch_doc.items()}

        with torch.inference_mode():
            outputs = self.model(**batch_doc)

        embeddings = self._extract_page_embeddings(outputs, batch_doc)

        # Clean up
        del batch_doc, outputs
        if self.device == "cuda":
            torch.cuda.empty_cache()

        return embeddings

    def embed_query(self, query: str) -> torch.Tensor:
        """Embed a query string, return [n_query_tokens, 128] float32 tensor."""
        self.load_model()

        batch_query = self.processor.process_queries(texts=[query])
        batch_query = {k: v.to(self.device) for k, v in batch_query.items()}

        with torch.inference_mode():
            outputs = self.model(**batch_query)

        # Query embeddings: all tokens [1, n_q, 128] -> [n_q, 128]
        query_embeddings = outputs[0].to(torch.float32).cpu()

        # Clean up
        del batch_query, outputs
        if self.device == "cuda":
            torch.cuda.empty_cache()

        return query_embeddings

    def maxsim(self, q: torch.Tensor, d: torch.Tensor) -> float:
        """Manual MaxSim: sum over query tokens of max similarity with doc tokens."""
        # q: [n_q, dim], d: [n_d, dim]
        # Sim matrix: [n_q, n_d]
        sim_matrix = torch.matmul(q, d.t())
        # Max per query token: [n_q]
        max_sim = sim_matrix.max(dim=-1).values
        # Sum over query tokens
        return max_sim.sum().item()

    def maxsim_naive(self, q: torch.Tensor, d: torch.Tensor) -> float:
        """Naive loop implementation of MaxSim for testing."""
        score = 0.0
        for i in range(q.shape[0]):
            # q[i]: [dim], d: [n_d, dim]
            dots = torch.matmul(d, q[i])  # [n_d]
            score += dots.max().item()
        return score

    def build_index(self, pages_dir: Path, index_dir: Path) -> None:
        """Build visual index from page images with progressive save."""
        self.load_model()

        metadata_path = pages_dir / "metadata.jsonl"
        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata not found: {metadata_path}")

        metadata = load_jsonl(metadata_path)
        ensure_dir(index_dir)

        # Paths for saved index components
        embeddings_path = index_dir / "visual_embeddings.pt"
        page_order_path = index_dir / "page_order.json"
        build_stats_path = index_dir / "build_stats.json"

        # Load existing index for resume
        embeddings_dict = {}
        page_order = []
        completed_pages = set()

        if embeddings_path.exists() and page_order_path.exists():
            print(f"Resuming from existing index: {embeddings_path}")
            saved = torch.load(embeddings_path, map_location="cpu")
            embeddings_dict = {k: v.to(torch.float32) for k, v in saved.items()}
            with open(page_order_path, "r") as f:
                page_order = json.load(f)
            completed_pages = set(page_order)
            print(f"  Already embedded: {len(completed_pages)} pages")

        # Track token stats
        token_counts = []

        start_time = time.time()
        peak_memory = 0

        for meta in tqdm(metadata, desc="Embedding pages"):
            page_id = self._get_page_id(meta["doc_name"], meta["page_num"])

            if page_id in completed_pages:
                # Already embedded, get token count from saved embedding
                token_counts.append(embeddings_dict[page_id].shape[0])
                continue

            # Load and embed page
            img_path = Path.cwd() / meta["image_path"]
            image = Image.open(img_path)

            embeddings = self.embed_page(image)  # [n_tokens, 128] float32
            n_tokens = embeddings.shape[0]
            token_counts.append(n_tokens)

            # Convert to fp16 for storage
            embeddings_fp16 = embeddings.to(torch.float16)
            embeddings_dict[page_id] = embeddings_fp16
            page_order.append(page_id)

            # Progressive save
            torch.save(embeddings_dict, embeddings_path)
            with open(page_order_path, "w") as f:
                json.dump(page_order, f)

            # Memory tracking
            if self.device == "cuda":
                mem = torch.cuda.max_memory_allocated() / 1e9
            else:
                import psutil
                process = psutil.Process()
                mem = process.memory_info().rss / 1e9
            peak_memory = max(peak_memory, mem)

            # Clean up
            del embeddings, embeddings_fp16, image
            gc.collect()
            if self.device == "cuda":
                torch.cuda.empty_cache()

        # Final save (in case any left)
        torch.save(embeddings_dict, embeddings_path)
        with open(page_order_path, "w") as f:
            json.dump(page_order, f)

        # Save build stats
        elapsed = time.time() - start_time
        stats = {
            "total_pages": len(page_order),
            "elapsed_seconds": elapsed,
            "peak_memory_gb": peak_memory,
            "token_counts": {
                "min": min(token_counts) if token_counts else 0,
                "max": max(token_counts) if token_counts else 0,
                "mean": sum(token_counts) / len(token_counts) if token_counts else 0,
            },
            "embedding_dim": 128,
            "model_name": self.model_name,
            "use_local_patches_only": self.use_local_patches_only,
        }
        with open(build_stats_path, "w") as f:
            json.dump(stats, f, indent=2)

        # Print summary
        print(f"\nIndex build complete:")
        print(f"  Pages indexed: {len(page_order)}")
        print(f"  Time: {elapsed:.1f}s ({elapsed/60:.1f} min)")
        print(f"  Peak RAM: {peak_memory:.2f} GB")
        print(f"  Tokens/page: min={stats['token_counts']['min']}, max={stats['token_counts']['max']}, mean={stats['token_counts']['mean']:.1f}")
        print(f"  Index size: {embeddings_path.stat().st_size / 1e6:.2f} MB")
        print(f"  Saved to: {index_dir}")

    def load_index(self, index_dir: Path) -> None:
        """Load embeddings and metadata from disk."""
        embeddings_path = index_dir / "visual_embeddings.pt"
        page_order_path = index_dir / "page_order.json"
        metadata_path = index_dir.parent / "pages" / "metadata.jsonl"

        if not embeddings_path.exists():
            raise FileNotFoundError(f"Embeddings not found: {embeddings_path}")
        if not page_order_path.exists():
            raise FileNotFoundError(f"Page order not found: {page_order_path}")

        # Load embeddings (fp16 -> float32)
        saved = torch.load(embeddings_path, map_location="cpu")
        self._embeddings = {k: v.to(torch.float32) for k, v in saved.items()}

        with open(page_order_path, "r") as f:
            self._page_order = json.load(f)

        if metadata_path.exists():
            self._metadata = load_jsonl(metadata_path)
        else:
            self._metadata = []

        print(f"Loaded index: {len(self._embeddings)} pages")

    def retrieve(self, query: str, k: int = 3) -> List[Tuple[str, float]]:
        """Retrieve top-k pages for a query."""
        if self._embeddings is None or self._page_order is None:
            # Try default index dir
            index_dir = Path(self.config["paths"]["index_dir"])
            self.load_index(index_dir)

        query_emb = self.embed_query(query)  # [n_q, 128]

        # Prepare for score_multi_vector
        qs = [query_emb]
        ps = [self._embeddings[pid] for pid in self._page_order]

        # Use library's MaxSim
        scores = self.processor.score_multi_vector(qs, ps)  # [1, n_pages]
        scores = scores[0].cpu().tolist()  # [n_pages]

        # Get top-k
        scored_pages = list(zip(self._page_order, scores))
        scored_pages.sort(key=lambda x: x[1], reverse=True)

        return scored_pages[:k]


def main():
    parser = argparse.ArgumentParser(description="Visual Retriever for VisionRAG")
    parser.add_argument(
        "--build_index", action="store_true", help="Build visual index from page images"
    )
    parser.add_argument(
        "--query", type=str, help="Query string to search"
    )
    parser.add_argument(
        "--k", type=int, default=3, help="Number of results to return"
    )
    parser.add_argument(
        "--pages_dir", type=str, default="data/pages", help="Directory with page images"
    )
    parser.add_argument(
        "--index_dir", type=str, default="data/index", help="Directory for index files"
    )
    parser.add_argument(
        "--config", type=str, default="config.yaml", help="Path to config.yaml"
    )

    args = parser.parse_args()

    config = load_config(args.config)
    retriever = VisualRetriever(config)

    pages_dir = Path(args.pages_dir)
    index_dir = Path(args.index_dir)

    if args.build_index:
        retriever.build_index(pages_dir, index_dir)
    elif args.query:
        results = retriever.retrieve(args.query, args.k)
        print(f"\nTop-{args.k} results for: {args.query}")
        for i, (page_id, score) in enumerate(results, 1):
            print(f"  {i}. {page_id}  score={score:.4f}")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
