# VisionRAG Compressed Visual Index Builder
# Transforms existing fp16 embeddings (from data/index/visual_embeddings.pt)
# into compressed formats: int8, mean_pool_N. NO re-embedding.
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from PIL import Image
from tqdm import tqdm

from src.utils import load_config, ensure_dir
from src.page_ids import build_page_id
from src.compression import CompressionMode, compress_embeddings, mean_pool_patches, decompress_embeddings, score_compressed


class CompressedVisualRetriever:
    """Visual retriever supporting compressed indexes (int8, mean_pool_N)."""

    def __init__(self, config: Dict[str, Any], compression_mode: str):
        self.config = config
        vr_cfg = config.get("retriever_visual", {})

        self.model_name = vr_cfg.get("model_name", "vidore/colSmol-256M")
        self.batch_size = vr_cfg.get("batch_size", 1)
        self.device_str = vr_cfg.get("device", "auto")
        self.max_length = vr_cfg.get("max_length", 512)
        self.use_local_patches_only = vr_cfg.get("use_local_patches_only", False)

        self.compression_mode = CompressionMode.from_string(compression_mode)

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
        return build_page_id(doc_name, page_num)

    def load_index(self, index_dir: Path) -> None:
        embeddings_path = index_dir / "visual_embeddings.pt"
        page_order_path = index_dir / "page_order.json"
        metadata_path = index_dir.parent / "pages" / "metadata.jsonl"

        if not embeddings_path.exists():
            raise FileNotFoundError(f"Embeddings not found: {embeddings_path}")
        if not page_order_path.exists():
            raise FileNotFoundError(f"Page order not found: {page_order_path}")

        saved = torch.load(embeddings_path, map_location="cpu")
        if isinstance(saved, dict) and "embeddings" in saved:
            self._embeddings = saved["embeddings"]
        else:
            self._embeddings = saved

        with open(page_order_path, "r") as f:
            self._page_order = json.load(f)

        metadata_path = index_dir.parent / "pages" / "metadata.jsonl"
        if metadata_path.exists():
            from src.utils import load_jsonl
            self._metadata = load_jsonl(metadata_path)
        else:
            self._metadata = []

        print(f"Loaded index: {len(self._embeddings)} pages, mode={self.compression_mode.value}")

    def _dequantize_all(self) -> None:
        """Pre-dequantize all int8 embeddings to float32 at load time."""
        if self.compression_mode != CompressionMode.INT8:
            return
        print("Pre-dequantizing int8 embeddings...")
        for pid in self._page_order:
            compressed_doc = self._embeddings[pid]
            if isinstance(compressed_doc, tuple):
                quantized, scales = compressed_doc
                # Dequantize: quantized * scales -> float32
                dequantized = quantized.to(torch.float32) * scales
                # Store as float16 for memory efficiency
                self._embeddings[pid] = dequantized.to(torch.float16)
        print("Dequantization complete")

    def retrieve(self, query: str, k: int = 3) -> List[Tuple[str, float]]:
        if self._embeddings is None or self._page_order is None:
            index_dir = Path(self.config["paths"]["index_dir"])
            self.load_index(index_dir)

        # Load model and processor if not loaded
        self.load_model()

        # Embed query
        batch_query = self.processor.process_queries(texts=[query])
        batch_query = {k: v.to(self.device) for k, v in batch_query.items()}

        with torch.inference_mode():
            outputs = self.model(**batch_query)

        query_emb = outputs[0].to(torch.float32).cpu()  # [n_q, 128]

        del batch_query, outputs
        if self.device == "cuda":
            torch.cuda.empty_cache()

        # Score against each page
        scores = []
        for pid in self._page_order:
            compressed_doc = self._embeddings[pid]

            # If int8 was pre-dequantized, embeddings are float16 tensors
            if self.compression_mode == CompressionMode.INT8 and isinstance(compressed_doc, torch.Tensor):
                # Already dequantized to float16, convert to float32 and score
                doc_emb = compressed_doc.to(torch.float32)
                sim_matrix = torch.matmul(query_emb.unsqueeze(0)[0], doc_emb.t())  # [n_q, n_d]
                max_sim = sim_matrix.max(dim=1).values
                score = max_sim.sum().item()
            else:
                score = score_compressed(
                    query_emb.unsqueeze(0),  # [1, n_q, 128]
                    compressed_doc,
                    self.compression_mode,
                )
            scores.append((pid, score))

        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:k]


def build_compressed_index(
    config: Dict[str, Any],
    compression_mode: CompressionMode,
    source_index_dir: Path,
    output_index_dir: Path,
) -> None:
    """Build compressed index from existing fp16 embeddings. NO re-embedding."""
    from src.utils import ensure_dir

    ensure_dir(output_index_dir)

    source_embeddings_path = source_index_dir / "visual_embeddings.pt"
    source_page_order_path = source_index_dir / "page_order.json"
    source_build_stats_path = source_index_dir / "build_stats.json"

    if not source_embeddings_path.exists():
        raise FileNotFoundError(f"Source embeddings not found: {source_embeddings_path}")
    if not source_page_order_path.exists():
        raise FileNotFoundError(f"Source page order not found: {source_page_order_path}")

    # Load source embeddings (fp16) and page order
    print(f"Loading source embeddings from {source_embeddings_path}...")
    source_saved = torch.load(source_embeddings_path, map_location="cpu")
    if isinstance(source_saved, dict) and "embeddings" in source_saved:
        source_embeddings_dict = source_saved["embeddings"]
    else:
        source_embeddings_dict = source_saved

    with open(source_page_order_path, "r") as f:
        page_order = json.load(f)

    print(f"Loaded {len(source_embeddings_dict)} pages from fp16 index")

    # Load source build stats for reference
    source_build_time = 0.0
    source_build_stats_path = source_index_dir / "build_stats.json"
    if source_build_stats_path.exists():
        with open(source_build_stats_path, "r", encoding="utf-8") as f:
            stats = json.load(f)
            source_build_time = stats.get("elapsed_seconds", 0.0)

    # Compress each page
    start_time = time.time()
    compressed_embeddings = {}

    for pid in tqdm(page_order, desc=f"Compressing to {compression_mode.value}"):
        if pid not in source_embeddings_dict:
            continue

        emb = source_embeddings_dict[pid]  # [n_tokens, 128] fp16 or float32
        emb_fp32 = emb.to(torch.float32)  # ensure float32 for processing

        if compression_mode == CompressionMode.FP16:
            # Baseline: just convert to fp16 if not already
            compressed_embeddings[pid] = emb_fp32.to(torch.float16)

        elif compression_mode.value.startswith("mean_pool_"):
            # Extract N from "mean_pool_N"
            n_vectors = int(compression_mode.value.split("_")[-1])
            compressed = mean_pool_patches(emb_fp32, n_vectors)
            # L2-normalize each pooled vector
            compressed = torch.nn.functional.normalize(compressed, p=2, dim=1)
            compressed_embeddings[pid] = compressed.to(torch.float16)

        elif compression_mode == CompressionMode.INT8:
            # Per-row quantization: one scale per token vector
            max_vals = emb_fp32.abs().max(dim=1, keepdim=True).values
            scales = (max_vals / 127.0).clamp(min=1e-8)
            quantized = (emb_fp32 / scales).round().clamp(-128, 127).to(torch.int8)
            compressed_embeddings[pid] = (quantized, scales)

        else:
            raise ValueError(f"Unsupported compression mode: {compression_mode}")

    # Save compressed index
    output_embeddings_path = output_index_dir / "visual_embeddings.pt"
    output_page_order_path = output_index_dir / "page_order.json"
    output_build_stats_path = output_index_dir / "build_stats.json"

    compressed_embeddings = {k: v.to(torch.float16) if not isinstance(v, tuple) else (v[0].to(torch.int8), v[1].to(torch.float32)) for k, v in compressed_embeddings.items()}

    torch.save(compressed_embeddings, output_index_dir / "visual_embeddings.pt")
    with open(output_index_dir / "page_order.json", "w") as f:
        json.dump(page_order, f)

    # Save build stats
    elapsed = time.time() - start_time
    stats = {
        "total_pages": len(page_order),
        "elapsed_seconds": elapsed,
        "compression_mode": compression_mode.value,
        "embedding_dim": 128,
    }
    with open(output_index_dir / "build_stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    size_mb = (output_index_dir / "visual_embeddings.pt").stat().st_size / 1e6
    print(f"\nCompressed index build complete:")
    print(f"  Pages: {len(page_order)}")
    print(f"  Time: {elapsed:.1f}s")
    print(f"  Index size: {size_mb:.2f} MB")
    print(f"  Compression: {compression_mode.value}")
    print(f"  Saved to: {output_index_dir}")


def main():
    parser = argparse.ArgumentParser(description="Build compressed visual indexes from existing fp16 index")
    parser.add_argument(
        "--mode", type=str, default="mean_pool_64",
        choices=["fp16", "int8", "mean_pool_8", "mean_pool_16", "mean_pool_64", "mean_pool_256"],
        help="Compression mode (fp16 = baseline uncompressed)"
    )
    parser.add_argument(
        "--source_dir", type=str, default="data/index",
        help="Source directory containing visual_embeddings.pt (fp16)"
    )
    parser.add_argument(
        "--output_dir", type=str, default=None,
        help="Output directory (default: data/index/visual_<mode>)"
    )
    parser.add_argument(
        "--config", type=str, default="config.yaml", help="Path to config.yaml"
    )
    parser.add_argument(
        "--local_patches", action="store_true",
        help="Use local patches only (ablation)"
    )

    args = parser.parse_args()

    config = load_config(args.config)

    # Override config with CLI args
    if args.local_patches:
        if "retriever_visual" not in config:
            config["retriever_visual"] = {}
        config["retriever_visual"]["use_local_patches_only"] = True

    source_dir = Path(args.source_dir)
    output_dir = Path(args.output_dir) if args.output_dir else Path("data/index") / f"visual_{args.mode}"
    if args.local_patches:
        output_dir = Path("data/index") / f"visual_{args.mode}_local"

    print(f"Building {args.mode} index from {args.source_dir}")
    if args.local_patches:
        print("  (local patches only ablation)")
    print(f"Output: {output_dir}")

    mode = CompressionMode.from_string(args.mode)
    build_compressed_index(config, CompressionMode(args.mode), Path(args.source_dir), Path(output_dir))


if __name__ == "__main__":
    main()