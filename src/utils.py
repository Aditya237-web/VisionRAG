# VisionRAG utilities
from __future__ import annotations

import faiss
import json
import yaml
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch


def load_config(config_path: str | Path = "config.yaml") -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_jsonl(path: str | Path) -> List[Dict[str, Any]]:
    """Load JSONL file as list of dicts."""
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def save_jsonl(data: List[Dict[str, Any]], path: str | Path) -> None:
    """Save list of dicts as JSONL."""
    with open(path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def get_device(device: str = "auto") -> str:
    """Resolve device string: 'auto' -> 'cuda' if available else 'cpu'."""
    if device != "auto":
        return device
    return "cuda" if torch.cuda.is_available() else "cpu"


def ensure_dir(path: str | Path) -> Path:
    """Ensure directory exists, return Path object."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def load_visual_index(index_dir: str | Path) -> Tuple[Dict[str, torch.Tensor], List[Dict], List[str]]:
    """Load visual index from disk.

    Returns:
        Tuple of (embeddings_dict, metadata_list, page_order)
        - embeddings_dict: {page_id: tensor[n_tokens, 128]} in float32
        - metadata_list: list of page metadata from metadata.jsonl
        - page_order: list of page_ids in consistent order
    """
    index_dir = Path(index_dir)
    embeddings_path = index_dir / "visual_embeddings.pt"
    page_order_path = index_dir / "page_order.json"
    metadata_path = index_dir.parent / "pages" / "metadata.jsonl"

    if not embeddings_path.exists():
        raise FileNotFoundError(f"Embeddings not found: {embeddings_path}")
    if not page_order_path.exists():
        raise FileNotFoundError(f"Page order not found: {page_order_path}")

    # Load embeddings (fp16 -> float32)
    saved = torch.load(embeddings_path, map_location="cpu")
    embeddings_dict = {k: v.to(torch.float32) for k, v in saved.items()}

    with open(page_order_path, "r") as f:
        page_order = json.load(f)

    metadata = []
    if metadata_path.exists():
        metadata = load_jsonl(metadata_path)

    return embeddings_dict, metadata, page_order


def load_text_index(index_dir: str | Path) -> Tuple[faiss.Index, List[Dict], List[str], List[str]]:
    """Load text index from disk.

    Returns:
        Tuple of (faiss_index, chunks, chunk_to_page, page_order)
        - faiss_index: FAISS IndexFlatIP
        - chunks: list of chunk dicts from chunks.jsonl
        - chunk_to_page: list mapping chunk_idx -> page_id
        - page_order: list of page_ids in consistent order
    """
    index_dir = Path(index_dir)
    index_path = index_dir / "faiss.index"
    chunks_path = index_dir / "chunks.jsonl"
    chunk_to_page_path = index_dir / "chunk_to_page.json"
    page_order_path = index_dir / "page_order.json"

    if not all(p.exists() for p in [index_path, chunks_path, chunk_to_page_path, page_order_path]):
        raise FileNotFoundError(f"Incomplete index in {index_dir}")

    index = faiss.read_index(str(index_path))
    chunks = load_jsonl(chunks_path)

    with open(chunk_to_page_path, "r") as f:
        chunk_to_page = json.load(f)

    with open(page_order_path, "r") as f:
        page_order = json.load(f)

    return index, chunks, chunk_to_page, page_order
