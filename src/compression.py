# VisionRAG Visual Index Compression
# Supports fp16 (baseline), int8 quantization, and mean-pooled patch vectors.
from __future__ import annotations

from enum import Enum

import torch


class CompressionMode(Enum):
    FP16 = "fp16"
    INT8 = "int8"
    MEAN_POOL_8 = "mean_pool_8"
    MEAN_POOL_16 = "mean_pool_16"
    MEAN_POOL_64 = "mean_pool_64"
    MEAN_POOL_256 = "mean_pool_256"

    @classmethod
    def from_string(cls, value: str) -> "CompressionMode":
        try:
            return cls(value.lower())
        except ValueError:
            # Handle case where value is already an enum
            if isinstance(value, CompressionMode):
                return value
            raise


def compress_embeddings(
    embeddings: torch.Tensor,
    mode: CompressionMode,
) -> torch.Tensor:
    """Compress page embeddings according to mode.

    Args:
        embeddings: Tensor of shape [n_tokens, 128] in float32
        mode: Compression mode

    Returns:
        Compressed tensor (storage format varies by mode)
    """
    if mode == CompressionMode.FP16:
        return embeddings.to(torch.float16)

    elif mode == CompressionMode.INT8:
        # Per-row quantization for better precision
        max_vals = embeddings.abs().max(dim=1, keepdim=True).values
        scales = max_vals / 127.0
        scales = scales.clamp(min=1e-8)  # Avoid division by zero
        quantized = (embeddings / scales).round().clamp(-128, 127).to(torch.int8)
        return quantized, scales  # Return per-row scales

    elif mode == CompressionMode.MEAN_POOL_8:
        return mean_pool_patches(embeddings, 8)

    elif mode == CompressionMode.MEAN_POOL_16:
        return mean_pool_patches(embeddings, 16)

    elif mode == CompressionMode.MEAN_POOL_64:
        return mean_pool_patches(embeddings, 64)

    elif mode == CompressionMode.MEAN_POOL_256:
        return mean_pool_patches(embeddings, 256)

    else:
        raise ValueError(f"Unknown compression mode: {mode}")


def decompress_embeddings(
    compressed: torch.Tensor,
    mode: CompressionMode | str,
    scale: float = None,
) -> torch.Tensor:
    """Decompress embeddings back to float32 for scoring."""
    # Accept both enum and string
    if isinstance(mode, str):
        mode = CompressionMode.from_string(mode)

    # Get the string value for comparison (works for both enum and string)
    mode_val = mode.value if isinstance(mode, CompressionMode) else mode

    if mode_val == "fp16":
        return compressed.to(torch.float32)

    elif mode_val == "int8":
        if isinstance(compressed, tuple):
            quantized, scale = compressed
        else:
            quantized = compressed
            scale = scale or 1.0
        return (quantized.to(torch.float32) * scale)

    elif mode_val in ("mean_pool_8", "mean_pool_16", "mean_pool_64", "mean_pool_256"):
        return compressed.to(torch.float32)

    else:
        raise ValueError(f"Unknown compression mode: {mode}")


def mean_pool_patches(embeddings: torch.Tensor, n_vectors: int) -> torch.Tensor:
    """Mean-pool patch embeddings into n_vectors by averaging contiguous groups.

    Args:
        embeddings: [n_tokens, 128] float32
        n_vectors: Number of pooled vectors (e.g., 64, 256)

    Returns:
        [n_vectors, 128] float32 pooled embeddings
    """
    n_tokens = embeddings.shape[0]
    if n_tokens <= n_vectors:
        padded = torch.zeros(n_vectors, 128, dtype=embeddings.dtype)
        padded[:n_tokens] = embeddings
        return padded

    # Split tokens into n_vectors contiguous groups and average each group
    group_size = n_tokens // n_vectors
    remainder = n_tokens % n_vectors

    pooled = []
    start = 0
    for i in range(n_vectors):
        size = group_size + (1 if i < n_tokens % n_vectors else 0)
        end = start + size
        if size > 0:
            group_mean = embeddings[start:end].mean(dim=0)
        else:
            group_mean = torch.zeros(128, dtype=embeddings.dtype)
        pooled.append(group_mean)
        start = end

    return torch.stack(pooled)  # [n_vectors, 128]


def score_compressed(
    query_emb: torch.Tensor,
    compressed_doc: torch.Tensor,
    mode: CompressionMode,
) -> float:
    """Score a query against a compressed document using MaxSim.

    Args:
        query_emb: [n_q, 128] or [1, n_q, 128] float32
        compressed_doc: Compressed document embedding
        mode: Compression mode

    Returns:
        MaxSim score (float)
    """
    doc_emb = decompress_embeddings(compressed_doc, mode)
    # Handle both [n_q, 128] and [1, n_q, 128] query shapes
    if query_emb.dim() == 3:
        query_emb = query_emb[0]  # [1, n_q, 128] -> [n_q, 128]
    sim_matrix = torch.matmul(query_emb, doc_emb.t())  # [n_q, n_d]
    max_sim = sim_matrix.max(dim=1).values  # [n_q]
    return max_sim.sum().item()


