# Tests for MaxSim implementation
from __future__ import annotations

import torch
import pytest

from src.retriever_visual import VisualRetriever
from src.utils import load_config


def test_maxsim_manual_vs_naive():
    """Test manual MaxSim matches naive loop implementation."""
    config = load_config()
    retriever = VisualRetriever(config)

    # Create random embeddings
    torch.manual_seed(42)
    q = torch.randn(5, 128)  # 5 query tokens
    d = torch.randn(100, 128)  # 100 doc tokens

    manual_score = retriever.maxsim(q, d)
    naive_score = retriever.maxsim_naive(q, d)

    # Allow slightly larger tolerance for fp32 differences
    assert abs(manual_score - naive_score) < 2e-6, \
        f"Manual: {manual_score}, Naive: {naive_score}"


def test_maxsim_manual_vs_library():
    """Test manual MaxSim matches library's score_multi_vector."""
    config = load_config()
    retriever = VisualRetriever(config)
    retriever.load_model()

    # Create test embeddings using the actual model
    from PIL import Image

    # Embed a test image
    img = Image.new("RGB", (224, 224), color="red")
    doc_emb = retriever.embed_page(img)  # [n_d, 128] float32

    # Embed a test query
    query_emb = retriever.embed_query("What is in this image?")  # [n_q, 128] float32

    # Manual MaxSim
    manual_score = retriever.maxsim(query_emb, doc_emb)

    # Library score_multi_vector
    qs = [query_emb]
    ps = [doc_emb]
    lib_scores = retriever.processor.score_multi_vector(qs, ps)  # [1, 1]
    lib_score = lib_scores[0, 0].item()

    assert abs(manual_score - lib_score) < 1e-4, \
        f"Manual: {manual_score}, Library: {lib_score}"


def test_maxsim_fp16_roundtrip():
    """Test fp16 round-trip preserves scores within tolerance."""
    config = load_config()
    retriever = VisualRetriever(config)
    retriever.load_model()

    from PIL import Image

    # Embed a test image
    img = Image.new("RGB", (224, 224), color="blue")
    doc_emb_fp32 = retriever.embed_page(img)  # [n_d, 128] float32

    # Embed a test query
    query_emb_fp32 = retriever.embed_query("Test query")  # [n_q, 128] float32

    # Score with original fp32
    orig_score = retriever.maxsim(query_emb_fp32, doc_emb_fp32)

    # Convert to fp16 and back
    doc_emb_fp16 = doc_emb_fp32.to(torch.float16)
    doc_emb_fp32_restored = doc_emb_fp16.to(torch.float32)

    # Score with restored
    restored_score = retriever.maxsim(query_emb_fp32, doc_emb_fp32_restored)

    assert abs(orig_score - restored_score) < 1e-3, \
        f"Original: {orig_score}, Restored: {restored_score}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
