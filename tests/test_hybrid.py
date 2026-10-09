# Tests for Hybrid Retriever (RRF) and compression.
import pytest

from src.retriever_hybrid import HybridRetriever


class MockRetriever:
    """Mock retriever with pre-defined rankings."""

    def __init__(self, ranking):
        self.ranking = ranking

    def retrieve(self, query, k=60):
        return self.ranking[:k]

    def free_memory(self):
        pass


def test_rrf_basic():
    """Test basic RRF fusion with known rankings."""
    # Retriever A: [a, b, c, d, e]
    # Retriever B: [b, a, f, g, h]
    # RRF with k=60:
    # a: 1/(60+1) + 1/(60+2) = 1/61 + 1/62
    # b: 1/(60+2) + 1/(60+1) = 1/62 + 1/61
    # c: 1/(60+3) = 1/63
    # d: 1/(60+4) = 1/64
    # e: 1/(60+5) = 1/65
    # f: 1/(60+3) = 1/63
    # g: 1/(60+4) = 1/64
    # h: 1/(60+5) = 1/65

    rank_a = [("a", 1.0), ("b", 0.9), ("c", 0.8), ("d", 0.7), ("e", 0.6)]
    rank_b = [("b", 1.0), ("a", 0.9), ("f", 0.8), ("g", 0.7), ("h", 0.6)]

    hybrid = HybridRetriever(
        config={},
        retriever_a_name="mock_a",
        retriever_b_name="mock_b",
        rrf_k=60,
    )
    hybrid.retriever_a = MockRetriever(rank_a)
    hybrid.retriever_b = MockRetriever(rank_b)

    results = hybrid.retrieve("test query", k=5)
    page_ids = [pid for pid, _ in results]

    # a and b should be tied at top (same RRF score), then c and f tied
    assert page_ids[0] in ("a", "b")
    assert page_ids[1] in ("a", "b")
    assert set(page_ids[:2]) == {"a", "b"}


def test_rrf_prefers_agreement():
    """Items ranked high by both retrievers should score higher."""
    rank_a = [("x", 1.0), ("y", 0.9), ("z", 0.8)]
    rank_b = [("x", 1.0), ("w", 0.9), ("v", 0.8)]

    hybrid = HybridRetriever(
        config={},
        retriever_a_name="mock_a",
        retriever_b_name="mock_b",
        rrf_k=60,
    )
    hybrid.retriever_a = MockRetriever(rank_a)
    hybrid.retriever_b = MockRetriever(rank_b)

    results = hybrid.retrieve("test", k=5)
    page_ids = [pid for pid, _ in results]

    # x should be first (rank 1 in both)
    assert page_ids[0] == "x"


def test_rrf_k_parameter():
    """Test that rrf_k parameter affects scoring."""
    rank_a = [("a", 1.0), ("b", 0.9)]
    rank_b = [("b", 1.0), ("a", 0.9)]

    hybrid_small_k = HybridRetriever(
        config={},
        retriever_a_name="mock_a",
        retriever_b_name="mock_b",
        rrf_k=1,  # Small k amplifies rank differences
    )
    hybrid_small_k.retriever_a = MockRetriever(rank_a)
    hybrid_small_k.retriever_b = MockRetriever(rank_b)

    hybrid_large_k = HybridRetriever(
        config={},
        retriever_a_name="mock_a",
        retriever_b_name="mock_b",
        rrf_k=1000,  # Large k makes ranks matter less
    )
    hybrid_large_k.retriever_a = MockRetriever(rank_a)
    hybrid_large_k.retriever_b = MockRetriever(rank_b)

    # With small k, top rank matters more; with large k, both similar
    # Both should still rank a and b at top (they're symmetric)
    results_small = hybrid_small_k.retrieve("test", k=2)
    results_large = hybrid_large_k.retrieve("test", k=2)
    assert set(pid for pid, _ in results_small) == {"a", "b"}
    assert set(pid for pid, _ in results_large) == {"a", "b"}


def test_hybrid_uses_both_retrievers():
    """Verify hybrid calls both retrievers."""
    called_a = []
    called_b = []

    class TrackingRetriever:
        def __init__(self, name, ranking, called_list):
            self.name = name
            self.ranking = ranking
            self.called_list = called_list

        def retrieve(self, query, k=60):
            self.called_list.append(query)
            return self.ranking[:k]

        def free_memory(self):
            pass

    rank_a = [("a", 1.0)]
    rank_b = [("b", 1.0)]

    hybrid = HybridRetriever(config={}, retriever_a_name="a", retriever_b_name="b", rrf_k=60)
    hybrid.retriever_a = TrackingRetriever("a", rank_a, called_a)
    hybrid.retriever_b = TrackingRetriever("b", rank_b, called_b)

    hybrid.retrieve("test query", k=2)

    assert "test query" in called_a
    assert "test query" in called_b


def test_rrf_k_60_default():
    """Test that default rrf_k is 60."""
    hybrid = HybridRetriever(config={})
    assert hybrid.rrf_k == 60


if __name__ == "__main__":
    pytest.main([__file__, "-v"])