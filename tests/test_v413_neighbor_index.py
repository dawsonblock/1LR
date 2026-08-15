"""v4.1.3 Neighbor index abstraction tests."""
from __future__ import annotations

import pytest
import torch

from lgae_v3.neighbor_index import (
    NeighborIndex, ExactChunkedKNN, KNNGraphResult,
    build_knn_graph, recall_at_k,
)


def test_exact_chunked_knn_basic():
    """ExactChunkedKNN should return k nearest neighbors."""
    torch.manual_seed(42)
    z = torch.randn(10, 4)
    index = ExactChunkedKNN(chunk_size=4)
    index.build(z)
    indices, distances = index.query(z, k=3)
    assert indices.shape == (10, 3)
    assert distances.shape == (10, 3)
    # No self-loops
    for i in range(10):
        assert i not in indices[i].tolist()


def test_exact_chunked_knn_matches_brute_force():
    """ExactChunkedKNN must match brute-force k-NN exactly."""
    torch.manual_seed(123)
    N, D = 20, 8
    z = torch.randn(N, D)
    index = ExactChunkedKNN(chunk_size=8)
    index.build(z)
    indices, distances = index.query(z, k=5)

    # Brute force
    dist_sq = torch.cdist(z, z, p=2) ** 2
    dist_sq.fill_diagonal_(float("inf"))
    bf_dist, bf_idx = torch.topk(dist_sq, 5, dim=1, largest=False)

    assert torch.equal(indices, bf_idx)
    assert torch.allclose(distances, torch.sqrt(bf_dist.clamp_min(0.0)), atol=1e-5)


def test_exact_chunked_knn_chunk_size_independence():
    """Results must be identical regardless of chunk size."""
    torch.manual_seed(777)
    z = torch.randn(30, 4)
    idx1 = ExactChunkedKNN(chunk_size=4)
    idx1.build(z)
    i1, d1 = idx1.query(z, k=5)

    idx2 = ExactChunkedKNN(chunk_size=32)
    idx2.build(z)
    i2, d2 = idx2.query(z, k=5)

    assert torch.equal(i1, i2)
    assert torch.allclose(d1, d2, atol=1e-6)


def test_should_rebuild_initial():
    """should_rebuild must return True when index is empty."""
    index = ExactChunkedKNN()
    z = torch.randn(5, 3)
    assert index.should_rebuild(z) is True


def test_should_rebuild_no_drift():
    """should_rebuild must return False when z hasn't changed."""
    z = torch.randn(5, 3)
    index = ExactChunkedKNN()
    index.build(z)
    assert index.should_rebuild(z) is False


def test_should_rebuild_after_drift():
    """should_rebuild must return True after significant drift."""
    z = torch.randn(5, 3)
    index = ExactChunkedKNN()
    index.build(z)
    z_drift = z + 10.0 * torch.randn(5, 3)
    assert index.should_rebuild(z_drift, drift_threshold=0.1) is True


def test_build_knn_graph_basic():
    """build_knn_graph should produce a valid sparse graph."""
    torch.manual_seed(42)
    z = torch.randn(10, 4)
    index = ExactChunkedKNN(chunk_size=10)
    index.build(z)
    result = build_knn_graph(index, z, k=3)
    assert result.num_nodes == 10
    assert result.src.numel() > 0
    assert result.dst.numel() > 0
    assert result.weight.numel() > 0
    # Weights should be positive
    assert (result.weight > 0).all()


def test_build_knn_graph_symmetric():
    """Symmetric k-NN graph should have reverse edges."""
    torch.manual_seed(42)
    z = torch.randn(8, 4)
    index = ExactChunkedKNN(chunk_size=8)
    index.build(z)
    result = build_knn_graph(index, z, k=3, symmetric=True)
    # Check that for each edge (i, j), (j, i) also exists
    edges = set(zip(result.src.tolist(), result.dst.tolist()))
    for (s, d) in edges:
        assert (d, s) in edges, f"Missing reverse edge ({d}, {s})"


def test_build_knn_graph_nonsymmetric():
    """Non-symmetric k-NN graph should have exactly N*k edges."""
    torch.manual_seed(42)
    N, k = 10, 3
    z = torch.randn(N, 4)
    index = ExactChunkedKNN(chunk_size=N)
    index.build(z)
    result = build_knn_graph(index, z, k=k, symmetric=False)
    assert result.src.numel() == N * k


def test_recall_at_k_perfect():
    """Recall@k should be 1.0 when ANN matches exact."""
    indices = torch.tensor([[1, 2, 3], [0, 2, 3]])
    exact = torch.tensor([[1, 2, 3], [0, 2, 3]])
    assert recall_at_k(indices, exact, k=3) == 1.0


def test_recall_at_k_partial():
    """Recall@k should be 2/3 when 2 of 3 match."""
    indices = torch.tensor([[1, 2, 4], [0, 2, 3]])
    exact = torch.tensor([[1, 2, 3], [0, 2, 3]])
    recall = recall_at_k(indices, exact, k=3)
    # First row: 2/3 match, second row: 3/3 match → avg = (2/3 + 1) / 2 = 5/6
    assert abs(recall - (2.0 / 3.0 + 1.0) / 2.0) < 1e-6


def test_recall_at_k_zero():
    """Recall@k should be 0.0 when nothing matches."""
    indices = torch.tensor([[4, 5, 6], [4, 5, 6]])
    exact = torch.tensor([[1, 2, 3], [0, 2, 3]])
    assert recall_at_k(indices, exact, k=3) == 0.0
