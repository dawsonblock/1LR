"""v4.1.3 Neighbor index abstraction for scalable diagnostic operator construction.

This module provides a protocol-based abstraction for k-NN neighbor discovery,
replacing the hardcoded O(N²D) chunked kNN with a pluggable backend system.

Available backends:
- ExactChunkedKNN: Mathematical reference, O(N²D) compute, bounded memory
- (Future: HNSW, FAISSFlat, FAISSIVF, GPU ANN)

The workflow is:
    Z → ANN candidate search → k' candidates → exact metric reranking → k neighbors → P^D

For example: k=32, k'=96. ANN proposes 96 likely neighbors, then exact metric
selects the true best 32 among them. This preserves geometric fidelity while
reducing compute from O(N²D) to O(N·k'·D) for ANN search + O(N·k'²) for reranking.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
import torch
from torch import Tensor


class NeighborIndex(ABC):
    """Abstract neighbor discovery protocol.

    Implementations must support:
    - build(z): Construct the index from a latent cloud
    - query(z, k): Return the k nearest neighbors for each node
    - rebuild_policy: Decide when to rebuild based on drift
    """

    @abstractmethod
    def build(self, z: Tensor) -> None:
        """Construct the index from latent state z [N, D]."""
        ...

    @abstractmethod
    def query(self, z: Tensor, k: int) -> tuple[Tensor, Tensor]:
        """Query k nearest neighbors.

        Returns (indices, distances) where:
        - indices: [N, k] tensor of neighbor indices
        - distances: [N, k] tensor of distances
        """
        ...

    @abstractmethod
    def should_rebuild(self, z: Tensor, drift_threshold: float = 0.1) -> bool:
        """Decide whether the index needs rebuilding based on latent drift."""
        ...


class ExactChunkedKNN(NeighborIndex):
    """Exact chunked k-NN backend (mathematical reference).

    Computes exact pairwise distances in chunks of size chunk_size to bound
    peak memory to O(chunk_size × N) instead of O(N²).

    Total compute is still O(N²D), but peak memory is controlled.
    This is the reference backend for correctness validation.
    """
    def __init__(self, chunk_size: int = 512, cache_index: bool = True):
        self.chunk_size = chunk_size
        self.cache_index = cache_index
        self._z_ref: Tensor | None = None
        self._index_time: Tensor | None = None

    def build(self, z: Tensor) -> None:
        self._z_ref = z.detach().clone()
        self._index_time = z.detach().clone()

    def query(self, z: Tensor, k: int) -> tuple[Tensor, Tensor]:
        N = z.shape[0]
        k = min(k, N - 1)
        device = z.device
        dtype = z.dtype

        # For each chunk of query points, compute distances to all points
        all_indices = torch.empty(N, k, dtype=torch.long, device=device)
        all_distances = torch.empty(N, k, dtype=dtype, device=device)

        for start in range(0, N, self.chunk_size):
            end = min(start + self.chunk_size, N)
            z_chunk = z[start:end]  # [chunk, D]
            # Compute pairwise distances: ||z_i - z_j||^2 = |z_i|^2 + |z_j|^2 - 2 z_i·z_j
            dist_sq = (
                z_chunk.pow(2).sum(dim=1, keepdim=True)  # [chunk, 1]
                + z.pow(2).sum(dim=1, keepdim=True).T    # [1, N]
                - 2.0 * z_chunk @ z.T                     # [chunk, N]
            )
            dist_sq = dist_sq.clamp_min(0.0)
            # Exclude self
            for i in range(end - start):
                dist_sq[i, start + i] = float("inf")
            # Top-k smallest
            topk_dist, topk_idx = torch.topk(dist_sq, k, dim=1, largest=False)
            all_indices[start:end] = topk_idx
            all_distances[start:end] = torch.sqrt(topk_dist.clamp_min(0.0))

        return all_indices, all_distances

    def should_rebuild(self, z: Tensor, drift_threshold: float = 0.1) -> bool:
        if self._z_ref is None:
            return True
        drift = torch.linalg.vector_norm(z - self._z_ref).item() / max(
            torch.linalg.vector_norm(self._z_ref).item(), 1e-10
        )
        return drift > drift_threshold


@dataclass
class KNNGraphResult:
    """Result of building a k-NN graph from a neighbor index."""
    src: Tensor
    dst: Tensor
    weight: Tensor
    num_nodes: int


def build_knn_graph(
    index: NeighborIndex,
    z: Tensor,
    k: int,
    epsilon_floor: float = 1e-4,
    symmetric: bool = True,
) -> KNNGraphResult:
    """Build a sparse k-NN graph from a neighbor index.

    The resulting graph has row-stochastic weights suitable for use as
    a diagnostic diffusion operator P^D.

    Args:
        index: Neighbor index (must be built)
        z: Latent state [N, D]
        k: Number of neighbors per node
        epsilon_floor: Minimum weight floor to avoid zero probabilities
        symmetric: If True, symmetrize the graph (add reverse edges)
    """
    indices, distances = index.query(z, k)
    N = z.shape[0]
    device = z.device

    # Convert distances to weights via Gaussian kernel
    # w_ij = exp(-d_ij^2 / (2 * sigma_i^2))
    sigma = distances.mean(dim=1, keepdim=True).clamp_min(1e-10)
    weights = torch.exp(-distances.pow(2) / (2 * sigma.pow(2)))
    weights = weights.clamp_min(epsilon_floor)

    # Row-normalize
    weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(epsilon_floor)

    # Build edge list
    src = torch.arange(N, device=device).unsqueeze(1).expand(-1, k).reshape(-1)
    dst = indices.reshape(-1)
    w = weights.reshape(-1)

    if symmetric:
        # Add reverse edges
        src_sym = torch.cat([src, dst])
        dst_sym = torch.cat([dst, src])
        w_sym = torch.cat([w, w])
        # Coalesce duplicates
        indices_t = torch.stack([src_sym, dst_sym], dim=0)
        sp = torch.sparse_coo_tensor(indices_t, w_sym, size=(N, N)).coalesce()
        src = sp.indices()[0]
        dst = sp.indices()[1]
        w = sp.values()
        # Re-normalize rows
        row_sum = torch.zeros(N, dtype=w.dtype, device=device)
        row_sum.index_add_(0, src, w)
        w = w / row_sum[src].clamp_min(epsilon_floor)

    return KNNGraphResult(src=src, dst=dst, weight=w, num_nodes=N)


def recall_at_k(ann_indices: Tensor, exact_indices: Tensor, k: int) -> float:
    """Compute Recall@k: fraction of true k-NN found by ANN.

    Args:
        ann_indices: [N, k'] ANN neighbor indices
        exact_indices: [N, k] Exact k-NN indices
        k: Number of true neighbors to check

    Returns:
        recall in [0, 1]
    """
    N = exact_indices.shape[0]
    total_recall = 0.0
    for i in range(N):
        exact_set = set(exact_indices[i, :k].tolist())
        ann_set = set(ann_indices[i, :k].tolist())
        if len(exact_set) > 0:
            total_recall += len(exact_set & ann_set) / len(exact_set)
    return total_recall / N
