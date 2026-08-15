"""v5.1 ANN-backed geometric neighbor index.

Provides an approximate nearest neighbor backend using either:
1. FAISS (if installed) — fast, optimized
2. HNSW via numpy fallback — pure Python, no external deps

The index is used to find candidate neighbors in latent space:
    latent Z → ANN index → 96 candidate neighbors → exact reranking → 32 diagnostic neighbors

Includes index refresh policies and Recall@k qualification.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
import time

import torch
from torch import Tensor
import numpy as np

from .neighbor_index import NeighborIndex, KNNGraphResult, recall_at_k


class FAISSIndex:
    """FAISS-backed ANN index.

    Uses IVF or HNSW depending on dataset size. Falls back to flat
    index for small datasets.
    """

    def __init__(self, dim: int, nlist: int = 100, nprobe: int = 10):
        try:
            import faiss
        except ImportError as e:
            raise ImportError(
                "FAISS is not installed. Install with: pip install faiss-cpu"
            ) from e
        self.faiss = faiss
        self.dim = dim
        self.nlist = nlist
        self.nprobe = nprobe
        self.index = None
        self._data = None

    def build(self, data: np.ndarray) -> None:
        """Build the index from data [N, dim]."""
        N = data.shape[0]
        if N < 1000:
            # Use flat index for small datasets
            self.index = self.faiss.IndexFlatL2(self.dim)
        else:
            # Use IVF for larger datasets
            quantizer = self.faiss.IndexFlatL2(self.dim)
            self.index = self.faiss.IndexIVFFlat(quantizer, self.dim, self.nlist)
            self.index.train(data)
            self.index.nprobe = self.nprobe
        self.index.add(data)
        self._data = data

    def search(self, query: np.ndarray, k: int = 10) -> tuple[np.ndarray, np.ndarray]:
        """Search for k nearest neighbors.

        Returns (distances, indices) of shape [N_query, k].
        """
        if self.index is None:
            raise RuntimeError("Index not built. Call build() first.")
        return self.index.search(query, k)

    def refresh(self, data: np.ndarray) -> None:
        """Rebuild the index with new data."""
        self.build(data)


class HNSWIndexNumpy:
    """Pure-numpy HNSW-like fallback ANN index.

    This is a simplified implementation that provides reasonable
    approximate search without external dependencies. It uses
    random projection trees for coarse partitioning followed by
    exact distance computation within partitions.
    """

    def __init__(self, dim: int, n_partitions: int = 16, max_leaf_size: int = 50):
        self.dim = dim
        self.n_partitions = n_partitions
        self.max_leaf_size = max_leaf_size
        self._data = None
        self._partitions: list[np.ndarray] = []  # indices into data
        self._projection: np.ndarray | None = None

    def build(self, data: np.ndarray) -> None:
        """Build the index from data [N, dim]."""
        N = data.shape[0]
        self._data = data.copy()

        if N <= self.max_leaf_size:
            self._partitions = [np.arange(N)]
            return

        # Random projection for partitioning
        rng = np.random.RandomState(42)
        self._projection = rng.randn(self.dim, self.n_partitions).astype(data.dtype)

        # Project data
        projections = data @ self._projection  # [N, n_partitions]

        # Assign each point to its best partition
        best_partition = np.argmax(projections, axis=1)
        self._partitions = []
        for p in range(self.n_partitions):
            mask = best_partition == p
            indices = np.where(mask)[0]
            if len(indices) > 0:
                self._partitions.append(indices)

    def search(self, query: np.ndarray, k: int = 10) -> tuple[np.ndarray, np.ndarray]:
        """Search for k nearest neighbors.

        Returns (distances, indices) of shape [N_query, k].
        """
        if self._data is None:
            raise RuntimeError("Index not built. Call build() first.")

        N_query = query.shape[0]
        all_distances = np.zeros((N_query, k), dtype=np.float32)
        all_indices = np.zeros((N_query, k), dtype=np.int64)

        for i in range(N_query):
            q = query[i]

            # Find the best partition
            if self._projection is not None:
                q_proj = q @ self._projection
                best_p = np.argmax(q_proj)
                # Also check neighboring partitions
                partition_scores = np.abs(q_proj)
                top_partitions = np.argsort(-partition_scores)[:3]
            else:
                top_partitions = [0]

            # Search within top partitions (filter out empty partitions)
            valid_partitions = [self._partitions[p] for p in top_partitions
                               if p < len(self._partitions) and len(self._partitions[p]) > 0]
            if valid_partitions:
                candidates = np.concatenate(valid_partitions)
            else:
                candidates = np.arange(len(self._data))
            if len(candidates) == 0:
                candidates = np.arange(len(self._data))

            # Exact distance computation within candidates
            diff = self._data[candidates] - q
            dists = np.sum(diff ** 2, axis=1)

            # Get top k
            k_actual = min(k, len(candidates))
            top_idx = np.argpartition(dists, k_actual - 1)[:k_actual]
            top_dists = dists[top_idx]
            top_indices = candidates[top_idx]

            # Sort by distance
            sort_order = np.argsort(top_dists)
            top_dists = top_dists[sort_order]
            top_indices = top_indices[sort_order]

            # Pad if needed
            if k_actual < k:
                pad_d = np.full(k - k_actual, np.inf, dtype=np.float32)
                pad_i = np.full(k - k_actual, -1, dtype=np.int64)
                top_dists = np.concatenate([top_dists, pad_d])
                top_indices = np.concatenate([top_indices, pad_i])

            all_distances[i] = top_dists
            all_indices[i] = top_indices

        return all_distances, all_indices

    def refresh(self, data: np.ndarray) -> None:
        """Rebuild the index with new data."""
        self.build(data)


class ANNNeighborIndex:
    """ANN-backed neighbor index implementing the NeighborIndex protocol.

    Pipeline:
        latent Z → ANN index → candidate neighbors → exact reranking → final neighbors

    The ANN index provides approximate candidates, which are then
    reranked with exact distances to ensure accuracy.
    """

    def __init__(
        self,
        dim: int,
        n_candidates: int = 96,
        n_final: int = 32,
        backend: str = "auto",  # "faiss", "numpy", or "auto"
        refresh_interval: int = 100,  # Rebuild every N steps
    ):
        self.dim = dim
        self.n_candidates = n_candidates
        self.n_final = n_final
        self.backend = backend
        self.refresh_interval = refresh_interval
        self._index = None
        self._step = 0
        self._data = None

    def _select_backend(self) -> Any:
        """Select the ANN backend."""
        if self.backend == "faiss":
            return FAISSIndex(self.dim)
        elif self.backend == "numpy":
            return HNSWIndexNumpy(self.dim)
        elif self.backend == "auto":
            try:
                return FAISSIndex(self.dim)
            except ImportError:
                return HNSWIndexNumpy(self.dim)
        else:
            raise ValueError(f"Unknown backend: {self.backend}")

    def build(self, z: Tensor) -> None:
        """Build the ANN index from latent states."""
        data = z.detach().cpu().numpy().astype(np.float32)
        self._data = data
        self._index = self._select_backend()
        self._index.build(data)

    def refresh(self, z: Tensor) -> None:
        """Refresh the index with updated latent states."""
        self.build(z)

    def search(
        self,
        z: Tensor,
        k: int | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Search for k nearest neighbors.

        Args:
            z: Latent states [N, dim]
            k: Number of neighbors (defaults to n_final)

        Returns:
            (distances, indices) of shape [N, k]
        """
        if self._index is None or self._step % self.refresh_interval == 0:
            self.refresh(z)
        self._step += 1

        k_final = k or self.n_final
        k_search = max(k_final, self.n_candidates)

        data = z.detach().cpu().numpy().astype(np.float32)
        distances, indices = self._index.search(data, k_search)

        # Exact reranking
        N = data.shape[0]
        final_distances = np.zeros((N, k_final), dtype=np.float32)
        final_indices = np.zeros((N, k_final), dtype=np.int64)

        if k_final < k_search:
            # Rerank candidates with exact distances
            for i in range(N):
                valid = indices[i] >= 0
                if valid.any():
                    diff = data[indices[i][valid]] - data[i]
                    exact_dists = np.sum(diff ** 2, axis=1)
                    # Get top k_final from valid candidates
                    k_valid = min(k_final, len(exact_dists))
                    top_idx = np.argpartition(exact_dists, k_valid - 1)[:k_valid]
                    sort_order = np.argsort(exact_dists[top_idx])
                    final_distances[i, :k_valid] = exact_dists[top_idx][sort_order]
                    final_indices[i, :k_valid] = indices[i][valid][top_idx][sort_order]
        else:
            final_distances = distances
            final_indices = indices

        return (
            torch.from_numpy(final_distances),
            torch.from_numpy(final_indices),
        )

    def build_knn_graph(
        self,
        z: Tensor,
        k: int | None = None,
        threshold: float | None = None,
    ) -> KNNGraphResult:
        """Build a k-NN graph from latent states."""
        k = k or self.n_final
        distances, indices = self.search(z, k)

        N = z.shape[0]
        src_list = []
        dst_list = []
        weight_list = []

        for i in range(N):
            for j_idx in range(k):
                j = int(indices[i, j_idx])
                if j < 0 or j == i:
                    continue
                d = float(distances[i, j_idx])
                if threshold is not None and d > threshold:
                    continue
                w = 1.0 / (1.0 + d)
                src_list.append(i)
                dst_list.append(j)
                weight_list.append(w)

        if not src_list:
            return KNNGraphResult(
                src=torch.zeros(0, dtype=torch.long),
                dst=torch.zeros(0, dtype=torch.long),
                weight=torch.zeros(0),
                num_nodes=N,
            )

        return KNNGraphResult(
            src=torch.tensor(src_list, dtype=torch.long),
            dst=torch.tensor(dst_list, dtype=torch.long),
            weight=torch.tensor(weight_list),
            num_nodes=N,
        )

    def measure_recall(
        self,
        z: Tensor,
        k: int = 10,
        exact_index: NeighborIndex | None = None,
    ) -> float:
        """Measure Recall@k against an exact index."""
        if exact_index is None:
            from .neighbor_index import ExactChunkedKNN
            exact_index = ExactChunkedKNN(self.dim)
            exact_index.build(z)

        # Get exact results: query returns (indices, distances)
        exact_indices, _ = exact_index.query(z, k)

        # Get ANN results
        ann_distances, ann_indices = self.search(z, k)

        return recall_at_k(ann_indices, exact_indices, k)
