"""v5.1 Hypergraph / higher-order relationships.

Extends the graph from pairwise edges to higher-order hyperedges
that connect 3+ nodes simultaneously. This captures relationships
that cannot be decomposed into pairwise interactions:

    Pairwise: (i, j) — binary relationship
    Hyperedge: (i, j, k) — ternary relationship (e.g., "i, j, k are co-activated")

The module provides:
- Hyperedge: a single hyperedge connecting k nodes
- HypergraphBuffers: sparse storage for hyperedges
- HypergraphLaplacian: diffusion operator for hypergraphs
- conversion utilities between hypergraphs and pairwise graphs
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
from torch import Tensor

from .version import VERSION


@dataclass
class Hyperedge:
    """A single hyperedge connecting k nodes."""
    nodes: tuple[int, ...]   # Node indices in this hyperedge
    weight: float = 1.0      # Hyperedge weight
    order: int = 2           # Number of nodes (2 = regular edge)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        self.order = len(self.nodes)

    def contains(self, node: int) -> bool:
        return node in self.nodes

    def pairs(self) -> list[tuple[int, int]]:
        """Decompose into all pairwise connections."""
        n = len(self.nodes)
        return [
            (self.nodes[i], self.nodes[j])
            for i in range(n)
            for j in range(i + 1, n)
        ]


class HypergraphBuffers:
    """Sparse storage for hyperedges of varying order.

    Uses a flat representation:
    - hyperedge_nodes: [total_indices] — flattened node indices
    - hyperedge_ptr: [num_hyperedges + 1] — pointers into hyperedge_nodes
    - hyperedge_weight: [num_hyperedges] — weight per hyperedge
    """

    def __init__(
        self,
        num_nodes: int,
        hyperedge_capacity: int,
        max_order: int = 4,
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.float32,
    ):
        self.num_nodes = int(num_nodes)
        self.hyperedge_capacity = int(hyperedge_capacity)
        self.max_order = int(max_order)
        self.device = device
        self.dtype = dtype

        # Flat storage
        total_indices = hyperedge_capacity * max_order
        self.hyperedge_nodes = torch.zeros(total_indices, dtype=torch.long, device=device)
        self.hyperedge_ptr = torch.zeros(hyperedge_capacity + 1, dtype=torch.long, device=device)
        self.hyperedge_weight = torch.zeros(hyperedge_capacity, dtype=dtype, device=device)
        self.hyperedge_order = torch.zeros(hyperedge_capacity, dtype=torch.long, device=device)
        self.hyperedge_valid = torch.zeros(hyperedge_capacity, dtype=torch.bool, device=device)
        self._num_active = 0

    @property
    def num_hyperedges(self) -> int:
        return int(self._num_active)

    def add_hyperedge(self, nodes: list[int], weight: float = 1.0) -> int:
        """Add a hyperedge. Returns the slot index."""
        order = len(nodes)
        if order < 2:
            raise ValueError("Hyperedge must connect at least 2 nodes")
        if order > self.max_order:
            raise ValueError(f"Hyperedge order {order} exceeds max_order {self.max_order}")
        if self._num_active >= self.hyperedge_capacity:
            raise RuntimeError("Hyperedge capacity exceeded")

        slot = self._num_active
        start = slot * self.max_order
        for i, node in enumerate(nodes):
            self.hyperedge_nodes[start + i] = node
        # Fill remaining with -1
        for i in range(order, self.max_order):
            self.hyperedge_nodes[start + i] = -1

        self.hyperedge_ptr[slot + 1] = start + order
        self.hyperedge_weight[slot] = weight
        self.hyperedge_order[slot] = order
        self.hyperedge_valid[slot] = True
        self._num_active += 1
        return slot

    def get_hyperedge(self, slot: int) -> Hyperedge | None:
        """Get a hyperedge by slot index."""
        if slot < 0 or slot >= self._num_active or not self.hyperedge_valid[slot]:
            return None
        start = slot * self.max_order
        order = int(self.hyperedge_order[slot])
        nodes = tuple(int(self.hyperedge_nodes[start + i]) for i in range(order))
        weight = float(self.hyperedge_weight[slot])
        return Hyperedge(nodes=nodes, weight=weight, order=order)

    def all_hyperedges(self) -> list[Hyperedge]:
        """Get all active hyperedges."""
        result = []
        for slot in range(self._num_active):
            he = self.get_hyperedge(slot)
            if he is not None:
                result.append(he)
        return result

    def to_pairwise_edges(self) -> tuple[Tensor, Tensor, Tensor]:
        """Decompose hyperedges into pairwise edges.

        Each hyperedge of order k produces C(k, 2) pairwise edges.
        The weight is distributed equally among the pairs.

        Returns:
            (src, dst, weight) tensors
        """
        src_list: list[int] = []
        dst_list: list[int] = []
        weight_list: list[float] = []

        for he in self.all_hyperedges():
            pairs = he.pairs()
            pair_weight = he.weight / len(pairs)
            for s, d in pairs:
                src_list.append(s)
                dst_list.append(d)
                weight_list.append(pair_weight)

        if not src_list:
            return (
                torch.zeros(0, dtype=torch.long),
                torch.zeros(0, dtype=torch.long),
                torch.zeros(0, dtype=self.dtype),
            )

        return (
            torch.tensor(src_list, dtype=torch.long),
            torch.tensor(dst_list, dtype=torch.long),
            torch.tensor(weight_list, dtype=self.dtype),
        )

    def clone(self) -> "HypergraphBuffers":
        """Create a copy of this hypergraph."""
        hg = HypergraphBuffers(
            self.num_nodes, self.hyperedge_capacity, self.max_order,
            device=self.device, dtype=self.dtype,
        )
        hg.hyperedge_nodes = self.hyperedge_nodes.clone()
        hg.hyperedge_ptr = self.hyperedge_ptr.clone()
        hg.hyperedge_weight = self.hyperedge_weight.clone()
        hg.hyperedge_order = self.hyperedge_order.clone()
        hg.hyperedge_valid = self.hyperedge_valid.clone()
        hg._num_active = self._num_active
        return hg

    def summary(self) -> dict[str, Any]:
        """Return a summary of the hypergraph."""
        orders = self.hyperedge_order[:self._num_active].tolist()
        order_counts: dict[int, int] = {}
        for o in orders:
            order_counts[o] = order_counts.get(o, 0) + 1
        return {
            "num_nodes": self.num_nodes,
            "num_hyperedges": self._num_active,
            "capacity": self.hyperedge_capacity,
            "max_order": self.max_order,
            "order_counts": order_counts,
            "version": VERSION,
        }


def hypergraph_laplacian_diffusion(
    z: Tensor,                          # [N, D]
    hypergraph: HypergraphBuffers,
    eta: float = 0.1,
    num_steps: int = 1,
) -> Tensor:
    """Diffusion on a hypergraph.

    For each hyperedge (i_1, ..., i_k), the diffusion updates each node's
    representation toward the centroid of the hyperedge:

        z_{i_j} ← z_{i_j} + η * w * (centroid - z_{i_j})

    where centroid = mean(z_{i_1}, ..., z_{i_k}).
    """
    N, D = z.shape
    z_out = z.clone()

    for _ in range(num_steps):
        for he in hypergraph.all_hyperedges():
            # Compute centroid
            nodes = list(he.nodes)
            centroid = z_out[nodes].mean(dim=0)

            # Update each node toward centroid
            for node in nodes:
                z_out[node] = z_out[node] + eta * he.weight * (centroid - z_out[node])

    return z_out


def clique_expansion(
    hypergraph: HypergraphBuffers,
) -> tuple[Tensor, Tensor, Tensor]:
    """Convert a hypergraph to a pairwise graph via clique expansion.

    Each hyperedge becomes a clique (all pairwise edges present).
    This is the standard reduction used in spectral hypergraph theory.
    """
    return hypergraph.to_pairwise_edges()


def star_expansion(
    hypergraph: HypergraphBuffers,
) -> tuple[Tensor, Tensor, Tensor, int]:
    """Convert a hypergraph to a pairwise graph via star expansion.

    Each hyperedge gets a new "anchor" node. All nodes in the hyperedge
    are connected to the anchor.

    Returns:
        (src, dst, weight, num_anchor_nodes)
    """
    src_list: list[int] = []
    dst_list: list[int] = []
    weight_list: list[float] = []
    N = hypergraph.num_nodes
    anchor_count = 0

    for he in hypergraph.all_hyperedges():
        anchor = N + anchor_count
        anchor_count += 1
        for node in he.nodes:
            src_list.append(node)
            dst_list.append(anchor)
            weight_list.append(he.weight)
            # Also add reverse edge
            src_list.append(anchor)
            dst_list.append(node)
            weight_list.append(he.weight)

    if not src_list:
        return (
            torch.zeros(0, dtype=torch.long),
            torch.zeros(0, dtype=torch.long),
            torch.zeros(0, dtype=torch.float32),
            0,
        )

    return (
        torch.tensor(src_list, dtype=torch.long),
        torch.tensor(dst_list, dtype=torch.long),
        torch.tensor(weight_list, dtype=torch.float32),
        anchor_count,
    )
