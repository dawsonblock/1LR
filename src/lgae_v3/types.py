from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Iterable
import hashlib
import json
import math

import torch
from torch import Tensor


class EdgeRole(str, Enum):
    GENERIC = "generic"
    CLUSTER = "cluster"
    BRIDGE = "bridge"
    HIERARCHY = "hierarchy"
    CAUSAL = "causal"
    MEMORY = "memory"


EDGE_ROLE_ORDER = tuple(EdgeRole)
ROLE_TO_CODE = {role: i for i, role in enumerate(EDGE_ROLE_ORDER)}
CODE_TO_ROLE = {i: role for role, i in ROLE_TO_CODE.items()}


def edge_role_code(role: EdgeRole | str | int) -> int:
    if isinstance(role, int):
        if role not in CODE_TO_ROLE:
            raise ValueError(f"unknown edge role code: {role}")
        return int(role)
    if isinstance(role, str):
        role = EdgeRole(role)
    return ROLE_TO_CODE[role]


def edge_role_from_code(code: int) -> EdgeRole:
    try:
        return CODE_TO_ROLE[int(code)]
    except KeyError as exc:
        raise ValueError(f"unknown edge role code: {code}") from exc


class MutationDecision(str, Enum):
    ACCEPT = "accept"
    QUARANTINE = "quarantine"
    REJECT = "reject"


@dataclass(slots=True)
class GraphBuffers:
    """Fixed-capacity undirected graph buffers for compile-stable actuation topology.

    Invariants are fail-closed for active slots: endpoints are in range, endpoints differ,
    weights and lengths are finite and strictly positive, and undirected duplicate edges
    are forbidden.

    **Metric–measure separation (v4.1):** The graph carries two independent edge scalars:

    - ``weight`` = **affinity** / conductance. Used by the Markov operator
      ``P_uv = a_uv / Σ_j a_uj``. A large weight means a strong connection
      (high transition probability).
    - ``length`` = **metric length**. Used by shortest-path distances
      ``d_ℓ(x,y) = shortestpath_ℓ(x,y)`` and curvature ground costs.
      A large length means endpoints are far apart.

    These are mathematically distinct: ``weight`` governs *how likely
    information is to move*, ``length`` governs *how far apart states are*.
    When only one scalar is provided at construction, the default inverse
    relationship ``length = 1/weight`` is used, but they evolve independently
    under mutation and Ricci flow.

    ``version`` is monotonically incremented by committed mutation operations.
    ``slot_generation`` increments monotonically whenever an individual edge
    slot changes identity or validity lifecycle.
    """

    num_nodes: int
    src: Tensor
    dst: Tensor
    weight: Tensor  # affinity / conductance
    valid: Tensor
    length: Tensor | None = None  # metric length (defaults to 1/weight)
    role: Tensor | None = None
    slot_generation: Tensor | None = None
    version: int = 0

    def __post_init__(self) -> None:
        if self.slot_generation is None:
            self.slot_generation = torch.zeros(self.src.numel(), dtype=torch.long, device=self.src.device)
        if self.length is None:
            # Default inverse relationship: length = 1/weight
            self.length = torch.where(
                self.weight > 0,
                1.0 / self.weight.clamp_min(1e-30),
                torch.zeros_like(self.weight),
            )
        self.validate()

    @property
    def capacity(self) -> int:
        return int(self.src.numel())

    @property
    def edge_count(self) -> int:
        return int(self.valid.sum().item())

    def validate(self, *, check_duplicates: bool = True) -> None:
        if self.num_nodes <= 0:
            raise ValueError("num_nodes must be positive")
        if self.src.ndim != 1 or self.dst.ndim != 1 or self.weight.ndim != 1 or self.valid.ndim != 1:
            raise ValueError("src, dst, weight, valid must be 1-D")
        n = self.src.numel()
        if not (self.dst.numel() == self.weight.numel() == self.valid.numel() == n):
            raise ValueError("graph buffers must have equal capacity")
        if self.src.dtype != torch.long or self.dst.dtype != torch.long:
            raise TypeError("src and dst must be torch.long")
        if self.valid.dtype != torch.bool:
            raise TypeError("valid must be boolean")
        if self.length is not None:
            if self.length.ndim != 1 or self.length.numel() != n:
                raise ValueError("length must be 1-D with graph buffer capacity")
            if self.length.dtype != self.weight.dtype:
                raise TypeError("length must match weight dtype")
        if self.slot_generation is not None:
            if self.slot_generation.ndim != 1 or self.slot_generation.numel() != n:
                raise ValueError("slot_generation must be 1-D with graph buffer capacity")
            if self.slot_generation.dtype != torch.long:
                raise TypeError("slot_generation must be torch.long")
            if bool((self.slot_generation < 0).any().item()):
                raise ValueError("slot_generation must be nonnegative")
        if self.role is not None:
            if self.role.ndim != 1 or self.role.numel() != n:
                raise ValueError("role must be 1-D with graph buffer capacity")
            if self.role.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64):
                raise TypeError("role must be an integer tensor")

        ids = torch.where(self.valid)[0]
        if ids.numel() == 0:
            return
        s = self.src[ids]
        d = self.dst[ids]
        w = self.weight[ids]
        if bool(((s < 0) | (s >= self.num_nodes) | (d < 0) | (d >= self.num_nodes)).any().item()):
            raise ValueError("active edge endpoint out of range")
        if bool((s == d).any().item()):
            raise ValueError("active self edges are not allowed")
        if bool((~torch.isfinite(w)).any().item()):
            raise ValueError("active edge weights must be finite")
        if bool((w <= 0).any().item()):
            raise ValueError("active edge weights must be strictly positive")
        if self.length is not None:
            ell = self.length[ids]
            if bool((~torch.isfinite(ell)).any().item()):
                raise ValueError("active edge lengths must be finite")
            if bool((ell <= 0).any().item()):
                raise ValueError("active edge lengths must be strictly positive")
        if self.role is not None:
            r = self.role[ids]
            if bool(((r < 0) | (r >= len(EDGE_ROLE_ORDER))).any().item()):
                raise ValueError("active edge role code out of range")
        if check_duplicates:
            pairs = [(min(int(u), int(v)), max(int(u), int(v))) for u, v in zip(s.tolist(), d.tolist())]
            if len(set(pairs)) != len(pairs):
                raise ValueError("duplicate active undirected edges are not allowed")

    def clone(self) -> "GraphBuffers":
        return GraphBuffers(
            num_nodes=self.num_nodes,
            src=self.src.clone(),
            dst=self.dst.clone(),
            weight=self.weight.clone(),
            valid=self.valid.clone(),
            length=None if self.length is None else self.length.clone(),
            role=None if self.role is None else self.role.clone(),
            slot_generation=None if self.slot_generation is None else self.slot_generation.clone(),
            version=int(self.version),
        )

    def active(self) -> tuple[Tensor, Tensor, Tensor]:
        """Return (src, dst, weight=affinity) for active edges."""
        return self.src[self.valid], self.dst[self.valid], self.weight[self.valid]

    def active_length(self) -> Tensor:
        """Return metric length for active edges."""
        if self.length is None:
            _, _, w = self.active()
            return 1.0 / w.clamp_min(1e-30)
        return self.length[self.valid]

    def active_roles(self) -> Tensor:
        if self.role is None:
            return torch.full(
                (self.edge_count,), edge_role_code(EdgeRole.GENERIC),
                dtype=torch.long, device=self.src.device,
            )
        return self.role[self.valid].to(torch.long)

    def bump_version(self) -> None:
        self.version = int(self.version) + 1

    def state_hash(self, *, include_version: bool = True) -> str:
        """Canonical state commitment over active edge rows.

        Each canonical edge row commits (slot_index, slot_generation, u, v,
        weight=affinity, length, role) so that slot-generation metadata and
        both edge scalars are cryptographically bound. Altering
        ``slot_generation`` without a corresponding edge identity change
        produces a different hash, preventing ABA-style slot reuse from
        going undetected.
        """
        src, dst, weight = self.active()
        lengths = self.active_length()
        roles = self.active_roles()
        active_ids = torch.where(self.valid)[0]
        rows = []
        for slot, u, v, w, ell, role in zip(
            active_ids.tolist(), src.tolist(), dst.tolist(),
            weight.tolist(), lengths.tolist(), roles.tolist()
        ):
            a, b = sorted((int(u), int(v)))
            g = int(self.slot_generation[int(slot)].item()) if self.slot_generation is not None else 0
            rows.append((int(slot), g, a, b, float(w), float(ell), int(role)))
        rows.sort()
        payload = {
            "schema": "LGAE_GRAPH_STATE_V4",
            "num_nodes": int(self.num_nodes),
            "edges": rows,
        }
        if include_version:
            payload["version"] = int(self.version)
        raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
        return hashlib.sha256(raw).hexdigest()

    def to_state_dict(self) -> dict[str, Any]:
        return {
            "num_nodes": int(self.num_nodes),
            "src": self.src.detach().cpu(),
            "dst": self.dst.detach().cpu(),
            "weight": self.weight.detach().cpu(),
            "length": None if self.length is None else self.length.detach().cpu(),
            "valid": self.valid.detach().cpu(),
            "role": None if self.role is None else self.role.detach().cpu(),
            "slot_generation": None if self.slot_generation is None else self.slot_generation.detach().cpu(),
            "version": int(self.version),
            "state_hash": self.state_hash(),
        }

    @classmethod
    def from_state_dict(cls, payload: dict[str, Any], *, device=None) -> "GraphBuffers":
        cap = len(payload["src"])
        slot_gen = payload.get("slot_generation")
        length_data = payload.get("length")
        # Backward compat: if no length in payload, it will be derived from weight
        graph = cls(
            num_nodes=int(payload["num_nodes"]),
            src=torch.as_tensor(payload["src"], dtype=torch.long, device=device).clone(),
            dst=torch.as_tensor(payload["dst"], dtype=torch.long, device=device).clone(),
            weight=torch.as_tensor(payload["weight"], device=device).clone(),
            valid=torch.as_tensor(payload["valid"], dtype=torch.bool, device=device).clone(),
            length=None if length_data is None else torch.as_tensor(length_data, device=device).clone(),
            role=None if payload.get("role") is None else torch.as_tensor(payload["role"], dtype=torch.long, device=device).clone(),
            slot_generation=torch.zeros(cap, dtype=torch.long, device=device) if slot_gen is None else torch.as_tensor(slot_gen, dtype=torch.long, device=device).clone(),
            version=int(payload.get("version", 0)),
        )
        expected = payload.get("state_hash")
        if expected is not None and graph.state_hash() != expected:
            raise ValueError("graph state hash mismatch during restore")
        return graph


@dataclass(slots=True)
class AuditSnapshot:
    lambda2: float
    operator_discrepancy: float
    integral_lly_deficit: float | None = None
    weak_entropic_min: float | None = None
    bakry_min: float | None = None
    cde_residual: float | None = None
    topology_signature: dict[str, float] = field(default_factory=dict)
    details: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class MutationResult:
    decision: MutationDecision
    reasons: list[str]
    before: AuditSnapshot | None = None
    after: AuditSnapshot | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


def make_graph_buffers(
    num_nodes: int,
    edges: list[tuple[int, int]] | list[tuple[int, int, float]] | list[tuple[int, int, float, float]],
    *,
    capacity: int | None = None,
    roles: Iterable[EdgeRole | str | int] | None = None,
    device=None,
    dtype: torch.dtype = torch.float32,
) -> GraphBuffers:
    """Create fixed-capacity graph buffers from an undirected edge list.

    Edge tuples may be:
    - ``(u, v)`` — uniform affinity=1, length=1
    - ``(u, v, w)`` — affinity=w, length=1/w (inverse default)
    - ``(u, v, a, ell)`` — affinity=a, length=ell (explicit metric-measure)
    """
    if num_nodes <= 0:
        raise ValueError("num_nodes must be positive")
    cap = max(capacity or len(edges), len(edges))
    src = torch.zeros(cap, dtype=torch.long, device=device)
    dst = torch.zeros(cap, dtype=torch.long, device=device)
    weight = torch.zeros(cap, dtype=dtype, device=device)
    length = torch.zeros(cap, dtype=dtype, device=device)
    valid = torch.zeros(cap, dtype=torch.bool, device=device)
    role_tensor = torch.full((cap,), edge_role_code(EdgeRole.GENERIC), dtype=torch.long, device=device)
    slot_gen = torch.zeros(cap, dtype=torch.long, device=device)
    if len(edges) > 0:
        slot_gen[:len(edges)] = 1
    role_list = list(roles) if roles is not None else None
    if role_list is not None and len(role_list) != len(edges):
        raise ValueError("roles length must match edges length")
    seen: set[tuple[int, int]] = set()
    for i, e in enumerate(edges):
        if len(e) == 2:
            u, v = e
            af = 1.0
            ell = 1.0
        elif len(e) == 3:
            u, v, w = e
            af = float(w)
            ell = 1.0 / af if af > 0 else 1.0  # inverse default
        else:
            u, v, af, ell = e
            af = float(af)
            ell = float(ell)
        u, v = int(u), int(v)
        if not (0 <= u < num_nodes and 0 <= v < num_nodes):
            raise ValueError("edge endpoint out of range")
        if u == v:
            raise ValueError("self edge not allowed")
        if not math.isfinite(af) or af <= 0:
            raise ValueError("edge affinity must be finite and positive")
        if not math.isfinite(ell) or ell <= 0:
            raise ValueError("edge length must be finite and positive")
        key = (min(u, v), max(u, v))
        if key in seen:
            raise ValueError("duplicate undirected edge")
        seen.add(key)
        src[i] = u
        dst[i] = v
        weight[i] = af
        length[i] = ell
        valid[i] = True
        if role_list is not None:
            role_tensor[i] = edge_role_code(role_list[i])
    return GraphBuffers(num_nodes, src, dst, weight, valid, length=length, role=role_tensor, slot_generation=slot_gen, version=0)


def round_edge_capacity(edge_count: int, bucket_size: int = 256, reserve_buckets: int = 1) -> int:
    """Round edge storage to fixed-capacity buckets for compile-stable mutation."""
    if edge_count < 0 or bucket_size <= 0 or reserve_buckets < 0:
        raise ValueError("invalid edge bucket parameters")
    base = ((max(int(edge_count), 1) + int(bucket_size) - 1) // int(bucket_size)) * int(bucket_size)
    return base + int(reserve_buckets) * int(bucket_size)


def make_bucketed_graph_buffers(
    num_nodes: int,
    edges: list[tuple[int, int]] | list[tuple[int, int, float]] | list[tuple[int, int, float, float]],
    *,
    bucket_size: int = 256,
    reserve_buckets: int = 1,
    roles: Iterable[EdgeRole | str | int] | None = None,
    device=None,
    dtype: torch.dtype = torch.float32,
) -> GraphBuffers:
    capacity = round_edge_capacity(len(edges), bucket_size=bucket_size, reserve_buckets=reserve_buckets)
    return make_graph_buffers(num_nodes, edges, capacity=capacity, roles=roles, device=device, dtype=dtype)
