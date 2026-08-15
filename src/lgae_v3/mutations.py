from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Protocol
import math

import torch

from .types import EdgeRole, GraphBuffers, edge_role_code


class GraphMutation(Protocol):
    name: str
    def apply(self, graph: GraphBuffers) -> dict: ...


class StructuralMutation(Protocol):
    """v4.1.3: Protocol for mutations that change structural state.

    Both graph and fiber mutations implement this protocol, allowing
    the governor to apply the same multi-horizon certification to both.

    Implementations:
    - AddEdge, PruneEdge, ReweightAffinity, ReweightLength, CoupledReweight
    - FiberBirth, FiberDeath (future)
    """
    name: str
    def apply(self, graph: GraphBuffers) -> dict: ...
    def touched_region(self) -> set[int]:
        """Return the set of node indices touched by this mutation."""
        ...


def canonical_edge(u: int, v: int) -> tuple[int, int]:
    return (min(int(u), int(v)), max(int(u), int(v)))


def _validate_endpoint(graph: GraphBuffers, u: int, v: int) -> None:
    if not (0 <= int(u) < graph.num_nodes and 0 <= int(v) < graph.num_nodes):
        raise ValueError("edge endpoint out of range")
    if int(u) == int(v):
        raise ValueError("self edge not allowed")


def _find_edge(graph: GraphBuffers, u: int, v: int) -> torch.Tensor:
    return torch.where(graph.valid & (((graph.src == u) & (graph.dst == v)) | ((graph.src == v) & (graph.dst == u))))[0]


@dataclass(slots=True)
class AddEdge:
    u: int
    v: int
    weight: float = 1.0  # affinity
    length: float | None = None  # metric length; defaults to 1/weight
    role: EdgeRole | str | int = EdgeRole.GENERIC
    name: str = "add_edge"

    def apply(self, graph: GraphBuffers) -> dict:
        _validate_endpoint(graph, self.u, self.v)
        w = float(self.weight)
        if not math.isfinite(w) or w <= 0:
            raise ValueError("edge weight (affinity) must be finite and positive")
        ell = float(self.length) if self.length is not None else (1.0 / w)
        if not math.isfinite(ell) or ell <= 0:
            raise ValueError("edge length must be finite and positive")
        role_code = edge_role_code(self.role)
        existing = _find_edge(graph, self.u, self.v)
        if existing.numel():
            i = int(existing[0].item())
            new_w = float(graph.weight[i].item()) + w
            if not math.isfinite(new_w) or new_w <= 0:
                raise ValueError("reweighted edge became invalid")
            graph.weight[i] = new_w
            if graph.length is not None:
                # When merging, use harmonic mean of lengths (resistors in parallel)
                old_ell = float(graph.length[i].item())
                if old_ell > 0 and ell > 0:
                    graph.length[i] = 1.0 / (1.0 / old_ell + 1.0 / ell)
            if graph.role is not None:
                graph.role[i] = role_code
            graph.bump_version()
            graph.validate()
            return {"slot": i, "reweighted_existing": True, "new_weight": new_w, "role": role_code, "affected_edges": [canonical_edge(self.u, self.v)]}
        free = torch.where(~graph.valid)[0]
        if not free.numel():
            raise RuntimeError("graph edge buffer capacity exhausted")
        i = int(free[0].item())
        graph.src[i] = int(self.u)
        graph.dst[i] = int(self.v)
        graph.weight[i] = w
        if graph.length is not None:
            graph.length[i] = ell
        graph.valid[i] = True
        if graph.slot_generation is not None:
            graph.slot_generation[i] += 1
        if graph.role is not None:
            graph.role[i] = role_code
        graph.bump_version()
        graph.validate()
        gen = int(graph.slot_generation[i].item()) if graph.slot_generation is not None else 0
        return {"slot": i, "slot_generation": gen, "reweighted_existing": False, "role": role_code, "affected_edges": [canonical_edge(self.u, self.v)]}

    def touched_region(self) -> set[int]:
        return {int(self.u), int(self.v)}


@dataclass(slots=True)
class ReweightEdge:
    """Reweight affinity. When length is present, inverse-update it: factor>1
    (stronger connection) shortens length by 1/factor."""
    u: int
    v: int
    factor: float = 1.1
    min_weight: float = 1e-3
    max_weight: float = 10.0
    name: str = "reweight_edge"

    def apply(self, graph: GraphBuffers) -> dict:
        _validate_endpoint(graph, self.u, self.v)
        factor = float(self.factor)
        if not math.isfinite(factor) or factor <= 0:
            raise ValueError("reweight factor must be finite and positive")
        if not (math.isfinite(self.min_weight) and math.isfinite(self.max_weight) and 0 < self.min_weight <= self.max_weight):
            raise ValueError("invalid weight clamp")
        ids = _find_edge(graph, self.u, self.v)
        if not ids.numel():
            raise ValueError("edge not found")
        i = int(ids[0].item())
        graph.weight[i] = torch.clamp(graph.weight[i] * factor, self.min_weight, self.max_weight)
        if graph.length is not None:
            # Inverse update: stronger affinity → shorter length
            graph.length[i] = torch.clamp(graph.length[i] / factor, 1.0 / self.max_weight, 1.0 / self.min_weight)
        graph.bump_version()
        graph.validate()
        return {"slot": i, "new_weight": float(graph.weight[i].item()), "affected_edges": [canonical_edge(self.u, self.v)]}

    def touched_region(self) -> set[int]:
        return {int(self.u), int(self.v)}


@dataclass(slots=True)
class ReweightAffinity:
    """Reweight only the affinity/conductance field, leaving metric length unchanged.

    This is the v4.1.1 split: affinity and length are independent fields.
    Use this mutation when the diffusion measure should change without
    altering the shortest-path metric.
    """
    u: int
    v: int
    factor: float = 1.1
    min_weight: float = 1e-3
    max_weight: float = 10.0
    name: str = "reweight_affinity"

    def apply(self, graph: GraphBuffers) -> dict:
        _validate_endpoint(graph, self.u, self.v)
        factor = float(self.factor)
        if not math.isfinite(factor) or factor <= 0:
            raise ValueError("reweight factor must be finite and positive")
        if not (math.isfinite(self.min_weight) and math.isfinite(self.max_weight) and 0 < self.min_weight <= self.max_weight):
            raise ValueError("invalid weight clamp")
        ids = _find_edge(graph, self.u, self.v)
        if not ids.numel():
            raise ValueError("edge not found")
        i = int(ids[0].item())
        graph.weight[i] = torch.clamp(graph.weight[i] * factor, self.min_weight, self.max_weight)
        # Length is NOT modified — affinity and length are independent
        graph.bump_version()
        graph.validate()
        return {"slot": i, "new_weight": float(graph.weight[i].item()), "field": "affinity", "affected_edges": [canonical_edge(self.u, self.v)]}

    def touched_region(self) -> set[int]:
        return {int(self.u), int(self.v)}


@dataclass(slots=True)
class ReweightLength:
    """Reweight only the metric length field, leaving affinity unchanged.

    Use this mutation when the shortest-path metric should change without
    altering the diffusion measure.
    """
    u: int
    v: int
    factor: float = 1.1
    min_length: float = 1e-3
    max_length: float = 1e6
    name: str = "reweight_length"

    def apply(self, graph: GraphBuffers) -> dict:
        _validate_endpoint(graph, self.u, self.v)
        factor = float(self.factor)
        if not math.isfinite(factor) or factor <= 0:
            raise ValueError("reweight factor must be finite and positive")
        if not (math.isfinite(self.min_length) and math.isfinite(self.max_length) and 0 < self.min_length <= self.max_length):
            raise ValueError("invalid length clamp")
        ids = _find_edge(graph, self.u, self.v)
        if not ids.numel():
            raise ValueError("edge not found")
        i = int(ids[0].item())
        if graph.length is None:
            raise ValueError("graph has no length field; cannot reweight length")
        graph.length[i] = torch.clamp(graph.length[i] * factor, self.min_length, self.max_length)
        # Affinity is NOT modified
        graph.bump_version()
        graph.validate()
        return {"slot": i, "new_length": float(graph.length[i].item()), "field": "length", "affected_edges": [canonical_edge(self.u, self.v)]}

    def touched_region(self) -> set[int]:
        return {int(self.u), int(self.v)}


@dataclass(slots=True)
class CoupledReweight:
    """Reweight both affinity and length with an explicit coupling policy.

    The coupling factor controls how length responds to affinity changes:
    - coupling="inverse": stronger affinity → shorter length (legacy behavior)
    - coupling="direct": stronger affinity → longer length
    - coupling="none": length unchanged (equivalent to ReweightAffinity)
    """
    u: int
    v: int
    affinity_factor: float = 1.1
    length_factor: float | None = None  # if None, derived from coupling
    coupling: str = "inverse"
    min_weight: float = 1e-3
    max_weight: float = 10.0
    min_length: float = 1e-3
    max_length: float = 1e6
    name: str = "coupled_reweight"

    def apply(self, graph: GraphBuffers) -> dict:
        _validate_endpoint(graph, self.u, self.v)
        af = float(self.affinity_factor)
        if not math.isfinite(af) or af <= 0:
            raise ValueError("affinity_factor must be finite and positive")
        if self.coupling not in ("inverse", "direct", "none"):
            raise ValueError("coupling must be 'inverse', 'direct', or 'none'")
        ids = _find_edge(graph, self.u, self.v)
        if not ids.numel():
            raise ValueError("edge not found")
        i = int(ids[0].item())
        graph.weight[i] = torch.clamp(graph.weight[i] * af, self.min_weight, self.max_weight)
        new_length = None
        if graph.length is not None:
            if self.length_factor is not None:
                lf = float(self.length_factor)
            elif self.coupling == "inverse":
                lf = 1.0 / af
            elif self.coupling == "direct":
                lf = af
            else:
                lf = 1.0
            if math.isfinite(lf) and lf > 0:
                graph.length[i] = torch.clamp(graph.length[i] * lf, self.min_length, self.max_length)
                new_length = float(graph.length[i].item())
        graph.bump_version()
        graph.validate()
        return {
            "slot": i,
            "new_weight": float(graph.weight[i].item()),
            "new_length": new_length,
            "coupling": self.coupling,
            "affected_edges": [canonical_edge(self.u, self.v)],
        }

    def touched_region(self) -> set[int]:
        return {int(self.u), int(self.v)}


@dataclass(slots=True)
class PruneEdge:
    u: int
    v: int
    name: str = "prune_edge"

    def apply(self, graph: GraphBuffers) -> dict:
        _validate_endpoint(graph, self.u, self.v)
        ids = _find_edge(graph, self.u, self.v)
        if not ids.numel():
            raise ValueError("edge not found")
        i = int(ids[0].item())
        graph.valid[i] = False
        graph.weight[i] = 0.0
        if graph.length is not None:
            graph.length[i] = 0.0
        if graph.slot_generation is not None:
            graph.slot_generation[i] += 1
        graph.bump_version()
        graph.validate()
        gen = int(graph.slot_generation[i].item()) if graph.slot_generation is not None else 0
        return {"slot": i, "slot_generation": gen, "affected_edges": [canonical_edge(self.u, self.v)]}


@dataclass(slots=True)
class RicciFlowReweight:
    """Log-conformal Ricci-flow update over existing edge slots.

    The update is w' = clamp(w * exp(-dt * (kappa-target)), [min,max]), so positive
    weights remain positive by construction. ``curvatures`` uses canonical undirected
    edge tuples and may cover only a subset of active edges.

    ``target_field`` selects which edge scalar the flow modifies:
    - ``"weight"`` (default, backward compat): modifies affinity/conductance.
    - ``"length"``: modifies metric length (geometrically canonical for Ricci flow).
    When one field is modified, the other is inverse-updated to preserve the
    default relationship unless ``coupled=False``.
    """
    curvatures: dict[tuple[int, int], float]
    target_curvature: float = 0.0
    dt: float = 0.05
    min_weight: float = 1e-3
    max_weight: float = 10.0
    target_field: str = "weight"  # "weight" (affinity) or "length"
    coupled: bool = True  # inverse-update the other field
    name: str = "ricci_flow_reweight"

    def apply(self, graph: GraphBuffers) -> dict:
        if not (math.isfinite(self.dt) and self.dt > 0):
            raise ValueError("dt must be finite and positive")
        if not (math.isfinite(self.target_curvature) and 0 < self.min_weight <= self.max_weight):
            raise ValueError("invalid Ricci-flow parameters")
        if self.target_field not in ("weight", "length"):
            raise ValueError("target_field must be 'weight' or 'length'")
        changed: list[tuple[int, int]] = []
        slots: list[int] = []
        for edge, kval in self.curvatures.items():
            u, v = canonical_edge(*edge)
            _validate_endpoint(graph, u, v)
            kappa = float(kval)
            if not math.isfinite(kappa):
                raise ValueError("curvature field must be finite")
            ids = _find_edge(graph, u, v)
            if not ids.numel():
                continue
            i = int(ids[0].item())
            exponent = max(-50.0, min(50.0, -float(self.dt) * (kappa - float(self.target_curvature))))
            factor = math.exp(exponent)
            if self.target_field == "weight":
                new_w = float(graph.weight[i].item()) * factor
                new_w = max(float(self.min_weight), min(float(self.max_weight), new_w))
                graph.weight[i] = new_w
                if self.coupled and graph.length is not None:
                    graph.length[i] = torch.clamp(
                        graph.length[i] / factor,
                        1.0 / self.max_weight, 1.0 / self.min_weight,
                    )
            else:  # target_field == "length"
                if graph.length is None:
                    continue
                new_ell = float(graph.length[i].item()) * factor
                new_ell = max(1.0 / self.max_weight, min(1.0 / self.min_weight, new_ell))
                graph.length[i] = new_ell
                if self.coupled:
                    graph.weight[i] = torch.clamp(
                        graph.weight[i] / factor, self.min_weight, self.max_weight,
                    )
            changed.append((u, v))
            slots.append(i)
        if changed:
            graph.bump_version()
            graph.validate()
        return {"slots": slots, "affected_edges": changed, "updated_edges": len(changed)}

    def touched_region(self) -> set[int]:
        """Return all endpoints touched by this Ricci flow update."""
        region: set[int] = set()
        for (u, v) in self.curvatures:
            region.add(int(u))
            region.add(int(v))
        return region


def affected_edges(mutation: Any) -> list[tuple[int, int]]:
    if isinstance(mutation, (AddEdge, ReweightEdge, PruneEdge)):
        return [canonical_edge(mutation.u, mutation.v)]
    if isinstance(mutation, RicciFlowReweight):
        return [canonical_edge(*e) for e in mutation.curvatures]
    return []


@dataclass(slots=True)
class MutationCooldownTracker:
    cooldown_steps: int = 20
    last_modified: dict[tuple[int, int], int] = field(default_factory=dict)

    def remaining(self, u: int, v: int, step: int) -> int:
        last = self.last_modified.get(canonical_edge(u, v))
        if last is None:
            return 0
        return max(0, int(self.cooldown_steps) - (int(step) - int(last)))

    def allows(self, mutation: Any, step: int) -> tuple[bool, dict[tuple[int, int], int]]:
        blocked: dict[tuple[int, int], int] = {}
        for e in affected_edges(mutation):
            rem = self.remaining(*e, step)
            if rem > 0:
                blocked[e] = rem
        return (not blocked), blocked

    def record(self, mutation: Any, step: int) -> None:
        for e in affected_edges(mutation):
            self.last_modified[e] = int(step)

    def surgery_action(self, curvature: float, *, add_threshold: float, deadband: float, prune_threshold: float) -> str | None:
        k = float(curvature)
        if k < float(add_threshold):
            return "add"
        if k > float(prune_threshold):
            return "prune"
        if -float(deadband) <= k <= float(deadband):
            return None
        return None

    def to_state_dict(self) -> dict[str, Any]:
        return {
            "cooldown_steps": int(self.cooldown_steps),
            "last_modified": [[u, v, int(step)] for (u, v), step in sorted(self.last_modified.items())],
        }

    @classmethod
    def from_state_dict(cls, payload: dict[str, Any]) -> "MutationCooldownTracker":
        obj = cls(cooldown_steps=int(payload.get("cooldown_steps", 20)))
        for u, v, step in payload.get("last_modified", []):
            obj.last_modified[canonical_edge(u, v)] = int(step)
        return obj


def mutation_to_spec(mutation: Any) -> dict[str, Any]:
    if isinstance(mutation, RicciFlowReweight):
        return {
            "type": "RicciFlowReweight",
            "curvatures": [[u, v, float(k)] for (u, v), k in sorted((canonical_edge(*e), v) for e, v in mutation.curvatures.items())],
            "target_curvature": float(mutation.target_curvature),
            "dt": float(mutation.dt),
            "min_weight": float(mutation.min_weight),
            "max_weight": float(mutation.max_weight),
            "target_field": str(mutation.target_field),
            "coupled": bool(mutation.coupled),
            "name": mutation.name,
        }
    if not isinstance(mutation, (AddEdge, ReweightEdge, ReweightAffinity, ReweightLength, CoupledReweight, PruneEdge)):
        raise TypeError(f"unsupported mutation type: {type(mutation).__name__}")
    payload = asdict(mutation)
    payload["type"] = type(mutation).__name__
    if isinstance(payload.get("role"), EdgeRole):
        payload["role"] = payload["role"].value
    return payload


def mutation_from_spec(payload: dict[str, Any]):
    data = dict(payload)
    kind = data.pop("type")
    if kind == "AddEdge":
        return AddEdge(**data)
    if kind == "ReweightEdge":
        return ReweightEdge(**data)
    if kind == "ReweightAffinity":
        return ReweightAffinity(**data)
    if kind == "ReweightLength":
        return ReweightLength(**data)
    if kind == "CoupledReweight":
        return CoupledReweight(**data)
    if kind == "PruneEdge":
        return PruneEdge(**data)
    if kind == "RicciFlowReweight":
        rows = data.pop("curvatures")
        data["curvatures"] = {canonical_edge(int(u), int(v)): float(k) for u, v, k in rows}
        # Backward compat: pre-v4.1 specs don't have target_field/coupled
        data.setdefault("target_field", "weight")
        data.setdefault("coupled", True)
        return RicciFlowReweight(**data)
    raise ValueError(f"unknown mutation type: {kind}")
