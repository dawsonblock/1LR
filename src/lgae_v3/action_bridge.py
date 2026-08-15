"""v5.1 Bridge between StructuralAction proposals and concrete mutations.

The structural executive proposes actions as `StructuralAction` enum values.
The governor certifies concrete mutation objects (AddEdge, PruneEdge, etc.).
This module bridges the two worlds:

    StructuralAction → concrete Mutation object → governor.evaluate_mutation

This keeps the executive as a pure proposal generator while ensuring
all proposals go through the authoritative governor for certification.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor

from .benchmark.tasks import StructuralAction
from .mutations import (
    AddEdge, PruneEdge, ReweightAffinity, ReweightLength,
    CoupledReweight,
)
from .types import GraphBuffers, MutationResult, MutationDecision


@dataclass
class ActionBridgeResult:
    """Result of bridging a StructuralAction through the governor."""
    action: StructuralAction
    mutation: Any | None         # The concrete mutation object, or None if NO_OP
    governor_result: MutationResult | None
    shadow_graph: GraphBuffers | None
    executed: bool
    reasons: list[str]
    metadata: dict[str, Any] | None = None


def action_to_mutation(
    action: StructuralAction,
    graph: GraphBuffers,
    z: Tensor,
    **kwargs: Any,
) -> Any | None:
    """Convert a StructuralAction to a concrete mutation object.

    Args:
        action: The proposed structural action
        graph: Current graph state (for selecting edges/nodes)
        z: Current latent state
        **kwargs: Additional parameters (e.g., u, v, weight, factor)

    Returns:
        A mutation object, or None for NO_OP / actions that don't map
        to a single mutation.
    """
    if action == StructuralAction.NO_OP:
        return None

    elif action == StructuralAction.ADD_EDGE:
        u = kwargs.get("u")
        v = kwargs.get("v")
        if u is None or v is None:
            # Pick a reasonable default: connect two disconnected nodes
            u, v = _find_disconnected_pair(graph)
        if u is None:
            return None
        weight = kwargs.get("weight", 1.0)
        return AddEdge(u, v, weight=weight)

    elif action == StructuralAction.PRUNE_EDGE:
        u = kwargs.get("u")
        v = kwargs.get("v")
        if u is None or v is None:
            # Pick the weakest edge
            u, v = _find_weakest_edge(graph)
        if u is None:
            return None
        return PruneEdge(u, v)

    elif action == StructuralAction.REWEIGHT_AFFINITY:
        u = kwargs.get("u")
        v = kwargs.get("v")
        if u is None or v is None:
            u, v = _find_weakest_edge(graph)
        if u is None:
            return None
        factor = kwargs.get("factor", 2.0)
        return ReweightAffinity(u, v, factor=factor)

    elif action == StructuralAction.REWEIGHT_LENGTH:
        u = kwargs.get("u")
        v = kwargs.get("v")
        if u is None or v is None:
            u, v = _find_longest_edge(graph)
        if u is None:
            return None
        factor = kwargs.get("factor", 0.5)
        return ReweightLength(u, v, factor=factor)

    elif action == StructuralAction.COUPLED_REWEIGHT:
        u = kwargs.get("u")
        v = kwargs.get("v")
        if u is None or v is None:
            u, v = _find_weakest_edge(graph)
        if u is None:
            return None
        affinity_factor = kwargs.get("factor", 2.0)
        coupling = kwargs.get("coupling", "inverse")
        return CoupledReweight(u, v, affinity_factor=affinity_factor, coupling=coupling)

    elif action in (StructuralAction.SPAWN_FIBER, StructuralAction.PRUNE_FIBER,
                    StructuralAction.CHANGE_GAUGE):
        # These don't map to edge mutations; they require fiber/gauge operations
        return None

    return None


def certify_action_through_governor(
    action: StructuralAction,
    graph: GraphBuffers,
    z: Tensor,
    governor: Any,
    *,
    seed: int = 0,
    gauge_bank: Any = None,
    **kwargs: Any,
) -> ActionBridgeResult:
    """Bridge a StructuralAction through the governor for certification.

    This is the critical integration point: the executive proposes, the
    governor certifies. The governor's decision is authoritative.

    Args:
        action: The proposed structural action
        graph: Current graph state
        z: Current latent state
        governor: The GeometryGovernor instance
        seed: Random seed for the governor's audit
        gauge_bank: Optional gauge bank for shadow rollout
        **kwargs: Parameters for the mutation (u, v, weight, etc.)

    Returns:
        ActionBridgeResult with the governor's decision
    """
    # Convert action to mutation
    mutation = action_to_mutation(action, graph, z, **kwargs)

    if mutation is None:
        # NO_OP or unmappable action
        return ActionBridgeResult(
            action=action,
            mutation=None,
            governor_result=None,
            shadow_graph=None,
            executed=False,
            reasons=["no_op_or_unmappable"],
        )

    # Submit to governor for certification
    result, shadow_graph = governor.evaluate_mutation(
        graph, z, mutation, seed=seed, gauge_bank=gauge_bank,
    )

    executed = result.decision == MutationDecision.ACCEPT

    return ActionBridgeResult(
        action=action,
        mutation=mutation,
        governor_result=result,
        shadow_graph=shadow_graph,
        executed=executed,
        reasons=list(result.reasons),
        metadata=result.metadata,
    )


def _find_disconnected_pair(graph: GraphBuffers) -> tuple[int | None, int | None]:
    """Find two nodes that are not directly connected."""
    N = graph.num_nodes
    valid = graph.valid.bool()
    edges = set()
    for i in range(graph.src.shape[0]):
        if valid[i]:
            u, v = int(graph.src[i]), int(graph.dst[i])
            edges.add((min(u, v), max(u, v)))

    for u in range(N):
        for v in range(u + 1, N):
            if (u, v) not in edges:
                return u, v
    return None, None


def _find_weakest_edge(graph: GraphBuffers) -> tuple[int | None, int | None]:
    """Find the edge with the lowest weight."""
    valid = graph.valid.bool()
    if not valid.any():
        return None, None

    weights = graph.weight[valid]
    src = graph.src[valid]
    dst = graph.dst[valid]

    min_idx = int(weights.argmin().item())
    return int(src[min_idx]), int(dst[min_idx])


def _find_longest_edge(graph: GraphBuffers) -> tuple[int | None, int | None]:
    """Find the edge with the longest metric length."""
    valid = graph.valid.bool()
    if not valid.any():
        return None, None

    if graph.length is None:
        # Fall back to weakest edge
        return _find_weakest_edge(graph)

    lengths = graph.length[valid]
    src = graph.src[valid]
    dst = graph.dst[valid]

    max_idx = int(lengths.argmax().item())
    return int(src[max_idx]), int(dst[max_idx])
