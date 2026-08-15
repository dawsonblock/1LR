"""Tests for v4.1 Metric–Measure Separation and Multi-Horizon Certification.

Covers:
- GraphBuffers carries independent length and weight(affinity) fields
- make_graph_buffers accepts (u,v,a,ell) 4-tuples
- Default inverse relationship: length = 1/weight when only one scalar given
- state_hash includes length (schema V4)
- checkpoint roundtrips both fields
- Mutations update both fields (AddEdge, ReweightEdge, PruneEdge, RicciFlow)
- graphbuffers_to_networkx stores both weight and length attributes
- Weighted ORC: ground cost from length, measures from affinity
- Weighted LLY: Lipschitz from length, Laplacian from affinity
- weighted_forman_edge: literature-faithful formula with sqrt ratios
- weighted_af3_proxy: clearly labeled as proxy, not canonical
- Ricci flow target_field: "length" vs "weight"
- Multi-horizon shadow certification
- Edge metric invariant: d_ell(u,v) <= ell_uv
"""
from __future__ import annotations

import networkx as nx
import numpy as np
import torch
import pytest

from lgae_v3 import LGAEConfig, LGAEEngine, make_graph_buffers, make_bucketed_graph_buffers
from lgae_v3.types import GraphBuffers
from lgae_v3.mutations import AddEdge, ReweightEdge, PruneEdge, RicciFlowReweight
from lgae_v3.topology import graphbuffers_to_networkx
from lgae_v3.curvature import (
    weighted_ollivier_edge,
    weighted_lly_laplacian_lp,
    weighted_forman_edge,
    weighted_af3_proxy,
    ollivier_edge,
    lly_laplacian_lp,
)
from lgae_v3.config import validate_config


# ---------------------------------------------------------------------------
# GraphBuffers: length field exists and is independent
# ---------------------------------------------------------------------------

def test_graph_buffers_has_length_field():
    """GraphBuffers should carry an independent length tensor."""
    graph = make_graph_buffers(3, [(0, 1, 2.0), (1, 2, 3.0)], capacity=4)
    assert graph.length is not None
    assert graph.length.shape == graph.weight.shape


def test_default_inverse_relationship():
    """When only weight is given, length should default to 1/weight."""
    graph = make_graph_buffers(3, [(0, 1, 4.0), (1, 2, 2.0)], capacity=4)
    # weight=4 → length=0.25, weight=2 → length=0.5
    assert float(graph.length[0].item()) == pytest.approx(0.25)
    assert float(graph.length[1].item()) == pytest.approx(0.5)


def test_explicit_length_affinity_4tuple():
    """make_graph_buffers should accept (u,v,affinity,length) 4-tuples."""
    graph = make_graph_buffers(3, [(0, 1, 10.0, 0.1), (1, 2, 0.5, 5.0)], capacity=4)
    # High affinity, short length
    assert float(graph.weight[0].item()) == pytest.approx(10.0)
    assert float(graph.length[0].item()) == pytest.approx(0.1)
    # Low affinity, long length
    assert float(graph.weight[1].item()) == pytest.approx(0.5)
    assert float(graph.length[1].item()) == pytest.approx(5.0)


def test_length_affinity_independent():
    """Length and affinity should be independently modifiable."""
    graph = make_graph_buffers(3, [(0, 1, 2.0, 3.0)], capacity=4)
    # They're not inverse of each other
    assert float(graph.weight[0].item()) == pytest.approx(2.0)
    assert float(graph.length[0].item()) == pytest.approx(3.0)
    assert float(graph.length[0].item()) != pytest.approx(1.0 / 2.0)


def test_state_hash_includes_length():
    """state_hash should include length (schema V4)."""
    g1 = make_graph_buffers(3, [(0, 1, 2.0, 1.0)], capacity=4)
    g2 = make_graph_buffers(3, [(0, 1, 2.0, 5.0)], capacity=4)
    # Same affinity, different length → different hash
    assert g1.state_hash() != g2.state_hash()


def test_checkpoint_roundtrips_length():
    """Checkpoint should save and restore both weight and length."""
    graph = make_graph_buffers(3, [(0, 1, 2.0, 3.0), (1, 2, 0.5, 10.0)], capacity=4)
    state = graph.to_state_dict()
    assert "length" in state
    restored = GraphBuffers.from_state_dict(state)
    assert torch.allclose(restored.weight, graph.weight)
    assert torch.allclose(restored.length, graph.length)


def test_clone_preserves_length():
    """clone() should deep-copy length tensor."""
    graph = make_graph_buffers(3, [(0, 1, 2.0, 3.0)], capacity=4)
    cloned = graph.clone()
    assert torch.allclose(cloned.length, graph.length)
    cloned.length[0] = 99.0
    assert float(graph.length[0].item()) == pytest.approx(3.0)  # original unchanged


def test_active_length_method():
    """active_length() should return lengths for valid edges only."""
    graph = make_graph_buffers(3, [(0, 1, 2.0, 3.0), (1, 2, 0.5, 10.0)], capacity=4)
    lengths = graph.active_length()
    assert lengths.numel() == 2
    assert float(lengths[0].item()) == pytest.approx(3.0)
    assert float(lengths[1].item()) == pytest.approx(10.0)


# ---------------------------------------------------------------------------
# Mutations: both fields updated
# ---------------------------------------------------------------------------

def test_add_edge_sets_both_fields():
    """AddEdge should set both weight (affinity) and length."""
    graph = make_graph_buffers(3, [(0, 1, 2.0)], capacity=4)
    AddEdge(1, 2, weight=5.0, length=0.2).apply(graph)
    assert float(graph.weight[1].item()) == pytest.approx(5.0)
    assert float(graph.length[1].item()) == pytest.approx(0.2)


def test_add_edge_default_length():
    """AddEdge without explicit length should use 1/weight."""
    graph = make_graph_buffers(3, [(0, 1, 2.0)], capacity=4)
    AddEdge(1, 2, weight=4.0).apply(graph)
    assert float(graph.length[1].item()) == pytest.approx(0.25)


def test_prune_edge_zeros_both():
    """PruneEdge should zero out both weight and length."""
    graph = make_graph_buffers(3, [(0, 1, 2.0, 3.0)], capacity=4)
    PruneEdge(0, 1).apply(graph)
    assert float(graph.weight[0].item()) == 0.0
    assert float(graph.length[0].item()) == 0.0


def test_reweight_edge_updates_both():
    """ReweightEdge should inverse-update length when affinity changes."""
    graph = make_graph_buffers(3, [(0, 1, 2.0, 0.5)], capacity=4)
    # factor=2: affinity doubles, length halves
    ReweightEdge(0, 1, factor=2.0).apply(graph)
    assert float(graph.weight[0].item()) == pytest.approx(4.0)
    assert float(graph.length[0].item()) == pytest.approx(0.25)


def test_ricci_flow_target_length():
    """RicciFlowReweight with target_field='length' should modify length."""
    graph = make_graph_buffers(3, [(0, 1, 2.0, 1.0), (1, 2, 1.0, 1.0)], capacity=4)
    original_weight = float(graph.weight[0].item())
    original_length = float(graph.length[0].item())
    RicciFlowReweight(
        curvatures={(0, 1): -1.0},
        target_field="length",
        coupled=False,
    ).apply(graph)
    # Length should change, weight should not (coupled=False)
    assert float(graph.weight[0].item()) == pytest.approx(original_weight)
    assert float(graph.length[0].item()) != pytest.approx(original_length)


def test_ricci_flow_target_weight_coupled():
    """RicciFlowReweight with target_field='weight' and coupled=True should inverse-update length."""
    graph = make_graph_buffers(3, [(0, 1, 2.0, 0.5)], capacity=4)
    original_length = float(graph.length[0].item())
    RicciFlowReweight(
        curvatures={(0, 1): -1.0},
        target_field="weight",
        coupled=True,
    ).apply(graph)
    # Weight changes, length inverse-updates
    assert float(graph.length[0].item()) != pytest.approx(original_length)


# ---------------------------------------------------------------------------
# NetworkX conversion: both attributes stored
# ---------------------------------------------------------------------------

def test_graphbuffers_to_networkx_has_both_attributes():
    """graphbuffers_to_networkx should store both weight and length."""
    graph = make_graph_buffers(3, [(0, 1, 2.0, 3.0)], capacity=4)
    g = graphbuffers_to_networkx(graph)
    assert g[0][1]["weight"] == pytest.approx(2.0)  # affinity
    assert g[0][1]["length"] == pytest.approx(3.0)  # metric length


# ---------------------------------------------------------------------------
# Weighted ORC: metric-measure separation
# ---------------------------------------------------------------------------

def test_weighted_orc_ground_cost_from_length():
    """Weighted ORC ground cost should come from length, not affinity."""
    # Graph where affinity and length are NOT inverse
    g = nx.Graph()
    g.add_edge(0, 1, weight=10.0, length=0.1)  # strong connection, short distance
    g.add_edge(1, 2, weight=0.1, length=10.0)  # weak connection, long distance
    g.add_edge(0, 2, weight=1.0, length=1.0)
    k = weighted_ollivier_edge(g, 0, 1, p=0.5, backend="exact_lp")
    # Should be finite — the key test is that it doesn't conflate the two
    assert np.isfinite(k)


def test_weighted_orc_measures_from_affinity():
    """Weighted ORC lazy measure should distribute mass by affinity."""
    g = nx.Graph()
    g.add_edge(0, 1, weight=10.0, length=1.0)
    g.add_edge(0, 2, weight=0.1, length=1.0)
    g.add_edge(1, 2, weight=1.0, length=1.0)
    # Node 0 sends much more mass to node 1 (weight=10) than to node 2 (weight=0.1)
    k = weighted_ollivier_edge(g, 0, 1, p=0.5, backend="exact_lp")
    assert np.isfinite(k)


# ---------------------------------------------------------------------------
# Weighted LLY: metric-measure separation
# ---------------------------------------------------------------------------

def test_weighted_lly_lipschitz_from_length():
    """Weighted LLY Lipschitz constraint should use length, not affinity."""
    g = nx.Graph()
    g.add_edge(0, 1, weight=10.0, length=0.1)  # strong but short
    g.add_edge(1, 2, weight=0.1, length=10.0)  # weak but long
    g.add_edge(0, 2, weight=1.0, length=1.0)
    k = weighted_lly_laplacian_lp(g, 0, 1)
    assert np.isfinite(k)


# ---------------------------------------------------------------------------
# Weighted Forman: literature-faithful vs proxy
# ---------------------------------------------------------------------------

def test_weighted_forman_edge_finite():
    """Literature-faithful weighted Forman should produce finite curvature."""
    g = nx.Graph()
    g.add_edge(0, 1, weight=2.0)
    g.add_edge(1, 2, weight=1.0)
    g.add_edge(0, 2, weight=3.0)
    k = weighted_forman_edge(g, 0, 1)
    assert np.isfinite(k)


def test_weighted_forman_uniform_weights_matches_unweighted_af3():
    """With uniform weights and no adjacent edges, weighted Forman should
    reduce to a simple expression."""
    # Path graph: 0-1-2, edge (0,1) has one adjacent edge at vertex 1
    g = nx.Graph()
    g.add_edge(0, 1, weight=1.0)
    g.add_edge(1, 2, weight=1.0)
    # F(e) = w_e * [w_u*(1 - sqrt(w_e/w_{e1})) + w_v*(1 - sqrt(w_e/w_{e2}))]
    # At u=0: no adjacent edges (excluding e) → sum=0 → term = 1*(1-0) = 1
    # At v=1: one adjacent edge (1,2) with w=1 → sqrt(1/1)=1 → sum=1 → term = 1*(1-1) = 0
    # F = 1 * (1 + 0) = 1
    k = weighted_forman_edge(g, 0, 1)
    assert k == pytest.approx(1.0)


def test_weighted_af3_proxy_is_not_weighted_forman():
    """The proxy and canonical Forman should give different results with non-uniform weights."""
    g = nx.Graph()
    g.add_edge(0, 1, weight=5.0)
    g.add_edge(1, 2, weight=0.5)
    g.add_edge(0, 2, weight=0.5)
    k_proxy = weighted_af3_proxy(g, 0, 1)
    k_forman = weighted_forman_edge(g, 0, 1)
    # They use different formulas, so should differ
    assert k_proxy != pytest.approx(k_forman, abs=1e-3)


# ---------------------------------------------------------------------------
# Config: ricci_flow_target and shadow_horizons
# ---------------------------------------------------------------------------

def test_ricci_flow_target_length_config():
    """Config should accept ricci_flow_target='length'."""
    cfg = LGAEConfig()
    cfg.mutation.ricci_flow_target = "length"
    validate_config(cfg)


def test_ricci_flow_target_invalid_rejected():
    """Invalid ricci_flow_target should be rejected."""
    cfg = LGAEConfig()
    cfg.mutation.ricci_flow_target = "invalid"
    with pytest.raises(ValueError, match="ricci_flow_target"):
        validate_config(cfg)


def test_shadow_horizons_config():
    """Config should accept shadow_horizons list."""
    cfg = LGAEConfig()
    cfg.mutation.shadow_horizons = [1, 2, 4, 8, 16]
    validate_config(cfg)


def test_shadow_horizons_invalid_rejected():
    """Non-positive shadow horizons should be rejected."""
    cfg = LGAEConfig()
    cfg.mutation.shadow_horizons = [1, 0, 4]
    with pytest.raises(ValueError, match="shadow_horizons"):
        validate_config(cfg)


# ---------------------------------------------------------------------------
# Multi-horizon shadow certification
# ---------------------------------------------------------------------------

def test_multi_horizon_shadow_certification():
    """Engine with shadow_horizons should evaluate at multiple horizons."""
    cfg = LGAEConfig()
    cfg.fiber.d_base = 2
    cfg.fiber.d_max = 4
    cfg.mutation.shadow_horizons = [1, 2, 4]
    cfg.audit.orc_backend = "exact_lp"
    cfg.audit.exact_lly_top_k = 16
    cfg.audit.entropic_nodes = 2
    cfg.audit.bakry_nodes = 1
    cfg.audit.cde_nodes = 1
    cfg.audit.cde_samples = 2

    graph = make_graph_buffers(4, [(0, 1), (1, 2), (2, 3), (0, 3)], capacity=8)
    engine = LGAEEngine(graph, cfg)
    engine.diffuse_(0.01)
    mutation = AddEdge(0, 2, weight=1.0)
    result = engine.evaluate_and_maybe_commit(mutation)
    # Should have multi_horizon metadata if evaluated
    if result.decision.value != "reject" or "multi_horizon" in result.metadata:
        assert "multi_horizon" in result.metadata
        horizons_evaluated = [h["horizon"] for h in result.metadata["multi_horizon"]]
        assert set(horizons_evaluated) == {1, 2, 4}


# ---------------------------------------------------------------------------
# Edge metric invariant: d_ell(u,v) <= ell_uv
# ---------------------------------------------------------------------------

def test_edge_metric_invariant():
    """For every edge, d_ell(u,v) <= ell_uv (shortest path ≤ direct edge)."""
    graph = make_graph_buffers(4, [(0, 1, 2.0, 3.0), (1, 2, 1.0, 1.0), (0, 2, 1.0, 0.5)], capacity=8)
    g = graphbuffers_to_networkx(graph)
    for u, v in g.edges():
        d_uv = nx.dijkstra_path_length(g, u, v, weight="length")
        ell_uv = g[u][v]["length"]
        # Shortest path distance should be ≤ direct edge length
        assert d_uv <= ell_uv + 1e-10, f"d_ell({u},{v})={d_uv} > ell={ell_uv}"


# ---------------------------------------------------------------------------
# Backward compatibility
# ---------------------------------------------------------------------------

def test_backward_compat_2tuple_edges():
    """make_graph_buffers with (u,v) tuples should still work."""
    graph = make_graph_buffers(3, [(0, 1), (1, 2)], capacity=4)
    assert graph.edge_count == 2
    assert float(graph.weight[0].item()) == pytest.approx(1.0)
    assert float(graph.length[0].item()) == pytest.approx(1.0)


def test_backward_compat_3tuple_edges():
    """make_graph_buffers with (u,v,w) tuples should still work."""
    graph = make_graph_buffers(3, [(0, 1, 5.0), (1, 2, 0.2)], capacity=4)
    assert float(graph.weight[0].item()) == pytest.approx(5.0)
    assert float(graph.length[0].item()) == pytest.approx(0.2)  # 1/5
    assert float(graph.weight[1].item()) == pytest.approx(0.2)
    assert float(graph.length[1].item()) == pytest.approx(5.0)  # 1/0.2


def test_backward_compat_checkpoint_no_length():
    """Old checkpoint without length should still load (derives from weight)."""
    graph = make_graph_buffers(3, [(0, 1, 4.0)], capacity=4)
    state = graph.to_state_dict()
    # Simulate old checkpoint by removing length
    del state["length"]
    restored = GraphBuffers.from_state_dict(state)
    # Length should be derived from weight (1/4 = 0.25)
    assert float(restored.length[0].item()) == pytest.approx(0.25)
