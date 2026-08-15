"""v4.1.2 adversarial and numerical edge-case qualification.

All tests verify that the system fails closed when injected with:
- NaN, ±Inf, 0, 1e-20, 1e20 in affinity, length, latent, gauge, curvature, P
- Disconnected graphs, single-node graphs, self-loops
- Extreme config values
"""
from __future__ import annotations

import math
import pytest
import torch
import numpy as np
import networkx as nx

from lgae_v3 import LGAEConfig, LGAEEngine, make_graph_buffers
from lgae_v3.types import GraphBuffers
from lgae_v3.mutations import (
    AddEdge, ReweightEdge, ReweightAffinity, ReweightLength, CoupledReweight,
    PruneEdge, RicciFlowReweight, mutation_to_spec, mutation_from_spec,
)
from lgae_v3.operators import (
    actuation_operator, diagnostic_diffusion_operator,
    actuation_markov_edges, diagnostic_diffusion_edges,
    sparse_operator_discrepancy,
)
from lgae_v3.curvature import (
    weighted_forman_edge, weighted_af3_proxy, af3_edge,
    validate_reversible_markov,
)
from lgae_v3.curvature.bakry_emery import normalized_markov_generator
from lgae_v3.config import config_governance_hash, validate_config


# ---------------------------------------------------------------------------
# Affinity injection: NaN, Inf, 0, 1e-20, 1e20
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("bad_value", [float("nan"), float("inf"), float("-inf"), 0.0])
def test_affinity_nan_inf_zero_rejected(bad_value):
    """GraphBuffers must reject non-positive or non-finite affinity."""
    with pytest.raises((ValueError, RuntimeError)):
        g = make_graph_buffers(3, [(0, 1, float(bad_value)), (1, 2, 1.0)], capacity=4)
        g.validate()


@pytest.mark.parametrize("extreme_value", [1e-20, 1e20])
def test_affinity_extreme_but_valid_accepted(extreme_value):
    """GraphBuffers should accept extreme but positive finite affinity."""
    g = make_graph_buffers(3, [(0, 1, float(extreme_value)), (1, 2, 1.0)], capacity=4)
    g.validate()  # should not raise


@pytest.mark.parametrize("bad_value", [float("nan"), float("inf"), float("-inf"), 0.0, -1e-10])
def test_length_nan_inf_zero_rejected(bad_value):
    """GraphBuffers must reject non-positive or non-finite length."""
    with pytest.raises((ValueError, RuntimeError)):
        g = make_graph_buffers(3, [(0, 1, 1.0, float(bad_value)), (1, 2, 1.0, 1.0)], capacity=4)
        g.validate()


# ---------------------------------------------------------------------------
# Latent state injection
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("bad_value", [float("nan"), float("inf"), float("-inf")])
def test_latent_nan_inf_rejected_by_audit(bad_value):
    """Governor audit must reject NaN/Inf latent state."""
    cfg = LGAEConfig()
    cfg.fiber.d_base = 2; cfg.fiber.d_max = 4
    graph = make_graph_buffers(4, [(0, 1), (1, 2), (2, 3)], capacity=8)
    engine = LGAEEngine(graph, cfg)
    z = torch.randn(4, 2)
    z[0, 0] = bad_value
    with pytest.raises(ValueError, match="NaN|Inf|finite"):
        engine.governor.audit(graph, z)


@pytest.mark.parametrize("bad_value", [float("nan"), float("inf"), float("-inf")])
def test_latent_nan_inf_rejected_by_shadow_rollout(bad_value):
    """Shadow rollout must reject NaN/Inf latent state."""
    cfg = LGAEConfig()
    cfg.fiber.d_base = 2; cfg.fiber.d_max = 4
    graph = make_graph_buffers(4, [(0, 1), (1, 2), (2, 3)], capacity=8)
    engine = LGAEEngine(graph, cfg)
    z = torch.randn(4, 2)
    z[1, 1] = bad_value
    with pytest.raises((ValueError, RuntimeError, FloatingPointError)):
        engine.governor.shadow_rollout(graph, z)


# ---------------------------------------------------------------------------
# Mutation factor injection
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("bad_factor", [float("nan"), float("inf"), float("-inf"), 0.0, -1.0])
def test_reweight_bad_factor_rejected(bad_factor):
    """ReweightEdge must reject non-positive or non-finite factor."""
    graph = make_graph_buffers(3, [(0, 1, 1.0), (1, 2, 1.0)], capacity=4)
    mut = ReweightEdge(0, 1, factor=bad_factor)
    with pytest.raises(ValueError):
        mut.apply(graph)


@pytest.mark.parametrize("bad_factor", [float("nan"), float("inf"), 0.0, -1.0])
def test_reweight_affinity_bad_factor_rejected(bad_factor):
    """ReweightAffinity must reject bad factor."""
    graph = make_graph_buffers(3, [(0, 1, 1.0, 1.0), (1, 2, 1.0, 1.0)], capacity=4)
    mut = ReweightAffinity(0, 1, factor=bad_factor)
    with pytest.raises(ValueError):
        mut.apply(graph)


@pytest.mark.parametrize("bad_factor", [float("nan"), float("inf"), 0.0, -1.0])
def test_reweight_length_bad_factor_rejected(bad_factor):
    """ReweightLength must reject bad factor."""
    graph = make_graph_buffers(3, [(0, 1, 1.0, 1.0), (1, 2, 1.0, 1.0)], capacity=4)
    mut = ReweightLength(0, 1, factor=bad_factor)
    with pytest.raises(ValueError):
        mut.apply(graph)


def test_coupled_reweight_bad_coupling_rejected():
    """CoupledReweight must reject invalid coupling policy."""
    graph = make_graph_buffers(3, [(0, 1, 1.0, 1.0), (1, 2, 1.0, 1.0)], capacity=4)
    mut = CoupledReweight(0, 1, coupling="invalid_policy")
    with pytest.raises(ValueError):
        mut.apply(graph)


# ---------------------------------------------------------------------------
# Markov generator injection
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("bad_value", [float("nan"), float("inf"), float("-inf")])
def test_markov_generator_nan_inf_rejected(bad_value):
    """normalized_markov_generator must reject NaN/Inf."""
    P = torch.tensor([[0.0, 1.0, 0.0], [0.5, 0.0, 0.5], [0.0, 1.0, 0.0]])
    P[0, 1] = bad_value
    with pytest.raises((ValueError, RuntimeError)):
        normalized_markov_generator(P)


def test_markov_generator_non_stochastic_rejected():
    """validate_reversible_markov must reject non-stochastic rows."""
    P = torch.tensor([[0.3, 0.8, 0.0], [0.5, 0.0, 0.5], [0.0, 1.0, 0.0]])  # row 0 sums to 1.1
    with pytest.raises((ValueError, AssertionError)):
        validate_reversible_markov(P)


# ---------------------------------------------------------------------------
# Disconnected / degenerate graphs
# ---------------------------------------------------------------------------

def test_disconnected_graph_audit():
    """Audit must handle disconnected graphs without crash."""
    cfg = LGAEConfig()
    cfg.fiber.d_base = 2; cfg.fiber.d_max = 4
    cfg.audit.entropic_nodes = 1; cfg.audit.bakry_nodes = 1; cfg.audit.cde_nodes = 1
    # Two disconnected components
    graph = make_graph_buffers(4, [(0, 1), (2, 3)], capacity=8)
    engine = LGAEEngine(graph, cfg)
    z = torch.randn(4, 2)
    snap = engine.governor.audit(graph, z)
    # lambda2 should be ~0 for disconnected
    assert snap.lambda2 <= 1e-6


def test_single_edge_graph_audit():
    """Audit must handle a single-edge graph."""
    cfg = LGAEConfig()
    cfg.fiber.d_base = 2; cfg.fiber.d_max = 4
    cfg.audit.entropic_nodes = 1; cfg.audit.bakry_nodes = 1; cfg.audit.cde_nodes = 1
    graph = make_graph_buffers(2, [(0, 1)], capacity=4)
    engine = LGAEEngine(graph, cfg)
    z = torch.randn(2, 2)
    snap = engine.governor.audit(graph, z)
    assert np.isfinite(snap.lambda2)


def test_bridge_prune_rejected():
    """Pruning a bridge must be rejected."""
    cfg = LGAEConfig()
    cfg.fiber.d_base = 2; cfg.fiber.d_max = 4
    cfg.audit.preserve_beta0 = True
    cfg.audit.local_disconnect_gate = True
    # 0-1-2: edge (1,2) is a bridge
    graph = make_graph_buffers(3, [(0, 1), (1, 2)], capacity=6)
    engine = LGAEEngine(graph, cfg)
    engine.diffuse_(0.01)
    mut = PruneEdge(1, 2)
    result = engine.evaluate_and_maybe_commit(mut)
    assert result.decision.value == "reject"
    assert any("bridge" in r or "component" in r for r in result.reasons)


# ---------------------------------------------------------------------------
# ReweightAffinity / ReweightLength / CoupledReweight semantics
# ---------------------------------------------------------------------------

def test_reweight_affinity_preserves_length():
    """ReweightAffinity must not change the length field."""
    graph = make_graph_buffers(3, [(0, 1, 2.0, 5.0), (1, 2, 1.0, 1.0)], capacity=4)
    original_length = graph.length.clone()
    mut = ReweightAffinity(0, 1, factor=2.0)
    mut.apply(graph)
    assert torch.allclose(graph.length, original_length)
    assert graph.weight[0] > 2.0  # affinity changed


def test_reweight_length_preserves_affinity():
    """ReweightLength must not change the affinity field."""
    graph = make_graph_buffers(3, [(0, 1, 2.0, 5.0), (1, 2, 1.0, 1.0)], capacity=4)
    original_weight = graph.weight.clone()
    mut = ReweightLength(0, 1, factor=2.0)
    mut.apply(graph)
    assert torch.allclose(graph.weight, original_weight)
    assert graph.length[0] > 5.0  # length changed


def test_coupled_reweight_inverse():
    """CoupledReweight with inverse coupling: stronger affinity → shorter length."""
    graph = make_graph_buffers(3, [(0, 1, 2.0, 5.0), (1, 2, 1.0, 1.0)], capacity=4)
    mut = CoupledReweight(0, 1, affinity_factor=2.0, coupling="inverse")
    mut.apply(graph)
    assert graph.weight[0] > 2.0
    assert graph.length[0] < 5.0


def test_coupled_reweight_direct():
    """CoupledReweight with direct coupling: stronger affinity → longer length."""
    graph = make_graph_buffers(3, [(0, 1, 2.0, 5.0), (1, 2, 1.0, 1.0)], capacity=4)
    mut = CoupledReweight(0, 1, affinity_factor=2.0, coupling="direct")
    mut.apply(graph)
    assert graph.weight[0] > 2.0
    assert graph.length[0] > 5.0


def test_coupled_reweight_none():
    """CoupledReweight with none coupling: length unchanged."""
    graph = make_graph_buffers(3, [(0, 1, 2.0, 5.0), (1, 2, 1.0, 1.0)], capacity=4)
    original_length = graph.length.clone()
    mut = CoupledReweight(0, 1, affinity_factor=2.0, coupling="none")
    mut.apply(graph)
    assert torch.allclose(graph.length, original_length)


def test_new_mutation_spec_roundtrips():
    """All new mutation types must roundtrip through spec."""
    for mut in [
        ReweightAffinity(0, 1, factor=1.5),
        ReweightLength(0, 1, factor=2.0),
        CoupledReweight(0, 1, affinity_factor=1.5, coupling="direct"),
    ]:
        spec = mutation_to_spec(mut)
        mut2 = mutation_from_spec(spec)
        assert type(mut2) is type(mut)
        assert mut2.u == mut.u
        assert mut2.v == mut.v


# ---------------------------------------------------------------------------
# Geometry-mode tier config
# ---------------------------------------------------------------------------

def test_geometry_mode_tiers_validate():
    """Tier config must validate."""
    cfg = LGAEConfig()
    cfg.audit.candidate_geometry_mode = "metric_measure"
    cfg.audit.audit_geometry_mode = "weighted"
    cfg.audit.certificate_geometry_mode = "unweighted_reference"
    validate_config(cfg)  # should not raise


def test_geometry_mode_tiers_invalid_rejected():
    """Invalid tier values must be rejected."""
    cfg = LGAEConfig()
    cfg.audit.candidate_geometry_mode = "invalid_mode"
    with pytest.raises(ValueError):
        validate_config(cfg)


def test_geometry_mode_tiers_affect_governance_hash():
    """Tier config must affect governance hash."""
    cfg_a = LGAEConfig()
    cfg_b = LGAEConfig()
    cfg_b.audit.candidate_geometry_mode = "metric_measure"
    assert config_governance_hash(cfg_a) != config_governance_hash(cfg_b)


def test_geometry_mode_tiers_default_to_curvature_weight_mode():
    """Empty tier values should fall back to curvature_weight_mode."""
    cfg = LGAEConfig()
    cfg.audit.curvature_weight_mode = "weighted"
    # Empty tier means "follow curvature_weight_mode"
    assert cfg.audit.candidate_geometry_mode == ""
    # The governor should use "weighted" for all tiers
    graph = make_graph_buffers(4, [(0, 1, 1.0, 1.0), (1, 2, 1.0, 1.0), (2, 3, 1.0, 1.0)], capacity=8)
    engine = LGAEEngine(graph, cfg)
    z = torch.randn(4, 2)
    snap = engine.governor.audit(graph, z)
    # audit_geometry_mode in details should reflect the fallback
    assert snap.details["audit_geometry_mode"] == "weighted"


# ---------------------------------------------------------------------------
# Multi-horizon fiber mutation
# ---------------------------------------------------------------------------

def test_fiber_multi_horizon_with_config():
    """Fiber mutations with shadow_horizons should include multi_horizon metadata."""
    cfg = LGAEConfig()
    cfg.fiber.d_base = 2; cfg.fiber.d_max = 4
    cfg.mutation.shadow_horizons = [1, 2, 4]
    cfg.audit.entropic_nodes = 1; cfg.audit.bakry_nodes = 1; cfg.audit.cde_nodes = 1
    cfg.audit.cde_samples = 2
    graph = make_graph_buffers(4, [(0, 1), (1, 2), (2, 3)], capacity=8)
    engine = LGAEEngine(graph, cfg)
    z_before = torch.randn(4, 2)
    z_after = z_before + 0.1 * torch.randn(4, 2)
    result = engine.governor.evaluate_latent_transition(
        graph, z_before, z_after, name="test_fiber"
    )
    assert "multi_horizon" in result.metadata
    assert len(result.metadata["multi_horizon"]) == 3


# ---------------------------------------------------------------------------
# Extreme config values
# ---------------------------------------------------------------------------

def test_extreme_shadow_horizons():
    """Very large shadow horizons should not crash."""
    cfg = LGAEConfig()
    cfg.fiber.d_base = 2; cfg.fiber.d_max = 4
    cfg.mutation.shadow_horizons = [1, 100]
    cfg.audit.entropic_nodes = 1; cfg.audit.bakry_nodes = 1; cfg.audit.cde_nodes = 1
    cfg.audit.cde_samples = 2
    graph = make_graph_buffers(4, [(0, 1), (1, 2), (2, 3)], capacity=8)
    engine = LGAEEngine(graph, cfg)
    engine.diffuse_(0.01)
    mut = AddEdge(0, 2, weight=1.0)
    result = engine.evaluate_and_maybe_commit(mut)
    # Should complete without crash
    assert result.decision.value in ("accept", "quarantine", "reject")


def test_zero_shadow_steps():
    """Zero shadow steps should return original latent."""
    cfg = LGAEConfig()
    cfg.fiber.d_base = 2; cfg.fiber.d_max = 4
    graph = make_graph_buffers(4, [(0, 1), (1, 2), (2, 3)], capacity=8)
    engine = LGAEEngine(graph, cfg)
    z = torch.randn(4, 2)
    z_out = engine.governor.shadow_rollout(graph, z, steps=0)
    assert torch.allclose(z_out, z)


# ---------------------------------------------------------------------------
# Sparse operator edge cases
# ---------------------------------------------------------------------------

def test_sparse_discrepancy_empty_operator():
    """Sparse discrepancy with empty operators should return 0."""
    N = 4
    empty = torch.tensor([], dtype=torch.long)
    empty_w = torch.tensor([], dtype=torch.float32)
    D = sparse_operator_discrepancy(empty, empty, empty_w, empty, empty, empty_w, N)
    assert D.item() == 0.0


def test_sparse_discrepancy_self_loop():
    """Sparse discrepancy with self-loops should work."""
    N = 3
    src = torch.tensor([0, 1, 2])
    dst = torch.tensor([0, 1, 2])
    w = torch.tensor([1.0, 1.0, 1.0])
    D = sparse_operator_discrepancy(src, dst, w, src, dst, w, N)
    assert D.item() == 0.0  # identical operators


# ---------------------------------------------------------------------------
# Weighted Forman edge cases
# ---------------------------------------------------------------------------

def test_weighted_forman_extreme_weights():
    """Weighted Forman should handle extreme but valid weights."""
    g = nx.Graph()
    g.add_edge(0, 1, weight=1e10, length=1e-10)
    k = weighted_forman_edge(g, 0, 1)
    assert math.isfinite(k)


def test_weighted_forman_negative_weight_rejected():
    """Weighted Forman should reject negative affinity."""
    g = nx.Graph()
    g.add_edge(0, 1, weight=-1.0, length=1.0)
    with pytest.raises(ValueError):
        weighted_forman_edge(g, 0, 1)


def test_weighted_forman_zero_length_rejected():
    """Weighted Forman should reject zero length."""
    g = nx.Graph()
    g.add_edge(0, 1, weight=1.0, length=0.0)
    with pytest.raises(ValueError):
        weighted_forman_edge(g, 0, 1)
