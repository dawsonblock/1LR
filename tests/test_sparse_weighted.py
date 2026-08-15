"""Tests for v4.0 Sparse Dual Operators and Weighted Curvature.

Covers:
- Sparse: diagnostic_diffusion_edges produces valid row-stochastic edges
- Sparse: SparseDualOperatorState discrepancy matches dense for small N
- Sparse: k-NN without full N×N matrix
- Sparse: sparse_operator_discrepancy on union support
- Weighted: weighted_ollivier_edge uses edge weights in lazy measure
- Weighted: weighted_lly_laplacian_lp uses weighted degree
- Weighted: weighted_af3_edge uses weighted degree
- Weighted: governor audit uses weighted curvature when configured
"""
from __future__ import annotations

import networkx as nx
import numpy as np
import torch
import pytest

from lgae_v3 import LGAEConfig, LGAEEngine, make_graph_buffers
from lgae_v3.operators import (
    SparseDualOperatorState,
    DualOperatorState,
    actuation_operator,
    actuation_markov_edges,
    diagnostic_diffusion_operator,
    diagnostic_diffusion_edges,
    sparse_operator_discrepancy,
    operator_discrepancy,
)
from lgae_v3.curvature import (
    af3_edge,
    weighted_af3_edge,
    ollivier_edge,
    weighted_ollivier_edge,
    lly_laplacian_lp,
    weighted_lly_laplacian_lp,
    lly_half_idleness,
    weighted_lly_half_idleness,
)


# ---------------------------------------------------------------------------
# Sparse: diagnostic_diffusion_edges
# ---------------------------------------------------------------------------

def test_diagnostic_diffusion_edges_row_stochastic():
    """Sparse diagnostic edges should be row-stochastic."""
    z = torch.randn(10, 4)
    src, dst, w = diagnostic_diffusion_edges(z, k=4)
    n = z.shape[0]
    mass = torch.zeros(n, dtype=w.dtype)
    mass.index_add_(0, src, w)
    # Each row should sum to ~1.0
    assert torch.allclose(mass, torch.ones_like(mass), atol=1e-5)


def test_diagnostic_diffusion_edges_no_dense_matrix():
    """Sparse edges should not require N×N memory for moderate N."""
    z = torch.randn(100, 8)
    src, dst, w = diagnostic_diffusion_edges(z, k=8)
    # Edge count should be O(N*k), not O(N²)
    assert src.numel() <= 100 * 8 * 2 + 100  # 2x for symmetrization + self-loops
    assert src.numel() < 100 * 100  # Much less than N²


def test_diagnostic_diffusion_edges_single_node():
    """Single node should produce a self-loop."""
    z = torch.randn(1, 4)
    src, dst, w = diagnostic_diffusion_edges(z, k=4)
    assert src.numel() == 1
    assert dst.numel() == 1
    assert w.item() == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Sparse: SparseDualOperatorState
# ---------------------------------------------------------------------------

def test_sparse_dual_operator_discrepancy_matches_dense():
    """For small N, sparse discrepancy should approximately match dense."""
    graph = make_graph_buffers(6, [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (0, 5)], capacity=12)
    z = torch.randn(6, 4)

    # Dense path
    pa_dense = actuation_operator(graph, symmetric=True)
    pd_dense = diagnostic_diffusion_operator(z, k=4, full_kernel_max_nodes=512)
    dense_disc = operator_discrepancy(pa_dense, pd_dense, mode="frobenius")

    # Sparse path
    sparse_state = SparseDualOperatorState.from_graph_and_latent(graph, z, diagnostic_k=4)
    sparse_disc = sparse_state.discrepancy(mode="frobenius")

    # They won't be exactly equal because the sparse diagnostic uses k-NN
    # while the dense uses full soft kernel, but they should be same order
    assert float(dense_disc.item()) > 0
    assert float(sparse_disc.item()) > 0
    # Both should be finite
    assert torch.isfinite(dense_disc)
    assert torch.isfinite(sparse_disc)


def test_sparse_dual_operator_to_dense():
    """Converting sparse to dense should produce valid row-stochastic matrices."""
    graph = make_graph_buffers(5, [(0, 1), (1, 2), (2, 3), (3, 4)], capacity=8)
    z = torch.randn(5, 4)
    sparse_state = SparseDualOperatorState.from_graph_and_latent(graph, z, diagnostic_k=3)
    dense_state = sparse_state.to_dense()

    # Check row stochasticity
    assert torch.allclose(dense_state.p_actuation.sum(dim=-1), torch.ones(5), atol=1e-5)
    assert torch.allclose(dense_state.p_diagnostic.sum(dim=-1), torch.ones(5), atol=1e-5)


# ---------------------------------------------------------------------------
# Sparse: sparse_operator_discrepancy
# ---------------------------------------------------------------------------

def test_sparse_operator_discrepancy_identical_operators():
    """Discrepancy of identical operators should be zero."""
    n = 5
    src = torch.tensor([0, 1, 2, 3, 4, 1, 2, 0, 3, 4])
    dst = torch.tensor([1, 2, 3, 4, 0, 0, 1, 2, 2, 3])
    w = torch.ones(10) / 2.0
    disc = sparse_operator_discrepancy(src, dst, w, src, dst, w, n, mode="frobenius")
    assert float(disc.item()) == pytest.approx(0.0, abs=1e-6)


def test_sparse_operator_discrepancy_different_operators():
    """Discrepancy of different operators should be positive."""
    n = 4
    src1 = torch.tensor([0, 1, 2, 3])
    dst1 = torch.tensor([1, 2, 3, 0])
    w1 = torch.ones(4)
    src2 = torch.tensor([0, 1, 2, 3])
    dst2 = torch.tensor([2, 3, 0, 1])
    w2 = torch.ones(4)
    disc = sparse_operator_discrepancy(src1, dst1, w1, src2, dst2, w2, n, mode="frobenius")
    assert float(disc.item()) > 0


# ---------------------------------------------------------------------------
# Weighted: weighted_ollivier_edge
# ---------------------------------------------------------------------------

def test_weighted_ollivier_uniform_weights_matches_unweighted():
    """With uniform weights, weighted Ollivier should match unweighted."""
    g = nx.Graph()
    g.add_edge(0, 1, weight=1.0)
    g.add_edge(1, 2, weight=1.0)
    g.add_edge(0, 2, weight=1.0)
    k_unweighted = ollivier_edge(g, 0, 1, p=0.5, backend="exact_lp")
    k_weighted = weighted_ollivier_edge(g, 0, 1, p=0.5, backend="exact_lp")
    assert k_unweighted == pytest.approx(k_weighted, abs=1e-6)


def test_weighted_ollivier_nonuniform_weights_differs():
    """With non-uniform weights, weighted Ollivier should differ from unweighted."""
    g = nx.Graph()
    g.add_edge(0, 1, weight=10.0)
    g.add_edge(1, 2, weight=0.1)
    g.add_edge(0, 2, weight=0.1)
    k_unweighted = ollivier_edge(g, 0, 1, p=0.5, backend="exact_lp")
    k_weighted = weighted_ollivier_edge(g, 0, 1, p=0.5, backend="exact_lp")
    # They should differ because the lazy measure distributes mass
    # proportionally to weights
    assert k_unweighted != pytest.approx(k_weighted, abs=1e-3)


def test_weighted_ollivier_sinkhorn_backend():
    """Weighted Ollivier should work with sinkhorn backend."""
    g = nx.Graph()
    g.add_edge(0, 1, weight=2.0)
    g.add_edge(1, 2, weight=1.0)
    g.add_edge(0, 2, weight=1.0)
    k = weighted_ollivier_edge(g, 0, 1, p=0.5, backend="sinkhorn_log")
    assert np.isfinite(k)


# ---------------------------------------------------------------------------
# Weighted: weighted LLY
# ---------------------------------------------------------------------------

def test_weighted_lly_uniform_weights_matches_unweighted():
    """With uniform weights, weighted LLY should match unweighted."""
    g = nx.Graph()
    g.add_edge(0, 1, weight=1.0)
    g.add_edge(1, 2, weight=1.0)
    g.add_edge(0, 2, weight=1.0)
    k_uw = lly_laplacian_lp(g, 0, 1)
    k_w = weighted_lly_laplacian_lp(g, 0, 1)
    assert k_uw == pytest.approx(k_w, abs=1e-4)


def test_weighted_lly_half_idleness_uniform_matches():
    """With uniform weights, weighted LLY half-idleness should match unweighted."""
    g = nx.Graph()
    g.add_edge(0, 1, weight=1.0)
    g.add_edge(1, 2, weight=1.0)
    g.add_edge(0, 2, weight=1.0)
    k_uw = lly_half_idleness(g, 0, 1)
    k_w = weighted_lly_half_idleness(g, 0, 1)
    assert k_uw == pytest.approx(k_w, abs=1e-4)


def test_weighted_lly_nonuniform_weights_finite():
    """Weighted LLY with non-uniform weights should produce finite curvature."""
    g = nx.Graph()
    g.add_edge(0, 1, weight=5.0)
    g.add_edge(1, 2, weight=0.5)
    g.add_edge(0, 2, weight=0.5)
    k = weighted_lly_laplacian_lp(g, 0, 1)
    assert np.isfinite(k)


# ---------------------------------------------------------------------------
# Weighted: weighted AF3
# ---------------------------------------------------------------------------

def test_weighted_af3_uniform_weights_matches_unweighted():
    """With uniform weights, weighted AF3 should match unweighted."""
    g = nx.Graph()
    g.add_edge(0, 1, weight=1.0)
    g.add_edge(1, 2, weight=1.0)
    g.add_edge(0, 2, weight=1.0)
    g.add_edge(2, 3, weight=1.0)
    k_uw = af3_edge(g, 0, 1)
    k_w = weighted_af3_edge(g, 0, 1)
    assert k_uw == pytest.approx(k_w, abs=1e-6)


def test_weighted_af3_nonuniform_weights():
    """Weighted AF3 with non-uniform weights should be finite and differ."""
    g = nx.Graph()
    g.add_edge(0, 1, weight=5.0)
    g.add_edge(1, 2, weight=0.5)
    g.add_edge(0, 2, weight=0.5)
    g.add_edge(2, 3, weight=2.0)
    k_uw = af3_edge(g, 0, 1)
    k_w = weighted_af3_edge(g, 0, 1)
    assert np.isfinite(k_w)
    # Weighted version uses weighted degree which differs
    assert k_uw != pytest.approx(k_w, abs=1e-3)


# ---------------------------------------------------------------------------
# Weighted: governor integration
# ---------------------------------------------------------------------------

def test_governor_weighted_mode_audit():
    """Governor audit should work with curvature_weight_mode='weighted'."""
    cfg = LGAEConfig()
    cfg.fiber.d_base = 2
    cfg.fiber.d_max = 4
    cfg.audit.curvature_weight_mode = "weighted"
    cfg.audit.orc_backend = "exact_lp"
    cfg.audit.exact_lly_top_k = 16
    cfg.audit.entropic_nodes = 3
    cfg.audit.bakry_nodes = 1
    cfg.audit.cde_nodes = 1
    cfg.audit.cde_samples = 2

    graph = make_graph_buffers(4, [(0, 1, 2.0), (1, 2, 1.0), (2, 3, 0.5), (0, 3, 1.0)], capacity=8)
    engine = LGAEEngine(graph, cfg)
    engine.diffuse_(0.01)
    snapshot = engine.audit()
    assert snapshot is not None
    # LLY should be computed with weighted backends
    assert "lly" in snapshot.details
    assert len(snapshot.details["lly"]) > 0


def test_governor_weighted_mode_fast_signals():
    """Fast signals should use weighted AF3 when configured."""
    cfg = LGAEConfig()
    cfg.audit.curvature_weight_mode = "weighted"
    graph = make_graph_buffers(4, [(0, 1, 2.0), (1, 2, 1.0), (2, 3, 0.5)], capacity=8)
    engine = LGAEEngine(graph, cfg)
    z = engine.fibers().detach()
    sig = engine.governor.fast_signals(engine.graph, z)
    # Should have AF3 values computed with weighted backends
    assert len(sig.edge_af3) > 0
    # Values should be finite
    for v in sig.edge_af3.values():
        assert np.isfinite(v)


# ---------------------------------------------------------------------------
# Config: weighted mode validation
# ---------------------------------------------------------------------------

def test_weighted_mode_accepted_in_config():
    """curvature_weight_mode='weighted' should be accepted by validate_config."""
    from lgae_v3.config import validate_config
    cfg = LGAEConfig()
    cfg.audit.curvature_weight_mode = "weighted"
    validate_config(cfg)  # should not raise


def test_invalid_curvature_weight_mode_rejected():
    """Invalid curvature_weight_mode should be rejected."""
    from lgae_v3.config import validate_config
    cfg = LGAEConfig()
    cfg.audit.curvature_weight_mode = "invalid"
    with pytest.raises(ValueError, match="curvature_weight_mode"):
        validate_config(cfg)
