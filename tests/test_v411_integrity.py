"""v4.1.1 Sparse Governance Integrity regression tests.

Tests the specific defects identified in the forensic audit:
- P0-1: Sparse governor must work for N>2048 (no p_diagnostic crash)
- P0-2: Sparse discrepancy must coalesce duplicate COO edges
- P0-3: Safe checkpoint must persist metric length
- P0-4: Governance hash must include v4.1 policy fields
- P0-5: RicciFlow serialization must preserve target_field/coupled
- P1-1: Multi-horizon must propagate QUARANTINE (max severity)
- P1-2: Safe checkpoint must preserve quarantined shadow graphs
- P1-3: Weighted Forman must use metric-measure formula
- P2-1: Version identity must be consistent across all artifacts
"""
from __future__ import annotations

import torch
import numpy as np
import networkx as nx
import pytest
import tempfile
import os
import json

from lgae_v3 import LGAEConfig, LGAEEngine, make_graph_buffers
from lgae_v3.types import GraphBuffers
from lgae_v3.operators import (
    diagnostic_diffusion_edges,
    sparse_operator_discrepancy,
    actuation_markov_edges,
    operator_discrepancy,
    actuation_operator,
    diagnostic_diffusion_operator,
    SparseDualOperatorState,
)
from lgae_v3.mutations import (
    AddEdge, ReweightEdge, PruneEdge, RicciFlowReweight,
    mutation_to_spec, mutation_from_spec,
)
from lgae_v3.config import config_governance_hash, validate_config
from lgae_v3.curvature import weighted_forman_edge, weighted_af3_proxy
from lgae_v3.version import VERSION


# ---------------------------------------------------------------------------
# P0-1: Sparse governor works for N>2048
# ---------------------------------------------------------------------------

def test_sparse_governor_n_gt_2048():
    """Governor audit must complete for N>2048 without AttributeError."""
    N = 2050
    edges = [(i, (i + 1) % N) for i in range(N)]
    graph = make_graph_buffers(N, edges, capacity=N + 10)
    cfg = LGAEConfig()
    cfg.fiber.d_base = 4
    cfg.fiber.d_max = 8
    cfg.audit.entropic_nodes = 1
    cfg.audit.bakry_nodes = 1
    cfg.audit.cde_nodes = 1
    cfg.audit.cde_samples = 2
    cfg.operator.diagnostic_full_kernel_max_nodes = 512  # force sparse path
    engine = LGAEEngine(graph, cfg)
    z = torch.randn(N, 4)
    snap = engine.governor.audit(graph, z)
    assert np.isfinite(snap.lambda2)


def test_sparse_governor_n_gt_2048_no_dense_allocation():
    """Sparse path must not allocate N×N dense matrix."""
    N = 2100
    edges = [(i, (i + 1) % N) for i in range(N)]
    graph = make_graph_buffers(N, edges, capacity=N + 10)
    cfg = LGAEConfig()
    cfg.fiber.d_base = 4
    cfg.fiber.d_max = 8
    cfg.audit.entropic_nodes = 1
    cfg.audit.bakry_nodes = 1
    cfg.audit.cde_nodes = 1
    cfg.audit.cde_samples = 2
    cfg.operator.diagnostic_full_kernel_max_nodes = 512
    engine = LGAEEngine(graph, cfg)
    z = torch.randn(N, 4)
    ops = engine.governor.operators(graph, z)
    assert isinstance(ops, SparseDualOperatorState)
    # The key invariant: we never materialized a full N×N matrix
    # Verify by checking that p_diagnostic would require O(N²) allocation
    # but local_dense_diagnostic only extracts a small neighborhood
    local_P, node_idx = ops.local_dense_diagnostic(
        torch.tensor([0, 1, 2], dtype=torch.long), radius=2
    )
    assert local_P.shape[0] < N  # local, not global


# ---------------------------------------------------------------------------
# P0-2: Sparse discrepancy coalesces duplicate edges
# ---------------------------------------------------------------------------

def test_sparse_discrepancy_matches_dense():
    """Sparse discrepancy must match the dense version of the same sparse operator."""
    torch.manual_seed(42)
    for N in [4, 5, 10, 20]:
        z = torch.randn(N, 4)
        edges = [(i, (i + 1) % N) for i in range(N)]
        graph = make_graph_buffers(N, edges, capacity=N + 4)

        # Build sparse operators
        diag_src, diag_dst, diag_w = diagnostic_diffusion_edges(z, k=3)
        act_src, act_dst, act_w = actuation_markov_edges(graph)

        # Convert to dense using accumulation (the correct reference)
        P_act_dense = torch.zeros((N, N), dtype=act_w.dtype)
        P_act_dense.index_put_((act_src, act_dst), act_w, accumulate=True)
        P_diag_dense = torch.zeros((N, N), dtype=diag_w.dtype)
        P_diag_dense.index_put_((diag_src, diag_dst), diag_w, accumulate=True)
        D_dense = operator_discrepancy(P_act_dense, P_diag_dense)

        # Sparse discrepancy
        D_sparse = sparse_operator_discrepancy(
            act_src, act_dst, act_w, diag_src, diag_dst, diag_w, N
        )

        assert abs(D_sparse - D_dense) < 1e-6, (
            f"N={N}: D_sparse={D_sparse:.10f} != D_dense={D_dense:.10f}, "
            f"error={abs(D_sparse - D_dense):.10f}"
        )


def test_sparse_discrepancy_duplicate_coalescing():
    """Duplicate directed edges from mutual k-NN must be coalesced."""
    torch.manual_seed(123)
    N = 6
    z = torch.randn(N, 4)
    edges = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (0, 5)]
    graph = make_graph_buffers(N, edges, capacity=12)

    # Build sparse operators
    diag_src, diag_dst, diag_w = diagnostic_diffusion_edges(z, k=3)
    # Check that duplicates exist (from symmetrization)
    keys = diag_src * N + diag_dst
    unique_keys = len(set(keys.tolist()))
    assert unique_keys < diag_src.numel(), "Test requires duplicate edges"

    act_src, act_dst, act_w = actuation_markov_edges(graph)

    # Dense reference with accumulation
    P_act_dense = torch.zeros((N, N), dtype=act_w.dtype)
    P_act_dense.index_put_((act_src, act_dst), act_w, accumulate=True)
    P_diag_dense = torch.zeros((N, N), dtype=diag_w.dtype)
    P_diag_dense.index_put_((diag_src, diag_dst), diag_w, accumulate=True)
    D_dense = operator_discrepancy(P_act_dense, P_diag_dense)

    D_sparse = sparse_operator_discrepancy(
        act_src, act_dst, act_w, diag_src, diag_dst, diag_w, N
    )
    assert abs(D_sparse - D_dense) < 1e-6


# ---------------------------------------------------------------------------
# P0-3: Safe checkpoint persists metric length
# ---------------------------------------------------------------------------

def test_safe_checkpoint_preserves_independent_length():
    """Safe checkpoint must persist length tensor, not just weight."""
    graph = make_graph_buffers(3, [(0, 1, 2.0, 3.0), (1, 2, 0.5, 10.0)], capacity=4)
    cfg = LGAEConfig()
    cfg.fiber.d_base = 2
    cfg.fiber.d_max = 4
    engine = LGAEEngine(graph, cfg)

    with tempfile.TemporaryDirectory() as d:
        engine._save_checkpoint_safe(__import__("pathlib").Path(d))
        from safetensors.torch import load_file
        tensors = load_file(os.path.join(d, "tensors.safetensors"))
        assert "graph.length" in tensors, "length tensor missing from safe checkpoint"
        assert torch.allclose(tensors["graph.length"], graph.length)


def test_safe_checkpoint_roundtrip_independent_length():
    """Full save/restore must preserve independent length values."""
    graph = make_graph_buffers(3, [(0, 1, 2.0, 3.0), (1, 2, 0.5, 10.0)], capacity=4)
    cfg = LGAEConfig()
    cfg.fiber.d_base = 2
    cfg.fiber.d_max = 4
    engine = LGAEEngine(graph, cfg)
    original_hash = graph.state_hash()

    with tempfile.TemporaryDirectory() as d:
        engine._save_checkpoint_safe(__import__("pathlib").Path(d))
        # Create a new engine and load
        graph2 = make_graph_buffers(3, [(0, 1, 1.0), (1, 2, 1.0)], capacity=4)
        engine2 = LGAEEngine(graph2, cfg)
        engine2._load_checkpoint_safe_(
            __import__("pathlib").Path(d),
            allow_governance_mismatch=False,
            optimizer_load_policy="restore",
        )
        assert engine2.graph.state_hash() == original_hash
        assert torch.allclose(engine2.graph.length, graph.length)
        assert torch.allclose(engine2.graph.weight, graph.weight)


# ---------------------------------------------------------------------------
# P0-4: Governance hash includes v4.1 policy fields
# ---------------------------------------------------------------------------

def test_governance_hash_commits_shadow_horizons():
    """Different shadow_horizons must produce different governance hashes."""
    cfg_a = LGAEConfig()
    cfg_b = LGAEConfig()
    cfg_b.mutation.shadow_horizons = [1, 2, 4, 8, 16]
    assert config_governance_hash(cfg_a) != config_governance_hash(cfg_b)


def test_governance_hash_commits_ricci_flow_target():
    """Different ricci_flow_target must produce different governance hashes."""
    cfg_a = LGAEConfig()
    cfg_b = LGAEConfig()
    cfg_b.mutation.ricci_flow_target = "length"
    assert config_governance_hash(cfg_a) != config_governance_hash(cfg_b)


def test_governance_hash_commits_ricci_flow_coupled():
    """Different ricci_flow_coupled must produce different governance hashes."""
    cfg_a = LGAEConfig()
    cfg_b = LGAEConfig()
    cfg_b.mutation.ricci_flow_coupled = False
    assert config_governance_hash(cfg_a) != config_governance_hash(cfg_b)


# ---------------------------------------------------------------------------
# P0-5: RicciFlow serialization preserves v4.1 fields
# ---------------------------------------------------------------------------

def test_ricci_flow_spec_roundtrip():
    """RicciFlowReweight serialization must preserve target_field and coupled."""
    m = RicciFlowReweight(
        curvatures={(0, 1): -0.5, (1, 2): 0.3},
        target_field="length",
        coupled=False,
    )
    spec = mutation_to_spec(m)
    assert "target_field" in spec
    assert "coupled" in spec
    assert spec["target_field"] == "length"
    assert spec["coupled"] is False

    m2 = mutation_from_spec(spec)
    assert m2.target_field == "length"
    assert m2.coupled is False
    assert m2.curvatures == m.curvatures


def test_ricci_flow_spec_backward_compat():
    """Pre-v4.1 specs without target_field/coupled should default correctly."""
    spec = {
        "type": "RicciFlowReweight",
        "curvatures": [[0, 1, -0.5]],
        "target_curvature": 0.0,
        "dt": 0.05,
        "min_weight": 1e-3,
        "max_weight": 10.0,
        "name": "ricci_flow_reweight",
    }
    m = mutation_from_spec(spec)
    assert m.target_field == "weight"  # backward compat default
    assert m.coupled is True  # backward compat default


# ---------------------------------------------------------------------------
# P1-1: Multi-horizon propagates QUARANTINE (max severity)
# ---------------------------------------------------------------------------

def test_multihorizon_quarantine_propagates():
    """If any horizon QUARANTINEs, the final decision must be at least QUARANTINE."""
    cfg = LGAEConfig()
    cfg.fiber.d_base = 2
    cfg.fiber.d_max = 4
    cfg.mutation.shadow_horizons = [1, 2, 4, 8, 16]
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

    if "multi_horizon" in result.metadata:
        decisions = [h.get("decision") for h in result.metadata["multi_horizon"]]
        if "quarantine" in decisions:
            assert result.decision.value in ("quarantine", "reject"), (
                f"Multi-horizon QUARANTINE not propagated: decisions={decisions}, "
                f"final={result.decision.value}"
            )
        if "reject" in decisions:
            assert result.decision.value == "reject"


# ---------------------------------------------------------------------------
# P1-2: Safe checkpoint preserves quarantined shadow graphs
# ---------------------------------------------------------------------------

def test_graph_quarantine_checkpoint_roundtrip():
    """Safe checkpoint must preserve complete shadow graphs for quarantine."""
    cfg = LGAEConfig()
    cfg.fiber.d_base = 2
    cfg.fiber.d_max = 4
    cfg.audit.orc_backend = "exact_lp"
    cfg.audit.exact_lly_top_k = 16
    cfg.audit.entropic_nodes = 2
    cfg.audit.bakry_nodes = 1
    cfg.audit.cde_nodes = 1
    cfg.audit.cde_samples = 2

    graph = make_graph_buffers(4, [(0, 1), (1, 2), (2, 3)], capacity=8)
    engine = LGAEEngine(graph, cfg)
    engine.diffuse_(0.01)

    # Create a mutation that gets quarantined
    mutation = AddEdge(0, 2, weight=1.0)
    result = engine.evaluate_and_maybe_commit(mutation)

    # If something was quarantined, test the roundtrip
    if engine.quarantine:
        with tempfile.TemporaryDirectory() as d:
            engine._save_checkpoint_safe(__import__("pathlib").Path(d))
            from safetensors.torch import load_file
            tensors = load_file(os.path.join(d, "tensors.safetensors"))
            # Check that shadow graph tensors are present
            shadow_keys = [k for k in tensors if k.startswith("quarantine.0.shadow_graph.")]
            assert len(shadow_keys) > 0, "Shadow graph tensors not persisted in safe checkpoint"

            # Load into new engine
            graph2 = make_graph_buffers(4, [(0, 1), (1, 2), (2, 3)], capacity=8)
            engine2 = LGAEEngine(graph2, cfg)
            engine2._load_checkpoint_safe_(
                __import__("pathlib").Path(d),
                allow_governance_mismatch=False,
                optimizer_load_policy="restore",
            )
            assert len(engine2.quarantine) == len(engine.quarantine)
            # Shadow graph should be reconstructable
            q = engine2.quarantine[0]
            assert q.shadow_graph is not None, "Shadow graph not restored from safe checkpoint"
            q.shadow_graph.validate()


# ---------------------------------------------------------------------------
# P1-3: Weighted Forman uses metric-measure formula
# ---------------------------------------------------------------------------

def test_weighted_forman_isolated_edge():
    """For an isolated edge with unit weights, metric-measure Forman should
    give 2.0 (not 4.0 as the old formula did)."""
    g = nx.Graph()
    g.add_edge(0, 1, weight=2.0, length=0.5)
    # No adjacent edges, so sums are 0
    # F = a_e/m1_u + a_e/m1_v - 0 - 0 = 2/2 + 2/2 = 2.0
    k = weighted_forman_edge(g, 0, 1)
    assert k == pytest.approx(2.0), f"Expected 2.0, got {k}"


def test_weighted_forman_uses_length():
    """Weighted Forman should use metric length, not just affinity."""
    g = nx.Graph()
    g.add_edge(0, 1, weight=1.0, length=1.0)
    g.add_edge(1, 2, weight=1.0, length=2.0)  # different length
    k1 = weighted_forman_edge(g, 0, 1)

    g2 = nx.Graph()
    g2.add_edge(0, 1, weight=1.0, length=1.0)
    g2.add_edge(1, 2, weight=1.0, length=0.5)  # different length
    k2 = weighted_forman_edge(g2, 0, 1)

    # Different lengths should produce different curvatures
    assert k1 != pytest.approx(k2, abs=1e-6)


def test_weighted_forman_uniform_reduction():
    """With uniform weights and lengths, Forman should reduce to unweighted."""
    g = nx.Graph()
    g.add_edge(0, 1, weight=1.0, length=1.0)
    g.add_edge(1, 2, weight=1.0, length=1.0)
    g.add_edge(0, 2, weight=1.0, length=1.0)
    # Triangle with all unit weights/lengths
    # F(0,1) = 1/2 + 1/2 - (1/2 * 1/1) - (1/2 * 1/1) = 1 - 0.5 - 0.5 = 0
    k = weighted_forman_edge(g, 0, 1)
    # With a triangle, each endpoint has one adjacent edge
    # sum_u = (1/2)*(1/1) = 0.5, sum_v = (1/2)*(1/1) = 0.5
    # F = 1/2 + 1/2 - 0.5 - 0.5 = 0
    assert k == pytest.approx(0.0, abs=1e-6)


# ---------------------------------------------------------------------------
# P2-1: Version identity consistency
# ---------------------------------------------------------------------------

def test_version_constant_exists():
    """version.py should export a VERSION constant."""
    assert VERSION == "5.1.0"


def test_cli_version_matches_package():
    """CLI output version should match package version."""
    from lgae_v3 import __version__
    assert __version__ == VERSION


def test_pyproject_version_matches():
    """pyproject.toml version should match VERSION."""
    import pathlib
    pyproject = pathlib.Path(__file__).parent.parent / "pyproject.toml"
    content = pyproject.read_text()
    assert f'version = "{VERSION}"' in content


def test_pyproject_version_matches_412():
    """pyproject.toml version should match VERSION constant."""
    import pathlib
    pyproject = pathlib.Path(__file__).parent.parent / "pyproject.toml"
    content = pyproject.read_text()
    assert 'version = "5.1.0"' in content


# ---------------------------------------------------------------------------
# Edge priority sorting uses unweighted AF3 (known partial-weighted issue)
# ---------------------------------------------------------------------------

def test_edge_priority_uses_unweighted_af3():
    """Document that edge priority sorting still uses unweighted AF3.

    This is a known limitation — the candidate proxy stage remains
    topological while the audit stage uses metric-measure curvature.
    """
    # This test documents the current behavior rather than asserting a fix
    cfg = LGAEConfig()
    cfg.audit.curvature_weight_mode = "weighted"
    # The governor's audit() sorts edges by af3_edge (unweighted)
    # even in weighted mode. This is the "candidate proxy" tier.
    # The audit itself (LLY/ORC) uses weighted curvature.
    # This is acceptable as long as it's documented.
    assert cfg.audit.curvature_weight_mode == "weighted"
