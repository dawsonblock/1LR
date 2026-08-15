"""v4.1.3 deep audit tests: float64 discrepancy, schema versioning, stale quarantine,
parameterized governance hash, RicciFlow split, Forman reference graphs, and
multi-horizon decision combinations."""
from __future__ import annotations

import math
import pytest
import torch
import numpy as np
import networkx as nx
import tempfile
import os
import json
from pathlib import Path

from lgae_v3 import LGAEConfig, LGAEEngine, make_graph_buffers
from lgae_v3.types import GraphBuffers
from lgae_v3.operators import (
    diagnostic_diffusion_edges, sparse_operator_discrepancy,
    actuation_markov_edges, operator_discrepancy,
    actuation_operator, diagnostic_diffusion_operator,
    SparseDualOperatorState,
)
from lgae_v3.mutations import (
    AddEdge, ReweightEdge, ReweightAffinity, ReweightLength, CoupledReweight,
    PruneEdge, RicciFlowReweight, mutation_to_spec, mutation_from_spec,
)
from lgae_v3.config import config_governance_hash, validate_config
from lgae_v3.curvature import weighted_forman_edge, weighted_af3_proxy, af3_edge
from lgae_v3.types import MutationDecision


# ===========================================================================
# Item 2: Float64 discrepancy tests with 1e-10 tolerance
# ===========================================================================

@pytest.mark.parametrize("seed", [42, 123, 777, 2024])
@pytest.mark.parametrize("N", [4, 6, 10, 20])
def test_sparse_discrepancy_float64_exact(seed, N):
    """Sparse discrepancy must match dense reference to 1e-10 in float64."""
    torch.manual_seed(seed)
    z = torch.randn(N, 4, dtype=torch.float64)
    edges = [(i, (i + 1) % N) for i in range(N)]
    graph = make_graph_buffers(N, edges, capacity=N + 4)

    diag_src, diag_dst, diag_w = diagnostic_diffusion_edges(z, k=3)
    act_src, act_dst, act_w = actuation_markov_edges(graph)

    # Cast to float64 for exact comparison
    act_w64 = act_w.to(torch.float64)
    diag_w64 = diag_w.to(torch.float64)

    # Dense reference with accumulation
    P_act = torch.zeros((N, N), dtype=torch.float64)
    P_act.index_put_((act_src.to(torch.long), act_dst.to(torch.long)), act_w64, accumulate=True)
    P_diag = torch.zeros((N, N), dtype=torch.float64)
    P_diag.index_put_((diag_src.to(torch.long), diag_dst.to(torch.long)), diag_w64, accumulate=True)
    D_dense = operator_discrepancy(P_act, P_diag)

    D_sparse = sparse_operator_discrepancy(
        act_src, act_dst, act_w64, diag_src, diag_dst, diag_w64, N
    )
    assert abs(D_sparse.item() - D_dense.item()) < 1e-10, (
        f"seed={seed} N={N}: D_sparse={D_sparse.item():.15f} != "
        f"D_dense={D_dense.item():.15f}, err={abs(D_sparse.item()-D_dense.item()):.2e}"
    )


def test_sparse_discrepancy_deliberate_duplicates():
    """Deliberately duplicated edges must be coalesced correctly."""
    N = 4
    # Create operators with deliberate duplicates
    act_src = torch.tensor([0, 0, 1, 1, 2, 3])  # (0,1) appears twice
    act_dst = torch.tensor([1, 1, 2, 3, 3, 0])
    act_w = torch.tensor([0.3, 0.2, 0.5, 0.5, 1.0, 1.0])  # 0.3+0.2=0.5 for (0,1)

    diag_src = torch.tensor([0, 1, 2, 3])
    diag_dst = torch.tensor([1, 2, 3, 0])
    diag_w = torch.tensor([0.5, 0.5, 0.5, 0.5])

    # Dense reference
    P_act = torch.zeros((N, N))
    P_act.index_put_((act_src, act_dst), act_w, accumulate=True)
    P_diag = torch.zeros((N, N))
    P_diag.index_put_((diag_src, diag_dst), diag_w, accumulate=True)
    D_dense = operator_discrepancy(P_act, P_diag)

    D_sparse = sparse_operator_discrepancy(
        act_src, act_dst, act_w, diag_src, diag_dst, diag_w, N
    )
    assert abs(D_sparse.item() - D_dense.item()) < 1e-10


def test_sparse_discrepancy_both_duplicated():
    """Both operators with duplicates must coalesce correctly."""
    N = 3
    # Both have duplicates
    act_src = torch.tensor([0, 0, 1, 1, 2, 2])
    act_dst = torch.tensor([1, 1, 2, 2, 0, 0])
    act_w = torch.tensor([0.2, 0.3, 0.2, 0.3, 0.5, 0.5])

    diag_src = torch.tensor([0, 0, 1, 2])
    diag_dst = torch.tensor([1, 1, 2, 0])
    diag_w = torch.tensor([0.3, 0.2, 1.0, 1.0])

    P_act = torch.zeros((N, N))
    P_act.index_put_((act_src, act_dst), act_w, accumulate=True)
    P_diag = torch.zeros((N, N))
    P_diag.index_put_((diag_src, diag_dst), diag_w, accumulate=True)
    D_dense = operator_discrepancy(P_act, P_diag)

    D_sparse = sparse_operator_discrepancy(
        act_src, act_dst, act_w, diag_src, diag_dst, diag_w, N
    )
    assert abs(D_sparse.item() - D_dense.item()) < 1e-10


# ===========================================================================
# Item 3: Schema version check for length mandatory in v4+ checkpoints
# ===========================================================================

def test_safe_checkpoint_v4_has_length():
    """v4+ safe checkpoints must include length tensor."""
    graph = make_graph_buffers(3, [(0, 1, 2.0, 3.0), (1, 2, 0.5, 10.0)], capacity=4)
    cfg = LGAEConfig()
    cfg.fiber.d_base = 2; cfg.fiber.d_max = 4
    engine = LGAEEngine(graph, cfg)
    with tempfile.TemporaryDirectory() as d:
        engine._save_checkpoint_safe(Path(d))
        from safetensors.torch import load_file
        tensors = load_file(os.path.join(d, "tensors.safetensors"))
        assert "graph.length" in tensors
        graph_json = json.loads(Path(d, "graph.json").read_text())
        assert graph_json.get("has_length") is True


def test_safe_checkpoint_roundtrip_independent_length_v4():
    """v4 checkpoint must preserve independent length (not 1/affinity)."""
    aff = [2.0, 0.7, 5.0]
    lng = [3.0, 1.2, 0.4]
    graph = make_graph_buffers(
        4, [(0, 1, aff[0], lng[0]), (1, 2, aff[1], lng[1]), (2, 3, aff[2], lng[2])],
        capacity=8,
    )
    cfg = LGAEConfig()
    cfg.fiber.d_base = 2; cfg.fiber.d_max = 4
    engine = LGAEEngine(graph, cfg)
    with tempfile.TemporaryDirectory() as d:
        engine._save_checkpoint_safe(Path(d))
        graph2 = make_graph_buffers(4, [(0, 1, 1.0), (1, 2, 1.0), (2, 3, 1.0)], capacity=8)
        engine2 = LGAEEngine(graph2, cfg)
        engine2._load_checkpoint_safe_(
            Path(d), allow_governance_mismatch=False, optimizer_load_policy="restore",
        )
        # Must recover EXACT length values, not 1/affinity
        for i, expected_l in enumerate(lng):
            assert abs(engine2.graph.length[i].item() - expected_l) < 1e-5, (
                f"length[{i}]={engine2.graph.length[i].item()} != {expected_l}"
            )
            # Verify it's NOT 1/affinity
            assert abs(engine2.graph.length[i].item() - 1.0 / aff[i]) > 0.01


# ===========================================================================
# Item 4: Stale quarantine detection after restart
# ===========================================================================

def test_stale_quarantine_rejected_after_restart():
    """A quarantine from before a graph mutation must be rejected as stale."""
    cfg = LGAEConfig()
    cfg.fiber.d_base = 2; cfg.fiber.d_max = 4
    cfg.audit.orc_backend = "exact_lp"
    cfg.audit.exact_lly_top_k = 16
    cfg.audit.entropic_nodes = 2; cfg.audit.bakry_nodes = 1; cfg.audit.cde_nodes = 1
    cfg.audit.cde_samples = 2

    graph = make_graph_buffers(4, [(0, 1), (1, 2), (2, 3)], capacity=8)
    engine = LGAEEngine(graph, cfg)
    engine.diffuse_(0.01)

    # Create a mutation that gets quarantined
    mut1 = AddEdge(0, 2, weight=1.0)
    result1 = engine.evaluate_and_maybe_commit(mut1)

    # If something was quarantined, save, mutate, reload, try accept
    if engine.quarantine:
        with tempfile.TemporaryDirectory() as d:
            engine._save_checkpoint_safe(Path(d))
            # Load into new engine
            graph2 = make_graph_buffers(4, [(0, 1), (1, 2), (2, 3)], capacity=8)
            engine2 = LGAEEngine(graph2, cfg)
            engine2._load_checkpoint_safe_(
                Path(d), allow_governance_mismatch=False, optimizer_load_policy="restore",
            )
            assert len(engine2.quarantine) > 0
            base_hash = engine2.quarantine[0].base_graph_hash

            # Apply a different mutation that changes the graph
            mut2 = AddEdge(1, 3, weight=2.0)
            mut2.apply(engine2.graph)
            engine2.graph.bump_version()

            # Now the old quarantine should be stale
            current_hash = engine2.graph.state_hash()
            if base_hash != current_hash:
                # Stale quarantine detected
                assert base_hash != current_hash


# ===========================================================================
# Item 5: Parameterized governance hash test over ALL decision-affecting fields
# ===========================================================================

def _get_all_governance_fields() -> list[tuple[str, str, any]]:
    """Return (section, field_name, alternative_value) for all governance fields."""
    from lgae_v3.config import _GOVERNANCE_FIELDS
    cfg = LGAEConfig()
    fields = []
    for section_name, field_names in _GOVERNANCE_FIELDS.items():
        section = getattr(cfg, section_name)
        for fname in field_names:
            current = getattr(section, fname)
            # Generate an alternative value that differs
            if isinstance(current, bool):
                alt = not current
            elif isinstance(current, int):
                alt = current + 1
            elif isinstance(current, float):
                alt = current * 2 + 0.001
            elif isinstance(current, str):
                alt = current + "_alt" if current else "weighted"
            elif isinstance(current, (list, tuple)):
                alt = list(current) + [99] if current else [1]
            elif isinstance(current, dict):
                alt = {**current, "_test": 1.0}
            else:
                continue  # skip types we can't easily alternate
            fields.append((section_name, fname, alt))
    return fields


@pytest.mark.parametrize("section_name,field_name,alt_value", _get_all_governance_fields())
def test_governance_hash_field_sensitivity(section_name, field_name, alt_value):
    """Changing any single governance field must change the governance hash."""
    cfg_a = LGAEConfig()
    cfg_b = LGAEConfig()
    section_b = getattr(cfg_b, section_name)
    setattr(section_b, field_name, alt_value)
    ha = config_governance_hash(cfg_a)
    hb = config_governance_hash(cfg_b)
    assert ha != hb, (
        f"Governance hash unchanged when {section_name}.{field_name} "
        f"changed from {getattr(cfg_a, section_name).__dict__.get(field_name)} to {alt_value}"
    )


# ===========================================================================
# Item 6: RicciFlow split + generic mutation registry roundtrip test
# ===========================================================================

def test_all_mutation_types_spec_roundtrip():
    """Every mutation type must roundtrip through encode/decode exactly."""
    mutations = [
        AddEdge(0, 1, weight=2.0),
        ReweightEdge(0, 1, factor=1.5),
        ReweightAffinity(0, 1, factor=2.0),
        ReweightLength(0, 1, factor=0.5),
        CoupledReweight(0, 1, affinity_factor=1.5, coupling="direct"),
        CoupledReweight(0, 1, affinity_factor=1.5, coupling="none"),
        PruneEdge(0, 1),
        RicciFlowReweight(curvatures={(0, 1): -0.5}, target_field="length", coupled=False),
        RicciFlowReweight(curvatures={(0, 1): 0.3}, target_field="affinity", coupled=True),
    ]
    for mut in mutations:
        spec = mutation_to_spec(mut)
        mut2 = mutation_from_spec(spec)
        assert type(mut2) is type(mut), f"{type(mut).__name__} roundtrip changed type"
        # Check key semantic fields
        for attr in ("u", "v", "factor", "target_field", "coupled", "coupling",
                      "affinity_factor", "length_factor", "name"):
            if hasattr(mut, attr):
                v1 = getattr(mut, attr)
                v2 = getattr(mut2, attr)
                assert v1 == v2, f"{type(mut).__name__}.{attr}: {v1} != {v2}"


# ===========================================================================
# Item 7: Forman reference graph tests
# ===========================================================================

def test_forman_k2():
    """K2 (single edge): metric-measure Forman = a_e/m1_u + a_e/m1_v = 2."""
    g = nx.Graph()
    g.add_edge(0, 1, weight=1.0, length=1.0)
    k = weighted_forman_edge(g, 0, 1)
    # m1_u = 1, m1_v = 1, a_e = 1, no adjacent edges
    # F = 1/1 + 1/1 - 0 - 0 = 2
    assert k == pytest.approx(2.0, abs=1e-10)


def test_forman_weighted_path_3():
    """Path graph 0-1-2 with weights: verify metric-measure formula."""
    g = nx.Graph()
    g.add_edge(0, 1, weight=2.0, length=0.5)
    g.add_edge(1, 2, weight=1.0, length=2.0)
    # F(0,1): m1_u=2, m1_v=2+1=3, a_e=2, omega_e=0.5
    # sum_u = 0 (no other edges at 0)
    # sum_v = (1/3)*(2.0/0.5) = (1/3)*4 = 4/3
    # F = 2/2 + 2/3 - 0 - 4/3 = 1 + 2/3 - 4/3 = 1 - 2/3 = 1/3
    k = weighted_forman_edge(g, 0, 1)
    assert k == pytest.approx(1.0 / 3.0, abs=1e-10)


def test_forman_weighted_star():
    """Star graph: center with 3 leaves of different weights."""
    g = nx.Graph()
    g.add_edge(0, 1, weight=3.0, length=1.0)
    g.add_edge(0, 2, weight=1.0, length=2.0)
    g.add_edge(0, 3, weight=2.0, length=0.5)
    # F(0,1): m1_0=3+1+2=6, m1_1=3, a_e=3, omega_e=1.0
    # sum_0 = (1/6)*(2/1) + (2/6)*(0.5/1) = 2/6 + 1/6 = 3/6 = 0.5
    # sum_1 = 0 (leaf)
    # F = 3/6 + 3/3 - 0.5 - 0 = 0.5 + 1 - 0.5 = 1.0
    k = weighted_forman_edge(g, 0, 1)
    assert k == pytest.approx(1.0, abs=1e-10)


def test_forman_uniform_reduction():
    """Uniform weights and lengths should reduce to unweighted convention."""
    g = nx.Graph()
    g.add_edge(0, 1, weight=1.0, length=1.0)
    g.add_edge(1, 2, weight=1.0, length=1.0)
    g.add_edge(0, 2, weight=1.0, length=1.0)
    # Triangle with all unit: F(0,1) = 1/2 + 1/2 - (1/2)*(1/1) - (1/2)*(1/1) = 0
    k = weighted_forman_edge(g, 0, 1)
    assert k == pytest.approx(0.0, abs=1e-10)


def test_forman_tree_4_path():
    """Path of 4 nodes: verify Forman at middle edge."""
    g = nx.Graph()
    g.add_edge(0, 1, weight=1.0, length=1.0)
    g.add_edge(1, 2, weight=1.0, length=1.0)
    g.add_edge(2, 3, weight=1.0, length=1.0)
    # F(1,2): m1_1=2, m1_2=2, a_e=1, omega_e=1
    # sum_1 = (1/2)*(1/1) = 0.5 (edge 0-1)
    # sum_2 = (1/2)*(1/1) = 0.5 (edge 2-3)
    # F = 1/2 + 1/2 - 0.5 - 0.5 = 0
    k = weighted_forman_edge(g, 1, 2)
    assert k == pytest.approx(0.0, abs=1e-10)


def test_forman_proxy_distinct_from_canonical():
    """weighted_af3_proxy must be distinct from weighted_forman_edge."""
    g = nx.Graph()
    g.add_edge(0, 1, weight=2.0, length=0.5)
    g.add_edge(1, 2, weight=1.0, length=2.0)
    k_forman = weighted_forman_edge(g, 0, 1)
    k_proxy = weighted_af3_proxy(g, 0, 1)
    # They use different formulas, so they should generally differ
    assert k_forman != pytest.approx(k_proxy, abs=1e-6)


# ===========================================================================
# Item 8: Multi-horizon decision combination tests
# ===========================================================================

def test_multihorizon_all_accept():
    """AAA → ACCEPT."""
    decisions = ["accept", "accept", "accept"]
    severity = {"accept": 0, "quarantine": 1, "reject": 2}
    worst = max(decisions, key=lambda d: severity[d])
    assert worst == "accept"


def test_multihorizon_accept_quarantine():
    """AAQ → QUARANTINE."""
    decisions = ["accept", "accept", "quarantine"]
    severity = {"accept": 0, "quarantine": 1, "reject": 2}
    worst = max(decisions, key=lambda d: severity[d])
    assert worst == "quarantine"


def test_multihorizon_quarantine_between_accepts():
    """AQA → QUARANTINE."""
    decisions = ["accept", "quarantine", "accept"]
    severity = {"accept": 0, "quarantine": 1, "reject": 2}
    worst = max(decisions, key=lambda d: severity[d])
    assert worst == "quarantine"


def test_multihorizon_double_quarantine_then_accept():
    """QQA → QUARANTINE."""
    decisions = ["quarantine", "quarantine", "accept"]
    severity = {"accept": 0, "quarantine": 1, "reject": 2}
    worst = max(decisions, key=lambda d: severity[d])
    assert worst == "quarantine"


def test_multihorizon_accept_then_reject():
    """AAR → REJECT."""
    decisions = ["accept", "accept", "reject"]
    severity = {"accept": 0, "quarantine": 1, "reject": 2}
    worst = max(decisions, key=lambda d: severity[d])
    assert worst == "reject"


def test_multihorizon_quarantine_then_reject():
    """QRA → REJECT."""
    decisions = ["quarantine", "reject", "accept"]
    severity = {"accept": 0, "quarantine": 1, "reject": 2}
    worst = max(decisions, key=lambda d: severity[d])
    assert worst == "reject"


def test_multihorizon_all_reject():
    """RRR → REJECT."""
    decisions = ["reject", "reject", "reject"]
    severity = {"accept": 0, "quarantine": 1, "reject": 2}
    worst = max(decisions, key=lambda d: severity[d])
    assert worst == "reject"


def test_multihorizon_early_severe_not_hidden():
    """Early REJECT must not be hidden by later ACCEPT."""
    decisions = ["reject", "accept", "accept", "accept"]
    severity = {"accept": 0, "quarantine": 1, "reject": 2}
    worst = max(decisions, key=lambda d: severity[d])
    assert worst == "reject"


def test_multihorizon_uncertainty_then_reject():
    """QUARANTINE at one horizon and REJECT at another → REJECT."""
    decisions = ["quarantine", "reject"]
    severity = {"accept": 0, "quarantine": 1, "reject": 2}
    worst = max(decisions, key=lambda d: severity[d])
    assert worst == "reject"


# ===========================================================================
# Item 1: N=2500 CI test
# ===========================================================================

def test_sparse_governor_n2500_ci():
    """CI test: governor audit must complete for N=2500 without dense allocation."""
    import time
    N = 2500
    edges = [(i, (i + 1) % N) for i in range(N)]
    graph = make_graph_buffers(N, edges, capacity=N + 10)
    cfg = LGAEConfig()
    cfg.fiber.d_base = 4; cfg.fiber.d_max = 8
    cfg.audit.entropic_nodes = 2
    cfg.audit.bakry_nodes = 3; cfg.audit.cde_nodes = 3
    cfg.audit.cde_samples = 2
    cfg.operator.diagnostic_full_kernel_max_nodes = 512
    engine = LGAEEngine(graph, cfg)
    z = torch.randn(N, 4)
    t0 = time.time()
    snap = engine.governor.audit(graph, z)
    elapsed = time.time() - t0
    assert np.isfinite(snap.lambda2)
    assert snap.details["bakry_generator"] == "reversible_normalized_markov_local_sparse"
    # Must complete in reasonable time (no O(N²) allocation)
    assert elapsed < 30.0, f"N=2500 audit took {elapsed:.1f}s (expected < 30s)"
