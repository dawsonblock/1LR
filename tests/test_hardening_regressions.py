import math
from types import SimpleNamespace

import networkx as nx
import pytest
import torch

from lgae_v3 import EdgeRole, LGAEConfig, LGAEEngine
from lgae_v3.config import validate_config
from lgae_v3.curvature import bakry_emery_curvature, weak_entropic_node
from lgae_v3.metrics import spawn_score_from_pressure, edge_diffusion_metrics, gamma_vector, diffusion_radius, local_variance
from lgae_v3.mutations import AddEdge, PruneEdge, ReweightEdge
from lgae_v3.operators import actuation_markov_edges, actuation_operator
from lgae_v3.types import GraphBuffers, MutationDecision, make_graph_buffers


def generator(g):
    n = len(g)
    P = torch.zeros(n, n, dtype=torch.float64)
    for u in g:
        for v in g.neighbors(u):
            P[u, v] = 1.0 / g.degree[u]
    return P - torch.eye(n, dtype=torch.float64)


@pytest.mark.parametrize("node,expected", [(0, 1.0), (1, 0.2928932188134524), (2, 0.2928932188134524), (3, 1.0)])
def test_bakry_p4_schur_oracle(node, expected):
    assert bakry_emery_curvature(generator(nx.path_graph(4)), node) == pytest.approx(expected, abs=1e-9)


def test_bakry_k2_remains_two():
    assert bakry_emery_curvature(generator(nx.path_graph(2)), 0) == pytest.approx(2.0, abs=1e-9)


def test_waf3_pressure_materially_changes_spawn_score():
    gamma = torch.tensor([0.2, 0.4, 0.8, 1.2])
    radius = torch.tensor([1.0, 1.1, 1.2, 1.3])
    var = torch.tensor([0.3, 0.4, 0.5, 0.6])
    residual = torch.arange(4, dtype=torch.float32)
    uncertainty = torch.zeros(4)
    capacity = torch.ones(4)
    zero = spawn_score_from_pressure(gamma, radius, var, torch.zeros(4), residual, uncertainty, capacity)
    pressured = spawn_score_from_pressure(gamma, radius, var, torch.tensor([0.0, 1.0, 5.0, 10.0]), residual, uncertainty, capacity)
    assert not torch.allclose(zero, pressured)
    assert pressured[-1] > zero[-1]


def test_default_governor_rejects_disconnection():
    cfg = LGAEConfig()
    cfg.fiber.d_base = 2
    cfg.fiber.d_max = 4
    cfg.operator.diagnostic_k = 3
    cfg.audit.exact_lly_top_k = 32
    cfg.audit.entropic_nodes = 4
    cfg.audit.bakry_nodes = 2
    cfg.audit.cde_nodes = 1
    cfg.audit.cde_samples = 2
    engine = LGAEEngine(make_graph_buffers(4, [(0, 1), (1, 2), (2, 3)], capacity=8), cfg)
    result = engine.evaluate_and_maybe_commit(PruneEdge(1, 2))
    assert result.decision is MutationDecision.REJECT
    assert "connected_component_increase" in result.reasons
    assert engine.graph.edge_count == 3


def test_graph_rejects_invalid_active_weight():
    g = make_graph_buffers(3, [(0, 1)], capacity=2)
    g.weight[0] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        g.validate()


@pytest.mark.parametrize("mutation", [
    AddEdge(-1, 1), AddEdge(0, 3), AddEdge(1, 1), AddEdge(0, 2, float("nan")),
    ReweightEdge(0, 1, factor=-1),
])
def test_mutation_inputs_fail_closed(mutation):
    g = make_graph_buffers(3, [(0, 1)], capacity=4)
    with pytest.raises((ValueError, RuntimeError)):
        mutation.apply(g)


def test_duplicate_constructor_rejected():
    with pytest.raises(ValueError, match="duplicate"):
        make_graph_buffers(3, [(0, 1), (1, 0)])


def test_roles_survive_graph_state_roundtrip():
    g = make_graph_buffers(3, [(0, 1), (1, 2)], roles=[EdgeRole.BRIDGE, EdgeRole.CLUSTER])
    restored = GraphBuffers.from_state_dict(g.to_state_dict())
    assert restored.state_hash() == g.state_hash()
    assert restored.active_roles().tolist() == g.active_roles().tolist()


def test_sparse_metrics_match_dense_reference():
    g = make_graph_buffers(5, [(0, 1), (1, 2), (2, 3), (3, 4), (0, 4)])
    z = torch.randn(5, 6)
    p = actuation_operator(g)
    src, dst, pw = actuation_markov_edges(g)
    m = edge_diffusion_metrics(z, src, dst, pw, 5)
    assert torch.allclose(m["gamma"], gamma_vector(z, p), atol=1e-6)
    assert torch.allclose(m["radius"], diffusion_radius(z, p), atol=1e-6)
    assert torch.allclose(m["local_var"], local_variance(z, p), atol=1e-6)


def test_entropic_empty_two_hop_is_positive_infinity():
    assert math.isinf(weak_entropic_node(nx.complete_graph(3), 0))


def test_entropic_solver_failure_is_not_optimistic(monkeypatch):
    import lgae_v3.curvature.entropic as ent
    monkeypatch.setattr(ent, "minimize", lambda *a, **k: SimpleNamespace(success=False, message="forced failure"))
    assert ent.weak_entropic_node(nx.path_graph(4), 1) is None
    detailed = ent.weak_entropic_node_detailed(nx.path_graph(4), 1)
    assert detailed.status == "solver_failed"


def test_unweighted_reference_mode_is_explicit():
    cfg = LGAEConfig()
    cfg.audit.curvature_weight_mode = "weighted"
    with pytest.raises(ValueError, match="unweighted_reference"):
        validate_config(cfg)
