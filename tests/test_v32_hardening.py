import math

import networkx as nx
import numpy as np
import pytest
import torch

from lgae_v3 import LGAEConfig, LGAEEngine, SOConnectionBank, make_graph_buffers, make_bucketed_graph_buffers, round_edge_capacity
from lgae_v3.curvature import log_sinkhorn_wasserstein, normalized_markov_generator, validate_reversible_markov
from lgae_v3.curvature.ollivier import _transport_lp
from lgae_v3.fibers import project_to_so_d
from lgae_v3.mutations import MutationCooldownTracker, PruneEdge, RicciFlowReweight, ReweightEdge
from lgae_v3.operators import spectral_gap_graphbuffers


def test_so_connection_bank_stays_special_orthogonal_after_optimizer_step():
    bank = SOConnectionBank(5, 3, parameterization="cayley", dtype=torch.float64)
    opt = torch.optim.Adam(bank.parameters(), lr=0.2)
    for _ in range(3):
        opt.zero_grad(set_to_none=True)
        r = bank.matrices()
        loss = (r[:, 0, 1] - 0.7).square().mean() + (r[:, 2, 0] + 0.2).square().mean()
        loss.backward()
        opt.step()
    orth, det = bank.invariant_error()
    assert float(orth.detach().max()) < 1e-10
    assert float(det.detach().max()) < 1e-10


def test_project_to_so_d_repairs_reflection():
    u = torch.eye(3, dtype=torch.float64).repeat(2, 1, 1)
    u[0, -1, -1] = -1.0
    u[1] += 0.05 * torch.randn(3, 3, dtype=torch.float64)
    r = project_to_so_d(u)
    eye = torch.eye(3, dtype=torch.float64)
    assert torch.allclose(r.transpose(-1, -2) @ r, eye.expand_as(r), atol=1e-10)
    assert torch.all(torch.linalg.det(r) > 0.999999)


def test_engine_gauge_transport_is_finite_and_shape_stable():
    cfg = LGAEConfig()
    cfg.fiber.d_base = 3
    cfg.fiber.d_max = 4
    cfg.fiber.gauge_dim = 3
    cfg.fiber.gauge_parameterization = "exp"
    graph = make_graph_buffers(4, [(0, 1), (1, 2), (2, 3)], capacity=8)
    eng = LGAEEngine(graph, cfg, dtype=torch.float64)
    assert eng.gauge_connections is not None
    before_shape = tuple(eng.gauge_connections.raw_generators.shape)
    out = eng.diffuse_(eta=0.01)
    assert torch.isfinite(out).all()
    assert tuple(eng.gauge_connections.raw_generators.shape) == before_shape
    orth, det = eng.gauge_connections.invariant_error()
    assert float(orth.detach().max()) < 1e-10
    assert float(det.detach().max()) < 1e-10


def test_log_sinkhorn_is_finite_at_small_epsilon_large_metric_and_tracks_lp():
    C = np.array([[0.0, 1000.0], [1000.0, 0.0]], dtype=float)
    a = np.array([0.999, 0.001], dtype=float)
    b = np.array([0.001, 0.999], dtype=float)
    exact = _transport_lp(C, a, b)
    approx = log_sinkhorn_wasserstein(C, a, b, epsilon=0.005, max_iter=2000, tolerance=1e-11)
    assert math.isfinite(approx)
    assert approx == pytest.approx(exact, rel=2e-3, abs=2.0)


def test_reversible_markov_generator_uses_volume_measure():
    P = torch.tensor([
        [0.0, 1.0, 0.0],
        [0.5, 0.0, 0.5],
        [0.0, 1.0, 0.0],
    ], dtype=torch.float64)
    m = validate_reversible_markov(P)
    assert torch.allclose(m, torch.tensor([0.25, 0.5, 0.25], dtype=torch.float64), atol=1e-10)
    Q, m2 = normalized_markov_generator(P)
    assert torch.allclose(Q.sum(-1), torch.zeros(3, dtype=torch.float64), atol=1e-12)
    assert torch.allclose(m, m2)


def test_log_ricci_flow_never_creates_nonpositive_weights():
    graph = make_graph_buffers(3, [(0, 1, 0.5), (1, 2, 0.5)], capacity=4)
    mutation = RicciFlowReweight(
        {(0, 1): 1000.0, (1, 2): -1000.0}, target_curvature=0.0, dt=1.0,
        min_weight=1e-3, max_weight=10.0,
    )
    mutation.apply(graph)
    _, _, w = graph.active()
    assert torch.all(w >= 1e-3)
    assert torch.all(w <= 10.0)
    assert torch.isfinite(w).all()


def test_cooldown_and_deadband_prevent_surgery_thrashing():
    tracker = MutationCooldownTracker(cooldown_steps=20)
    m = ReweightEdge(1, 2, factor=1.1)
    assert tracker.allows(m, 0)[0]
    tracker.record(m, 0)
    allowed, blocked = tracker.allows(m, 5)
    assert not allowed and blocked[(1, 2)] == 15
    assert tracker.allows(m, 20)[0]
    assert tracker.surgery_action(-0.3, add_threshold=-0.2, deadband=0.05, prune_threshold=0.2) == "add"
    assert tracker.surgery_action(0.0, add_threshold=-0.2, deadband=0.05, prune_threshold=0.2) is None
    assert tracker.surgery_action(0.3, add_threshold=-0.2, deadband=0.05, prune_threshold=0.2) == "prune"


def test_sparse_lobpcg_gap_matches_exact_cycle():
    n = 24
    graph = make_graph_buffers(n, [(u, v) for u, v in nx.cycle_graph(n).edges()], dtype=torch.float64)
    exact, method1 = spectral_gap_graphbuffers(graph, solver="exact")
    approx, method2 = spectral_gap_graphbuffers(graph, solver="lobpcg", lobpcg_min_nodes=6, niter=200, tol=1e-9, seed=7)
    assert method1 == "exact"
    assert method2 == "lobpcg"
    assert approx == pytest.approx(exact, rel=2e-3, abs=1e-5)


def test_spectral_gap_returns_zero_for_isolated_vertex_without_nan():
    graph = make_graph_buffers(4, [(0, 1), (1, 2)], capacity=8)
    gap, method = spectral_gap_graphbuffers(graph, solver="auto", lobpcg_min_nodes=6)
    assert gap == 0.0
    assert method == "isolated_vertex"


def test_fixed_capacity_bucket_rounding():
    assert round_edge_capacity(257, bucket_size=256, reserve_buckets=1) == 768
    graph = make_bucketed_graph_buffers(3, [(0, 1), (1, 2)], bucket_size=4, reserve_buckets=1)
    assert graph.capacity == 8
    assert graph.edge_count == 2


def test_local_bridge_gate_rejects_before_global_audit():
    cfg = LGAEConfig()
    cfg.fiber.d_base = 2; cfg.fiber.d_max = 4
    eng = LGAEEngine(make_graph_buffers(4, [(0, 1), (1, 2), (2, 3)], capacity=8), cfg)
    r = eng.evaluate_and_maybe_commit(PruneEdge(1, 2))
    assert r.decision.value == "reject"
    assert r.before is None
    assert "local_bridge_prune_would_disconnect" in r.reasons


def test_engine_cooldown_rejects_immediate_repeat_reweight():
    cfg = LGAEConfig()
    cfg.fiber.d_base = 2; cfg.fiber.d_max = 4
    cfg.mutation.edge_cooldown_steps = 20
    cfg.audit.exact_lly_top_k = 16; cfg.audit.entropic_nodes = 3; cfg.audit.bakry_nodes = 1; cfg.audit.cde_nodes = 1; cfg.audit.cde_samples = 2
    eng = LGAEEngine(make_graph_buffers(3, [(0, 1), (1, 2), (0, 2)], capacity=8), cfg)
    first = eng.evaluate_and_maybe_commit(ReweightEdge(0, 1, factor=1.05, min_weight=1e-3, max_weight=10.0))
    assert first.decision.value == "accept"
    second = eng.evaluate_and_maybe_commit(ReweightEdge(0, 1, factor=1.05, min_weight=1e-3, max_weight=10.0))
    assert second.decision.value == "reject"
    assert "edge_mutation_cooldown" in second.reasons


def test_gauge_slot_reset_on_committed_prune():
    cfg = LGAEConfig()
    cfg.fiber.d_base = 2; cfg.fiber.d_max = 3; cfg.fiber.gauge_dim = 2
    cfg.audit.exact_lly_top_k = 16; cfg.audit.entropic_nodes = 3; cfg.audit.bakry_nodes = 1; cfg.audit.cde_nodes = 1; cfg.audit.cde_samples = 2
    eng = LGAEEngine(make_graph_buffers(3, [(0, 1), (1, 2), (0, 2)], capacity=8), cfg)
    assert eng.gauge_connections is not None
    with torch.no_grad():
        eng.gauge_connections.raw_generators[0, 0, 1] = 2.0
    r = eng.evaluate_and_maybe_commit(PruneEdge(0, 1))
    assert r.decision.value == "accept"
    assert torch.count_nonzero(eng.gauge_connections.raw_generators[0]).item() == 0


def test_padded_edge_refresh_keeps_static_shapes_after_mutation():
    from lgae_v3.training import padded_markov_edges, refresh_padded_markov_edges_
    graph = make_graph_buffers(4, [(0, 1), (1, 2), (2, 3)], capacity=8)
    src, dst, weight, valid = padded_markov_edges(graph, max_edges=32)
    shapes = (src.shape, dst.shape, weight.shape, valid.shape)
    # Add a nonduplicate edge directly to authoritative test graph.
    from lgae_v3.mutations import AddEdge
    AddEdge(0, 2).apply(graph)
    count = refresh_padded_markov_edges_(graph, src, dst, weight, valid)
    assert count > 0
    assert (src.shape, dst.shape, weight.shape, valid.shape) == shapes


def test_log_sinkhorn_preserves_exact_zero_mass_support():
    C = np.array([[0.0, 3.0], [3.0, 0.0]], dtype=float)
    a = np.array([1.0, 0.0], dtype=float)
    b = np.array([0.0, 1.0], dtype=float)
    value = log_sinkhorn_wasserstein(C, a, b, epsilon=0.01, max_iter=32, tolerance=1e-12)
    assert value == pytest.approx(3.0, abs=1e-10)


def test_log_sinkhorn_fails_closed_on_nonconvergence():
    C = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=float)
    a = np.array([0.9, 0.1], dtype=float)
    b = np.array([0.2, 0.8], dtype=float)
    with pytest.raises(RuntimeError, match="did not converge"):
        log_sinkhorn_wasserstein(C, a, b, epsilon=0.01, max_iter=1, tolerance=1e-14)


def test_stationary_measure_power_iteration_matches_path_volume():
    P = torch.tensor([
        [0.0, 1.0, 0.0, 0.0],
        [0.5, 0.0, 0.5, 0.0],
        [0.0, 0.5, 0.0, 0.5],
        [0.0, 0.0, 1.0, 0.0],
    ], dtype=torch.float64)
    m = validate_reversible_markov(P)
    expected = torch.tensor([1.0, 2.0, 2.0, 1.0], dtype=torch.float64) / 6.0
    assert torch.allclose(m, expected, atol=1e-8, rtol=1e-8)


def test_diffusion_does_not_accumulate_hidden_inactive_latent_state():
    cfg = LGAEConfig()
    cfg.fiber.d_base = 1
    cfg.fiber.d_max = 3
    cfg.fiber.gauge_dim = 0
    graph = make_graph_buffers(3, [(0, 1), (1, 2)], capacity=8)
    eng = LGAEEngine(graph, cfg, dtype=torch.float64)
    with torch.no_grad():
        eng.fibers.latent[:, 1:] = 1e9
    eng.diffuse_(eta=0.01)
    assert torch.count_nonzero(eng.fibers.latent[:, 1:]).item() == 0


def test_normalized_generator_removes_float32_row_sum_roundoff():
    from lgae_v3.curvature import normalized_markov_generator
    K = torch.tensor([
        [0.0, 0.7, 0.2, 0.1],
        [0.7, 0.0, 0.2, 0.1],
        [0.2, 0.2, 0.0, 0.6],
        [0.1, 0.1, 0.6, 0.0],
    ], dtype=torch.float32)
    P = K / K.sum(-1, keepdim=True)
    Q, m = normalized_markov_generator(P)
    assert Q.dtype == torch.float64
    assert torch.allclose(Q.sum(-1), torch.zeros(4, dtype=torch.float64), atol=1e-14, rtol=0)
    assert torch.all(m > 0)


def test_gauge_connections_receive_integrated_training_gradients():
    from torch import nn
    from lgae_v3.training import LGAETrainCore, padded_markov_edges_with_slots

    cfg = LGAEConfig()
    cfg.fiber.d_base = 2
    cfg.fiber.d_max = 3
    cfg.fiber.gauge_dim = 2
    cfg.fiber.gauge_parameterization = "cayley"
    graph = make_graph_buffers(3, [(0, 1), (1, 2), (0, 2)], capacity=8)
    eng = LGAEEngine(graph, cfg, dtype=torch.float64)
    assert eng.gauge_connections is not None
    decoder = nn.Linear(3, 2, dtype=torch.float64)
    core = LGAETrainCore(
        eng.fibers,
        decoder,
        gauge_bank=eng.gauge_connections,
        gauge_dim=2,
    )
    src, dst, w, valid, slot, reverse = padded_markov_edges_with_slots(graph, max_edges=32)
    target = torch.zeros(3, 2, dtype=torch.float64)
    pressure = torch.zeros(3, dtype=torch.float64)
    out = core(
        target=target,
        src=src,
        dst=dst,
        weight=w,
        valid=valid,
        bottleneck_pressure=pressure,
        edge_slot=slot,
        reverse=reverse,
    )
    out["loss"].backward()
    grad = eng.gauge_connections.raw_generators.grad
    assert grad is not None
    assert torch.isfinite(grad).all()
    assert torch.count_nonzero(grad).item() > 0
    opt = torch.optim.Adam(core.parameters(), lr=0.05)
    opt.step()
    orth, det = eng.gauge_connections.invariant_error()
    assert float(orth.detach().max()) < 1e-10
    assert float(det.detach().max()) < 1e-10
