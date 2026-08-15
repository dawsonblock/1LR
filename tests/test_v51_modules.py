"""v5.1 module tests: dynamic gauge, timescales, sheaf diffusion, ANN, causal, hypergraph."""
from __future__ import annotations

import pytest
import torch
import numpy as np

from lgae_v3 import (
    # Dynamic gauge
    DynamicGaugeNetwork, DynamicGaugeBank, StaticGaugeAdapter,
    gauge_transport, gauge_alignment_loss,
    # Timescales
    Timescale, TimescaleSchedule, MultiTimescaleController,
    # Sheaf diffusion
    sheaf_laplacian_diffusion, sheaf_adjacency_diffusion,
    gated_sheaf_diffusion, agreement_gate, compare_diffusion_methods,
    # ANN
    ANNNeighborIndex, HNSWIndexNumpy,
    # Causal
    EdgeSemantics, CausalEdge, CausalEdgeRegistry, infer_causality_from_temporal,
    # Hypergraph
    Hyperedge, HypergraphBuffers, hypergraph_laplacian_diffusion,
    clique_expansion, star_expansion,
    # Existing
    SOConnectionBank, make_graph_buffers,
)


# ===========================================================================
# Dynamic gauge tests
# ===========================================================================

class TestDynamicGauge:
    """Test dynamic gauge connections."""

    def test_dynamic_gauge_network_initialization(self):
        net = DynamicGaugeNetwork(latent_dim=4, context_dim=2, hidden_dim=32)
        z_i = torch.randn(3, 4)
        z_j = torch.randn(3, 4)
        ctx = torch.randn(3, 2)
        A = net(z_i, z_j, ctx)
        assert A.shape == (3, 4, 4)
        # Should be skew-symmetric
        assert torch.allclose(A, -A.transpose(-1, -2), atol=1e-5)

    def test_dynamic_gauge_network_no_context(self):
        net = DynamicGaugeNetwork(latent_dim=4, context_dim=0, hidden_dim=32)
        z_i = torch.randn(2, 4)
        z_j = torch.randn(2, 4)
        A = net(z_i, z_j)
        assert A.shape == (2, 4, 4)
        assert torch.allclose(A, -A.transpose(-1, -2), atol=1e-5)

    def test_dynamic_gauge_bank_matrices(self):
        bank = DynamicGaugeBank(edge_capacity=10, dim=4, context_dim=2, hidden_dim=32)
        z = torch.randn(6, 4)
        src = torch.tensor([0, 1, 2, 3])
        dst = torch.tensor([1, 2, 3, 4])
        ctx = torch.randn(2)
        U = bank.matrices(z, src, dst, ctx)
        assert U.shape == (4, 4, 4)
        # Should be in SO(4): U @ U^T = I
        eye = torch.eye(4).unsqueeze(0).expand(4, 4, 4)
        assert torch.allclose(torch.bmm(U, U.transpose(-1, -2)), eye, atol=1e-4)

    def test_dynamic_gauge_bank_forward(self):
        bank = DynamicGaugeBank(edge_capacity=10, dim=4, hidden_dim=32)
        z = torch.randn(5, 4)
        src = torch.tensor([0, 1, 2])
        dst = torch.tensor([1, 2, 3])
        U = bank(z, src, dst)
        assert U.shape == (3, 4, 4)

    def test_gauge_transport_src_to_dst(self):
        z = torch.randn(4, 3)
        src = torch.tensor([0, 1])
        dst = torch.tensor([1, 2])
        U = torch.eye(3).unsqueeze(0).expand(2, 3, 3)
        transported = gauge_transport(z, src, dst, U, "src_to_dst")
        assert transported.shape == (2, 3)
        # With identity U, transported = z_src
        assert torch.allclose(transported, z[src], atol=1e-5)

    def test_gauge_alignment_loss(self):
        z = torch.randn(5, 4)
        src = torch.tensor([0, 1, 2])
        dst = torch.tensor([1, 2, 3])
        U = torch.eye(4).unsqueeze(0).expand(3, 4, 4)
        loss = gauge_alignment_loss(z, src, dst, U)
        assert loss.item() >= 0.0
        assert loss.requires_grad is False  # No grad since no parameters

    def test_static_gauge_adapter(self):
        bank = SOConnectionBank(edge_capacity=10, dim=4)
        adapter = StaticGaugeAdapter(bank)
        U = adapter.matrices()
        assert U.shape[0] == 10
        assert U.shape[1] == 4

    def test_dynamic_gauge_reset_slots_is_noop(self):
        bank = DynamicGaugeBank(edge_capacity=10, dim=4)
        # Should not raise
        bank.reset_slots([0, 1, 2])


# ===========================================================================
# Timescale tests
# ===========================================================================

class TestTimescales:
    """Test multi-timescale adaptation controller."""

    def test_timescale_schedule_basic(self):
        sched = TimescaleSchedule(fast_interval=1, medium_interval=10, slow_interval=100)
        assert sched.is_active(Timescale.FAST, 0)
        assert sched.is_active(Timescale.MEDIUM, 0)
        assert sched.is_active(Timescale.SLOW, 0)
        assert sched.is_active(Timescale.FAST, 1)
        assert not sched.is_active(Timescale.MEDIUM, 1)
        assert not sched.is_active(Timescale.SLOW, 1)
        assert sched.is_active(Timescale.MEDIUM, 10)
        assert sched.is_active(Timescale.SLOW, 100)

    def test_active_timescales(self):
        sched = TimescaleSchedule(fast_interval=1, medium_interval=10, slow_interval=100)
        active = sched.active_timescales(0)
        assert Timescale.FAST in active
        assert Timescale.MEDIUM in active
        assert Timescale.SLOW in active

    def test_controller_min_fast_before_medium(self):
        ctrl = MultiTimescaleController(
            schedule=TimescaleSchedule(fast_interval=1, medium_interval=1, slow_interval=100),
            min_fast_before_medium=5,
        )
        # At step 0: fast is active, medium is not (not enough fast updates)
        active = ctrl.update(0)
        assert Timescale.FAST in active
        assert Timescale.MEDIUM not in active
        # After 5 fast updates, medium becomes active
        for step in range(1, 5):
            ctrl.update(step)
        active = ctrl.update(5)
        assert Timescale.MEDIUM in active

    def test_controller_min_medium_before_slow(self):
        ctrl = MultiTimescaleController(
            schedule=TimescaleSchedule(fast_interval=1, medium_interval=1, slow_interval=1),
            min_fast_before_medium=0,
            min_medium_before_slow=3,
        )
        # Allow medium immediately
        for _ in range(3):
            ctrl.update(0)
        # Now slow should be allowed
        # Actually, we need to call update at different steps
        ctrl2 = MultiTimescaleController(
            schedule=TimescaleSchedule(fast_interval=1, medium_interval=1, slow_interval=1),
            min_fast_before_medium=0,
            min_medium_before_slow=3,
        )
        ctrl2.update(0)  # fast + medium (slow blocked)
        ctrl2.update(1)  # fast + medium
        ctrl2.update(2)  # fast + medium
        active = ctrl2.update(3)  # Now slow should be active
        assert Timescale.SLOW in active

    def test_controller_summary(self):
        ctrl = MultiTimescaleController()
        ctrl.update(0)
        ctrl.update(1)
        summary = ctrl.summary()
        assert summary["step"] == 1
        assert summary["fast_updates"] == 2

    def test_can_adapt_methods(self):
        ctrl = MultiTimescaleController(
            schedule=TimescaleSchedule(fast_interval=1, medium_interval=10, slow_interval=100),
            min_fast_before_medium=0,
            min_medium_before_slow=0,
        )
        # can_adapt_* are read-only checks (do not advance state)
        assert ctrl.can_adapt_gauge(0)
        assert ctrl.can_adapt_affinity(0)
        assert ctrl.can_adapt_length(0)
        # Step 1: only fast is active
        assert ctrl.can_adapt_gauge(1)
        assert not ctrl.can_adapt_affinity(1)
        assert not ctrl.can_adapt_length(1)
        # Step 10: fast + medium
        assert ctrl.can_adapt_gauge(10)
        assert ctrl.can_adapt_affinity(10)
        # Step 100: all active
        assert ctrl.can_adapt_gauge(100)
        assert ctrl.can_adapt_affinity(100)
        assert ctrl.can_adapt_length(100)


# ===========================================================================
# Sheaf diffusion tests
# ===========================================================================

class TestSheafDiffusion:
    """Test sheaf-adjacency diffusion."""

    def test_sheaf_laplacian_diffusion(self):
        N, D = 6, 4
        z = torch.randn(N, D)
        src = torch.tensor([0, 1, 2, 3, 4])
        dst = torch.tensor([1, 2, 3, 4, 5])
        U = torch.eye(D).unsqueeze(0).expand(5, D, D)
        weight = torch.ones(5)
        z_out = sheaf_laplacian_diffusion(z, src, dst, U, weight, num_steps=3, eta=0.1)
        assert z_out.shape == z.shape
        # Should not diverge
        assert torch.isfinite(z_out).all()

    def test_sheaf_adjacency_diffusion(self):
        N, D = 6, 4
        z = torch.randn(N, D)
        src = torch.tensor([0, 1, 2, 3, 4])
        dst = torch.tensor([1, 2, 3, 4, 5])
        U = torch.eye(D).unsqueeze(0).expand(5, D, D)
        weight = torch.ones(5)
        z_out = sheaf_adjacency_diffusion(z, src, dst, U, weight, num_steps=3, eta=0.1)
        assert z_out.shape == z.shape
        assert torch.isfinite(z_out).all()

    def test_gated_sheaf_diffusion(self):
        N, D = 5, 3
        z = torch.randn(N, D)
        src = torch.tensor([0, 1, 2, 3])
        dst = torch.tensor([1, 2, 3, 4])
        U = torch.eye(D).unsqueeze(0).expand(4, D, D)
        weight = torch.ones(4)

        def gate_fn(z_i, z_j):
            return torch.ones(z_i.shape[0])  # All gates open

        z_out = gated_sheaf_diffusion(z, src, dst, U, weight, gate_fn, num_steps=2, eta=0.1)
        assert z_out.shape == z.shape
        assert torch.isfinite(z_out).all()

    def test_agreement_gate(self):
        z_i = torch.randn(3, 4)
        z_j = torch.randn(3, 4)
        gate = agreement_gate(z_i, z_j)
        assert gate.shape == (3,)
        assert (gate >= 0).all() and (gate <= 1).all()

    def test_compare_diffusion_methods(self):
        N, D = 8, 4
        z = torch.randn(N, D)
        src = torch.tensor([0, 1, 2, 3, 4, 5])
        dst = torch.tensor([1, 2, 3, 4, 5, 6])
        U = torch.eye(D).unsqueeze(0).expand(6, D, D)
        weight = torch.ones(6)
        result = compare_diffusion_methods(z, src, dst, U, weight, num_steps=5, eta=0.1)
        assert "laplacian" in result
        assert "adjacency" in result
        assert "variance_ratio_laplacian" in result
        assert "variance_ratio_adjacency" in result
        assert result["variance_ratio_laplacian"] >= 0
        assert result["variance_ratio_adjacency"] >= 0


# ===========================================================================
# ANN index tests
# ===========================================================================

class TestANNIndex:
    """Test ANN-backed neighbor index."""

    def test_hnsw_index_build_and_search(self):
        data = np.random.randn(100, 4).astype(np.float32)
        idx = HNSWIndexNumpy(dim=4, n_partitions=8)
        idx.build(data)
        query = data[:5]
        distances, indices = idx.search(query, k=5)
        assert distances.shape == (5, 5)
        assert indices.shape == (5, 5)

    def test_ann_neighbor_index_numpy_backend(self):
        z = torch.randn(50, 4)
        ann = ANNNeighborIndex(dim=4, n_candidates=10, n_final=5, backend="numpy")
        distances, indices = ann.search(z, k=5)
        assert distances.shape == (50, 5)
        assert indices.shape == (50, 5)

    def test_ann_build_knn_graph(self):
        z = torch.randn(30, 4)
        ann = ANNNeighborIndex(dim=4, n_candidates=10, n_final=5, backend="numpy")
        result = ann.build_knn_graph(z, k=3)
        assert result.src.shape[0] > 0
        assert result.dst.shape[0] > 0
        assert result.weight.shape[0] > 0

    def test_ann_measure_recall(self):
        z = torch.randn(50, 4)
        ann = ANNNeighborIndex(dim=4, n_candidates=20, n_final=10, backend="numpy")
        recall = ann.measure_recall(z, k=5)
        assert 0.0 <= recall <= 1.0
        # With small dataset, recall should be reasonable
        assert recall > 0.3


# ===========================================================================
# Causal edge tests
# ===========================================================================

class TestCausalEdges:
    """Test causal edge semantics."""

    def test_edge_semantics_enum(self):
        assert EdgeSemantics.ASSOCIATION.value == "association"
        assert EdgeSemantics.CAUSAL.value == "causal"
        assert EdgeSemantics.BIDIRECTIONAL.value == "bidirectional"

    def test_causal_edge_registry_register(self):
        reg = CausalEdgeRegistry()
        reg.register(0, 1, EdgeSemantics.CAUSAL, confidence=0.9)
        assert reg.is_causal(0, 1)
        assert not reg.is_causal(1, 0)
        assert not reg.is_association(0, 1)

    def test_causal_edge_registry_association(self):
        reg = CausalEdgeRegistry()
        reg.register(0, 1, EdgeSemantics.ASSOCIATION)
        assert reg.is_association(0, 1)
        assert not reg.is_causal(0, 1)

    def test_causal_parents_children(self):
        reg = CausalEdgeRegistry()
        reg.register(0, 2, EdgeSemantics.CAUSAL)
        reg.register(1, 2, EdgeSemantics.CAUSAL)
        reg.register(2, 3, EdgeSemantics.CAUSAL)
        assert reg.causal_parents(2) == {0, 1}
        assert reg.causal_children(2) == {3}

    def test_causal_paths(self):
        reg = CausalEdgeRegistry()
        reg.register(0, 1, EdgeSemantics.CAUSAL)
        reg.register(1, 2, EdgeSemantics.CAUSAL)
        reg.register(2, 3, EdgeSemantics.CAUSAL)
        paths = reg.causal_paths(0, 3)
        assert len(paths) == 1
        assert paths[0] == [0, 1, 2, 3]

    def test_causal_paths_multiple(self):
        reg = CausalEdgeRegistry()
        reg.register(0, 1, EdgeSemantics.CAUSAL)
        reg.register(0, 2, EdgeSemantics.CAUSAL)
        reg.register(1, 3, EdgeSemantics.CAUSAL)
        reg.register(2, 3, EdgeSemantics.CAUSAL)
        paths = reg.causal_paths(0, 3)
        assert len(paths) == 2

    def test_intervene(self):
        reg = CausalEdgeRegistry()
        reg.register(0, 1, EdgeSemantics.CAUSAL)
        z = torch.randn(3, 4)
        new_val = torch.ones(4)
        z_new = reg.intervene(0, new_val, z)
        # Node 0 is set to the new value
        assert torch.allclose(z_new[0], new_val)
        # Node 1 (causal child) is influenced by the intervention
        # It should have shifted toward node 0's new value
        assert not torch.allclose(z_new[1], z[1])
        # Node 2 (no causal connection) is unchanged
        assert torch.allclose(z_new[2], z[2])

    def test_serialize_deserialize(self):
        reg = CausalEdgeRegistry()
        reg.register(0, 1, EdgeSemantics.CAUSAL, confidence=0.9)
        reg.register(1, 2, EdgeSemantics.ASSOCIATION)
        data = reg.to_dict()
        reg2 = CausalEdgeRegistry.from_dict(data)
        assert reg2.is_causal(0, 1)
        assert reg2.is_association(1, 2)

    def test_summary(self):
        reg = CausalEdgeRegistry()
        reg.register(0, 1, EdgeSemantics.CAUSAL)
        reg.register(1, 2, EdgeSemantics.ASSOCIATION)
        summary = reg.summary()
        assert summary["total_edges"] == 2
        assert summary["semantic_counts"]["causal"] == 1
        assert summary["semantic_counts"]["association"] == 1

    def test_infer_causality_from_temporal(self):
        T, N, D = 20, 3, 4
        z_history = torch.randn(T, N, D)
        # Make node 0 lead node 1
        z_history[1:, 1] = z_history[:-1, 0] * 0.5 + z_history[1:, 1] * 0.5
        result = infer_causality_from_temporal(z_history, 0, 1, lag=1)
        assert isinstance(result, EdgeSemantics)


# ===========================================================================
# Hypergraph tests
# ===========================================================================

class TestHypergraph:
    """Test hypergraph / higher-order relationships."""

    def test_hyperedge_basic(self):
        he = Hyperedge(nodes=(0, 1, 2), weight=1.0)
        assert he.order == 3
        assert he.contains(0)
        assert not he.contains(5)
        pairs = he.pairs()
        assert len(pairs) == 3  # C(3, 2) = 3

    def test_hypergraph_buffers_add(self):
        hg = HypergraphBuffers(num_nodes=10, hyperedge_capacity=20, max_order=4)
        slot = hg.add_hyperedge([0, 1, 2], weight=1.5)
        assert slot == 0
        assert hg.num_hyperedges == 1

    def test_hypergraph_buffers_get(self):
        hg = HypergraphBuffers(num_nodes=10, hyperedge_capacity=20, max_order=4)
        slot = hg.add_hyperedge([0, 1, 2], weight=1.5)
        he = hg.get_hyperedge(slot)
        assert he is not None
        assert he.nodes == (0, 1, 2)
        assert he.weight == 1.5

    def test_hypergraph_to_pairwise(self):
        hg = HypergraphBuffers(num_nodes=10, hyperedge_capacity=20, max_order=4)
        hg.add_hyperedge([0, 1, 2], weight=1.0)
        hg.add_hyperedge([1, 2, 3], weight=2.0)
        src, dst, w = hg.to_pairwise_edges()
        # 2 hyperedges of order 3 → 2 * C(3,2) = 6 pairwise edges
        assert src.shape[0] == 6
        assert dst.shape[0] == 6
        assert w.shape[0] == 6

    def test_hypergraph_clone(self):
        hg = HypergraphBuffers(num_nodes=10, hyperedge_capacity=20, max_order=4)
        hg.add_hyperedge([0, 1, 2], weight=1.0)
        hg2 = hg.clone()
        assert hg2.num_hyperedges == 1
        he = hg2.get_hyperedge(0)
        assert he is not None
        assert he.nodes == (0, 1, 2)

    def test_hypergraph_diffusion(self):
        hg = HypergraphBuffers(num_nodes=6, hyperedge_capacity=10, max_order=4)
        hg.add_hyperedge([0, 1, 2], weight=1.0)
        hg.add_hyperedge([3, 4, 5], weight=1.0)
        z = torch.randn(6, 4)
        z_out = hypergraph_laplacian_diffusion(z, hg, eta=0.1, num_steps=3)
        assert z_out.shape == z.shape
        assert torch.isfinite(z_out).all()

    def test_clique_expansion(self):
        hg = HypergraphBuffers(num_nodes=10, hyperedge_capacity=20, max_order=4)
        hg.add_hyperedge([0, 1, 2], weight=1.0)
        src, dst, w = clique_expansion(hg)
        assert src.shape[0] == 3  # C(3, 2) = 3

    def test_star_expansion(self):
        hg = HypergraphBuffers(num_nodes=10, hyperedge_capacity=20, max_order=4)
        hg.add_hyperedge([0, 1, 2], weight=1.0)
        src, dst, w, n_anchors = star_expansion(hg)
        assert n_anchors == 1
        # 3 nodes * 2 (bidirectional) = 6 edges
        assert src.shape[0] == 6

    def test_summary(self):
        hg = HypergraphBuffers(num_nodes=10, hyperedge_capacity=20, max_order=4)
        hg.add_hyperedge([0, 1], weight=1.0)
        hg.add_hyperedge([0, 1, 2], weight=1.0)
        hg.add_hyperedge([0, 1, 2, 3], weight=1.0)
        summary = hg.summary()
        assert summary["num_hyperedges"] == 3
        assert summary["order_counts"][2] == 1
        assert summary["order_counts"][3] == 1
        assert summary["order_counts"][4] == 1

    def test_max_order_exceeded_raises(self):
        hg = HypergraphBuffers(num_nodes=10, hyperedge_capacity=20, max_order=3)
        with pytest.raises(ValueError, match="exceeds max_order"):
            hg.add_hyperedge([0, 1, 2, 3])

    def test_capacity_exceeded_raises(self):
        hg = HypergraphBuffers(num_nodes=10, hyperedge_capacity=2, max_order=3)
        hg.add_hyperedge([0, 1], weight=1.0)
        hg.add_hyperedge([1, 2], weight=1.0)
        with pytest.raises(RuntimeError, match="capacity"):
            hg.add_hyperedge([2, 0], weight=1.0)
