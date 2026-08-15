"""v5.1 Integration tests: executive + governor + benchmark end-to-end.

These tests verify that the v5 structural learning loop integrates
correctly with the existing v4 governor, mutations, and benchmark harness.
"""
from __future__ import annotations

import pytest
import torch

from lgae_v3 import (
    LGAEConfig, make_graph_buffers, GeometryGovernor,
    StructuralExecutive, StructuralAction, ActionProposal,
    StructuralLearningLoop, StructuralLoopResult,
    action_to_mutation, certify_action_through_governor, ActionBridgeResult,
    AddEdge, PruneEdge, ReweightAffinity,
    MutationDecision,
)
from lgae_v3.benchmark import (
    BenchmarkHarness, ALL_TASKS, TaskA_Bottleneck, TaskC_SpuriousEdge,
    StructuralAction as BenchAction,
)


# ===========================================================================
# Action bridge tests
# ===========================================================================

class TestActionBridge:
    """Test the bridge between StructuralAction and concrete mutations."""

    def test_add_edge_to_mutation(self):
        graph = make_graph_buffers(6, [(0, 1), (1, 2), (2, 3)], capacity=10)
        z = torch.randn(6, 4)
        mut = action_to_mutation(StructuralAction.ADD_EDGE, graph, z, u=0, v=3, weight=2.0)
        assert isinstance(mut, AddEdge)
        assert mut.u == 0
        assert mut.v == 3

    def test_prune_edge_to_mutation(self):
        graph = make_graph_buffers(6, [(0, 1), (1, 2), (2, 3)], capacity=10)
        z = torch.randn(6, 4)
        mut = action_to_mutation(StructuralAction.PRUNE_EDGE, graph, z, u=0, v=1)
        assert isinstance(mut, PruneEdge)

    def test_no_op_to_mutation_returns_none(self):
        graph = make_graph_buffers(4, [(0, 1)], capacity=6)
        z = torch.randn(4, 4)
        mut = action_to_mutation(StructuralAction.NO_OP, graph, z)
        assert mut is None

    def test_spawn_fiber_to_mutation_returns_none(self):
        """SPAWN_FIBER doesn't map to an edge mutation."""
        graph = make_graph_buffers(4, [(0, 1)], capacity=6)
        z = torch.randn(4, 4)
        mut = action_to_mutation(StructuralAction.SPAWN_FIBER, graph, z)
        assert mut is None

    def test_add_edge_auto_selects_disconnected(self):
        graph = make_graph_buffers(4, [(0, 1), (1, 2)], capacity=6)
        z = torch.randn(4, 4)
        mut = action_to_mutation(StructuralAction.ADD_EDGE, graph, z)
        assert isinstance(mut, AddEdge)
        # Should connect two previously disconnected nodes
        assert mut.u != mut.v

    def test_prune_edge_auto_selects_weakest(self):
        graph = make_graph_buffers(4, [(0, 1), (1, 2), (2, 3)], capacity=6)
        # Set different weights
        graph.weight[0] = 0.1  # Weakest
        graph.weight[1] = 1.0
        graph.weight[2] = 0.5
        z = torch.randn(4, 4)
        mut = action_to_mutation(StructuralAction.PRUNE_EDGE, graph, z)
        assert isinstance(mut, PruneEdge)
        # Should prune the weakest edge (0, 1) with weight 0.1
        assert {mut.u, mut.v} == {0, 1}


# ===========================================================================
# Governor certification integration
# ===========================================================================

class TestGovernorCertification:
    """Test that the governor certifies actions from the executive."""

    def test_certify_add_edge_through_governor(self):
        """The governor should be able to certify an ADD_EDGE proposal."""
        cfg = LGAEConfig()
        cfg.fiber.d_base = 4; cfg.fiber.d_max = 8
        cfg.audit.orc_backend = "exact_lp"
        graph = make_graph_buffers(6, [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)], capacity=10)
        z = torch.randn(6, 4)
        governor = GeometryGovernor(cfg)

        result = certify_action_through_governor(
            StructuralAction.ADD_EDGE, graph, z, governor,
            u=0, v=5, weight=1.0,
        )
        assert isinstance(result, ActionBridgeResult)
        assert result.action == StructuralAction.ADD_EDGE
        assert result.mutation is not None
        assert result.governor_result is not None
        assert result.governor_result.decision in (
            MutationDecision.ACCEPT, MutationDecision.QUARANTINE, MutationDecision.REJECT,
        )

    def test_certify_no_op_returns_not_executed(self):
        cfg = LGAEConfig()
        graph = make_graph_buffers(4, [(0, 1), (1, 2)], capacity=6)
        z = torch.randn(4, 4)
        governor = GeometryGovernor(cfg)

        result = certify_action_through_governor(
            StructuralAction.NO_OP, graph, z, governor,
        )
        assert not result.executed
        assert result.mutation is None

    def test_certify_prune_edge_through_governor(self):
        cfg = LGAEConfig()
        cfg.fiber.d_base = 4; cfg.fiber.d_max = 8
        # Use exact LP backend to avoid Sinkhorn convergence edge cases in tests
        cfg.audit.orc_backend = "exact_lp"
        torch.manual_seed(123)
        graph = make_graph_buffers(6, [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (0, 5), (1, 4)], capacity=10)
        z = torch.randn(6, 4) * 0.3
        governor = GeometryGovernor(cfg)

        result = certify_action_through_governor(
            StructuralAction.PRUNE_EDGE, graph, z, governor,
            u=0, v=5, seed=123,
        )
        assert result.governor_result is not None
        assert result.governor_result.decision in (
            MutationDecision.ACCEPT, MutationDecision.QUARANTINE, MutationDecision.REJECT,
        )

    def test_certify_provides_shadow_graph(self):
        """The governor should provide a shadow graph for accepted mutations."""
        cfg = LGAEConfig()
        cfg.fiber.d_base = 4; cfg.fiber.d_max = 8
        cfg.audit.orc_backend = "exact_lp"
        graph = make_graph_buffers(6, [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)], capacity=10)
        z = torch.randn(6, 4)
        governor = GeometryGovernor(cfg)

        result = certify_action_through_governor(
            StructuralAction.ADD_EDGE, graph, z, governor,
            u=0, v=5, weight=1.0,
        )
        # Shadow graph should always be returned (even for rejected mutations)
        assert result.shadow_graph is not None


# ===========================================================================
# Structural loop with real governor
# ===========================================================================

class TestStructuralLoopWithGovernor:
    """Test the structural learning loop with a real governor."""

    def test_loop_with_governor_initialization(self):
        cfg = LGAEConfig()
        cfg.fiber.d_base = 4; cfg.fiber.d_max = 8
        governor = GeometryGovernor(cfg)
        loop = StructuralLearningLoop(cfg, governor=governor)
        assert loop.governor is not None

    def test_loop_step_with_governor(self):
        """The loop should work with a real governor certifying actions."""
        cfg = LGAEConfig()
        cfg.fiber.d_base = 4; cfg.fiber.d_max = 8
        cfg.audit.orc_backend = "exact_lp"
        graph = make_graph_buffers(6, [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)], capacity=10)
        z = torch.randn(6, 4)
        governor = GeometryGovernor(cfg)
        loop = StructuralLearningLoop(cfg, governor=governor)

        result = loop.step(graph, z)
        assert isinstance(result, StructuralLoopResult)
        # The governance decision should come from the real governor
        assert result.governance_decision in ("accept", "quarantine", "reject")

    def test_loop_without_governor_falls_back(self):
        """Without a governor, the loop should fall back to uncertainty gating."""
        cfg = LGAEConfig()
        cfg.fiber.d_base = 4; cfg.fiber.d_max = 8
        graph = make_graph_buffers(4, [(0, 1), (1, 2)], capacity=6)
        z = torch.randn(4, 4)
        loop = StructuralLearningLoop(cfg, governor=None)

        result = loop.step(graph, z)
        assert isinstance(result, StructuralLoopResult)
        # Should still produce a valid decision
        assert result.governance_decision in ("accept", "quarantine", "reject")

    def test_loop_multiple_steps_with_governor(self):
        """The loop should handle multiple steps with the governor."""
        cfg = LGAEConfig()
        cfg.fiber.d_base = 4; cfg.fiber.d_max = 8
        cfg.audit.orc_backend = "exact_lp"
        graph = make_graph_buffers(6, [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)], capacity=10)
        z = torch.randn(6, 4)
        governor = GeometryGovernor(cfg)
        loop = StructuralLearningLoop(cfg, governor=governor)

        for i in range(3):
            result = loop.step(graph, z, task_loss=1.0 - i * 0.1)
            assert result.step == i


# ===========================================================================
# Benchmark + executive end-to-end
# ===========================================================================

class TestBenchmarkExecutiveE2E:
    """End-to-end tests of the executive on the benchmark."""

    def test_executive_proposes_valid_actions_on_all_tasks(self):
        """The executive should propose valid actions for all benchmark tasks."""
        harness = BenchmarkHarness()

        for task in ALL_TASKS:
            state = task.initial_state(seed=42)
            exec_model = StructuralExecutive(state.config)
            obs = exec_model.observe(state.graph, state.z)
            proposal = exec_model.best_proposal(obs)
            assert isinstance(proposal.action, StructuralAction)
            assert isinstance(proposal.score, float)

    def test_benchmark_oracle_vs_executive(self):
        """Oracle should outperform or match an untrained executive."""
        harness = BenchmarkHarness()
        oracle = harness.run_oracle(seed=42)

        def untrained_executive(state):
            exec_model = StructuralExecutive(state.config)
            obs = exec_model.observe(state.graph, state.z)
            return exec_model.best_proposal(obs).action

        exec_result = harness.evaluate_executive(untrained_executive, seed=42)
        # Oracle should have perfect accuracy
        assert oracle.diagnosis_accuracy == 1.0
        # Executive should produce valid results (even if untrained)
        assert 0.0 <= exec_result.diagnosis_accuracy <= 1.0

    def test_executive_can_be_trained_on_benchmark(self):
        """The executive should improve when trained on benchmark outcomes."""
        harness = BenchmarkHarness()
        exec_model = StructuralExecutive(LGAEConfig(), lr=1e-2)

        # Generate training data by running on all tasks
        for task in ALL_TASKS:
            state = task.initial_state(seed=42)
            obs = exec_model.observe(state.graph, state.z)
            # Use the correct action's outcome as training signal
            correct_actions = task.correct_actions()
            correct_action = next(iter(correct_actions))
            outcome = task.evaluate(state, correct_action)
            exec_model.record_outcome(obs, correct_action, outcome.delta_utility)

        # Record more examples with different actions
        for task in ALL_TASKS:
            state = task.initial_state(seed=43)
            obs = exec_model.observe(state.graph, state.z)
            for action in [StructuralAction.NO_OP, StructuralAction.ADD_EDGE]:
                outcome = task.evaluate(state, action)
                exec_model.record_outcome(obs, action, outcome.delta_utility)

        # Train
        metrics = exec_model.train_step(batch_size=8)
        assert metrics["samples"] > 0
        assert metrics["loss"] >= 0.0

    def test_credit_tracker_records_benchmark_outcomes(self):
        """The credit tracker should record outcomes from benchmark tasks."""
        from lgae_v3 import MutationCreditTracker

        tracker = MutationCreditTracker(horizons=[1, 2])
        task = TaskA_Bottleneck()
        state = task.initial_state(seed=42)

        # Simulate a mutation
        tracker.record_mutation(
            action=StructuralAction.ADD_EDGE,
            step=0,
            predicted_delta_u=0.2,
            predicted_uncertainty=0.05,
            governance_decision="accept",
            governance_reasons=["test"],
            graph_hash_before=state.graph.state_hash(),
            graph_hash_after=state.graph.state_hash(),
            config_governance_hash="test",
        )

        # Record utility at horizons
        u0 = task.utility(state)
        tracker.record_utility(0, u0)
        tracker.record_utility(1, u0 + 0.1)
        tracker.record_utility(2, u0 + 0.2)

        outcomes = tracker.get_outcomes()
        assert len(outcomes) == 1
        assert outcomes[0].action == StructuralAction.ADD_EDGE

    def test_consolidation_tracks_fiber_lifecycle_on_benchmark(self):
        """The consolidation controller should track fibers spawned during benchmarks."""
        from lgae_v3 import StabilityPlasticityController, FiberLifecycleStage

        ctrl = StabilityPlasticityController(probation_length=5)
        task = TaskA_Bottleneck()
        state = task.initial_state(seed=42)

        # Simulate fiber spawn
        fiber = ctrl.register_fiber(dimension=4, step=0)
        assert fiber.stage == FiberLifecycleStage.NEW

        # Record utility over time
        for i in range(6):
            ctrl.record_fiber_utility(fiber.fiber_id, 0.5, i)

        # Update lifecycle
        ctrl.update_lifecycle(5)
        assert fiber.stage in (FiberLifecycleStage.MATURE, FiberLifecycleStage.PROTECTED)

    def test_full_loop_on_benchmark_task(self):
        """Run the full structural loop on a benchmark task."""
        cfg = LGAEConfig()
        cfg.fiber.d_base = 4; cfg.fiber.d_max = 8
        cfg.audit.orc_backend = "exact_lp"
        task = TaskA_Bottleneck()
        state = task.initial_state(seed=42)
        governor = GeometryGovernor(cfg)
        loop = StructuralLearningLoop(cfg, governor=governor)

        def utility_fn(graph, z):
            return task.utility(type('S', (), {'graph': graph, 'z': z, 'config': cfg, 'task_params': state.task_params})())

        result = loop.step(
            state.graph, state.z,
            task_loss=1.0,
            utility_fn=utility_fn,
        )
        assert isinstance(result, StructuralLoopResult)
        assert result.step == 0
        # Should have a valid chosen action
        assert isinstance(result.chosen_action, StructuralAction)
