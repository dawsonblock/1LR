"""v5.0 Structural learning loop tests.

Tests the closed loop:
    geometry observes → learned executive predicts →
    counterfactuals compete → governor certifies →
    outcomes train the executive
"""
from __future__ import annotations

import pytest
import torch
import tempfile
import os

from lgae_v3 import (
    LGAEConfig, make_graph_buffers, LGAEEngine,
    StructuralExecutive, ActionProposal, StructuralObservation,
    StructuralAction, ACTION_LIST, ACTION_TO_IDX, NUM_ACTIONS,
    EnsembleUncertainty, ConformalCalibrator, UncertaintyEstimate,
    uncertainty_gated_decision,
    MutationCreditTracker, MutationReceipt, MutationOutcome,
    StabilityPlasticityController, FiberState, FiberLifecycleStage,
    StructuralCounterfactualEngine, CounterfactualResult,
    StructuralLearningLoop, StructuralLoopResult,
)
from lgae_v3.benchmark import (
    BenchmarkHarness, ALL_TASKS, TaskA_Bottleneck,
    evaluate_diagnosis_accuracy, evaluate_mutation_regret, run_benchmark,
)


# ===========================================================================
# Executive tests
# ===========================================================================

class TestStructuralExecutive:
    """Test the learned structural executive."""

    def test_executive_initialization(self):
        exec = StructuralExecutive()
        assert exec.network is not None
        assert len(ACTION_LIST) == NUM_ACTIONS

    def test_observe_returns_observation(self):
        cfg = LGAEConfig()
        cfg.fiber.d_base = 4; cfg.fiber.d_max = 8
        graph = make_graph_buffers(6, [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)], capacity=10)
        z = torch.randn(6, 4)
        exec = StructuralExecutive(cfg)
        obs = exec.observe(graph, z)
        assert isinstance(obs, StructuralObservation)
        vec = obs.to_vector()
        assert vec.shape[0] == exec._obs_dim

    def test_propose_returns_sorted_proposals(self):
        cfg = LGAEConfig()
        graph = make_graph_buffers(6, [(0, 1), (1, 2), (2, 3)], capacity=8)
        z = torch.randn(6, 4)
        exec = StructuralExecutive(cfg)
        obs = exec.observe(graph, z)
        proposals = exec.propose(obs)
        assert len(proposals) == NUM_ACTIONS
        # Sorted by score descending
        for i in range(len(proposals) - 1):
            assert proposals[i].score >= proposals[i + 1].score

    def test_best_proposal_returns_highest_score(self):
        cfg = LGAEConfig()
        graph = make_graph_buffers(4, [(0, 1), (1, 2), (2, 3)], capacity=6)
        z = torch.randn(4, 4)
        exec = StructuralExecutive(cfg)
        obs = exec.observe(graph, z)
        best = exec.best_proposal(obs)
        all_props = exec.propose(obs)
        assert best.score == all_props[0].score

    def test_record_mutation_updates_history(self):
        exec = StructuralExecutive()
        exec.record_mutation(StructuralAction.ADD_EDGE)
        assert exec._mutation_history[ACTION_TO_IDX[StructuralAction.ADD_EDGE]] > 0

    def test_record_outcome_stores_experience(self):
        cfg = LGAEConfig()
        graph = make_graph_buffers(4, [(0, 1), (1, 2)], capacity=6)
        z = torch.randn(4, 4)
        exec = StructuralExecutive(cfg)
        obs = exec.observe(graph, z)
        exec.record_outcome(obs, StructuralAction.ADD_EDGE, 0.5)
        assert len(exec._experience) == 1

    def test_train_step_with_experience(self):
        cfg = LGAEConfig()
        graph = make_graph_buffers(4, [(0, 1), (1, 2)], capacity=6)
        z = torch.randn(4, 4)
        exec = StructuralExecutive(cfg, lr=1e-2)
        obs = exec.observe(graph, z)
        # Record enough experience for a batch
        for i in range(35):
            exec.record_outcome(obs, StructuralAction.ADD_EDGE, float(i) * 0.01)
        metrics = exec.train_step(batch_size=32)
        assert metrics["samples"] == 32
        assert metrics["loss"] >= 0.0

    def test_save_load_state(self):
        exec = StructuralExecutive()
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            path = f.name
        try:
            exec.save_state(path)
            exec2 = StructuralExecutive()
            exec2.load_state(path)
            assert exec2._obs_dim == exec._obs_dim
        finally:
            os.unlink(path)


# ===========================================================================
# Uncertainty tests
# ===========================================================================

class TestUncertainty:
    """Test calibrated uncertainty estimation."""

    def test_ensemble_uncertainty_estimate(self):
        exec = StructuralExecutive()
        ens = EnsembleUncertainty(exec, ensemble_size=3)
        obs_vec = torch.randn(exec._obs_dim)
        est = ens.estimate(obs_vec, action_idx=0)
        assert isinstance(est, UncertaintyEstimate)
        assert est.method == "ensemble"
        assert est.lcb <= est.mean <= est.ucb

    def test_conformal_calibrator(self):
        cal = ConformalCalibrator(alpha=0.1)
        predicted = [1.0, 2.0, 3.0, 4.0, 5.0]
        actual = [1.1, 2.2, 2.8, 3.9, 5.1]
        q = cal.calibrate(predicted, actual)
        assert q >= 0.0
        lower, upper = cal.interval(3.0)
        assert lower < 3.0 < upper

    def test_uncertainty_gated_decision_accept(self):
        """High LCB, low uncertainty → accept."""
        from lgae_v3.executive import ActionProposal
        prop = ActionProposal(
            action=StructuralAction.ADD_EDGE,
            expected_delta_utility=1.0, information_gain=0.0,
            cost=0.0, risk=0.0, score=1.0, uncertainty=0.1, lcb=0.8,
        )
        unc = UncertaintyEstimate(mean=1.0, std=0.1, lcb=0.8, ucb=1.2)
        decision = uncertainty_gated_decision(prop, unc, lcb_threshold=0.0, quarantine_uncertainty=0.5)
        assert decision == "accept"

    def test_uncertainty_gated_decision_quarantine_high_uncertainty(self):
        """Positive LCB but high uncertainty → quarantine."""
        prop = ActionProposal(
            action=StructuralAction.SPAWN_FIBER,
            expected_delta_utility=0.5, information_gain=0.0,
            cost=0.0, risk=0.0, score=0.5, uncertainty=0.8, lcb=0.3,
        )
        unc = UncertaintyEstimate(mean=0.5, std=0.8, lcb=0.3, ucb=0.7)
        decision = uncertainty_gated_decision(prop, unc, lcb_threshold=0.0, quarantine_uncertainty=0.5)
        assert decision == "quarantine"

    def test_uncertainty_gated_decision_quarantine_interesting(self):
        """Low LCB but high UCB → quarantine (uncertain but interesting)."""
        prop = ActionProposal(
            action=StructuralAction.SPAWN_FIBER,
            expected_delta_utility=0.0, information_gain=0.0,
            cost=0.0, risk=0.0, score=0.0, uncertainty=1.0, lcb=-0.5,
        )
        unc = UncertaintyEstimate(mean=0.0, std=1.0, lcb=-0.5, ucb=0.5)
        decision = uncertainty_gated_decision(prop, unc, lcb_threshold=0.0, quarantine_uncertainty=0.5)
        assert decision == "quarantine"

    def test_uncertainty_gated_decision_reject(self):
        """Low LCB and low UCB → reject."""
        prop = ActionProposal(
            action=StructuralAction.PRUNE_EDGE,
            expected_delta_utility=-1.0, information_gain=0.0,
            cost=0.0, risk=0.0, score=-1.0, uncertainty=0.1, lcb=-1.1,
        )
        unc = UncertaintyEstimate(mean=-1.0, std=0.1, lcb=-1.1, ucb=-0.9)
        decision = uncertainty_gated_decision(prop, unc, lcb_threshold=0.0, quarantine_uncertainty=0.5)
        assert decision == "reject"


# ===========================================================================
# Credit assignment tests
# ===========================================================================

class TestCreditAssignment:
    """Test long-term mutation credit assignment."""

    def test_record_mutation_returns_receipt(self):
        tracker = MutationCreditTracker()
        receipt = tracker.record_mutation(
            action=StructuralAction.ADD_EDGE,
            step=0,
            predicted_delta_u=0.5,
            predicted_uncertainty=0.1,
            governance_decision="accept",
            governance_reasons=["all_passed"],
            graph_hash_before="abc",
            graph_hash_after="def",
            config_governance_hash="ghi",
        )
        assert receipt.receipt_id == 0
        assert receipt.action == StructuralAction.ADD_EDGE

    def test_record_utility_tracks_history(self):
        tracker = MutationCreditTracker(horizons=[2, 5])
        tracker.record_mutation(
            action=StructuralAction.ADD_EDGE, step=0,
            predicted_delta_u=0.5, predicted_uncertainty=0.1,
            governance_decision="accept", governance_reasons=[],
            graph_hash_before="a", graph_hash_after="b",
            config_governance_hash="c",
        )
        tracker.record_utility(2, 1.0)
        tracker.record_utility(5, 2.0)
        outcomes = tracker.get_outcomes()
        assert len(outcomes) == 1
        assert outcomes[0].utility_at_16 is None  # Not a horizon
        assert outcomes[0].discounted_return > 0

    def test_training_data_generation(self):
        tracker = MutationCreditTracker(horizons=[1, 2])
        tracker.record_mutation(
            action=StructuralAction.ADD_EDGE, step=0,
            predicted_delta_u=0.5, predicted_uncertainty=0.1,
            governance_decision="accept", governance_reasons=[],
            graph_hash_before="a", graph_hash_after="b",
            config_governance_hash="c",
        )
        tracker.record_utility(1, 1.0)
        tracker.record_utility(2, 1.5)
        training = tracker.get_training_data()
        assert len(training) == 1
        assert training[0]["action"] == StructuralAction.ADD_EDGE

    def test_summary(self):
        tracker = MutationCreditTracker(horizons=[1, 2])
        tracker.record_mutation(
            action=StructuralAction.NO_OP, step=0,
            predicted_delta_u=0.0, predicted_uncertainty=0.0,
            governance_decision="accept", governance_reasons=[],
            graph_hash_before="a", graph_hash_after="a",
            config_governance_hash="c",
        )
        tracker.record_utility(1, 0.5)
        tracker.record_utility(2, 0.5)
        summary = tracker.summary()
        assert summary["total_mutations"] == 1
        assert summary["finalized"] == 1

    def test_save_load_state(self):
        tracker = MutationCreditTracker(horizons=[1, 2])
        tracker.record_mutation(
            action=StructuralAction.ADD_EDGE, step=0,
            predicted_delta_u=0.5, predicted_uncertainty=0.1,
            governance_decision="accept", governance_reasons=["test"],
            graph_hash_before="a", graph_hash_after="b",
            config_governance_hash="c",
        )
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
            path = f.name
        try:
            tracker.save_state(path)
            tracker2 = MutationCreditTracker()
            tracker2.load_state(path)
            assert len(tracker2.get_receipts()) == 1
        finally:
            os.unlink(path)


# ===========================================================================
# Consolidation tests
# ===========================================================================

class TestConsolidation:
    """Test stability/plasticity controller and fiber consolidation."""

    def test_register_fiber(self):
        ctrl = StabilityPlasticityController()
        fiber = ctrl.register_fiber(dimension=4, step=0)
        assert fiber.stage == FiberLifecycleStage.NEW
        assert ctrl.budget.total_fiber_dim == 4

    def test_remove_fiber(self):
        ctrl = StabilityPlasticityController()
        fiber = ctrl.register_fiber(dimension=4, step=0)
        ctrl.remove_fiber(fiber.fiber_id)
        assert ctrl.budget.total_fiber_dim == 0

    def test_lifecycle_new_to_probation(self):
        ctrl = StabilityPlasticityController(probation_length=10)
        fiber = ctrl.register_fiber(dimension=4, step=0)
        ctrl.update_lifecycle(1)  # Age 1
        assert fiber.stage == FiberLifecycleStage.PROBATION

    def test_lifecycle_probation_to_mature(self):
        """Fiber with moderate utility past probation → MATURE (not PROTECTED)."""
        ctrl = StabilityPlasticityController(
            probation_length=10, prune_threshold=0.01, protect_threshold=0.8,
        )
        fiber = ctrl.register_fiber(dimension=4, step=0)
        # Record moderate utility (above prune, below protect)
        for i in range(15):
            ctrl.record_fiber_utility(fiber.fiber_id, 0.5, i)
        ctrl.update_lifecycle(10)  # Past probation
        assert fiber.stage == FiberLifecycleStage.MATURE

    def test_lifecycle_mature_to_unused(self):
        """Fiber with declining utility → UNUSED."""
        ctrl = StabilityPlasticityController(
            probation_length=5, prune_threshold=0.1, protect_threshold=0.8,
        )
        fiber = ctrl.register_fiber(dimension=4, step=0)
        # Good utility during probation
        for i in range(6):
            ctrl.record_fiber_utility(fiber.fiber_id, 0.5, i)
        ctrl.update_lifecycle(5)
        assert fiber.stage == FiberLifecycleStage.MATURE
        # Then bad utility
        for i in range(6, 16):
            ctrl.record_fiber_utility(fiber.fiber_id, 0.001, i)
        ctrl.update_lifecycle(15)
        assert fiber.stage == FiberLifecycleStage.UNUSED

    def test_lifecycle_mature_to_protected(self):
        """Fiber with high utility → PROTECTED."""
        ctrl = StabilityPlasticityController(
            probation_length=5, protect_threshold=0.3,
        )
        fiber = ctrl.register_fiber(dimension=4, step=0)
        for i in range(6):
            ctrl.record_fiber_utility(fiber.fiber_id, 0.6, i)
        ctrl.update_lifecycle(5)
        # With cascaded transitions and high utility, goes to PROTECTED
        assert fiber.stage == FiberLifecycleStage.PROTECTED

    def test_gate_value_increases_during_probation(self):
        ctrl = StabilityPlasticityController(g_growth_rate=0.1)
        fiber = ctrl.register_fiber(dimension=4, step=0)
        assert fiber.g_value == 0.0
        ctrl.update_lifecycle(1)
        g1 = fiber.g_value
        assert g1 > 0.0
        ctrl.update_lifecycle(2)
        g2 = fiber.g_value
        assert g2 > g1  # Should increase more

    def test_capacity_budget_check(self):
        ctrl = StabilityPlasticityController(max_budget=20.0, alpha=0.1)
        ctrl.budget.total_edges = 50
        # Budget = 0 + 0.1*50 = 5, can grow by 15
        assert ctrl.budget.can_grow(delta_dim=10, delta_edges=0)
        # But not by 20
        assert not ctrl.budget.can_grow(delta_dim=20, delta_edges=0)

    def test_growth_justification(self):
        ctrl = StabilityPlasticityController(tau_efficiency=0.1)
        # ΔU/ΔB = 0.5/2.0 = 0.25 > 0.1 → justified
        assert ctrl.evaluate_growth(delta_utility=0.5, delta_budget=2.0)
        # ΔU/ΔB = 0.05/2.0 = 0.025 < 0.1 → not justified
        assert not ctrl.evaluate_growth(delta_utility=0.05, delta_budget=2.0)

    def test_can_spawn_fiber(self):
        ctrl = StabilityPlasticityController(max_budget=100.0, tau_efficiency=0.01)
        # Good utility, within budget
        assert ctrl.can_spawn_fiber(dimension=4, predicted_delta_u=1.0)
        # Bad utility, not justified
        assert not ctrl.can_spawn_fiber(dimension=4, predicted_delta_u=0.001)

    def test_prune_candidates(self):
        ctrl = StabilityPlasticityController(probation_length=5, prune_threshold=0.1)
        f1 = ctrl.register_fiber(dimension=4, step=0)
        f2 = ctrl.register_fiber(dimension=4, step=0)
        # f1: bad utility → unused
        for i in range(6):
            ctrl.record_fiber_utility(f1.fiber_id, 0.001, i)
        # f2: good utility → mature
        for i in range(6):
            ctrl.record_fiber_utility(f2.fiber_id, 0.5, i)
        ctrl.update_lifecycle(5)
        ctrl.update_lifecycle(6)
        prune_candidates = ctrl.get_prune_candidates()
        assert f1.fiber_id in prune_candidates
        assert f2.fiber_id not in prune_candidates

    def test_summary(self):
        ctrl = StabilityPlasticityController()
        ctrl.register_fiber(dimension=4, step=0)
        summary = ctrl.summary()
        assert summary["total_fibers"] == 1
        assert summary["total_fiber_dim"] == 4


# ===========================================================================
# Counterfactual engine tests
# ===========================================================================

class TestCounterfactualEngine:
    """Test structural counterfactual comparison."""

    def test_evaluate_returns_result(self):
        exec = StructuralExecutive()
        engine = StructuralCounterfactualEngine(exec)
        obs = StructuralObservation()
        result = engine.evaluate(obs)
        assert isinstance(result, CounterfactualResult)
        assert len(result.proposals) > 0
        assert result.no_op_baseline.action == StructuralAction.NO_OP

    def test_best_action_returns_action(self):
        exec = StructuralExecutive()
        engine = StructuralCounterfactualEngine(exec)
        obs = StructuralObservation()
        action = engine.best_action(obs)
        assert isinstance(action, StructuralAction)

    def test_no_op_always_included(self):
        exec = StructuralExecutive()
        engine = StructuralCounterfactualEngine(exec, max_candidates=3)
        obs = StructuralObservation()
        result = engine.evaluate(obs)
        actions = [p.action for p in result.proposals]
        assert StructuralAction.NO_OP in actions

    def test_shadow_simulator_updates_scores(self):
        exec = StructuralExecutive()
        engine = StructuralCounterfactualEngine(exec, max_candidates=NUM_ACTIONS)
        obs = StructuralObservation()

        def simulator(action):
            if action == StructuralAction.ADD_EDGE:
                return 1.0
            return 0.0

        result = engine.evaluate(obs, shadow_simulator=simulator)
        assert "add_edge" in result.shadow_utilities
        assert result.shadow_utilities["add_edge"] == 1.0


# ===========================================================================
# Closed loop tests
# ===========================================================================

class TestStructuralLearningLoop:
    """Test the closed structural learning loop."""

    def test_loop_initialization(self):
        loop = StructuralLearningLoop()
        assert loop.executive is not None
        assert loop.uncertainty_estimator is not None
        assert loop.credit_tracker is not None
        assert loop.consolidation is not None
        assert loop.counterfactual is not None

    def test_loop_step_returns_result(self):
        cfg = LGAEConfig()
        cfg.fiber.d_base = 4; cfg.fiber.d_max = 8
        graph = make_graph_buffers(6, [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)], capacity=10)
        z = torch.randn(6, 4)
        loop = StructuralLearningLoop(cfg)
        result = loop.step(graph, z)
        assert isinstance(result, StructuralLoopResult)
        assert result.step == 0
        assert isinstance(result.chosen_action, StructuralAction)

    def test_loop_multiple_steps(self):
        cfg = LGAEConfig()
        cfg.fiber.d_base = 4; cfg.fiber.d_max = 8
        graph = make_graph_buffers(4, [(0, 1), (1, 2), (2, 3)], capacity=6)
        z = torch.randn(4, 4)
        loop = StructuralLearningLoop(cfg)
        for i in range(5):
            result = loop.step(graph, z, task_loss=1.0 - i * 0.1)
            assert result.step == i

    def test_loop_summary(self):
        cfg = LGAEConfig()
        graph = make_graph_buffers(4, [(0, 1), (1, 2)], capacity=6)
        z = torch.randn(4, 4)
        loop = StructuralLearningLoop(cfg)
        loop.step(graph, z)
        summary = loop.summary()
        assert summary["step"] == 1
        assert "credit" in summary
        assert "consolidation" in summary


# ===========================================================================
# Integration: benchmark + executive
# ===========================================================================

class TestBenchmarkExecutiveIntegration:
    """Test the executive against the benchmark harness."""

    def test_executive_can_be_evaluated_on_benchmark(self):
        """The executive should be evaluable on the benchmark harness."""
        harness = BenchmarkHarness()

        def executive_fn(state):
            cfg = LGAEConfig()
            exec_model = StructuralExecutive(cfg)
            obs = exec_model.observe(state.graph, state.z)
            return exec_model.best_proposal(obs).action

        result = harness.evaluate_executive(executive_fn, seed=42)
        assert 0.0 <= result.diagnosis_accuracy <= 1.0
        assert result.total_tasks == len(ALL_TASKS)

    def test_oracle_beats_random(self):
        """Oracle should have better diagnosis accuracy than random."""
        harness = BenchmarkHarness()
        oracle = harness.run_oracle(seed=42)
        random_result = harness.run_random(seed=42)
        assert oracle.diagnosis_accuracy >= random_result.diagnosis_accuracy

    def test_oracle_has_zero_regret(self):
        """Oracle should have zero or near-zero mean regret."""
        harness = BenchmarkHarness()
        oracle = harness.run_oracle(seed=42)
        assert oracle.mean_regret == pytest.approx(0.0, abs=1e-6)
