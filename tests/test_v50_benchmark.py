"""v5.0 Benchmark harness tests."""
from __future__ import annotations

import pytest
import torch

from lgae_v3.benchmark import (
    BenchmarkHarness, BenchmarkResult,
    StructuralDiagnosisResult, MutationRegretResult,
    evaluate_diagnosis_accuracy, evaluate_mutation_regret, run_benchmark,
    TaskA_Bottleneck, TaskB_RepComplexity, TaskC_SpuriousEdge,
    TaskD_GaugeMismatch, TaskE_DistributionShift, TaskF_NoOp,
    ALL_TASKS, StructuralAction,
)
from lgae_v3.benchmark.tasks import TaskState


class TestBenchmarkTasks:
    """Test each benchmark task's basic structure."""

    @pytest.mark.parametrize("task", ALL_TASKS, ids=[t.name for t in ALL_TASKS])
    def test_task_has_initial_state(self, task):
        state = task.initial_state(seed=42)
        assert state.graph is not None
        assert state.z is not None
        assert state.config is not None

    @pytest.mark.parametrize("task", ALL_TASKS, ids=[t.name for t in ALL_TASKS])
    def test_task_has_correct_actions(self, task):
        correct = task.correct_actions()
        assert len(correct) > 0
        assert all(isinstance(a, StructuralAction) for a in correct)

    @pytest.mark.parametrize("task", ALL_TASKS, ids=[t.name for t in ALL_TASKS])
    def test_task_utility_is_finite(self, task):
        state = task.initial_state(seed=42)
        u = task.utility(state)
        assert isinstance(u, float)
        assert u == u  # Not NaN

    @pytest.mark.parametrize("task", ALL_TASKS, ids=[t.name for t in ALL_TASKS])
    def test_no_op_does_not_crash(self, task):
        state = task.initial_state(seed=42)
        outcome = task.evaluate(state, StructuralAction.NO_OP)
        assert outcome.action == StructuralAction.NO_OP
        assert isinstance(outcome.delta_utility, float)


class TestTaskA_Bottleneck:
    """Test Task A: long-range bottleneck."""

    def test_add_edge_improves_utility(self):
        task = TaskA_Bottleneck()
        state = task.initial_state()
        outcome = task.evaluate(state, StructuralAction.ADD_EDGE)
        assert outcome.delta_utility > 0, "Adding alternate route should improve utility"

    def test_prune_edge_decreases_utility(self):
        task = TaskA_Bottleneck()
        state = task.initial_state()
        outcome = task.evaluate(state, StructuralAction.PRUNE_EDGE)
        assert outcome.delta_utility < 0, "Pruning the bridge should decrease utility"

    def test_add_edge_is_best_or_near_best(self):
        task = TaskA_Bottleneck()
        state = task.initial_state()
        outcomes = task.evaluate_all(state)
        add_edge_outcome = next(o for o in outcomes if o.action == StructuralAction.ADD_EDGE)
        no_op_outcome = next(o for o in outcomes if o.action == StructuralAction.NO_OP)
        assert add_edge_outcome.delta_utility > no_op_outcome.delta_utility


class TestTaskB_RepComplexity:
    """Test Task B: local representational complexity."""

    def test_spawn_fiber_improves_utility(self):
        task = TaskB_RepComplexity()
        state = task.initial_state()
        outcome = task.evaluate(state, StructuralAction.SPAWN_FIBER)
        assert outcome.delta_utility > 0, "Spawning fiber should improve utility"

    def test_no_op_does_not_improve(self):
        task = TaskB_RepComplexity()
        state = task.initial_state()
        outcome = task.evaluate(state, StructuralAction.NO_OP)
        assert outcome.delta_utility <= 0, "NO_OP should not improve utility"


class TestTaskC_SpuriousEdge:
    """Test Task C: noisy spurious edge."""

    def test_prune_edge_improves_utility(self):
        task = TaskC_SpuriousEdge()
        state = task.initial_state()
        outcome = task.evaluate(state, StructuralAction.PRUNE_EDGE)
        assert outcome.delta_utility > 0, "Pruning spurious edge should improve utility"

    def test_add_edge_does_not_help_as_much(self):
        task = TaskC_SpuriousEdge()
        state = task.initial_state()
        prune_outcome = task.evaluate(state, StructuralAction.PRUNE_EDGE)
        add_outcome = task.evaluate(state, StructuralAction.ADD_EDGE)
        assert prune_outcome.delta_utility > add_outcome.delta_utility


class TestTaskD_GaugeMismatch:
    """Test Task D: coordinate-frame mismatch."""

    def test_change_gauge_improves_utility(self):
        task = TaskD_GaugeMismatch()
        state = task.initial_state()
        outcome = task.evaluate(state, StructuralAction.CHANGE_GAUGE)
        assert outcome.delta_utility > 0, "Gauge adaptation should improve utility"

    def test_no_op_does_not_improve(self):
        task = TaskD_GaugeMismatch()
        state = task.initial_state()
        outcome = task.evaluate(state, StructuralAction.NO_OP)
        assert outcome.delta_utility <= 0.01, "NO_OP should not improve utility"


class TestTaskE_DistributionShift:
    """Test Task E: distribution shift."""

    def test_spawn_fiber_improves_utility(self):
        task = TaskE_DistributionShift()
        state = task.initial_state()
        outcome = task.evaluate(state, StructuralAction.SPAWN_FIBER)
        assert outcome.delta_utility > 0, "Spawning fiber should improve utility"


class TestTaskF_NoOp:
    """Test Task F: nothing wrong."""

    def test_no_op_is_correct(self):
        task = TaskF_NoOp()
        correct = task.correct_actions()
        assert StructuralAction.NO_OP in correct

    def test_no_op_has_zero_or_positive_regret(self):
        task = TaskF_NoOp()
        state = task.initial_state()
        outcome = task.evaluate(state, StructuralAction.NO_OP)
        # NO_OP should not make things worse
        assert outcome.delta_utility >= -0.01


class TestBenchmarkMetrics:
    """Test benchmark metric calculations."""

    def test_oracle_achieves_perfect_accuracy(self):
        result = run_benchmark(proposals=None, seed=42)
        assert result.diagnosis_accuracy == 1.0, "Oracle should have perfect accuracy"
        assert result.mean_regret == 0.0, "Oracle should have zero regret"

    def test_random_baseline_accuracy_is_not_perfect(self):
        result = run_benchmark(
            proposals={t.name: StructuralAction.NO_OP for t in ALL_TASKS},
            seed=42,
        )
        # NO_OP for all tasks: only Task F is correct
        assert result.diagnosis_accuracy == 1.0 / len(ALL_TASKS)

    def test_diagnosis_accuracy_calculation(self):
        task = TaskA_Bottleneck()
        result = evaluate_diagnosis_accuracy(task, StructuralAction.ADD_EDGE, seed=42)
        assert result.is_correct is True
        assert StructuralAction.ADD_EDGE in result.correct_actions

    def test_diagnosis_accuracy_wrong_action(self):
        task = TaskA_Bottleneck()
        result = evaluate_diagnosis_accuracy(task, StructuralAction.NO_OP, seed=42)
        assert result.is_correct is False

    def test_mutation_regret_zero_for_optimal(self):
        task = TaskA_Bottleneck()
        state = task.initial_state(seed=42)
        outcomes = task.evaluate_all(state)
        optimal = max(outcomes, key=lambda o: o.delta_utility)
        regret = evaluate_mutation_regret(task, optimal.action, seed=42)
        assert regret.regret == pytest.approx(0.0, abs=1e-10)

    def test_mutation_regret_positive_for_suboptimal(self):
        task = TaskA_Bottleneck()
        state = task.initial_state(seed=42)
        outcomes = task.evaluate_all(state)
        optimal = max(outcomes, key=lambda o: o.delta_utility)
        worst = min(outcomes, key=lambda o: o.delta_utility)
        if optimal.action != worst.action:
            regret = evaluate_mutation_regret(task, worst.action, seed=42)
            assert regret.regret > 0


class TestBenchmarkHarness:
    """Test the benchmark harness orchestration."""

    def test_harness_oracle(self):
        harness = BenchmarkHarness()
        result = harness.run_oracle(seed=42)
        assert result.diagnosis_accuracy == 1.0
        assert result.mean_regret == 0.0

    def test_harness_no_op_baseline(self):
        harness = BenchmarkHarness()
        result = harness.run_no_op_baseline(seed=42)
        # Only Task F has NO_OP as correct
        assert result.diagnosis_accuracy > 0
        assert result.diagnosis_accuracy <= 1.0

    def test_harness_random_baseline(self):
        harness = BenchmarkHarness()
        result = harness.run_random(seed=42)
        assert 0.0 <= result.diagnosis_accuracy <= 1.0
        assert result.mean_regret >= 0.0

    def test_harness_summary(self):
        harness = BenchmarkHarness()
        result = harness.run_oracle(seed=42)
        summary = harness.summary(result)
        assert "Diagnosis Accuracy" in summary
        assert "Mean Regret" in summary

    def test_harness_evaluate_executive(self):
        """Test evaluating a simple executive that always proposes ADD_EDGE."""
        harness = BenchmarkHarness()

        def simple_executive(state: TaskState) -> StructuralAction:
            return StructuralAction.ADD_EDGE

        result = harness.evaluate_executive(simple_executive, seed=42)
        # ADD_EDGE is correct for Task A only
        assert result.diagnosis_accuracy > 0
        assert result.total_tasks == len(ALL_TASKS)

    def test_all_tasks_have_unique_names(self):
        names = [t.name for t in ALL_TASKS]
        assert len(names) == len(set(names)), "Task names must be unique"

    def test_all_tasks_cover_all_action_types(self):
        """All major action types should be the correct action for at least one task."""
        all_correct = set()
        for task in ALL_TASKS:
            all_correct.update(task.correct_actions())
        # At minimum: ADD_EDGE, SPAWN_FIBER, PRUNE_EDGE, CHANGE_GAUGE, NO_OP
        required = {StructuralAction.ADD_EDGE, StructuralAction.SPAWN_FIBER,
                    StructuralAction.PRUNE_EDGE, StructuralAction.CHANGE_GAUGE,
                    StructuralAction.NO_OP}
        assert required.issubset(all_correct), (
            f"Missing actions: {required - all_correct}"
        )
