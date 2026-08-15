"""v5.0 Benchmark harness: orchestration for structural learning evaluation.

The harness provides a unified interface for:
- Running tasks with proposed actions
- Comparing against oracle baselines
- Measuring structural diagnosis accuracy and mutation regret
- Evaluating the learned executive against the known-optimal actions
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import torch
import numpy as np

from .tasks import BenchmarkTask, StructuralAction, TaskState, TaskOutcome, ALL_TASKS
from .metrics import (
    StructuralDiagnosisResult,
    MutationRegretResult,
    BenchmarkResult,
    evaluate_diagnosis_accuracy,
    evaluate_mutation_regret,
    run_benchmark,
)


class BenchmarkHarness:
    """Orchestrates benchmark evaluation for the structural learning loop.

    Usage:
        harness = BenchmarkHarness()
        # Oracle baseline (always correct)
        oracle_result = harness.run_oracle()
        # Random baseline
        random_result = harness.run_random(seed=42)
        # Custom proposals
        proposals = {"A_bottleneck": StructuralAction.ADD_EDGE, ...}
        custom_result = harness.run(proposals)
    """

    def __init__(self, tasks: list[BenchmarkTask] | None = None):
        self.tasks = tasks or ALL_TASKS

    def run(
        self, proposals: dict[str, StructuralAction], seed: int = 42,
    ) -> BenchmarkResult:
        """Run benchmark with custom proposals."""
        return run_benchmark(proposals=proposals, seed=seed)

    def run_oracle(self, seed: int = 42) -> BenchmarkResult:
        """Run with oracle proposals (always correct action)."""
        return run_benchmark(proposals=None, seed=seed)

    def run_random(self, seed: int = 42) -> BenchmarkResult:
        """Run with random proposals (uniform over actions)."""
        rng = np.random.RandomState(seed)
        all_actions = list(StructuralAction)
        proposals = {}
        for task in self.tasks:
            proposals[task.name] = all_actions[rng.randint(len(all_actions))]
        return run_benchmark(proposals=proposals, seed=seed)

    def run_no_op_baseline(self, seed: int = 42) -> BenchmarkResult:
        """Run with NO_OP for all tasks (do-nothing baseline)."""
        proposals = {task.name: StructuralAction.NO_OP for task in self.tasks}
        return run_benchmark(proposals=proposals, seed=seed)

    def evaluate_executive(
        self,
        executive: Callable[[TaskState], StructuralAction],
        seed: int = 42,
    ) -> BenchmarkResult:
        """Evaluate a learned structural executive.

        Args:
            executive: A function that takes a TaskState and returns a
                       StructuralAction proposal.
            seed: Random seed for reproducibility.

        Returns:
            BenchmarkResult with the executive's performance.
        """
        proposals: dict[str, StructuralAction] = {}
        for task in self.tasks:
            state = task.initial_state(seed=seed)
            proposals[task.name] = executive(state)
        return run_benchmark(proposals=proposals, seed=seed)

    def summary(self, result: BenchmarkResult) -> str:
        """Generate a human-readable summary of benchmark results."""
        lines = [
            f"Benchmark Results ({result.total_tasks} tasks)",
            f"  Diagnosis Accuracy: {result.diagnosis_accuracy:.1%}",
            f"  Mean Regret: {result.mean_regret:.4f}",
            "",
            "  Per-task breakdown:",
        ]
        for diag, regret in zip(result.diagnosis_results, result.regret_results):
            correct_str = "✓" if diag.is_correct else "✗"
            lines.append(
                f"    {diag.task_name}: {correct_str} "
                f"proposed={diag.proposed_action.value}, "
                f"correct={{{', '.join(a.value for a in diag.correct_actions)}}}, "
                f"regret={regret.regret:.4f}"
            )
        return "\n".join(lines)
