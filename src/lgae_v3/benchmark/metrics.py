"""v5.0 Benchmark metrics: structural diagnosis accuracy and mutation regret.

Structural diagnosis accuracy: does the system identify the correct
structural intervention for each task?

Mutation regret: R_t = U(m_t*) - U(m_t), where m_t* is the optimal
action and m_t is the action actually taken.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
import numpy as np

from .tasks import (
    BenchmarkTask, StructuralAction, TaskState, TaskOutcome, ALL_TASKS,
)


@dataclass
class StructuralDiagnosisResult:
    """Result of evaluating structural diagnosis accuracy on one task."""
    task_name: str
    proposed_action: StructuralAction
    correct_actions: set[StructuralAction]
    is_correct: bool
    utility_per_action: dict[str, float]
    best_action: StructuralAction
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class MutationRegretResult:
    """Result of evaluating mutation regret on one task."""
    task_name: str
    chosen_action: StructuralAction
    optimal_action: StructuralAction
    chosen_utility: float
    optimal_utility: float
    regret: float
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class BenchmarkResult:
    """Aggregate benchmark result across all tasks."""
    diagnosis_results: list[StructuralDiagnosisResult]
    regret_results: list[MutationRegretResult]
    diagnosis_accuracy: float
    mean_regret: float
    total_tasks: int
    metadata: dict[str, Any] = field(default_factory=dict)


def evaluate_diagnosis_accuracy(
    task: BenchmarkTask,
    proposed_action: StructuralAction,
    seed: int = 42,
) -> StructuralDiagnosisResult:
    """Evaluate whether a proposed action is correct for a task.

    Args:
        task: The benchmark task
        proposed_action: The action proposed by the structural executive
        seed: Random seed for task initialization

    Returns:
        StructuralDiagnosisResult with diagnosis accuracy info
    """
    state = task.initial_state(seed=seed)
    correct = task.correct_actions()

    # Evaluate all actions to find the best one
    outcomes = task.evaluate_all(state)
    utility_per_action = {o.action.value: o.delta_utility for o in outcomes}
    best_action = max(outcomes, key=lambda o: o.delta_utility).action

    return StructuralDiagnosisResult(
        task_name=task.name,
        proposed_action=proposed_action,
        correct_actions=correct,
        is_correct=proposed_action in correct,
        utility_per_action=utility_per_action,
        best_action=best_action,
        metadata={"description": task.description},
    )


def evaluate_mutation_regret(
    task: BenchmarkTask,
    chosen_action: StructuralAction,
    seed: int = 42,
) -> MutationRegretResult:
    """Evaluate mutation regret for a chosen action.

    Regret = U(m*) - U(m_chosen), where m* is the optimal action.

    Args:
        task: The benchmark task
        chosen_action: The action actually taken
        seed: Random seed for task initialization

    Returns:
        MutationRegretResult with regret calculation
    """
    state = task.initial_state(seed=seed)
    outcomes = task.evaluate_all(state)

    # Find optimal action (highest delta utility)
    optimal = max(outcomes, key=lambda o: o.delta_utility)
    chosen = next(o for o in outcomes if o.action == chosen_action)

    regret = optimal.delta_utility - chosen.delta_utility

    return MutationRegretResult(
        task_name=task.name,
        chosen_action=chosen_action,
        optimal_action=optimal.action,
        chosen_utility=chosen.delta_utility,
        optimal_utility=optimal.delta_utility,
        regret=max(0.0, regret),  # Regret is non-negative
        metadata={"description": task.description},
    )


def run_benchmark(
    proposals: dict[str, StructuralAction] | None = None,
    seed: int = 42,
) -> BenchmarkResult:
    """Run the full benchmark suite.

    Args:
        proposals: Mapping from task name to proposed action. If None,
                   uses the correct action for each task (oracle baseline).
        seed: Random seed for reproducibility.

    Returns:
        BenchmarkResult with aggregate metrics.
    """
    if proposals is None:
        # Oracle baseline: always propose the correct action
        proposals = {}
        for task in ALL_TASKS:
            correct = task.correct_actions()
            proposals[task.name] = next(iter(correct))  # Pick first correct

    diagnosis_results: list[StructuralDiagnosisResult] = []
    regret_results: list[MutationRegretResult] = []

    for task in ALL_TASKS:
        proposed = proposals.get(task.name, StructuralAction.NO_OP)

        diag = evaluate_diagnosis_accuracy(task, proposed, seed=seed)
        diagnosis_results.append(diag)

        regret = evaluate_mutation_regret(task, proposed, seed=seed)
        regret_results.append(regret)

    diagnosis_accuracy = sum(1 for d in diagnosis_results if d.is_correct) / len(diagnosis_results)
    mean_regret = float(np.mean([r.regret for r in regret_results]))

    return BenchmarkResult(
        diagnosis_results=diagnosis_results,
        regret_results=regret_results,
        diagnosis_accuracy=diagnosis_accuracy,
        mean_regret=mean_regret,
        total_tasks=len(ALL_TASKS),
        metadata={"seed": seed, "proposals": {k: v.value for k, v in proposals.items()}},
    )
