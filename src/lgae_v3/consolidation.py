"""v5.0 Stability–plasticity controller and fiber consolidation.

Prevents endless growth and premature pruning through:
- Capacity budget: B_t = Σ d_i + α|E|
- Growth justification: ΔU/ΔB > τ_efficiency
- Fiber lifecycle: NEW → PROBATION → MATURE → PROTECTED / UNUSED → PRUNE
- Probation gate: g(t) slowly increases during probation
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any
import math

import torch
from torch import Tensor

from .executive import StructuralAction
from .version import VERSION


class FiberLifecycleStage(Enum):
    """Lifecycle stages for a fiber."""
    NEW = "new"              # Just spawned, not yet evaluated
    PROBATION = "probation"  # Being evaluated, g(t) slowly increases
    MATURE = "mature"        # Passed probation, fully integrated
    PROTECTED = "protected"  # High utility, protected from pruning
    UNUSED = "unused"        # Low utility, candidate for pruning


@dataclass
class FiberState:
    """State of a single fiber in the consolidation system."""
    fiber_id: int
    dimension: int
    birth_step: int
    stage: FiberLifecycleStage = FiberLifecycleStage.NEW
    utility_history: list[float] = field(default_factory=list)
    last_used_step: int = 0
    g_value: float = 0.0  # Gate value (0 → 1 during probation)
    metadata: dict[str, Any] = field(default_factory=dict)

    def age(self, current_step: int) -> int:
        return current_step - self.birth_step

    def mean_utility(self) -> float:
        if not self.utility_history:
            return 0.0
        return sum(self.utility_history) / len(self.utility_history)

    def recent_utility(self, window: int = 10) -> float:
        if not self.utility_history:
            return 0.0
        recent = self.utility_history[-window:]
        return sum(recent) / len(recent)


@dataclass
class CapacityBudget:
    """Current capacity budget state."""
    total_fiber_dim: int = 0    # Σ d_i
    total_edges: int = 0        # |E|
    alpha: float = 0.1          # Edge weight in budget
    max_budget: float = float("inf")  # B_max (if set)

    @property
    def budget(self) -> float:
        """B_t = Σ d_i + α|E|"""
        return float(self.total_fiber_dim) + self.alpha * float(self.total_edges)

    def can_grow(self, delta_dim: int = 0, delta_edges: int = 0) -> bool:
        """Check if growth is within budget."""
        new_budget = (
            float(self.total_fiber_dim + delta_dim)
            + self.alpha * float(self.total_edges + delta_edges)
        )
        return new_budget <= self.max_budget


class StabilityPlasticityController:
    """Controls stability–plasticity balance and fiber consolidation.

    Enforces:
    1. Capacity budget: growth must stay within B_max
    2. Growth justification: ΔU/ΔB > τ_efficiency
    3. Fiber lifecycle: NEW → PROBATION → MATURE → PROTECTED / UNUSED → PRUNE
    4. Probation gate: g(t) slowly increases during probation
    """

    def __init__(
        self,
        alpha: float = 0.1,            # Edge weight in budget
        max_budget: float = float("inf"),  # B_max
        tau_efficiency: float = 0.01,  # Min ΔU/ΔB for growth
        probation_length: int = 100,   # Steps in probation
        prune_threshold: float = 0.01, # Utility below which → UNUSED
        protect_threshold: float = 0.5, # Utility above which → PROTECTED
        g_growth_rate: float = 0.01,   # g(t) growth per step during probation
    ):
        self.budget = CapacityBudget(alpha=alpha, max_budget=max_budget)
        self.tau_efficiency = tau_efficiency
        self.probation_length = probation_length
        self.prune_threshold = prune_threshold
        self.protect_threshold = protect_threshold
        self.g_growth_rate = g_growth_rate

        self._fibers: dict[int, FiberState] = {}
        self._next_fiber_id: int = 0

    def register_fiber(self, dimension: int, step: int) -> FiberState:
        """Register a newly spawned fiber."""
        fiber = FiberState(
            fiber_id=self._next_fiber_id,
            dimension=dimension,
            birth_step=step,
            stage=FiberLifecycleStage.NEW,
        )
        self._fibers[fiber.fiber_id] = fiber
        self._next_fiber_id += 1
        self.budget.total_fiber_dim += dimension
        return fiber

    def remove_fiber(self, fiber_id: int) -> FiberState | None:
        """Remove a fiber (pruning)."""
        fiber = self._fibers.pop(fiber_id, None)
        if fiber is not None:
            self.budget.total_fiber_dim -= fiber.dimension
        return fiber

    def record_fiber_utility(self, fiber_id: int, utility: float, step: int) -> None:
        """Record utility measurement for a fiber."""
        fiber = self._fibers.get(fiber_id)
        if fiber is None:
            return
        fiber.utility_history.append(utility)
        fiber.last_used_step = step

    def update_lifecycle(self, current_step: int) -> dict[int, FiberLifecycleStage]:
        """Update all fiber lifecycle stages based on age and utility.

        Returns a mapping of fiber_id → new stage.
        """
        changes: dict[int, FiberLifecycleStage] = {}
        for fid, fiber in self._fibers.items():
            old_stage = fiber.stage
            age = fiber.age(current_step)
            recent_util = fiber.recent_utility()

            # Update g(t) during probation
            if fiber.stage in (FiberLifecycleStage.NEW, FiberLifecycleStage.PROBATION):
                fiber.g_value = min(1.0, fiber.g_value + self.g_growth_rate)

            # Stage transitions (use independent if statements to allow
            # multiple transitions in a single update_lifecycle call)
            if fiber.stage == FiberLifecycleStage.NEW and age >= 1:
                fiber.stage = FiberLifecycleStage.PROBATION
            if fiber.stage == FiberLifecycleStage.PROBATION and age >= self.probation_length:
                if recent_util >= self.prune_threshold:
                    fiber.stage = FiberLifecycleStage.MATURE
                else:
                    fiber.stage = FiberLifecycleStage.UNUSED
            if fiber.stage == FiberLifecycleStage.MATURE:
                if recent_util >= self.protect_threshold:
                    fiber.stage = FiberLifecycleStage.PROTECTED
                elif recent_util < self.prune_threshold:
                    fiber.stage = FiberLifecycleStage.UNUSED
            if fiber.stage == FiberLifecycleStage.PROTECTED and recent_util < self.protect_threshold:
                fiber.stage = FiberLifecycleStage.MATURE

            if fiber.stage != old_stage:
                changes[fid] = fiber.stage

        return changes

    def get_prune_candidates(self) -> list[int]:
        """Return fiber IDs that are candidates for pruning (UNUSED stage)."""
        return [
            fid for fid, fiber in self._fibers.items()
            if fiber.stage == FiberLifecycleStage.UNUSED
        ]

    def get_protected_fibers(self) -> list[int]:
        """Return fiber IDs that are protected from pruning."""
        return [
            fid for fid, fiber in self._fibers.items()
            if fiber.stage == FiberLifecycleStage.PROTECTED
        ]

    def evaluate_growth(
        self,
        delta_utility: float,
        delta_budget: float,
    ) -> bool:
        """Evaluate whether growth is justified.

        Requires: ΔU/ΔB > τ_efficiency
        """
        if delta_budget <= 0:
            return True  # No growth, always allow
        efficiency = delta_utility / delta_budget
        return efficiency > self.tau_efficiency

    def can_spawn_fiber(
        self,
        dimension: int,
        delta_edges: int = 0,
        predicted_delta_u: float = 0.0,
    ) -> bool:
        """Check if a new fiber can be spawned.

        Combines budget check and growth justification.
        """
        delta_b = float(dimension) + self.budget.alpha * float(delta_edges)
        if not self.budget.can_grow(delta_dim=dimension, delta_edges=delta_edges):
            return False
        return self.evaluate_growth(predicted_delta_u, delta_b)

    def get_gate_value(self, fiber_id: int) -> float:
        """Get the current gate value g(t) for a fiber.

        During probation, g(t) slowly increases from 0 to 1.
        This gives new fibers time to learn before being fully active.
        """
        fiber = self._fibers.get(fiber_id)
        if fiber is None:
            return 1.0  # Default: fully active
        return fiber.g_value

    def summary(self) -> dict[str, Any]:
        """Return a summary of the consolidation state."""
        stage_counts: dict[str, int] = {}
        for fiber in self._fibers.values():
            stage_counts[fiber.stage.value] = stage_counts.get(fiber.stage.value, 0) + 1
        return {
            "total_fibers": len(self._fibers),
            "total_fiber_dim": self.budget.total_fiber_dim,
            "total_edges": self.budget.total_edges,
            "budget": self.budget.budget,
            "max_budget": self.budget.max_budget,
            "stage_counts": stage_counts,
            "prune_candidates": len(self.get_prune_candidates()),
            "protected_fibers": len(self.get_protected_fibers()),
            "version": VERSION,
        }
