"""v5.1 Multi-timescale adaptation controller.

Separates structural adaptation into three timescales to prevent
every kind of structure from moving simultaneously:

    fast (every step):     gauge U_ij, gates g_i
    medium (every ~100):   affinity a_ij, fibers F_i
    slow (every ~1000):    length ℓ_ij, topology E, V

This matches the practical observation that jointly changing topology
and representation can induce mutual drift, motivating alternating
or two-timescale optimization.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import torch
from torch import Tensor

from .version import VERSION


class Timescale(Enum):
    """Adaptation timescales."""
    FAST = "fast"       # Every step: gauge, gates
    MEDIUM = "medium"   # Every ~100 steps: affinity, fibers
    SLOW = "slow"       # Every ~1000 steps: length, topology


@dataclass
class TimescaleSchedule:
    """Schedule for when each timescale is active."""
    fast_interval: int = 1        # Every step
    medium_interval: int = 100    # Every 100 steps
    slow_interval: int = 1000     # Every 1000 steps

    def is_active(self, timescale: Timescale, step: int) -> bool:
        """Check if a timescale is active at the given step."""
        if timescale == Timescale.FAST:
            return step % self.fast_interval == 0
        elif timescale == Timescale.MEDIUM:
            return step % self.medium_interval == 0
        elif timescale == Timescale.SLOW:
            return step % self.slow_interval == 0
        return False

    def active_timescales(self, step: int) -> set[Timescale]:
        """Return all active timescales at the given step."""
        active: set[Timescale] = set()
        for ts in Timescale:
            if self.is_active(ts, step):
                active.add(ts)
        return active


@dataclass
class AdaptationState:
    """State of the multi-timescale controller."""
    step: int = 0
    fast_updates: int = 0
    medium_updates: int = 0
    slow_updates: int = 0
    last_fast_step: int = 0
    last_medium_step: int = 0
    last_slow_step: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


class MultiTimescaleController:
    """Controls which structural components can adapt at each step.

    This prevents mutual drift by ensuring that fast-adapting components
    (gauge, gates) converge before slower components (affinity, topology)
    are allowed to change.

    Usage:
        controller = MultiTimescaleController()
        for step in range(num_steps):
            active = controller.update(step)
            if Timescale.FAST in active:
                # Update gauge connections, gate values
                ...
            if Timescale.MEDIUM in active:
                # Update affinities, fiber states
                ...
            if Timescale.SLOW in active:
                # Update edge lengths, topology
                ...
    """

    def __init__(
        self,
        schedule: TimescaleSchedule | None = None,
        # Minimum fast updates before medium is allowed
        min_fast_before_medium: int = 50,
        # Minimum medium updates before slow is allowed
        min_medium_before_slow: int = 10,
    ):
        self.schedule = schedule or TimescaleSchedule()
        self.min_fast_before_medium = min_fast_before_medium
        self.min_medium_before_slow = min_medium_before_slow
        self.state = AdaptationState()

    def update(self, step: int) -> set[Timescale]:
        """Determine which timescales are active at this step.

        Returns the set of active timescales. Also enforces the
        minimum-convergence constraint: medium is not active until
        enough fast updates have occurred, and slow is not active
        until enough medium updates have occurred.
        """
        active = self.schedule.active_timescales(step)

        # Enforce minimum convergence constraints
        if Timescale.MEDIUM in active:
            if self.state.fast_updates < self.min_fast_before_medium:
                active.discard(Timescale.MEDIUM)

        if Timescale.SLOW in active:
            if self.state.medium_updates < self.min_medium_before_slow:
                active.discard(Timescale.SLOW)

        # Update counters
        if Timescale.FAST in active:
            self.state.fast_updates += 1
            self.state.last_fast_step = step
        if Timescale.MEDIUM in active:
            self.state.medium_updates += 1
            self.state.last_medium_step = step
        if Timescale.SLOW in active:
            self.state.slow_updates += 1
            self.state.last_slow_step = step

        self.state.step = step
        return active

    def can_adapt_gauge(self, step: int) -> bool:
        """Check if gauge connections can adapt at this step."""
        return Timescale.FAST in self.update(step)

    def can_adapt_affinity(self, step: int) -> bool:
        """Check if affinities can adapt at this step."""
        active = self.update(step)
        return Timescale.MEDIUM in active

    def can_adapt_length(self, step: int) -> bool:
        """Check if edge lengths can adapt at this step."""
        active = self.update(step)
        return Timescale.SLOW in active

    def can_adapt_topology(self, step: int) -> bool:
        """Check if topology (add/prune edges) can change at this step."""
        active = self.update(step)
        return Timescale.SLOW in active

    def can_spawn_fiber(self, step: int) -> bool:
        """Check if fibers can be spawned/pruned at this step."""
        active = self.update(step)
        return Timescale.MEDIUM in active

    def summary(self) -> dict[str, Any]:
        """Return a summary of the adaptation state."""
        return {
            "step": self.state.step,
            "fast_updates": self.state.fast_updates,
            "medium_updates": self.state.medium_updates,
            "slow_updates": self.state.slow_updates,
            "last_fast_step": self.state.last_fast_step,
            "last_medium_step": self.state.last_medium_step,
            "last_slow_step": self.state.last_slow_step,
            "schedule": {
                "fast_interval": self.schedule.fast_interval,
                "medium_interval": self.schedule.medium_interval,
                "slow_interval": self.schedule.slow_interval,
            },
            "version": VERSION,
        }
