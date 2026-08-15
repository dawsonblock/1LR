"""v5.0 Structural counterfactual engine.

Compares several candidate structural actions from the exact same state,
running shadow simulations for each. This gives a structural counterfactual:
"What would have happened if I had changed the representation rather than
the topology?"

The engine always includes NO_OP as a baseline. If no candidate beats
NO_OP after accounting for risk and cost, the system does nothing.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
from torch import Tensor

from .executive import StructuralExecutive, ActionProposal, StructuralObservation
from .benchmark.tasks import StructuralAction
from .types import GraphBuffers


@dataclass
class CounterfactualResult:
    """Result of a structural counterfactual comparison."""
    proposals: list[ActionProposal]
    winner: StructuralAction
    winner_proposal: ActionProposal | None = None
    no_op_baseline: ActionProposal | None = None
    beats_no_op: bool = False
    shadow_utilities: dict[str, float] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


class StructuralCounterfactualEngine:
    """Evaluates multiple structural actions from the same state.

    The engine:
    1. Observes the current state
    2. Proposes candidate actions (including NO_OP)
    3. Runs shadow simulations for each candidate
    4. Compares outcomes
    5. Returns the winner (which must beat NO_OP)

    This is the "counterfactual compete" step in the closed loop:
        geometry observes → learned executive predicts →
        counterfactuals compete → governor certifies →
        outcomes train the executive
    """

    def __init__(
        self,
        executive: StructuralExecutive,
        max_candidates: int = 5,
        no_op_penalty: float = 0.0,  # Extra penalty for non-NO_OP actions
    ):
        self.executive = executive
        self.max_candidates = max_candidates
        self.no_op_penalty = no_op_penalty

    def evaluate(
        self,
        observation: StructuralObservation,
        shadow_simulator: Any | None = None,
    ) -> CounterfactualResult:
        """Evaluate all candidate actions from the same state.

        Args:
            observation: Current structural observation
            shadow_simulator: Optional function that takes an action and
                             returns a simulated utility. If None, uses
                             the executive's predictions directly.

        Returns:
            CounterfactualResult with the winner and all proposals
        """
        # Get all proposals from the executive
        all_proposals = self.executive.propose(observation)

        # Take top candidates (including NO_OP)
        candidates = all_proposals[:self.max_candidates]

        # Ensure NO_OP is included
        no_op_proposal = next(
            (p for p in all_proposals if p.action == StructuralAction.NO_OP),
            all_proposals[-1],
        )
        if no_op_proposal not in candidates:
            candidates.append(no_op_proposal)

        # Run shadow simulations if available
        shadow_utilities: dict[str, float] = {}
        if shadow_simulator is not None:
            for prop in candidates:
                sim_utility = shadow_simulator(prop.action)
                shadow_utilities[prop.action.value] = sim_utility
                # Update the proposal's score with the simulated utility
                prop.score = sim_utility + self.executive.nu * prop.information_gain \
                           - self.executive.lam * prop.cost \
                           - self.executive.mu * prop.risk
                if prop.action != StructuralAction.NO_OP:
                    prop.score -= self.no_op_penalty

        # Find the winner (highest score)
        winner_proposal = max(candidates, key=lambda p: p.score)

        # Check if winner beats NO_OP
        beats_no_op = (
            winner_proposal.action == StructuralAction.NO_OP
            or winner_proposal.score > no_op_proposal.score
        )

        return CounterfactualResult(
            proposals=candidates,
            winner=winner_proposal.action,
            winner_proposal=winner_proposal,
            no_op_baseline=no_op_proposal,
            beats_no_op=beats_no_op,
            shadow_utilities=shadow_utilities,
            metadata={
                "max_candidates": self.max_candidates,
                "no_op_penalty": self.no_op_penalty,
            },
        )

    def best_action(
        self,
        observation: StructuralObservation,
        shadow_simulator: Any | None = None,
    ) -> StructuralAction:
        """Return the best action after counterfactual comparison.

        If no action beats NO_OP, returns NO_OP.
        """
        result = self.evaluate(observation, shadow_simulator)
        if result.beats_no_op:
            return result.winner
        return StructuralAction.NO_OP
