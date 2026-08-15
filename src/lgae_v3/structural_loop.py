"""v5.0 Closed structural learning loop.

The closed loop:
    geometry observes → learned executive predicts →
    counterfactuals compete → governor certifies →
    outcomes train the executive

This module ties together:
- StructuralExecutive (proposal generation)
- StructuralCounterfactualEngine (candidate comparison)
- EnsembleUncertainty (calibrated uncertainty)
- MutationCreditTracker (long-term credit assignment)
- StabilityPlasticityController (capacity budget, fiber lifecycle)
- GeometryGovernor (certification authority)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import torch
from torch import Tensor

from .executive import (
    StructuralExecutive, ActionProposal, StructuralObservation,
    StructuralAction, ACTION_TO_IDX,
)
from .counterfactual import StructuralCounterfactualEngine, CounterfactualResult
from .uncertainty import EnsembleUncertainty, ConformalCalibrator, uncertainty_gated_decision
from .credit import MutationCreditTracker, MutationReceipt
from .consolidation import StabilityPlasticityController, FiberLifecycleStage
from .types import GraphBuffers
from .config import LGAEConfig, config_governance_hash
from .version import VERSION


@dataclass
class StructuralLoopResult:
    """Result of one step of the structural learning loop."""
    step: int
    observation: StructuralObservation
    counterfactual: CounterfactualResult
    chosen_action: StructuralAction
    uncertainty_decision: str  # "accept", "quarantine", "reject"
    governance_decision: str   # Governor's decision
    executed: bool             # Whether the mutation was actually executed
    utility_before: float
    utility_after: float
    delta_utility: float
    metadata: dict[str, Any] = field(default_factory=dict)


class StructuralLearningLoop:
    """The closed structural learning loop.

    Usage:
        loop = StructuralLearningLoop(config)
        for step in range(num_steps):
            result = loop.step(graph, z, audit_snapshot, task_loss, task_loss_delta)
            # result.chosen_action is the action taken
            # result.delta_utility is the observed utility change
    """

    def __init__(
        self,
        config: LGAEConfig | None = None,
        # Executive hyperparameters
        executive: StructuralExecutive | None = None,
        # Uncertainty
        ensemble_size: int = 5,
        beta: float = 1.0,
        # Credit assignment
        gamma: float = 0.99,
        credit_horizons: list[int] | None = None,
        # Consolidation
        max_budget: float = float("inf"),
        tau_efficiency: float = 0.01,
        probation_length: int = 100,
        # Counterfactual
        max_candidates: int = 5,
        no_op_penalty: float = 0.0,
    ):
        self.config = config or LGAEConfig()
        self.executive = executive or StructuralExecutive(self.config)
        self.uncertainty_estimator = EnsembleUncertainty(
            self.executive, ensemble_size=ensemble_size, beta=beta,
        )
        self.calibrator = ConformalCalibrator(alpha=0.1)
        self.credit_tracker = MutationCreditTracker(
            gamma=gamma, horizons=credit_horizons or [16, 100, 1000],
        )
        self.consolidation = StabilityPlasticityController(
            max_budget=max_budget,
            tau_efficiency=tau_efficiency,
            probation_length=probation_length,
        )
        self.counterfactual = StructuralCounterfactualEngine(
            self.executive, max_candidates=max_candidates, no_op_penalty=no_op_penalty,
        )
        self._step: int = 0

    def step(
        self,
        graph: GraphBuffers,
        z: Tensor,
        audit_snapshot: Any | None = None,
        task_loss: float = 0.0,
        task_loss_delta: float = 0.0,
        epistemic_uncertainty: float = 0.0,
        utility_fn: Callable[[GraphBuffers, Tensor], float] | None = None,
        shadow_simulator: Callable[[StructuralAction], float] | None = None,
    ) -> StructuralLoopResult:
        """Execute one step of the structural learning loop.

        Args:
            graph: Current graph state
            z: Current latent state
            audit_snapshot: Latest audit from the governor
            task_loss: Current task loss
            task_loss_delta: Change in task loss since last step
            epistemic_uncertainty: Current epistemic uncertainty estimate
            utility_fn: Optional function to measure task utility
            shadow_simulator: Optional function to simulate action outcomes

        Returns:
            StructuralLoopResult with the full decision trace
        """
        step = self._step

        # 1. OBSERVE: Construct structural observation
        observation = self.executive.observe(
            graph, z, audit_snapshot, task_loss, task_loss_delta, epistemic_uncertainty,
        )

        # 2. PREDICT + COUNTERFACTUAL: Compare candidate actions
        counterfactual = self.counterfactual.evaluate(observation, shadow_simulator)
        chosen_action = counterfactual.winner if counterfactual.beats_no_op else StructuralAction.NO_OP

        # 3. UNCERTAINTY: Estimate epistemic uncertainty for the chosen action
        obs_vec = observation.to_vector()
        action_idx = ACTION_TO_IDX.get(chosen_action, 0)
        unc_estimate = self.uncertainty_estimator.estimate(obs_vec, action_idx)
        uncertainty_decision = uncertainty_gated_decision(
            ActionProposal(
                action=chosen_action,
                expected_delta_utility=unc_estimate.mean,
                information_gain=0.0, cost=0.0, risk=0.0,
                score=unc_estimate.mean, uncertainty=unc_estimate.std,
                lcb=unc_estimate.lcb,
            ),
            unc_estimate,
        )

        # 4. CERTIFY: The governor certifies (simulated here)
        # In full integration, this calls governor.evaluate_mutation
        governance_decision = "accept" if uncertainty_decision == "accept" else "quarantine"

        # 5. EXECUTE: Only if both uncertainty and governance agree
        executed = (
            uncertainty_decision in ("accept", "quarantine")
            and governance_decision in ("accept", "quarantine")
            and chosen_action != StructuralAction.NO_OP
        )

        # Measure utility
        u_before = utility_fn(graph, z) if utility_fn else 0.0
        u_after = u_before  # Default: no change
        delta_u = 0.0

        if executed and chosen_action != StructuralAction.NO_OP:
            # Record mutation in history
            self.executive.record_mutation(chosen_action)

            # Record graph hash before execution
            hash_before = graph.state_hash()

            # If SPAWN_FIBER, register in consolidation
            if chosen_action == StructuralAction.SPAWN_FIBER:
                new_dim = max(1, self.config.fiber.d_max - z.shape[1])
                self.consolidation.register_fiber(
                    dimension=new_dim,
                    step=step,
                )

            # Measure after execution (if utility_fn provided)
            if utility_fn is not None:
                u_after = utility_fn(graph, z)
            delta_u = u_after - u_before

            # Record graph hash after execution
            hash_after = graph.state_hash()

            # Record in credit tracker
            self.credit_tracker.record_mutation(
                action=chosen_action,
                step=step,
                predicted_delta_u=counterfactual.winner_proposal.expected_delta_utility
                                  if counterfactual.winner_proposal else 0.0,
                predicted_uncertainty=unc_estimate.std,
                governance_decision=governance_decision,
                governance_reasons=[],
                graph_hash_before=hash_before,
                graph_hash_after=hash_after,
                config_governance_hash=config_governance_hash(self.config),
            )

        # 6. TRAIN: Record outcome for executive learning
        if executed:
            self.executive.record_outcome(observation, chosen_action, delta_u)
            # Record utility at the mutation step (baseline)
            self.credit_tracker.record_utility(step, u_before)
            # Record utility after execution
            self.credit_tracker.record_utility(step + 1, u_after)

        # Update consolidation lifecycle
        self.consolidation.update_lifecycle(step)

        # Train executive
        if len(self.executive._experience) >= 32:
            self.executive.train_step()

        self._step += 1

        return StructuralLoopResult(
            step=step,
            observation=observation,
            counterfactual=counterfactual,
            chosen_action=chosen_action,
            uncertainty_decision=uncertainty_decision,
            governance_decision=governance_decision,
            executed=executed,
            utility_before=u_before,
            utility_after=u_after,
            delta_utility=delta_u,
            metadata={
                "version": VERSION,
                "uncertainty": {
                    "mean": unc_estimate.mean,
                    "std": unc_estimate.std,
                    "lcb": unc_estimate.lcb,
                    "ucb": unc_estimate.ucb,
                },
                "consolidation": self.consolidation.summary(),
            },
        )

    def summary(self) -> dict[str, Any]:
        """Return a summary of the entire loop state."""
        return {
            "step": self._step,
            "executive_experience": len(self.executive._experience),
            "credit": self.credit_tracker.summary(),
            "consolidation": self.consolidation.summary(),
            "version": VERSION,
        }
