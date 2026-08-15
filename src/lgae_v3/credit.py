"""v5.0 Long-term mutation credit assignment.

Tracks mutation receipts and their long-term outcomes. Each committed
mutation gets a receipt and later outcome measurements at multiple horizons.

R_mutation = Σ_{τ=0}^{T} γ^τ ΔU_{t+τ}

The executive learns from its own structural history by comparing
initial predictions to long-term outcomes.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from collections import deque
import time

import torch
import numpy as np

from .executive import StructuralAction
from .version import VERSION


@dataclass
class MutationReceipt:
    """Receipt for a committed mutation."""
    receipt_id: int
    action: StructuralAction
    step: int                          # Global step when mutation was committed
    predicted_delta_u: float           # Executive's initial prediction
    predicted_uncertainty: float       # Executive's initial uncertainty
    governance_decision: str           # "accept", "quarantine", "reject"
    governance_reasons: list[str]      # Governor's reasons
    graph_hash_before: str             # State hash before mutation
    graph_hash_after: str              # State hash after mutation
    config_governance_hash: str        # Governance config hash
    version: str = VERSION
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class MutationOutcome:
    """Long-term outcome of a mutation."""
    receipt_id: int
    action: StructuralAction
    # Utility measurements at multiple horizons
    utility_at_16: float | None = None       # 16-step outcome
    utility_at_100: float | None = None      # 100-step outcome
    utility_at_1000: float | None = None     # 1000-step outcome
    # Discounted return
    discounted_return: float = 0.0
    # Prediction error
    prediction_error: float = 0.0
    # Decision
    retained: bool = True
    # Metadata
    metadata: dict[str, Any] = field(default_factory=dict)


class MutationCreditTracker:
    """Tracks mutation receipts and long-term outcomes.

    Computes discounted returns:
        R = Σ_{τ=0}^{T} γ^τ ΔU_{t+τ}

    and feeds the results back to the executive for learning.
    """

    def __init__(
        self,
        gamma: float = 0.99,       # Discount factor
        max_history: int = 1000,   # Maximum receipts to keep
        horizons: list[int] = None,  # Evaluation horizons
    ):
        self.gamma = gamma
        self.max_history = max_history
        self.horizons = horizons or [16, 100, 1000]

        self._receipts: deque[MutationReceipt] = deque(maxlen=max_history)
        self._outcomes: deque[MutationOutcome] = deque(maxlen=max_history)
        self._utility_history: deque = deque(maxlen=max(self.horizons) + 100)
        self._next_id: int = 0
        self._pending: dict[int, dict] = {}  # receipt_id → pending tracking

    def record_mutation(
        self,
        action: StructuralAction,
        step: int,
        predicted_delta_u: float,
        predicted_uncertainty: float,
        governance_decision: str,
        governance_reasons: list[str],
        graph_hash_before: str,
        graph_hash_after: str,
        config_governance_hash: str,
        metadata: dict[str, Any] | None = None,
    ) -> MutationReceipt:
        """Record a committed mutation and return its receipt."""
        receipt = MutationReceipt(
            receipt_id=self._next_id,
            action=action,
            step=step,
            predicted_delta_u=predicted_delta_u,
            predicted_uncertainty=predicted_uncertainty,
            governance_decision=governance_decision,
            governance_reasons=governance_reasons,
            graph_hash_before=graph_hash_before,
            graph_hash_after=graph_hash_after,
            config_governance_hash=config_governance_hash,
            metadata=metadata or {},
        )
        self._receipts.append(receipt)
        self._pending[receipt.receipt_id] = {
            "action": action,
            "step": step,
            "predicted_delta_u": predicted_delta_u,
            "utility_samples": [],
            "baseline_utility": None,  # Set on first record_utility call
        }
        self._next_id += 1
        return receipt

    def record_utility(self, step: int, utility: float) -> None:
        """Record the current task utility at a given step."""
        self._utility_history.append((step, utility))

        # Check if any pending mutations have reached their horizons
        for rid, pending in list(self._pending.items()):
            mut_step = pending["step"]
            age = step - mut_step

            # Record baseline utility at age 0 (the step of the mutation)
            if age == 0 and pending["baseline_utility"] is None:
                pending["baseline_utility"] = utility

            if age in self.horizons:
                pending["utility_samples"].append((age, utility))

            # Check if all horizons are reached
            if len(pending["utility_samples"]) >= len(self.horizons):
                self._finalize_outcome(rid, pending)

    def _finalize_outcome(self, receipt_id: int, pending: dict) -> None:
        """Finalize a mutation outcome once all horizons are reached."""
        # Find the receipt
        receipt = next((r for r in self._receipts if r.receipt_id == receipt_id), None)
        if receipt is None:
            return

        # Get utility at each horizon
        utility_at = {}
        for age, util in pending["utility_samples"]:
            utility_at[age] = util

        # Compute discounted return
        # R = Σ γ^τ ΔU_τ where ΔU_τ = U_{t+τ} - U_baseline
        baseline = pending.get("baseline_utility", 0.0) or 0.0
        discounted_return = 0.0
        for age, util in pending["utility_samples"]:
            delta_u = util - baseline
            discounted_return += (self.gamma ** age) * delta_u

        # Prediction error
        prediction_error = abs(discounted_return - pending["predicted_delta_u"])

        # Retention decision: keep if discounted return is positive
        retained = discounted_return > 0

        outcome = MutationOutcome(
            receipt_id=receipt_id,
            action=receipt.action,
            utility_at_16=utility_at.get(16),
            utility_at_100=utility_at.get(100),
            utility_at_1000=utility_at.get(1000),
            discounted_return=discounted_return,
            prediction_error=prediction_error,
            retained=retained,
            metadata={"predicted_delta_u": pending["predicted_delta_u"]},
        )
        self._outcomes.append(outcome)
        del self._pending[receipt_id]

    def get_outcomes(self) -> list[MutationOutcome]:
        """Return all finalized outcomes."""
        return list(self._outcomes)

    def get_receipts(self) -> list[MutationReceipt]:
        """Return all receipts."""
        return list(self._receipts)

    def get_training_data(self) -> list[dict]:
        """Return training data for the executive.

        Each entry contains:
        - action: The structural action taken
        - predicted_delta_u: What the executive predicted
        - actual_return: The discounted return
        - prediction_error: |predicted - actual|
        """
        training_data = []
        for outcome in self._outcomes:
            receipt = next(
                (r for r in self._receipts if r.receipt_id == outcome.receipt_id),
                None,
            )
            if receipt is None:
                continue
            training_data.append({
                "action": outcome.action,
                "predicted_delta_u": receipt.predicted_delta_u,
                "actual_return": outcome.discounted_return,
                "prediction_error": outcome.prediction_error,
                "retained": outcome.retained,
            })
        return training_data

    def summary(self) -> dict[str, Any]:
        """Return a summary of mutation credit tracking."""
        outcomes = self.get_outcomes()
        if not outcomes:
            return {"total_mutations": len(self._receipts), "finalized": 0}

        returns = [o.discounted_return for o in outcomes]
        errors = [o.prediction_error for o in outcomes]
        retained = sum(1 for o in outcomes if o.retained)

        return {
            "total_mutations": len(self._receipts),
            "finalized": len(outcomes),
            "pending": len(self._pending),
            "mean_return": float(np.mean(returns)),
            "std_return": float(np.std(returns)),
            "mean_prediction_error": float(np.mean(errors)),
            "retention_rate": retained / len(outcomes),
            "version": VERSION,
        }

    def save_state(self, path: str) -> None:
        """Save credit tracker state."""
        import json
        state = {
            "gamma": self.gamma,
            "horizons": self.horizons,
            "next_id": self._next_id,
            "receipts": [
                {
                    "receipt_id": r.receipt_id,
                    "action": r.action.value,
                    "step": r.step,
                    "predicted_delta_u": r.predicted_delta_u,
                    "predicted_uncertainty": r.predicted_uncertainty,
                    "governance_decision": r.governance_decision,
                    "governance_reasons": r.governance_reasons,
                    "graph_hash_before": r.graph_hash_before,
                    "graph_hash_after": r.graph_hash_after,
                    "config_governance_hash": r.config_governance_hash,
                    "version": r.version,
                }
                for r in self._receipts
            ],
            "outcomes": [
                {
                    "receipt_id": o.receipt_id,
                    "action": o.action.value,
                    "utility_at_16": o.utility_at_16,
                    "utility_at_100": o.utility_at_100,
                    "utility_at_1000": o.utility_at_1000,
                    "discounted_return": o.discounted_return,
                    "prediction_error": o.prediction_error,
                    "retained": o.retained,
                }
                for o in self._outcomes
            ],
            "version": VERSION,
        }
        with open(path, "w") as f:
            json.dump(state, f, indent=2)

    def load_state(self, path: str) -> None:
        """Load credit tracker state."""
        import json
        with open(path) as f:
            state = json.load(f)
        self.gamma = state["gamma"]
        self.horizons = state["horizons"]
        self._next_id = state["next_id"]
        # Reconstruct receipts and outcomes
        self._receipts.clear()
        self._outcomes.clear()
        for r in state.get("receipts", []):
            receipt = MutationReceipt(
                receipt_id=r["receipt_id"],
                action=StructuralAction(r["action"]),
                step=r["step"],
                predicted_delta_u=r["predicted_delta_u"],
                predicted_uncertainty=r["predicted_uncertainty"],
                governance_decision=r["governance_decision"],
                governance_reasons=r["governance_reasons"],
                graph_hash_before=r["graph_hash_before"],
                graph_hash_after=r["graph_hash_after"],
                config_governance_hash=r["config_governance_hash"],
                version=r.get("version", VERSION),
            )
            self._receipts.append(receipt)
        for o in state.get("outcomes", []):
            outcome = MutationOutcome(
                receipt_id=o["receipt_id"],
                action=StructuralAction(o["action"]),
                utility_at_16=o.get("utility_at_16"),
                utility_at_100=o.get("utility_at_100"),
                utility_at_1000=o.get("utility_at_1000"),
                discounted_return=o["discounted_return"],
                prediction_error=o["prediction_error"],
                retained=o["retained"],
            )
            self._outcomes.append(outcome)
