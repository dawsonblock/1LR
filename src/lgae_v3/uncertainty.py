"""v5.0 Calibrated uncertainty for structural mutations.

Provides epistemic uncertainty estimates around mutation outcomes:
    p(ΔU | m, S)

Uses ensemble-based uncertainty estimation and conformal calibration
to produce statistically calibrated prediction intervals.

LCB acceptance gate:
    LCB(m) = E[ΔU_m] - β·σ_m
    Only positive LCB → automatic accept
    Uncertain but interesting → QUARANTINE
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
from torch import Tensor
import torch.nn as nn
import numpy as np

from .executive import StructuralExecutive, ActionProposal, StructuralAction


@dataclass
class UncertaintyEstimate:
    """Calibrated uncertainty estimate for a mutation outcome."""
    mean: float          # E[ΔU]
    std: float           # σ(ΔU)
    lcb: float           # Lower confidence bound
    ucb: float           # Upper confidence bound
    calibration_error: float = 0.0  # Conformal calibration error
    method: str = "ensemble"
    metadata: dict[str, Any] = field(default_factory=dict)


class EnsembleUncertainty:
    """Ensemble-based epistemic uncertainty estimation.

    Maintains K independently initialized executive networks.
    The disagreement across ensemble members provides epistemic uncertainty.

    σ_epistemic = std across ensemble members' predictions
    """

    def __init__(
        self,
        executive: StructuralExecutive,
        ensemble_size: int = 5,
        beta: float = 1.0,
    ):
        self.executive = executive
        self.ensemble_size = ensemble_size
        self.beta = beta

        # Create ensemble by perturbing the base network
        self.ensemble: list[dict] = []
        base_state = executive.network.state_dict()
        for i in range(ensemble_size):
            # Create a perturbed copy
            perturbed = {k: v + 0.1 * torch.randn_like(v) for k, v in base_state.items()}
            self.ensemble.append(perturbed)

    def estimate(
        self, observation_vec: Tensor, action_idx: int,
    ) -> UncertaintyEstimate:
        """Estimate uncertainty for a specific action.

        Args:
            observation_vec: Feature vector from StructuralObservation.to_vector()
            action_idx: Index of the action to evaluate

        Returns:
            UncertaintyEstimate with mean, std, LCB, UCB
        """
        # Save original network state
        original_state = self.executive.network.state_dict()

        # Collect predictions from each ensemble member
        predictions: list[float] = []
        for member_state in self.ensemble:
            self.executive.network.load_state_dict(member_state)
            self.executive.network.eval()
            with torch.no_grad():
                preds = self.executive.network(observation_vec)
                predictions.append(float(preds["delta_u"][action_idx].item()))

        # Restore original state
        self.executive.network.load_state_dict(original_state)

        mean = float(np.mean(predictions))
        std = float(np.std(predictions))
        lcb = mean - self.beta * std
        ucb = mean + self.beta * std

        return UncertaintyEstimate(
            mean=mean, std=std, lcb=lcb, ucb=ucb,
            method="ensemble",
            metadata={"ensemble_size": self.ensemble_size, "predictions": predictions},
        )


class ConformalCalibrator:
    """Conformal prediction-based calibration for mutation outcomes.

    Uses split conformal prediction to produce calibrated prediction intervals
    with coverage guarantees.
    """

    def __init__(self, alpha: float = 0.1):
        """Initialize conformal calibrator.

        Args:
            alpha: Miscoverage rate (1 - alpha = coverage probability).
                   E.g., alpha=0.1 → 90% coverage intervals.
        """
        self.alpha = alpha
        self._residuals: list[float] = []
        self._quantile: float | None = None

    def calibrate(self, predicted: list[float], actual: list[float]) -> float:
        """Calibrate using a held-out calibration set.

        Args:
            predicted: Predicted ΔU values
            actual: Actual ΔU values

        Returns:
            The conformal quantile (half-width of the prediction interval)
        """
        if len(predicted) != len(actual):
            raise ValueError(
                f"predicted and actual must have same length: {len(predicted)} vs {len(actual)}"
            )
        residuals = [abs(p - a) for p, a in zip(predicted, actual)]
        self._residuals = residuals
        if len(residuals) == 0:
            self._quantile = 0.0
            return 0.0
        # Conformal quantile: ceil((n+1)*(1-alpha)/n)th order statistic
        n = len(residuals)
        idx = int(np.ceil((n + 1) * (1 - self.alpha) / n)) - 1
        idx = max(0, min(idx, n - 1))
        sorted_residuals = sorted(residuals)
        self._quantile = sorted_residuals[idx]
        return self._quantile

    def interval(self, prediction: float) -> tuple[float, float]:
        """Produce a calibrated prediction interval.

        Args:
            prediction: Point prediction (E[ΔU])

        Returns:
            (lower, upper) bounds of the conformal prediction interval
        """
        if self._quantile is None:
            return (prediction, prediction)
        return (prediction - self._quantile, prediction + self._quantile)

    def lcb(self, prediction: float, beta: float = 1.0) -> float:
        """Lower confidence bound using conformal calibration."""
        lower, _ = self.interval(prediction)
        return lower


def uncertainty_gated_decision(
    proposal: ActionProposal,
    uncertainty: UncertaintyEstimate,
    lcb_threshold: float = 0.0,
    quarantine_uncertainty: float = 0.5,
) -> str:
    """Make an uncertainty-gated decision on a proposal.

    Decision logic:
    - LCB > threshold AND σ < quarantine_uncertainty → "accept"
    - LCB > threshold AND σ >= quarantine_uncertainty → "quarantine"
    - LCB <= threshold AND UCB > threshold → "quarantine" (uncertain but interesting)
    - LCB <= threshold AND UCB <= threshold → "reject"

    Returns:
        "accept", "quarantine", or "reject"
    """
    lcb = uncertainty.lcb
    ucb = uncertainty.ucb
    sigma = uncertainty.std

    if lcb > lcb_threshold:
        if sigma < quarantine_uncertainty:
            return "accept"
        else:
            return "quarantine"
    elif ucb > lcb_threshold:
        # Uncertain but potentially interesting
        return "quarantine"
    else:
        return "reject"
