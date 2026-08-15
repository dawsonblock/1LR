"""v5.0 Learned structural executive.

A proposal model that observes local geometry, task residuals, uncertainty,
capacity, edge role, and recent mutation history, then scores structural
actions. This is a proposal generator only; the governor remains the authority.

Objective:
    m* = argmax_m [E[ΔU(m)] + ν·IG(m) - λ·C(m) - μ·R(m)]

where:
    ΔU(m)  = predicted downstream task improvement
    IG(m)  = expected information gain
    C(m)   = compute/memory cost
    R(m)   = structural risk

The executive does NOT execute mutations. It proposes them. The governor
certifies them. Outcomes train the executive.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F

from .benchmark.tasks import StructuralAction
from .types import GraphBuffers
from .config import LGAEConfig


# Action space
ACTION_LIST: list[StructuralAction] = [
    StructuralAction.NO_OP,
    StructuralAction.ADD_EDGE,
    StructuralAction.PRUNE_EDGE,
    StructuralAction.REWEIGHT_AFFINITY,
    StructuralAction.REWEIGHT_LENGTH,
    StructuralAction.SPAWN_FIBER,
    StructuralAction.PRUNE_FIBER,
    StructuralAction.CHANGE_GAUGE,
    StructuralAction.COUPLED_REWEIGHT,
]
ACTION_TO_IDX: dict[StructuralAction, int] = {a: i for i, a in enumerate(ACTION_LIST)}
NUM_ACTIONS: int = len(ACTION_LIST)


@dataclass
class StructuralObservation:
    """Observation vector for the structural executive.

    Encodes local geometry, task residuals, uncertainty, capacity, edge role,
    and recent mutation history into a fixed-size feature vector.
    """
    # Graph geometry features
    spectral_gap: float = 0.0
    mean_affinity: float = 0.0
    std_affinity: float = 0.0
    mean_length: float = 1.0
    std_length: float = 0.0
    num_edges: float = 0.0
    num_nodes: float = 0.0
    # Curvature features
    mean_gamma: float = 0.0
    max_gamma: float = 0.0
    min_lly: float = 0.0
    mean_lly: float = 0.0
    # Operator features
    operator_discrepancy: float = 0.0
    lambda2: float = 0.0
    # Fiber features
    fiber_capacity: float = 0.0
    fiber_utilization: float = 0.0
    # Task features
    task_loss: float = 0.0
    task_loss_delta: float = 0.0
    # Uncertainty
    epistemic_uncertainty: float = 0.0
    # Recent mutation history (one-hot per action, decaying)
    recent_mutations: list[float] = field(default_factory=lambda: [0.0] * NUM_ACTIONS)

    def to_vector(self) -> Tensor:
        """Convert to fixed-size feature vector [D]."""
        base = [
            self.spectral_gap, self.mean_affinity, self.std_affinity,
            self.mean_length, self.std_length, self.num_edges, self.num_nodes,
            self.mean_gamma, self.max_gamma, self.min_lly, self.mean_lly,
            self.operator_discrepancy, self.lambda2,
            self.fiber_capacity, self.fiber_utilization,
            self.task_loss, self.task_loss_delta,
            self.epistemic_uncertainty,
        ]
        return torch.tensor(base + self.recent_mutations, dtype=torch.float32)


@dataclass
class ActionProposal:
    """A single action proposal from the executive."""
    action: StructuralAction
    expected_delta_utility: float
    information_gain: float
    cost: float
    risk: float
    score: float  # Combined objective value
    uncertainty: float = 0.0  # Epistemic uncertainty on ΔU
    lcb: float = 0.0  # Lower confidence bound
    metadata: dict[str, Any] = field(default_factory=dict)


class ExecutiveNetwork(nn.Module):
    """Neural network for the structural executive.

    Input:  StructuralObservation feature vector
    Output: Per-action scores (expected ΔU, IG, cost, risk)

    The network is intentionally small to avoid overfitting on the
    limited structural history.
    """

    def __init__(self, obs_dim: int, hidden_dim: int = 64, num_actions: int = NUM_ACTIONS):
        super().__init__()
        self.num_actions = num_actions
        # Shared encoder
        self.encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        # Heads
        self.delta_u_head = nn.Linear(hidden_dim, num_actions)  # Expected ΔU
        self.ig_head = nn.Linear(hidden_dim, num_actions)       # Information gain
        self.cost_head = nn.Linear(hidden_dim, num_actions)     # Compute cost
        self.risk_head = nn.Linear(hidden_dim, num_actions)     # Structural risk
        self.uncertainty_head = nn.Linear(hidden_dim, num_actions)  # Epistemic uncertainty

    def forward(self, obs: Tensor) -> dict[str, Tensor]:
        """Forward pass returning per-action predictions.

        Returns:
            dict with keys: delta_u, ig, cost, risk, uncertainty
            Each is [num_actions] for a single observation.
        """
        h = self.encoder(obs)
        return {
            "delta_u": self.delta_u_head(h),
            "ig": F.softplus(self.ig_head(h)),  # IG ≥ 0
            "cost": F.softplus(self.cost_head(h)),  # Cost ≥ 0
            "risk": F.softplus(self.risk_head(h)),  # Risk ≥ 0
            "uncertainty": F.softplus(self.uncertainty_head(h)),  # σ ≥ 0
        }


class StructuralExecutive:
    """Learned structural executive: proposes mutations.

    The executive observes the current state and proposes the best structural
    action. It does NOT execute mutations — the governor remains the authority.

    The objective is:
        m* = argmax_m [E[ΔU(m)] + ν·IG(m) - λ·C(m) - μ·R(m)]

    With uncertainty-aware acceptance via LCB:
        LCB(m) = E[ΔU_m] - β·σ_m
        Only positive LCB → automatic proposal
        Uncertain but interesting → quarantine proposal
    """

    def __init__(
        self,
        config: LGAEConfig | None = None,
        hidden_dim: int = 64,
        # Objective weights
        nu: float = 0.1,    # Information gain weight
        lam: float = 0.05,  # Cost weight
        mu: float = 0.1,    # Risk weight
        # Uncertainty
        beta: float = 1.0,  # LCB confidence parameter
        lcb_threshold: float = 0.0,  # Minimum LCB for auto-propose
        quarantine_uncertainty: float = 0.5,  # σ above which → quarantine
        # Learning
        lr: float = 1e-3,
    ):
        self.config = config or LGAEConfig()
        self.nu = nu
        self.lam = lam
        self.mu = mu
        self.beta = beta
        self.lcb_threshold = lcb_threshold
        self.quarantine_uncertainty = quarantine_uncertainty

        # Build observation dimension
        self._obs_dim = self._compute_obs_dim()
        self.network = ExecutiveNetwork(self._obs_dim, hidden_dim=hidden_dim)
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=lr)

        # Recent mutation history (exponential decay)
        self._mutation_history = torch.zeros(NUM_ACTIONS)

        # Training data buffer
        self._experience: list[dict] = []

    def _compute_obs_dim(self) -> int:
        """Compute observation vector dimension."""
        obs = StructuralObservation()
        return obs.to_vector().shape[0]

    def observe(
        self,
        graph: GraphBuffers,
        z: Tensor,
        audit_snapshot: Any | None = None,
        task_loss: float = 0.0,
        task_loss_delta: float = 0.0,
        epistemic_uncertainty: float = 0.0,
    ) -> StructuralObservation:
        """Construct a structural observation from the current state."""
        obs = StructuralObservation()

        # Graph features
        obs.num_nodes = float(graph.num_nodes)
        valid_mask = graph.valid.bool()
        obs.num_edges = float(valid_mask.sum().item())
        if obs.num_edges > 0:
            w = graph.weight[valid_mask]
            obs.mean_affinity = float(w.mean().item())
            obs.std_affinity = float(w.std().item()) if w.numel() > 1 else 0.0
            if graph.length is not None:
                l = graph.length[valid_mask]
                obs.mean_length = float(l.mean().item())
                obs.std_length = float(l.std().item()) if l.numel() > 1 else 0.0

        # Audit features
        if audit_snapshot is not None:
            obs.lambda2 = float(audit_snapshot.lambda2)
            obs.operator_discrepancy = float(
                audit_snapshot.details.get("operator_discrepancy", 0.0)
            )
            gamma = audit_snapshot.details.get("gamma")
            if gamma is not None:
                if isinstance(gamma, Tensor):
                    obs.mean_gamma = float(gamma.mean().item())
                    obs.max_gamma = float(gamma.max().item())
                elif isinstance(gamma, dict):
                    vals = list(gamma.values())
                    if vals:
                        obs.mean_gamma = float(sum(vals) / len(vals))
                        obs.max_gamma = float(max(vals))
            lly = audit_snapshot.details.get("lly")
            if lly and isinstance(lly, dict):
                vals = list(lly.values())
                if vals:
                    obs.min_lly = float(min(vals))
                    obs.mean_lly = float(sum(vals) / len(vals))

        # Fiber features
        obs.fiber_capacity = float(self.config.fiber.d_max)
        obs.fiber_utilization = float(z.shape[1]) if z.ndim == 2 else 0.0

        # Task features
        obs.task_loss = task_loss
        obs.task_loss_delta = task_loss_delta

        # Uncertainty
        obs.epistemic_uncertainty = epistemic_uncertainty

        # Recent mutation history (decaying)
        obs.recent_mutations = self._mutation_history.tolist()

        return obs

    def propose(self, observation: StructuralObservation) -> list[ActionProposal]:
        """Propose all candidate actions with scores.

        Returns a list of ActionProposal objects, sorted by score (descending).
        The best action is proposals[0].
        """
        self.network.eval()
        with torch.no_grad():
            obs_vec = observation.to_vector()
            preds = self.network(obs_vec)

        proposals: list[ActionProposal] = []
        for i, action in enumerate(ACTION_LIST):
            delta_u = float(preds["delta_u"][i].item())
            ig = float(preds["ig"][i].item())
            cost = float(preds["cost"][i].item())
            risk = float(preds["risk"][i].item())
            sigma = float(preds["uncertainty"][i].item())

            # Combined objective: E[ΔU] + ν·IG - λ·C - μ·R
            score = delta_u + self.nu * ig - self.lam * cost - self.mu * risk

            # LCB
            lcb = delta_u - self.beta * sigma

            proposals.append(ActionProposal(
                action=action,
                expected_delta_utility=delta_u,
                information_gain=ig,
                cost=cost,
                risk=risk,
                score=score,
                uncertainty=sigma,
                lcb=lcb,
                metadata={"action_idx": i},
            ))

        # Sort by score descending
        proposals.sort(key=lambda p: p.score, reverse=True)
        return proposals

    def best_proposal(self, observation: StructuralObservation) -> ActionProposal:
        """Return the best action proposal (highest score)."""
        return self.propose(observation)[0]

    def should_quarantine(self, proposal: ActionProposal) -> bool:
        """Decide whether a proposal should be quarantined based on uncertainty."""
        return (
            proposal.uncertainty > self.quarantine_uncertainty
            and proposal.lcb < self.lcb_threshold
        )

    def record_mutation(self, action: StructuralAction) -> None:
        """Record that a mutation was executed (for history tracking)."""
        idx = ACTION_TO_IDX.get(action)
        if idx is not None:
            # One-hot encode and decay old history
            self._mutation_history *= 0.9  # Exponential decay
            self._mutation_history[idx] += 1.0

    def record_outcome(
        self,
        observation: StructuralObservation,
        action: StructuralAction,
        actual_delta_utility: float,
    ) -> None:
        """Record a training example for the executive.

        The executive learns from the difference between its predicted ΔU
        and the actual ΔU.
        """
        self._experience.append({
            "observation": observation.to_vector().detach().clone(),
            "action_idx": ACTION_TO_IDX[action],
            "predicted_delta_u": None,  # Filled during training
            "actual_delta_u": actual_delta_utility,
        })

    def train_step(self, batch_size: int = 32) -> dict[str, float]:
        """Train the executive on recorded experience.

        Returns training metrics.
        """
        if len(self._experience) < batch_size:
            return {"loss": 0.0, "samples": 0}

        self.network.train()

        # Sample a batch
        import random
        batch = random.sample(self._experience, min(batch_size, len(self._experience)))

        total_loss = 0.0
        for exp in batch:
            obs_vec = exp["observation"]
            action_idx = exp["action_idx"]
            actual_du = exp["actual_delta_u"]

            preds = self.network(obs_vec)
            predicted_du = preds["delta_u"][action_idx]

            # MSE loss on ΔU prediction
            loss = F.mse_loss(predicted_du, torch.tensor(actual_du))
            total_loss += loss.item()

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        return {
            "loss": total_loss / len(batch),
            "samples": len(batch),
        }

    def save_state(self, path: str) -> None:
        """Save the executive's network state."""
        torch.save({
            "network_state": self.network.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "mutation_history": self._mutation_history,
            "experience_count": len(self._experience),
        }, path)

    def load_state(self, path: str) -> None:
        """Load the executive's network state."""
        state = torch.load(path, weights_only=False)
        self.network.load_state_dict(state["network_state"])
        self.optimizer.load_state_dict(state["optimizer_state"])
        self._mutation_history = state["mutation_history"]
