"""v5.1 Sheaf-adjacency diffusion.

Replaces pure sheaf Laplacian propagation with a sheaf-adjacency
formulation plus normalization and gating:

    Pure Laplacian:    Z_{t+1} = Z_t - η L_F Z_t
    Sheaf-adjacency:   Z_{t+1} = Z_t + η A_F Z_t  (with normalization/gating)

The sheaf-adjacency form avoids the problem that repeated diffusion
can suppress useful disagreement signals at depth. Instead of
smoothing toward the mean, it allows structured amplification of
agreement while gating disagreement.

Both formulations are provided so the choice can be empirical.
"""
from __future__ import annotations

from typing import Any

import torch
from torch import Tensor
import torch.nn.functional as F


def sheaf_laplacian_diffusion(
    z: Tensor,              # [N, D]
    src: Tensor,            # [E]
    dst: Tensor,            # [E]
    U: Tensor,              # [E, D, D] connection matrices
    weight: Tensor,         # [E]
    num_steps: int = 1,
    eta: float = 0.1,
) -> Tensor:
    """Pure sheaf Laplacian diffusion.

    Z_{t+1} = Z_t - η L_F Z_t

    where L_F is the sheaf Laplacian:
        L_F = D - A_F
        (A_F)_{ij} = w_{ij} U_{ij}  (weighted connection)

    This is the standard formulation. Repeated application smooths
    the signal, which can suppress useful disagreement at depth.
    """
    N, D = z.shape
    z_out = z.clone()

    for _ in range(num_steps):
        # Transport: z_i → U_ij @ z_i for each edge
        z_src = z_out[src]  # [E, D]
        transported = torch.bmm(U, z_src.unsqueeze(-1)).squeeze(-1)  # [E, D]

        # Weighted aggregation to destination
        diff = weight.unsqueeze(-1) * (transported - z_out[dst])  # [E, D]
        delta = torch.zeros_like(z_out)
        delta.index_add_(0, dst, diff)

        z_out = z_out - eta * delta

    return z_out


def sheaf_adjacency_diffusion(
    z: Tensor,              # [N, D]
    src: Tensor,            # [E]
    dst: Tensor,            # [E]
    U: Tensor,              # [E, D, D] connection matrices
    weight: Tensor,         # [E]
    num_steps: int = 1,
    eta: float = 0.1,
    gate: Tensor | None = None,  # [E] optional gating
    normalize: bool = True,
) -> Tensor:
    """Sheaf-adjacency diffusion with normalization and gating.

    Z_{t+1} = Z_t + η A_F Z_t  (with normalization/gating)

    where A_F is the sheaf adjacency:
        (A_F Z)_j = Σ_{i→j} w_{ij} g_{ij} U_{ij} z_i

    with optional gating g_{ij} and row normalization.

    Unlike the Laplacian form, this does not subtract the self-term,
    allowing structured amplification rather than pure smoothing.
    """
    N, D = z.shape
    z_out = z.clone()

    for _ in range(num_steps):
        # Transport: z_i → U_ij @ z_i
        z_src = z_out[src]  # [E, D]
        transported = torch.bmm(U, z_src.unsqueeze(-1)).squeeze(-1)  # [E, D]

        # Apply weights
        w = weight
        if gate is not None:
            w = w * gate  # Element-wise gating

        msg = w.unsqueeze(-1) * transported  # [E, D]

        # Aggregate
        agg = torch.zeros_like(z_out)
        agg.index_add_(0, dst, msg)

        if normalize:
            # Row-normalize by in-degree weight sum
            deg = torch.zeros(N, dtype=w.dtype, device=w.device)
            deg.index_add_(0, dst, w)
            agg = agg / deg.unsqueeze(-1).clamp_min(1e-8)

        # Update with residual
        z_out = z_out + eta * (agg - z_out)

    return z_out


def gated_sheaf_diffusion(
    z: Tensor,
    src: Tensor,
    dst: Tensor,
    U: Tensor,
    weight: Tensor,
    gate_fn: Any,          # Callable: (z_i, z_j) → gate ∈ [0, 1]
    num_steps: int = 1,
    eta: float = 0.1,
    normalize: bool = True,
) -> Tensor:
    """Sheaf-adjacency diffusion with learned gating.

    The gate function determines how much of each edge's transported
    signal is allowed through. This enables selective message passing:
    edges with high agreement pass more signal, edges with disagreement
    are gated down.
    """
    N, D = z.shape
    z_out = z.clone()

    for _ in range(num_steps):
        z_src = z_out[src]
        z_dst = z_out[dst]

        # Compute gates
        gate = gate_fn(z_src, z_dst)  # [E]

        z_out = sheaf_adjacency_diffusion(
            z_out, src, dst, U, weight,
            num_steps=1, eta=eta, gate=gate, normalize=normalize,
        )

    return z_out


def agreement_gate(z_i: Tensor, z_j: Tensor, U: Tensor | None = None) -> Tensor:
    """Default gate function: agreement between transported representations.

    gate = sigmoid(cosine_similarity(U @ z_i, z_j))

    High agreement → gate opens (signal passes).
    Low agreement → gate closes (signal blocked).
    """
    if U is not None:
        z_i_transformed = torch.bmm(U, z_i.unsqueeze(-1)).squeeze(-1)
    else:
        z_i_transformed = z_i

    cos_sim = F.cosine_similarity(z_i_transformed, z_j, dim=-1)
    return torch.sigmoid(cos_sim)


def compare_diffusion_methods(
    z: Tensor,
    src: Tensor,
    dst: Tensor,
    U: Tensor,
    weight: Tensor,
    num_steps: int = 5,
    eta: float = 0.1,
) -> dict[str, Tensor]:
    """Compare Laplacian vs sheaf-adjacency diffusion.

    Returns both results so the choice can be empirical.
    """
    z_lap = sheaf_laplacian_diffusion(z, src, dst, U, weight, num_steps, eta)
    z_adj = sheaf_adjacency_diffusion(z, src, dst, U, weight, num_steps, eta)

    # Measure signal preservation (variance ratio)
    var_input = z.var(dim=0).mean().item()
    var_lap = z_lap.var(dim=0).mean().item()
    var_adj = z_adj.var(dim=0).mean().item()

    return {
        "laplacian": z_lap,
        "adjacency": z_adj,
        "variance_ratio_laplacian": var_lap / max(var_input, 1e-10),
        "variance_ratio_adjacency": var_adj / max(var_input, 1e-10),
        "input_variance": var_input,
        "laplacian_variance": var_lap,
        "adjacency_variance": var_adj,
    }
