"""v5.1 Dynamic gauge connections.

Context-conditioned SO(d) transport where the connection matrix depends
on the latent states of the endpoints and a context vector:

    A_ij = skew(f_θ(z_i, z_j, c_t))
    U_ij = exp(A_ij)  ∈ SO(d)

This makes the gauge connection dynamic — it changes with local
spatiotemporal conditions rather than being a persistent edge parameter.

The module provides:
- DynamicGaugeNetwork: f_θ that maps (z_i, z_j, c_t) → skew matrix
- DynamicGaugeBank: Drop-in replacement for SOConnectionBank that
  computes context-conditioned connections
- StaticGaugeAdapter: Wraps a static SOConnectionBank for backward
  compatibility when dynamic gauge is not needed
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F

from .fibers import skew_symmetric, cayley_so, project_to_so_d


class DynamicGaugeNetwork(nn.Module):
    """Neural network that generates context-conditioned skew-symmetric matrices.

    f_θ: (z_i, z_j, c_t) → A_ij ∈ so(d)

    where so(d) is the space of skew-symmetric d×d matrices.

    The network takes the concatenation [z_i, z_j, c_t] and outputs
    the upper-triangular elements of a skew-symmetric matrix, which
    are then assembled into the full skew matrix.
    """

    def __init__(
        self,
        latent_dim: int,
        context_dim: int = 0,
        hidden_dim: int = 64,
        num_layers: int = 2,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.context_dim = context_dim
        self.hidden_dim = hidden_dim

        # Input: [z_i, z_j, c_t] → [2*latent_dim + context_dim]
        input_dim = 2 * latent_dim + context_dim

        # Number of upper-triangular elements in d×d skew matrix
        # = d*(d-1)/2
        self.skew_dim = latent_dim * (latent_dim - 1) // 2

        # Build MLP
        layers: list[nn.Module] = []
        d = input_dim
        for _ in range(num_layers):
            layers.append(nn.Linear(d, hidden_dim))
            layers.append(nn.ReLU())
            d = hidden_dim
        layers.append(nn.Linear(d, self.skew_dim))
        self.mlp = nn.Sequential(*layers)

    def forward(
        self,
        z_i: Tensor,    # [*, latent_dim]
        z_j: Tensor,    # [*, latent_dim]
        context: Tensor | None = None,  # [*, context_dim]
    ) -> Tensor:
        """Generate skew-symmetric matrices from endpoint latents and context.

        Args:
            z_i: Source node latent state
            z_j: Target node latent state
            context: Optional context vector (e.g., task embedding, time encoding)

        Returns:
            Skew-symmetric matrices of shape [*, latent_dim, latent_dim]
        """
        # Concatenate inputs
        if context is not None:
            x = torch.cat([z_i, z_j, context], dim=-1)
        else:
            # If no context, pad with zeros
            ctx_pad = torch.zeros(
                *z_i.shape[:-1], self.context_dim,
                dtype=z_i.dtype, device=z_i.device,
            )
            x = torch.cat([z_i, z_j, ctx_pad], dim=-1)

        # Generate upper-triangular elements
        skew_elements = self.mlp(x)  # [*, skew_dim]

        # Assemble into skew-symmetric matrix
        return self._assemble_skew(skew_elements)

    def _assemble_skew(self, elements: Tensor) -> Tensor:
        """Assemble upper-triangular elements into a skew-symmetric matrix.

        For a d×d skew matrix, the upper triangle has d*(d-1)/2 elements.
        We fill the upper triangle, then set the lower triangle as the
        negative transpose.
        """
        d = self.latent_dim
        batch_shape = elements.shape[:-1]
        mat = torch.zeros(*batch_shape, d, d, dtype=elements.dtype, device=elements.device)

        # Fill upper triangle
        idx = 0
        for i in range(d):
            for j in range(i + 1, d):
                mat[..., i, j] = elements[..., idx]
                mat[..., j, i] = -elements[..., idx]
                idx += 1

        return mat


class DynamicGaugeBank(nn.Module):
    """Context-conditioned SO(d) connection bank.

    Unlike SOConnectionBank which stores persistent per-edge parameters,
    DynamicGaugeBank computes connection matrices on-the-fly from the
    endpoint latent states and an optional context vector:

        A_ij = f_θ(z_i, z_j, c_t)  ∈ so(d)
        U_ij = exp(A_ij)  ∈ SO(d)

    This allows the gauge connection to adapt to changing latent states
    and task context, enabling richer message passing than static edges.
    """

    def __init__(
        self,
        edge_capacity: int,
        dim: int,
        context_dim: int = 0,
        hidden_dim: int = 64,
        num_layers: int = 2,
        parameterization: str = "cayley",
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        self.edge_capacity = int(edge_capacity)
        self.dim = int(dim)
        self.context_dim = int(context_dim)
        self.parameterization = parameterization

        self.net = DynamicGaugeNetwork(
            latent_dim=dim,
            context_dim=context_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
        )

        # Initialize network weights to near-zero so initial connections ≈ I
        for p in self.net.parameters():
            nn.init.normal_(p, mean=0.0, std=0.01)

    def matrices(
        self,
        z: Tensor,                    # [N, dim]
        src: Tensor,                  # [E]
        dst: Tensor,                  # [E]
        context: Tensor | None = None,  # [context_dim] or [batch, context_dim]
    ) -> Tensor:
        """Compute connection matrices for all edges.

        Args:
            z: Node latent states [N, dim]
            src: Source node indices [E]
            dst: Destination node indices [E]
            context: Optional context vector

        Returns:
            Connection matrices [E, dim, dim] in SO(d)
        """
        z_i = z[src]  # [E, dim]
        z_j = z[dst]  # [E, dim]

        # Expand context if needed
        if context is not None and context.ndim == 1:
            context = context.unsqueeze(0).expand(src.shape[0], -1)

        # Generate skew matrices
        A = self.net(z_i, z_j, context)  # [E, dim, dim]

        # Map to SO(d)
        if self.parameterization == "exp":
            return torch.matrix_exp(A)
        return cayley_so(A)

    def forward(
        self,
        z: Tensor,
        src: Tensor,
        dst: Tensor,
        context: Tensor | None = None,
    ) -> Tensor:
        """Forward pass: compute connection matrices."""
        return self.matrices(z, src, dst, context)

    @torch.no_grad()
    def reset_slots(
        self,
        slot_ids: Tensor | list[int] | tuple[int, ...],
        optimizers: Any = None,
    ) -> None:
        """Reset is a no-op for dynamic gauge (no per-slot parameters).

        The network weights are shared across all edges, so there are no
        per-slot parameters to reset. This method exists for API
        compatibility with SOConnectionBank.
        """
        pass


class StaticGaugeAdapter(nn.Module):
    """Adapter to use a static SOConnectionBank where a DynamicGaugeBank is expected.

    This provides the same interface as DynamicGaugeBank but delegates
    to a pre-existing SOConnectionBank, ignoring the latent states and
    context. Useful for backward compatibility and comparison.
    """

    def __init__(self, bank: Any):
        super().__init__()
        self.bank = bank
        self.dim = bank.dim
        self.edge_capacity = bank.edge_capacity

    def matrices(
        self,
        z: Tensor | None = None,
        src: Tensor | None = None,
        dst: Tensor | None = None,
        context: Tensor | None = None,
    ) -> Tensor:
        """Return static connection matrices (ignoring z and context)."""
        return self.bank.matrices()

    def forward(
        self,
        z: Tensor | None = None,
        src: Tensor | None = None,
        dst: Tensor | None = None,
        context: Tensor | None = None,
    ) -> Tensor:
        return self.matrices(z, src, dst, context)


def gauge_transport(
    z: Tensor,           # [N, dim]
    src: Tensor,         # [E]
    dst: Tensor,         # [E]
    U: Tensor,           # [E, dim, dim]
    direction: str = "src_to_dst",
) -> Tensor:
    """Transport latent vectors across edges using gauge connections.

    For each edge (i, j) with connection U_ij:
        src_to_dst: z_i transported to j's frame = U_ij @ z_i
        dst_to_src: z_j transported to i's frame = U_ij^T @ z_j

    Args:
        z: Node latent states [N, dim]
        src: Source node indices [E]
        dst: Destination node indices [E]
        U: Connection matrices [E, dim, dim]
        direction: "src_to_dst" or "dst_to_src"

    Returns:
        Transported vectors [E, dim]
    """
    if direction == "src_to_dst":
        z_src = z[src]  # [E, dim]
        return torch.bmm(U, z_src.unsqueeze(-1)).squeeze(-1)
    elif direction == "dst_to_src":
        z_dst = z[dst]  # [E, dim]
        # Use transpose for inverse transport
        U_inv = U.transpose(-1, -2)
        return torch.bmm(U_inv, z_dst.unsqueeze(-1)).squeeze(-1)
    else:
        raise ValueError(f"Unknown direction: {direction}")


def gauge_alignment_loss(
    z: Tensor,           # [N, dim]
    src: Tensor,         # [E]
    dst: Tensor,         # [E]
    U: Tensor,           # [E, dim, dim]
) -> Tensor:
    """Compute alignment loss: transported source should match destination.

    L = Σ_e ||U_e @ z_{src(e)} - z_{dst(e)}||^2

    This loss encourages the gauge connections to align the coordinate
    frames of connected nodes.
    """
    transported = gauge_transport(z, src, dst, U, direction="src_to_dst")
    z_dst = z[dst]
    return (transported - z_dst).pow(2).sum(dim=-1).mean()
