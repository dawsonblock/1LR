from __future__ import annotations

from typing import Dict
import torch
from torch import Tensor


def gamma_vector(z: Tensor, p: Tensor) -> Tensor:
    """Dense reference Γ_Z(i)=1/2 Σ_j P_ij ||z_j-z_i||²."""
    d2 = torch.cdist(z, z, p=2).square()
    return 0.5 * (p * d2).sum(dim=-1)


def diffusion_radius(z: Tensor, p: Tensor, eps: float = 1e-12) -> Tensor:
    dist = torch.cdist(z, z, p=2).clamp_min(eps)
    return (p * dist).sum(dim=-1)


def local_variance(z: Tensor, p: Tensor) -> Tensor:
    mean = p @ z
    second = p @ z.square()
    return (second - mean.square()).clamp_min(0.0).sum(dim=-1)


def edge_diffusion_metrics(
    z: Tensor,
    src: Tensor,
    dst: Tensor,
    weight: Tensor,
    num_nodes: int,
    eps: float = 1e-8,
    *,
    normalize_mass: bool = False,
    validate_weights: bool = True,
) -> Dict[str, Tensor]:
    """Sparse local Γ/radius/variance from a directed edge list.

    ``weight`` should normally be row-stochastic Markov mass. When ``normalize_mass``
    is true, raw nonnegative weights are normalized per source first. Zero-mass nodes
    contribute zeros and should normally be represented by an explicit self-loop.
    """
    if src.ndim != 1 or dst.ndim != 1 or weight.ndim != 1:
        raise ValueError("src, dst, weight must be 1-D")
    if not (src.numel() == dst.numel() == weight.numel()):
        raise ValueError("edge tensors must have equal length")
    if validate_weights:
        if bool((weight < 0).any().item()) or bool((~torch.isfinite(weight)).any().item()):
            raise ValueError("edge weights must be finite and nonnegative")

    mass = torch.zeros(num_nodes, device=z.device, dtype=z.dtype)
    mass.index_add_(0, src, weight.to(z.dtype))
    w = weight.to(z.dtype)
    if normalize_mass:
        w = w / mass[src].clamp_min(eps)
        mass_eff = torch.zeros_like(mass)
        mass_eff.index_add_(0, src, w)
    else:
        mass_eff = mass

    delta = z[dst] - z[src]
    dist_sq = delta.square().sum(dim=-1)
    dist = dist_sq.clamp_min(eps).sqrt()

    gamma = torch.zeros(num_nodes, device=z.device, dtype=z.dtype)
    radius = torch.zeros_like(gamma)
    gamma.index_add_(0, src, 0.5 * w * dist_sq)
    radius.index_add_(0, src, w * dist)

    local_mean = torch.zeros_like(z)
    local_mean.index_add_(0, src, w.unsqueeze(-1) * z[dst])
    denom = mass_eff.clamp_min(eps).unsqueeze(-1)
    if not normalize_mass:
        # Markov inputs have mass ~= 1. Divide only to make numerical drift harmless.
        local_mean = local_mean / denom
        gamma = gamma / mass_eff.clamp_min(eps)
        radius = radius / mass_eff.clamp_min(eps)

    centered = z[dst] - local_mean[src]
    local_var = torch.zeros_like(gamma)
    local_var.index_add_(0, src, w * centered.square().sum(dim=-1))
    if not normalize_mass:
        local_var = local_var / mass_eff.clamp_min(eps)

    return {"gamma": gamma, "radius": radius, "local_var": local_var, "mass": mass_eff}


def laplacian_energy_edges(z: Tensor, src: Tensor, dst: Tensor, weight: Tensor) -> Tensor:
    delta = z[dst] - z[src]
    return 0.5 * (weight.to(z.dtype) * delta.square().sum(dim=-1)).sum()


def transport_pressure(gamma: Tensor, baseline: Tensor | None = None, eps: float = 1e-8) -> Tensor:
    if baseline is None:
        med = gamma.median()
        baseline = torch.full_like(gamma, med)
    return gamma / baseline.clamp_min(eps)


def robust_normalize(x: Tensor, eps: float = 1e-6) -> Tensor:
    med = x.detach().median()
    mad = (x.detach() - med).abs().median().clamp_min(eps)
    return (x - med) / mad


def bottleneck_from_curvature(curvature: Tensor) -> Tensor:
    """Raw curvature -> nonnegative bottleneck pressure."""
    return torch.relu(-curvature)


def spawn_score_from_pressure(
    gamma: Tensor,
    radius: Tensor,
    local_var: Tensor,
    bottleneck_pressure: Tensor,
    residual: Tensor,
    uncertainty: Tensor,
    capacity: Tensor,
) -> Tensor:
    """Spawn score where ``bottleneck_pressure`` is already higher-is-worse.

    This avoids the former double-negation bug that erased the Forman/WAF3 signal.
    """
    q = robust_normalize(gamma)
    r = robust_normalize(radius)
    v = robust_normalize(local_var)
    f = robust_normalize(bottleneck_pressure)
    e = robust_normalize(residual)
    u = robust_normalize(uncertainty)
    c = robust_normalize(capacity.float())
    return q + 0.8 * f + 0.7 * r + 0.3 * v + 0.6 * e + 0.25 * u - 0.5 * c


def spawn_score(
    gamma: Tensor,
    radius: Tensor,
    local_var: Tensor,
    curvature: Tensor,
    residual: Tensor,
    uncertainty: Tensor,
    capacity: Tensor,
) -> Tensor:
    """Compatibility wrapper accepting raw curvature rather than pressure."""
    return spawn_score_from_pressure(
        gamma, radius, local_var, bottleneck_from_curvature(curvature), residual, uncertainty, capacity
    )


def edge_diffusion_metrics_gauge(
    z: Tensor,
    src: Tensor,
    dst: Tensor,
    weight: Tensor,
    connection: Tensor,
    gauge_dim: int,
    num_nodes: int,
    eps: float = 1e-8,
    *,
    normalize_mass: bool = False,
    validate_weights: bool = True,
) -> Dict[str, Tensor]:
    """Gauge-covariant sparse Γ/radius/variance on directed edges.

    Neighbor features are parallel transported into the source frame before local
    differences and moments are formed. Channels beyond ``gauge_dim`` use ordinary
    scalar transport. The operation is differentiable with respect to the SO(d)
    connection matrices.
    """
    if gauge_dim <= 0 or gauge_dim > z.shape[-1]:
        raise ValueError("invalid gauge_dim")
    if connection.shape != (src.numel(), gauge_dim, gauge_dim):
        raise ValueError("connection shape must be [E,gauge_dim,gauge_dim]")
    if validate_weights:
        if bool((weight < 0).any().item()) or bool((~torch.isfinite(weight)).any().item()):
            raise ValueError("edge weights must be finite and nonnegative")

    w = weight.to(z.dtype)
    mass = torch.zeros(num_nodes, device=z.device, dtype=z.dtype)
    mass.index_add_(0, src, w)
    if normalize_mass:
        w = w / mass[src].clamp_min(eps)
        mass_eff = torch.zeros_like(mass)
        mass_eff.index_add_(0, src, w)
    else:
        mass_eff = mass

    prefix = torch.einsum("eij,ej->ei", connection.to(z.dtype), z[dst, :gauge_dim])
    transported = prefix if gauge_dim == z.shape[-1] else torch.cat((prefix, z[dst, gauge_dim:]), dim=-1)
    delta = transported - z[src]
    dist_sq = delta.square().sum(dim=-1)
    dist = dist_sq.clamp_min(eps).sqrt()

    gamma = torch.zeros(num_nodes, device=z.device, dtype=z.dtype)
    radius = torch.zeros_like(gamma)
    gamma.index_add_(0, src, 0.5 * w * dist_sq)
    radius.index_add_(0, src, w * dist)

    local_mean = torch.zeros_like(z)
    local_mean.index_add_(0, src, w.unsqueeze(-1) * transported)
    denom = mass_eff.clamp_min(eps).unsqueeze(-1)
    if not normalize_mass:
        local_mean = local_mean / denom
        gamma = gamma / mass_eff.clamp_min(eps)
        radius = radius / mass_eff.clamp_min(eps)

    centered = transported - local_mean[src]
    local_var = torch.zeros_like(gamma)
    local_var.index_add_(0, src, w * centered.square().sum(dim=-1))
    if not normalize_mass:
        local_var = local_var / mass_eff.clamp_min(eps)
    return {"gamma": gamma, "radius": radius, "local_var": local_var, "mass": mass_eff}
