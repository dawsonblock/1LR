from __future__ import annotations

import torch
from torch import Tensor


def _gamma(Q: Tensor, f: Tensor, g: Tensor) -> Tensor:
    return 0.5 * (Q @ (f*g) - f*(Q@g) - g*(Q@f))


def _gamma2(Q: Tensor, f: Tensor) -> Tensor:
    gam=_gamma(Q,f,f); qf=Q@f
    return 0.5*(Q@gam - 2.0*_gamma(Q,f,qf))


def sampled_cde_prime_residual(
    Q: Tensor,
    nodes: Tensor | list[int] | None = None,
    *,
    dimension: float = 16.0,
    K: float = 0.0,
    samples: int = 64,
    seed: int = 0,
) -> float:
    """Monte-Carlo violation statistic for CDE′; not an exact certificate.

    Uses modified Γ~2 = Γ2 - Γ(f, Γ(f)/f) and returns the maximum positive
    violation across sampled strictly-positive functions and requested vertices.
    """
    n=Q.shape[0]
    ids=torch.arange(n,device=Q.device) if nodes is None else torch.as_tensor(nodes,device=Q.device,dtype=torch.long)
    gen=torch.Generator(device=Q.device); gen.manual_seed(seed)
    worst=torch.tensor(0.0,dtype=Q.dtype,device=Q.device)
    for _ in range(samples):
        raw=torch.randn(n,generator=gen,device=Q.device,dtype=Q.dtype)
        f=torch.exp(0.5*raw).clamp_min(1e-6)
        gam=_gamma(Q,f,f)
        tilde=_gamma2(Q,f)-_gamma(Q,f,gam/f)
        rhs=(f.square()*(Q@torch.log(f)).square()/float(dimension)) + float(K)*gam
        residual=(rhs-tilde).clamp_min(0.0)
        worst=torch.maximum(worst,residual[ids].max())
    return float(worst.item())
