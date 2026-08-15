from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from .types import GraphBuffers


def row_normalize_dense(a: Tensor, eps: float = 1e-12) -> Tensor:
    if a.ndim != 2 or a.shape[0] != a.shape[1]:
        raise ValueError("expected square matrix")
    denom = a.sum(dim=-1, keepdim=True).clamp_min(eps)
    return a / denom


def graph_buffers_to_dense(graph: GraphBuffers, symmetric: bool = True) -> Tensor:
    graph.validate()
    a = torch.zeros((graph.num_nodes, graph.num_nodes), device=graph.weight.device, dtype=graph.weight.dtype)
    src, dst, w = graph.active()
    if src.numel():
        a.index_put_((src, dst), w, accumulate=True)
        if symmetric:
            a.index_put_((dst, src), w, accumulate=True)
    return a


def actuation_operator(graph: GraphBuffers, symmetric: bool = True, self_loop: float = 0.0) -> Tensor:
    a = graph_buffers_to_dense(graph, symmetric=symmetric)
    if self_loop:
        a = a + torch.eye(graph.num_nodes, device=a.device, dtype=a.dtype) * float(self_loop)
    isolated = a.sum(dim=-1) <= 0
    if isolated.any():
        a = a.clone()
        idx = torch.arange(graph.num_nodes, device=a.device)[isolated]
        a[idx, idx] = 1.0
    return row_normalize_dense(a)


def actuation_markov_edges(
    graph: GraphBuffers,
    *,
    symmetric: bool = True,
    self_loop: float = 0.0,
    eps: float = 1e-12,
) -> tuple[Tensor, Tensor, Tensor]:
    """Return directed row-stochastic actuation edges without a dense adjacency."""
    graph.validate()
    src, dst, w = graph.active()
    if symmetric:
        s = torch.cat([src, dst])
        d = torch.cat([dst, src])
        ww = torch.cat([w, w])
    else:
        s, d, ww = src.clone(), dst.clone(), w.clone()
    if self_loop > 0:
        ids = torch.arange(graph.num_nodes, device=graph.src.device)
        s = torch.cat([s, ids])
        d = torch.cat([d, ids])
        ww = torch.cat([ww, torch.full((graph.num_nodes,), float(self_loop), dtype=w.dtype, device=w.device)])

    mass = torch.zeros(graph.num_nodes, dtype=ww.dtype, device=ww.device)
    if s.numel():
        mass.index_add_(0, s, ww)
    isolated = mass <= 0
    if isolated.any():
        ids = torch.arange(graph.num_nodes, device=graph.src.device)[isolated]
        s = torch.cat([s, ids])
        d = torch.cat([d, ids])
        ww = torch.cat([ww, torch.ones(ids.numel(), dtype=w.dtype, device=w.device)])
        mass = torch.zeros(graph.num_nodes, dtype=ww.dtype, device=ww.device)
        mass.index_add_(0, s, ww)
    pweight = ww / mass[s].clamp_min(eps)
    return s, d, pweight


def sparse_markov_apply(z: Tensor, src: Tensor, dst: Tensor, pweight: Tensor, num_nodes: int) -> Tensor:
    out = torch.zeros((num_nodes, z.shape[-1]), dtype=z.dtype, device=z.device)
    out.index_add_(0, src, pweight.to(z.dtype).unsqueeze(-1) * z[dst])
    return out


def sparse_laplacian_step(
    z: Tensor,
    src: Tensor,
    dst: Tensor,
    pweight: Tensor,
    *,
    eta: float | Tensor,
    num_nodes: int,
) -> Tensor:
    pz = sparse_markov_apply(z, src, dst, pweight, num_nodes)
    return z - eta * (z - pz)


def positive_laplacian_from_markov(p: Tensor) -> Tensor:
    return torch.eye(p.shape[0], device=p.device, dtype=p.dtype) - p


def generator_from_markov(p: Tensor) -> Tensor:
    """Continuous-time generator Δ=P-I, matching Γ-calculus conventions."""
    return p - torch.eye(p.shape[0], device=p.device, dtype=p.dtype)


def pairwise_metric_sq(z: Tensor) -> Tensor:
    if z.ndim != 2:
        raise ValueError("z must have shape [N,D]")
    zz = (z * z).sum(dim=-1, keepdim=True)
    d2 = zz + zz.T - 2.0 * (z @ z.T)
    return d2.clamp_min(0.0)


def diagnostic_diffusion_operator(
    z: Tensor,
    k: int = 16,
    epsilon_floor: float = 1e-4,
    include_self: bool = False,
    *,
    full_kernel_max_nodes: int = 512,
) -> Tensor:
    """Gaussian diffusion operator on the feature cloud.

    For ``N <= full_kernel_max_nodes`` the support is fully soft, avoiding k-NN
    neighbor-order discontinuities. For larger N this reference backend uses top-k
    support as an explicit scalability approximation; production-scale deployments
    should replace the O(N²) distance construction with a stable ANN candidate cache.
    """
    n = z.shape[0]
    if n == 0:
        raise ValueError("empty latent cloud")
    d2 = pairwise_metric_sq(z)
    eye = torch.eye(n, dtype=torch.bool, device=z.device)

    if n <= int(full_kernel_max_nodes):
        scale_source = d2.masked_fill(eye, float("inf"))
        k_eff = min(max(int(k), 1), max(n - 1, 1))
        vals, _ = torch.topk(scale_source, k=k_eff, largest=False, dim=-1)
        local_scale = vals[:, -1].sqrt().clamp_min(epsilon_floor)
        eps_ij = (local_scale[:, None] * local_scale[None, :]).clamp_min(epsilon_floor ** 2)
        kernel = torch.exp(-0.5 * d2 / eps_ij)
        if not include_self:
            kernel = kernel.masked_fill(eye, 0.0)
    else:
        support_d2 = d2.clone()
        if not include_self:
            support_d2 = support_d2.masked_fill(eye, float("inf"))
        k_eff = min(max(int(k), 1), max(n - (0 if include_self else 1), 1))
        vals, idx = torch.topk(support_d2, k=k_eff, largest=False, dim=-1)
        finite_vals = torch.where(torch.isfinite(vals), vals, torch.zeros_like(vals))
        local_scale = finite_vals[:, -1].sqrt().clamp_min(epsilon_floor)
        eps_ij = (local_scale[:, None] * local_scale[idx]).clamp_min(epsilon_floor ** 2)
        kvals = torch.exp(-0.5 * finite_vals / eps_ij)
        kvals = torch.where(torch.isfinite(vals), kvals, torch.zeros_like(kvals))
        kernel = torch.zeros_like(d2)
        kernel.scatter_(1, idx, kvals)
        kernel = 0.5 * (kernel + kernel.T)
        if include_self:
            kernel.fill_diagonal_(1.0)

    isolated = kernel.sum(dim=-1) <= 0
    if isolated.any():
        kernel = kernel.clone()
        ids = torch.arange(n, device=z.device)[isolated]
        kernel[ids, ids] = 1.0
    return row_normalize_dense(kernel)


def operator_discrepancy(p_act: Tensor, p_diag: Tensor, mode: str = "frobenius") -> Tensor:
    if p_act.shape != p_diag.shape:
        raise ValueError("operator shapes differ")
    diff = p_act - p_diag
    if mode == "frobenius":
        return torch.linalg.matrix_norm(diff, ord="fro") / max(p_act.shape[0], 1) ** 0.5
    if mode == "mean_l1":
        return diff.abs().sum(dim=-1).mean()
    raise ValueError(f"unknown discrepancy mode: {mode}")


def spectral_gap_symmetric(p: Tensor) -> Tensor:
    """Return λ2 of a symmetric normalized Laplacian associated with P."""
    s = 0.5 * (p + p.T)
    s = row_normalize_dense(s.clamp_min(0.0))
    a = 0.5 * (s + s.T)
    deg = a.sum(dim=-1).clamp_min(1e-12)
    dinv = deg.rsqrt()
    sym = dinv[:, None] * a * dinv[None, :]
    l = torch.eye(p.shape[0], device=p.device, dtype=p.dtype) - sym
    vals = torch.linalg.eigvalsh(l)
    if vals.numel() < 2:
        return torch.tensor(0.0, device=p.device, dtype=p.dtype)
    return vals[1]


@dataclass(slots=True)
class DualOperatorState:
    p_actuation: Tensor
    p_diagnostic: Tensor

    @property
    def l_actuation(self) -> Tensor:
        return positive_laplacian_from_markov(self.p_actuation)

    @property
    def l_diagnostic(self) -> Tensor:
        return positive_laplacian_from_markov(self.p_diagnostic)

    def discrepancy(self, mode: str = "frobenius") -> Tensor:
        return operator_discrepancy(self.p_actuation, self.p_diagnostic, mode=mode)


def symmetric_normalized_laplacian_sparse(graph: GraphBuffers, eps: float = 1e-12) -> Tensor:
    """Sparse symmetric normalized Laplacian I-D^{-1/2} A D^{-1/2}.

    Isolated vertices are rejected here rather than silently normalized; the governor
    treats them as a disconnected-state failure before spectral certification.
    """
    graph.validate()
    n = graph.num_nodes
    src, dst, w = graph.active()
    deg = torch.zeros(n, dtype=w.dtype, device=w.device)
    if src.numel():
        deg.index_add_(0, src, w)
        deg.index_add_(0, dst, w)
    if bool((deg <= eps).any().item()):
        raise ValueError("normalized Laplacian undefined for isolated vertices")
    norm_w = w * deg[src].rsqrt() * deg[dst].rsqrt()
    ids = torch.arange(n, dtype=torch.long, device=src.device)
    row = torch.cat([ids, src, dst])
    col = torch.cat([ids, dst, src])
    val = torch.cat([torch.ones(n, dtype=w.dtype, device=w.device), -norm_w, -norm_w])
    return torch.sparse_coo_tensor(torch.stack([row, col]), val, (n, n), device=w.device, dtype=w.dtype).coalesce()


def spectral_gap_graphbuffers(
    graph: GraphBuffers,
    *,
    solver: str = "auto",
    lobpcg_min_nodes: int = 256,
    niter: int = 60,
    tol: float = 1e-6,
    seed: int = 0,
) -> tuple[float, str]:
    """Algebraic connectivity of the symmetric normalized Laplacian.

    Small graphs use an exact dense eigensolve. Larger graphs use sparse LOBPCG with a
    deterministic initial block. Any disconnected/isolated state returns zero rather
    than propagating NaNs. On LOBPCG failure, auto mode falls back to exact only for
    moderately sized graphs; explicit ``lobpcg`` mode fails closed.
    """
    if solver not in {"auto", "exact", "lobpcg"}:
        raise ValueError("unknown spectral solver")
    graph.validate()
    n = graph.num_nodes
    if n < 2:
        return 0.0, "trivial"

    src, dst, w = graph.active()
    deg = torch.zeros(n, dtype=w.dtype, device=w.device)
    if src.numel():
        deg.index_add_(0, src, w)
        deg.index_add_(0, dst, w)
    if bool((deg <= 0).any().item()):
        return 0.0, "isolated_vertex"

    use_lobpcg = solver == "lobpcg" or (solver == "auto" and n >= int(lobpcg_min_nodes))
    if not use_lobpcg:
        Ls = symmetric_normalized_laplacian_sparse(graph)
        vals = torch.linalg.eigvalsh(Ls.to_dense())
        return float(vals[1].clamp_min(0).item()), "exact"

    if n < 6:
        if solver == "lobpcg":
            raise ValueError("LOBPCG with k=2 requires at least 6 rows")
        Ls = symmetric_normalized_laplacian_sparse(graph)
        vals = torch.linalg.eigvalsh(Ls.to_dense())
        return float(vals[1].clamp_min(0).item()), "exact_small"

    Ls = symmetric_normalized_laplacian_sparse(graph)
    # Seeded dense initial block; torch.lobpcg accepts sparse A but X must be dense.
    gen = torch.Generator(device=graph.weight.device)
    gen.manual_seed(int(seed))
    X = torch.randn((n, 2), generator=gen, dtype=graph.weight.dtype, device=graph.weight.device)
    try:
        evals, _ = torch.lobpcg(Ls, k=2, X=X, largest=False, niter=int(niter), tol=float(tol), method="ortho")
        evals = torch.sort(evals).values
        if not bool(torch.isfinite(evals).all().item()):
            raise FloatingPointError("non-finite LOBPCG eigenvalue")
        # For a connected graph the first eigenvalue is ~0; clip tiny negative roundoff.
        return float(evals[1].clamp_min(0).item()), "lobpcg"
    except Exception:
        if solver == "lobpcg" or n > max(4 * int(lobpcg_min_nodes), 2048):
            raise
        vals = torch.linalg.eigvalsh(Ls.to_dense())
        return float(vals[1].clamp_min(0).item()), "exact_fallback"


def actuation_markov_edges_with_slots(
    graph: GraphBuffers,
    *,
    symmetric: bool = True,
    self_loop: float = 0.0,
    eps: float = 1e-12,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Directed Markov edges plus source graph slot and orientation flags."""
    graph.validate()
    slots = torch.where(graph.valid)[0]
    src = graph.src[slots]
    dst = graph.dst[slots]
    w = graph.weight[slots]
    if symmetric:
        s = torch.cat([src, dst]); d = torch.cat([dst, src]); ww = torch.cat([w, w])
        slot = torch.cat([slots, slots]); reverse = torch.cat([torch.zeros_like(slots, dtype=torch.bool), torch.ones_like(slots, dtype=torch.bool)])
    else:
        s, d, ww, slot = src.clone(), dst.clone(), w.clone(), slots.clone()
        reverse = torch.zeros_like(slot, dtype=torch.bool)
    if self_loop > 0:
        ids = torch.arange(graph.num_nodes, device=graph.src.device)
        s = torch.cat([s, ids]); d = torch.cat([d, ids])
        ww = torch.cat([ww, torch.full((graph.num_nodes,), float(self_loop), dtype=w.dtype, device=w.device)])
        slot = torch.cat([slot, torch.full((graph.num_nodes,), -1, dtype=torch.long, device=slot.device)])
        reverse = torch.cat([reverse, torch.zeros(graph.num_nodes, dtype=torch.bool, device=reverse.device)])
    mass = torch.zeros(graph.num_nodes, dtype=ww.dtype, device=ww.device)
    if s.numel():
        mass.index_add_(0, s, ww)
    isolated = mass <= 0
    if isolated.any():
        ids = torch.arange(graph.num_nodes, device=graph.src.device)[isolated]
        s = torch.cat([s, ids]); d = torch.cat([d, ids]); ww = torch.cat([ww, torch.ones(ids.numel(), dtype=w.dtype, device=w.device)])
        slot = torch.cat([slot, torch.full((ids.numel(),), -1, dtype=torch.long, device=slot.device)])
        reverse = torch.cat([reverse, torch.zeros(ids.numel(), dtype=torch.bool, device=reverse.device)])
        mass = torch.zeros(graph.num_nodes, dtype=ww.dtype, device=ww.device); mass.index_add_(0, s, ww)
    return s, d, ww / mass[s].clamp_min(eps), slot, reverse


def sparse_markov_apply_gauge(
    z: Tensor,
    src: Tensor,
    dst: Tensor,
    pweight: Tensor,
    connection: Tensor,
    *,
    gauge_dim: int,
    num_nodes: int,
) -> Tensor:
    """Markov aggregation with SO(d) parallel transport on a prefix of channels."""
    if connection.shape[0] != src.numel() or connection.shape[-2:] != (gauge_dim, gauge_dim):
        raise ValueError("connection shape does not match directed edges/gauge_dim")
    if gauge_dim <= 0 or gauge_dim > z.shape[-1]:
        raise ValueError("invalid gauge_dim")
    transported = z[dst].clone()
    transported[:, :gauge_dim] = torch.einsum("eij,ej->ei", connection.to(z.dtype), z[dst, :gauge_dim])
    out = torch.zeros((num_nodes, z.shape[-1]), dtype=z.dtype, device=z.device)
    out.index_add_(0, src, pweight.to(z.dtype).unsqueeze(-1) * transported)
    return out


def sparse_laplacian_step_gauge(
    z: Tensor,
    src: Tensor,
    dst: Tensor,
    pweight: Tensor,
    connection: Tensor,
    *,
    gauge_dim: int,
    eta: float | Tensor,
    num_nodes: int,
) -> Tensor:
    pz = sparse_markov_apply_gauge(z, src, dst, pweight, connection, gauge_dim=gauge_dim, num_nodes=num_nodes)
    return z - eta * (z - pz)
