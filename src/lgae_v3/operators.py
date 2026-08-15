from __future__ import annotations

from dataclasses import dataclass
from typing import Any

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


# ---------------------------------------------------------------------------
# Sparse dual operator system (v4.0)
#
# The dense DualOperatorState allocates O(N²) for both actuation and diagnostic
# operators. For large N this is the primary scaling bottleneck. The sparse
# system represents both operators as directed edge lists (src, dst, weight)
# and computes discrepancy on the union of supports without materializing N×N.
#
# For small N (<= sparse_threshold), the dense path is retained as an exact
# reference. For large N, the sparse path uses k-NN on the feature cloud
# without forming the full pairwise distance matrix.
# ---------------------------------------------------------------------------


def _knn_distances(z: Tensor, k: int) -> tuple[Tensor, Tensor]:
    """Compute k-nearest-neighbor distances and indices.

    For moderate N (≤4096), uses torch.cdist directly. For larger N, uses
    chunked computation to bound peak memory to O(B*N*D) where B is the
    chunk size, but total compute remains O(N²D).

    **Scalability note:** This is bounded-memory exact k-NN, not sub-quadratic
    ANN. The sparse *storage* is O(Nk), but the *construction* is still
    O(N²D) compute. True large-N scalability requires an ANN index (e.g.
    FAISS, HNSW) which is a future extension.

    Returns (distances, indices) of shape [N, k] where distances are squared
    Euclidean distances and indices are the column indices of the nearest
    neighbors (excluding self).
    """
    n, d = z.shape
    k_eff = min(int(k), max(n - 1, 1))
    if k_eff <= 0:
        return z.new_zeros((n, 0)), torch.zeros((n, 0), dtype=torch.long, device=z.device)

    # For moderate N, use cdist directly (it's optimized in C++)
    # For very large N, chunk the computation to bound memory
    if n <= 4096:
        d2 = torch.cdist(z, z, p=2).square()
        # Exclude self
        eye = torch.eye(n, dtype=torch.bool, device=z.device)
        d2.masked_fill_(eye, float("inf"))
        vals, idx = torch.topk(d2, k=k_eff, largest=False, dim=-1)
        return vals, idx
    else:
        # Chunked k-NN: process in blocks to bound peak memory
        vals_list: list[Tensor] = []
        idx_list: list[Tensor] = []
        chunk = max(1024, 4096 // max(d, 1))
        for start in range(0, n, chunk):
            end = min(start + chunk, n)
            z_chunk = z[start:end]
            d2_chunk = torch.cdist(z_chunk, z, p=2).square()
            # Exclude self
            for i in range(end - start):
                d2_chunk[i, start + i] = float("inf")
            vals_chunk, idx_chunk = torch.topk(d2_chunk, k=k_eff, largest=False, dim=-1)
            vals_list.append(vals_chunk)
            idx_list.append(idx_chunk)
        return torch.cat(vals_list, dim=0), torch.cat(idx_list, dim=0)


def diagnostic_diffusion_edges(
    z: Tensor,
    k: int = 16,
    epsilon_floor: float = 1e-4,
    *,
    include_self: bool = False,
) -> tuple[Tensor, Tensor, Tensor]:
    """Sparse Gaussian diffusion operator on the feature cloud.

    Returns directed edges (src, dst, weight) representing a row-stochastic
    diffusion kernel, without materializing an N×N matrix. Uses k-NN to
    select support, then applies a Gaussian kernel with local scaling.

    **Scalability:** The sparse *storage* is O(Nk), but the k-NN
    *construction* is O(N²D) compute with O(BN) peak memory via chunking.
    This is bounded-memory exact k-NN, not sub-quadratic ANN. True
    large-N behavior requires an ANN index (future extension).

    This is the sparse counterpart to ``diagnostic_diffusion_operator``.
    """
    n = z.shape[0]
    if n == 0:
        raise ValueError("empty latent cloud")
    if n == 1:
        # Single node: self-loop only
        s = torch.tensor([0], dtype=torch.long, device=z.device)
        d = torch.tensor([0], dtype=torch.long, device=z.device)
        w = torch.tensor([1.0], dtype=z.dtype, device=z.device)
        return s, d, w

    k_eff = min(max(int(k), 1), max(n - 1, 1))
    vals, idx = _knn_distances(z, k_eff)  # [N, k], squared distances

    # Local scaling: sigma_i = distance to k-th nearest neighbor
    local_scale = vals[:, -1].sqrt().clamp_min(epsilon_floor)  # [N]

    # Build edges: for each i, connect to its k nearest neighbors j
    src = torch.arange(n, device=z.device).unsqueeze(1).expand(n, k_eff).reshape(-1)
    dst = idx.reshape(-1)
    d2 = vals.reshape(-1)

    # Gaussian kernel with bilateral local scaling
    sigma_i = local_scale[src]
    sigma_j = local_scale[dst]
    eps_ij = (sigma_i * sigma_j).clamp_min(epsilon_floor ** 2)
    kernel_vals = torch.exp(-0.5 * d2 / eps_ij)

    # Symmetrize: add reverse edges (j -> i with same weight)
    src_sym = torch.cat([src, dst])
    dst_sym = torch.cat([dst, src])
    w_sym = torch.cat([kernel_vals, kernel_vals])

    if include_self:
        ids = torch.arange(n, device=z.device)
        src_sym = torch.cat([src_sym, ids])
        dst_sym = torch.cat([dst_sym, ids])
        w_sym = torch.cat([w_sym, torch.ones(n, dtype=z.dtype, device=z.device)])

    # Row-normalize to make row-stochastic
    mass = torch.zeros(n, dtype=w_sym.dtype, device=z.device)
    mass.index_add_(0, src_sym, w_sym)
    # Handle isolated nodes (no neighbors found)
    isolated = mass <= 0
    if bool(isolated.any().item()):
        ids = torch.arange(n, device=z.device)[isolated]
        src_sym = torch.cat([src_sym, ids])
        dst_sym = torch.cat([dst_sym, ids])
        w_sym = torch.cat([w_sym, torch.ones(ids.numel(), dtype=z.dtype, device=z.device)])
        mass = torch.zeros(n, dtype=w_sym.dtype, device=z.device)
        mass.index_add_(0, src_sym, w_sym)

    pweight = w_sym / mass[src_sym].clamp_min(1e-12)
    return src_sym, dst_sym, pweight


def sparse_operator_discrepancy(
    act_src: Tensor,
    act_dst: Tensor,
    act_weight: Tensor,
    diag_src: Tensor,
    diag_dst: Tensor,
    diag_weight: Tensor,
    num_nodes: int,
    mode: str = "frobenius",
) -> Tensor:
    """Compute operator discrepancy on sparse edge supports.

    Builds the difference P_act - P_diag only on the union of supports,
    avoiding a full N×N matrix. Uses ``torch.sparse_coo_tensor.coalesce()``
    to correctly accumulate duplicate directed edges (e.g. from mutual k-NN
    symmetrization) before comparison.

    Memory: O(|E_act| + |E_diag|) instead of O(N²).
    """
    import warnings

    # Build coalesced sparse tensors for both operators
    act_indices = torch.stack([act_src.to(torch.long), act_dst.to(torch.long)], dim=0)
    diag_indices = torch.stack([diag_src.to(torch.long), diag_dst.to(torch.long)], dim=0)

    with warnings.catch_warnings():
        # Suppress sparse invariant check warning on older PyTorch versions
        warnings.filterwarnings("ignore", message="Sparse invariant checks", category=UserWarning)
        P_act_sp = torch.sparse_coo_tensor(
            act_indices, act_weight, size=(num_nodes, num_nodes),
        ).coalesce()

        P_diag_sp = torch.sparse_coo_tensor(
            diag_indices, diag_weight, size=(num_nodes, num_nodes),
        ).coalesce()

        # Build the difference by concatenating indices and values
        # P_act - P_diag: act entries get +weight, diag entries get -weight
        diff_indices = torch.cat([P_act_sp.indices(), P_diag_sp.indices()], dim=1)
        diff_values = torch.cat([P_act_sp.values(), -P_diag_sp.values()])
        delta = torch.sparse_coo_tensor(
            diff_indices, diff_values, size=(num_nodes, num_nodes),
        ).coalesce()

    if mode == "frobenius":
        # ||P_act - P_diag||_F / sqrt(N)
        return torch.sqrt((delta.values() ** 2).sum().clamp_min(0.0)) / max(num_nodes, 1) ** 0.5
    if mode == "mean_l1":
        return delta.values().abs().sum() / max(num_nodes, 1)
    raise ValueError(f"unknown discrepancy mode: {mode}")


@dataclass(slots=True)
class SparseDualOperatorState:
    """Sparse dual operator representation using edge lists.

    Both actuation and diagnostic operators are stored as directed edge
    lists (src, dst, weight) where weight is row-stochastic. This avoids
    the O(N²) memory of the dense DualOperatorState.

    **Scalability:** Sparse storage is O(|E_act| + N*k), but the k-NN
    construction for the diagnostic operator is O(N²D) compute with
    bounded memory. This is not sub-quadratic ANN. The sparse discrepancy
    is exact for the truncated operator, not for the hypothetical full
    Gaussian kernel from which top-k edges were selected.
    """
    act_src: Tensor
    act_dst: Tensor
    act_weight: Tensor
    diag_src: Tensor
    diag_dst: Tensor
    diag_weight: Tensor
    num_nodes: int

    @classmethod
    def from_graph_and_latent(
        cls,
        graph: GraphBuffers,
        z: Tensor,
        *,
        symmetric: bool = True,
        self_loop: float = 0.0,
        diagnostic_k: int = 16,
        diagnostic_epsilon_floor: float = 1e-4,
    ) -> "SparseDualOperatorState":
        """Build sparse dual operators from graph and latent state."""
        # Actuation: reuse the existing sparse edge construction
        act_src, act_dst, act_w = actuation_markov_edges(
            graph, symmetric=symmetric, self_loop=self_loop,
        )
        # Diagnostic: k-NN based diffusion on the feature cloud
        diag_src, diag_dst, diag_w = diagnostic_diffusion_edges(
            z, k=diagnostic_k, epsilon_floor=diagnostic_epsilon_floor,
        )
        return cls(act_src, act_dst, act_w, diag_src, diag_dst, diag_w, graph.num_nodes)

    def discrepancy(self, mode: str = "frobenius") -> Tensor:
        return sparse_operator_discrepancy(
            self.act_src, self.act_dst, self.act_weight,
            self.diag_src, self.diag_dst, self.diag_weight,
            self.num_nodes, mode=mode,
        )

    def to_dense(self) -> "DualOperatorState":
        """Convert to dense DualOperatorState (for small N or testing)."""
        n = self.num_nodes
        p_act = torch.zeros((n, n), dtype=self.act_weight.dtype, device=self.act_weight.device)
        p_act.index_put_((self.act_src, self.act_dst), self.act_weight, accumulate=True)
        p_diag = torch.zeros((n, n), dtype=self.diag_weight.dtype, device=self.diag_weight.device)
        p_diag.index_put_((self.diag_src, self.diag_dst), self.diag_weight, accumulate=True)
        return DualOperatorState(p_act, p_diag)

    @property
    def p_diagnostic(self) -> Tensor:
        """Materialize the full dense diagnostic operator.

        WARNING: This allocates O(N²) memory. Only use for small N or testing.
        For large N, use ``local_dense_diagnostic`` instead.
        """
        return self.to_dense().p_diagnostic

    @property
    def p_actuation(self) -> Tensor:
        """Materialize the full dense actuation operator.

        WARNING: This allocates O(N²) memory. Only use for small N or testing.
        """
        return self.to_dense().p_actuation

    def local_dense_diagnostic(self, center_nodes: Tensor, radius: int = 2) -> tuple[Tensor, Tensor]:
        """Extract a local dense diagnostic operator for selected center nodes.

        For each center node, extracts the 2-hop neighborhood from the sparse
        diagnostic graph and materializes only that local submatrix. This is
        the key scaling mechanism: BE/CDE complexity depends on local degree,
        not global N.

        Returns (local_P, node_indices) where:
        - local_P is a block-diagonal-ish dense matrix over the union of neighborhoods
        - node_indices is the mapping from local indices to global node IDs
        """
        device = self.diag_src.device
        dtype = self.diag_weight.dtype

        # Build adjacency sets from sparse diagnostic edges
        nbrs: dict[int, set[int]] = {i: set() for i in range(self.num_nodes)}
        for s, d in zip(self.diag_src.tolist(), self.diag_dst.tolist()):
            nbrs[s].add(d)

        # BFS to radius hops from each center node
        selected: set[int] = set()
        for c in center_nodes.tolist():
            frontier = {int(c)}
            for _ in range(radius):
                next_frontier: set[int] = set()
                for node in frontier:
                    next_frontier.update(nbrs.get(node, set()))
                selected.update(frontier)
                frontier = next_frontier
            selected.update(frontier)

        sorted_nodes = sorted(selected)
        idx_map = {g: i for i, g in enumerate(sorted_nodes)}
        n_local = len(sorted_nodes)
        if n_local == 0:
            return torch.zeros((0, 0), dtype=dtype, device=device), torch.tensor([], dtype=torch.long, device=device)

        # Build local dense matrix from sparse edges
        local_P = torch.zeros((n_local, n_local), dtype=dtype, device=device)
        for s, d, w in zip(self.diag_src.tolist(), self.diag_dst.tolist(), self.diag_weight.tolist()):
            if s in idx_map and d in idx_map:
                local_P[idx_map[s], idx_map[d]] += w

        return local_P, torch.tensor(sorted_nodes, dtype=torch.long, device=device)


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
    return torch.sparse_coo_tensor(torch.stack([row, col]), val, (n, n), device=w.device, dtype=w.dtype, check_invariants=False).coalesce()


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
