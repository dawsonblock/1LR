from __future__ import annotations

import math
import networkx as nx
import numpy as np
from scipy.optimize import linprog
from scipy.special import logsumexp


def _lazy_measure(g: nx.Graph, x: int, p: float) -> tuple[list[int], np.ndarray]:
    nbrs = list(g.neighbors(x))
    if not nbrs:
        return [x], np.array([1.0], dtype=float)
    nodes = [x] + nbrs
    mass = np.full(len(nodes), (1.0 - p) / len(nbrs), dtype=float)
    mass[0] = p
    return nodes, mass


def _weighted_lazy_measure(g: nx.Graph, x: int, p: float) -> tuple[list[int], np.ndarray]:
    """Affinity-based lazy random-walk measure.

    The idle mass p stays at x; the remaining (1-p) is distributed to
    neighbors proportionally to edge **affinity** (the ``weight`` attribute).
    This is the correct lazy measure for the Markov chain P(a).
    """
    nbrs = list(g.neighbors(x))
    if not nbrs:
        return [x], np.array([1.0], dtype=float)
    nodes = [x] + nbrs
    weights = np.array([g[x][u].get("weight", 1.0) for u in nbrs], dtype=float)
    total = weights.sum()
    if total <= 0:
        # Degenerate: fall back to uniform
        mass = np.full(len(nodes), (1.0 - p) / len(nbrs), dtype=float)
    else:
        mass = np.zeros(len(nodes), dtype=float)
        mass[1:] = (1.0 - p) * weights / total
    mass[0] = p
    return nodes, mass


def _transport_lp(cost: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    m, n = cost.shape
    c = cost.reshape(-1)
    Aeq = []
    beq = []
    for i in range(m):
        row = np.zeros(m * n)
        row[i*n:(i+1)*n] = 1.0
        Aeq.append(row); beq.append(a[i])
    for j in range(n):
        row = np.zeros(m * n)
        row[j::n] = 1.0
        Aeq.append(row); beq.append(b[j])
    res = linprog(c, A_eq=np.asarray(Aeq), b_eq=np.asarray(beq), bounds=(0.0, None), method="highs")
    if not res.success:
        raise RuntimeError(f"Wasserstein LP failed: {res.message}")
    return float(res.fun)


def log_sinkhorn_wasserstein(
    cost: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    *,
    epsilon: float = 0.05,
    max_iter: int = 200,
    tolerance: float = 1e-6,
    normalize_cost: bool = True,
    require_convergence: bool = True,
) -> float:
    """Numerically stabilized entropic approximation of W1.

    Sinkhorn scaling is performed entirely in the log domain. Zero-mass rows and
    columns are removed exactly instead of being replaced by tiny positive mass.
    Convergence is certified against the *marginal residuals* of the recovered
    coupling, not merely changes in dual/scaling variables.

    The optimization can use a normalized ground cost for conditioning, while the
    returned transport cost is always evaluated in the original metric units.
    """
    C = np.asarray(cost, dtype=np.float64)
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    if C.ndim != 2 or C.shape != (a.size, b.size):
        raise ValueError("cost shape must match marginal sizes")
    if not np.isfinite(C).all() or (C < 0).any():
        raise ValueError("cost must be finite and nonnegative")
    if epsilon <= 0 or max_iter <= 0 or tolerance <= 0:
        raise ValueError("invalid Sinkhorn parameters")
    if (a < 0).any() or (b < 0).any() or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("marginals must be finite and nonnegative")
    sa, sb = float(a.sum()), float(b.sum())
    if sa <= 0 or sb <= 0:
        raise ValueError("marginals must have positive mass")
    a = a / sa
    b = b / sb

    # Removing exact zero-mass support is both more accurate and more stable than
    # injecting machine-tiny mass, especially for p=0 Ollivier measures.
    ma = a > 0.0
    mb = b > 0.0
    C = C[np.ix_(ma, mb)]
    a = a[ma]
    b = b[mb]

    scale = float(C.max()) if normalize_cost else 1.0
    if not math.isfinite(scale) or scale <= 0:
        return 0.0
    Cn = C / scale
    eps = float(epsilon)

    log_a = np.log(a)
    log_b = np.log(b)
    log_k = -Cn / eps
    log_u = np.zeros_like(log_a)
    log_v = np.zeros_like(log_b)
    residual = float("inf")
    log_plan = None

    for _ in range(int(max_iter)):
        log_u = log_a - logsumexp(log_k + log_v[None, :], axis=1)
        log_v = log_b - logsumexp(log_k + log_u[:, None], axis=0)
        log_plan = log_u[:, None] + log_k + log_v[None, :]
        row = np.exp(logsumexp(log_plan, axis=1))
        col = np.exp(logsumexp(log_plan, axis=0))
        residual = max(float(np.max(np.abs(row - a))), float(np.max(np.abs(col - b))))
        if residual <= tolerance:
            break

    if log_plan is None:
        raise RuntimeError("log-domain Sinkhorn executed zero iterations")
    if require_convergence and residual > tolerance:
        raise RuntimeError(
            f"log-domain Sinkhorn did not converge: marginal residual={residual:.3e} "
            f"after {int(max_iter)} iterations"
        )

    plan = np.exp(log_plan)
    value = float(np.sum(plan * C))
    if not math.isfinite(value):
        raise FloatingPointError("log-domain Sinkhorn produced non-finite transport cost")
    return value


def _edge_cost(g: nx.Graph, left: list[int], right: list[int]) -> np.ndarray:
    cost = np.empty((len(left), len(right)), dtype=float)
    for i, x in enumerate(left):
        lengths = nx.single_source_shortest_path_length(g, x)
        for j, y in enumerate(right):
            if y not in lengths:
                raise ValueError("Ollivier transport requires connected support metric")
            cost[i, j] = lengths[y]
    return cost


def _weighted_edge_cost(g: nx.Graph, left: list[int], right: list[int]) -> np.ndarray:
    """Metric-length shortest-path cost matrix.

    Uses Dijkstra with edge ``length`` attribute as distances. Falls back
    to ``1/weight`` (inverse affinity) when ``length`` is not present,
    matching the default metric-measure relationship.
    """
    # Ensure all edges have a length attribute (default: 1/weight)
    for a, b in g.edges():
        if "length" not in g[a][b]:
            w = g[a][b].get("weight", 1.0)
            g[a][b]["length"] = 1.0 / w if w > 0 else 1.0
    cost = np.empty((len(left), len(right)), dtype=float)
    for i, x in enumerate(left):
        lengths = nx.single_source_dijkstra_path_length(g, x, weight="length")
        for j, y in enumerate(right):
            if y not in lengths:
                raise ValueError("metric-length Ollivier transport requires connected support metric")
            cost[i, j] = lengths[y]
    return cost


def ollivier_edge(
    g: nx.Graph,
    u: int,
    v: int,
    p: float = 0.0,
    *,
    backend: str = "exact_lp",
    sinkhorn_epsilon: float = 0.05,
    sinkhorn_max_iter: int = 200,
    sinkhorn_tolerance: float = 1e-6,
) -> float:
    """p-idle Ollivier curvature on an unweighted graph edge.

    ``exact_lp`` is the qualification/reference backend. ``sinkhorn_log`` is a stable,
    entropically regularized approximation intended for larger online audits.
    """
    if not (0.0 <= p <= 1.0):
        raise ValueError("p must lie in [0,1]")
    if not g.has_edge(u, v):
        raise ValueError("ollivier_edge currently expects an edge")
    if backend not in {"exact_lp", "sinkhorn_log"}:
        raise ValueError("unknown Ollivier backend")
    left, a = _lazy_measure(g, u, p)
    right, b = _lazy_measure(g, v, p)
    cost = _edge_cost(g, left, right)
    if backend == "exact_lp":
        w1 = _transport_lp(cost, a, b)
    else:
        w1 = log_sinkhorn_wasserstein(
            cost, a, b, epsilon=sinkhorn_epsilon,
            max_iter=sinkhorn_max_iter, tolerance=sinkhorn_tolerance,
        )
    d = nx.shortest_path_length(g, u, v)
    return float(1.0 - w1 / float(d))


def ollivier_curvatures(g: nx.Graph, p: float = 0.0, edges=None, **kwargs) -> dict[tuple[int, int], float]:
    target = g.edges() if edges is None else edges
    return {(int(u), int(v)): ollivier_edge(g, int(u), int(v), p=p, **kwargs) for u, v in target}


def weighted_ollivier_edge(
    g: nx.Graph,
    u: int,
    v: int,
    p: float = 0.0,
    *,
    backend: str = "exact_lp",
    sinkhorn_epsilon: float = 0.05,
    sinkhorn_max_iter: int = 200,
    sinkhorn_tolerance: float = 1e-6,
) -> float:
    """p-idle Ollivier curvature with metric-measure separation.

    The lazy measure μ_x distributes (1-p) mass to neighbors proportionally
    to edge **affinity** (``weight`` attribute). The Wasserstein ground cost
    uses **metric length** (``length`` attribute) via Dijkstra shortest path.

    This cleanly separates:
    - How far apart are states? → d_ℓ (from length)
    - How likely is information to move? → P(a) (from affinity)

    κ(x,y) = 1 - W₁^{d_ℓ}(μ_x, μ_y) / d_ℓ(x,y)
    """
    if not (0.0 <= p <= 1.0):
        raise ValueError("p must lie in [0,1]")
    if not g.has_edge(u, v):
        raise ValueError("weighted_ollivier_edge currently expects an edge")
    if backend not in {"exact_lp", "sinkhorn_log"}:
        raise ValueError("unknown Ollivier backend")
    left, a = _weighted_lazy_measure(g, u, p)  # measures from affinity
    right, b = _weighted_lazy_measure(g, v, p)
    cost = _weighted_edge_cost(g, left, right)  # cost from length
    if backend == "exact_lp":
        w1 = _transport_lp(cost, a, b)
    else:
        w1 = log_sinkhorn_wasserstein(
            cost, a, b, epsilon=sinkhorn_epsilon,
            max_iter=sinkhorn_max_iter, tolerance=sinkhorn_tolerance,
        )
    # Ensure length attribute exists for distance computation
    if "length" not in g[u][v]:
        w = g[u][v].get("weight", 1.0)
        g[u][v]["length"] = 1.0 / w if w > 0 else 1.0
    d = nx.dijkstra_path_length(g, u, v, weight="length")  # d_ℓ(x,y)
    return float(1.0 - w1 / float(d))


def _uniform_ball_measure(g: nx.Graph, x: int, radius: int) -> tuple[list[int], np.ndarray]:
    if radius < 0:
        raise ValueError("radius must be nonnegative")
    lengths = nx.single_source_shortest_path_length(g, x, cutoff=int(radius))
    nodes = sorted(int(v) for v in lengths)
    if not nodes:
        nodes = [int(x)]
    mass = np.full(len(nodes), 1.0 / len(nodes), dtype=float)
    return nodes, mass


def multiscale_ollivier_edge(
    g: nx.Graph,
    u: int,
    v: int,
    *,
    radius: int = 2,
    backend: str = "exact_lp",
    sinkhorn_epsilon: float = 0.05,
    sinkhorn_max_iter: int = 200,
    sinkhorn_tolerance: float = 1e-6,
) -> float:
    """Mesoscopic Ollivier curvature using uniform closed-ball measures."""
    if not g.has_edge(u, v):
        raise ValueError("multiscale_ollivier_edge currently expects an edge")
    left, a = _uniform_ball_measure(g, u, int(radius))
    right, b = _uniform_ball_measure(g, v, int(radius))
    cost = _edge_cost(g, left, right)
    if backend == "exact_lp":
        w1 = _transport_lp(cost, a, b)
    elif backend == "sinkhorn_log":
        w1 = log_sinkhorn_wasserstein(
            cost, a, b, epsilon=sinkhorn_epsilon,
            max_iter=sinkhorn_max_iter, tolerance=sinkhorn_tolerance,
        )
    else:
        raise ValueError("unknown Ollivier backend")
    d = nx.shortest_path_length(g, u, v)
    return float(1.0 - w1 / float(d))
