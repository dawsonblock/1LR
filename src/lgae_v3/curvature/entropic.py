from __future__ import annotations

from dataclasses import dataclass
import math
import networkx as nx
import numpy as np
from scipy.optimize import minimize


@dataclass(frozen=True, slots=True)
class WeakEntropicNodeResult:
    node: int
    value: float | None
    status: str
    optimizer_success: bool
    message: str = ""


def _distance_two(g: nx.Graph, z: int) -> list[int]:
    d = nx.single_source_shortest_path_length(g, z, cutoff=2)
    return [int(v) for v, dist in d.items() if dist == 2]


def weak_entropic_node_detailed(
    g: nx.Graph,
    z: int,
    *,
    maxiter: int = 100,
    eps: float = 1e-10,
) -> WeakEntropicNodeResult:
    """Weak entropic curvature κ_w(z) for the unweighted adjacency generator L0.

    Solver failure is fail-closed: no optimistic feasible-point fallback is used.
    When the distance-two shell is empty the theoretical algorithm assigns +∞;
    this is preserved explicitly rather than dropped from graph-level summaries.
    """
    if z not in g:
        raise ValueError("node not present in graph")
    nbrs = list(g.neighbors(z))
    W = _distance_two(g, z)
    if not W:
        return WeakEntropicNodeResult(int(z), float("inf"), "empty_two_hop", True, "S2(z) is empty")
    if not nbrs:
        return WeakEntropicNodeResult(int(z), float("inf"), "isolated", True, "node has no neighbors")

    pos = {u: i for i, u in enumerate(nbrs)}
    terms: list[tuple[float, dict[int, float]]] = []
    for w in W:
        mids = [u for u in nbrs if g.has_edge(u, w)]
        L2 = float(len(mids))
        if L2 <= 0:
            continue
        terms.append((L2, {pos[u]: 2.0 / L2 for u in mids}))
    if not terms:
        return WeakEntropicNodeResult(int(z), None, "invalid_two_hop", False, "distance-two shell has no valid midpoint terms")

    def H(alpha: np.ndarray) -> float:
        total = 0.0
        aa = np.clip(alpha, eps, None)
        for L2, exps in terms:
            logprod = 0.0
            for i, power in exps.items():
                logprod += power * math.log(float(aa[i]))
            total += L2 * math.exp(logprod)
        return total

    x0 = np.full(len(nbrs), 1.0 / len(nbrs), dtype=float)
    cons = {"type": "eq", "fun": lambda a: float(np.sum(a) - 1.0)}
    try:
        res = minimize(
            lambda a: -H(a),
            x0,
            method="SLSQP",
            bounds=[(eps, 1.0)] * len(nbrs),
            constraints=[cons],
            options={"maxiter": maxiter, "ftol": 1e-10},
        )
    except Exception as exc:
        return WeakEntropicNodeResult(int(z), None, "solver_exception", False, str(exc))
    if not res.success:
        return WeakEntropicNodeResult(int(z), None, "solver_failed", False, str(res.message))
    h = H(res.x)
    if not math.isfinite(h) or h <= 0:
        return WeakEntropicNodeResult(int(z), None, "invalid_objective", False, f"H={h!r}")
    value = float(-2.0 * math.log(h))
    if not math.isfinite(value):
        return WeakEntropicNodeResult(int(z), None, "nonfinite_curvature", False, f"kappa={value!r}")
    return WeakEntropicNodeResult(int(z), value, "ok", True, str(res.message))


def weak_entropic_node(g: nx.Graph, z: int, **kwargs) -> float | None:
    """Compatibility wrapper returning κ_w, +∞ for empty S2, or None on failure."""
    return weak_entropic_node_detailed(g, z, **kwargs).value


def weak_entropic_graph_detailed(g: nx.Graph, nodes=None, **kwargs) -> dict[int, WeakEntropicNodeResult]:
    target = list(g.nodes() if nodes is None else nodes)
    return {int(z): weak_entropic_node_detailed(g, int(z), **kwargs) for z in target}


def weak_entropic_graph(g: nx.Graph, nodes=None, **kwargs) -> dict[int, float]:
    detailed = weak_entropic_graph_detailed(g, nodes=nodes, **kwargs)
    return {z: r.value for z, r in detailed.items() if r.value is not None}
