from __future__ import annotations

import networkx as nx


def af3_edge(g: nx.Graph, u: int, v: int) -> float:
    """Exact unweighted Augmented Forman-3 curvature.

    AF3(u,v) = 4 - deg(u) - deg(v) + 3*T(u,v), where T is the number
    of triangles containing edge (u,v).
    """
    if not g.has_edge(u, v):
        raise ValueError(f"({u},{v}) is not an edge")
    common = len(set(g.neighbors(u)).intersection(g.neighbors(v)))
    return float(4 - g.degree[u] - g.degree[v] + 3 * common)


def af3_curvatures(g: nx.Graph) -> dict[tuple[int, int], float]:
    return {(int(u), int(v)): af3_edge(g, int(u), int(v)) for u, v in g.edges()}


def degree_weighted_af3_proxy(g: nx.Graph, u: int, v: int) -> float:
    """Degree-weighted AF3 proxy used as a scalable candidate score.

    This is deliberately named a *proxy*: the accessible ICLR-2026 source
    confirms degree weighting f(d)=(1+d)^-1 as the best reported variant,
    but did not expose the complete WAF3 equation in retrievable text.
    We therefore do not claim this is paper-exact WAF3.
    """
    base = af3_edge(g, u, v)
    fu = 1.0 / (1.0 + float(g.degree[u]))
    fv = 1.0 / (1.0 + float(g.degree[v]))
    scale = 2.0 / max(fu + fv, 1e-12)
    return float(base / scale)
