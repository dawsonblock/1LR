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


def weighted_af3_edge(g: nx.Graph, u: int, v: int) -> float:
    """Edge-weight-aware Augmented Forman-3 curvature.

    For a weighted graph, the Forman curvature scales with the edge weight
    relative to the weighted degrees. The formula adapts AF3 to weighted
    edges by replacing unweighted degree with weighted degree (sum of edge
    weights) and scaling by the edge weight:

        κ_w(u,v) = w_{uv} * [4/w_{uv} - deg_w(u)/w_{uv} - deg_w(v)/w_{uv}
                              + 3*T(u,v)/w_{uv}]

    Simplified for the normalized case where w_{uv} is the reference scale:

        κ_w(u,v) = 4 - deg_w(u) - deg_w(v) + 3*T(u,v)

    where deg_w is the weighted degree (sum of incident edge weights) and
    T(u,v) is the number of triangles containing (u,v).

    This is a weighted adaptation of AF3, not a paper-exact weighted Forman
    curvature. The triangle count remains unweighted since AF3's triangle
    term counts shared neighbors, not weighted paths.
    """
    if not g.has_edge(u, v):
        raise ValueError(f"({u},{v}) is not an edge")
    w_uv = float(g[u][v].get("weight", 1.0))
    if w_uv <= 0:
        raise ValueError("edge weight must be positive for weighted AF3")

    # Weighted degree: sum of edge weights
    deg_w_u = float(sum(g[u][z].get("weight", 1.0) for z in g.neighbors(u)))
    deg_w_v = float(sum(g[v][z].get("weight", 1.0) for z in g.neighbors(v)))

    # Triangle count (unweighted — shared neighbors)
    common = len(set(g.neighbors(u)).intersection(g.neighbors(v)))

    # Scale the unweighted AF3 formula by the edge weight ratio.
    # The weighted degree replaces unweighted degree.
    return float(w_uv * (4.0 / w_uv - deg_w_u / w_uv - deg_w_v / w_uv + 3.0 * common / w_uv))


def weighted_af3_curvatures(g: nx.Graph) -> dict[tuple[int, int], float]:
    return {(int(u), int(v)): weighted_af3_edge(g, int(u), int(v)) for u, v in g.edges()}
