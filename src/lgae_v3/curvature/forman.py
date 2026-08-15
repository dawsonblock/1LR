from __future__ import annotations

import math
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


def weighted_af3_proxy(g: nx.Graph, u: int, v: int) -> float:
    """Weighted-degree AF3 proxy (not canonical weighted Forman).

    This is a cheap heuristic that substitutes weighted degree (sum of edge
    affinities) for unweighted degree in the AF3 formula. It is NOT the
    literature-faithful weighted Forman curvature, which requires explicit
    square-root weight ratios. Use ``weighted_forman_edge`` for the
    canonical formula.
    """
    if not g.has_edge(u, v):
        raise ValueError(f"({u},{v}) is not an edge")
    w_uv = float(g[u][v].get("weight", 1.0))
    if w_uv <= 0:
        raise ValueError("edge affinity must be positive for weighted AF3 proxy")

    deg_w_u = float(sum(g[u][z].get("weight", 1.0) for z in g.neighbors(u)))
    deg_w_v = float(sum(g[v][z].get("weight", 1.0) for z in g.neighbors(v)))
    common = len(set(g.neighbors(u)).intersection(g.neighbors(v)))
    return float(w_uv * (4.0 / w_uv - deg_w_u / w_uv - deg_w_v / w_uv + 3.0 * common / w_uv))


def weighted_af3_proxy_curvatures(g: nx.Graph) -> dict[tuple[int, int], float]:
    return {(int(u), int(v)): weighted_af3_proxy(g, int(u), int(v)) for u, v in g.edges()}


def weighted_forman_edge(g: nx.Graph, u: int, v: int) -> float:
    """Literature-faithful weighted Forman curvature for an edge.

    Uses the standard weighted Forman expression with explicit edge-weight
    square-root ratios:

        F(e) = w_e * [ w_u (1 - Σ_{e_u~e} √(w_e/w_{e_u}))
                     + w_v (1 - Σ_{e_v~e} √(w_e/w_{e_v})) ]

    where w_e is the edge weight, w_u/w_v are vertex weights (default 1),
    and the sums are over edges adjacent to e at endpoints u and v
    respectively (excluding e itself).

    This is the canonical weighted Forman curvature, distinct from the
    ``weighted_af3_proxy`` heuristic. The square-root weight ratios
    capture the relative importance of parallel vs perpendicular edges
    in a way that simple degree substitution cannot.
    """
    if not g.has_edge(u, v):
        raise ValueError(f"({u},{v}) is not an edge")
    w_e = float(g[u][v].get("weight", 1.0))
    if w_e <= 0:
        raise ValueError("edge weight must be positive for weighted Forman")

    # Vertex weights (default 1.0 if not specified)
    w_u = float(g.nodes[u].get("weight", 1.0))
    w_v = float(g.nodes[v].get("weight", 1.0))

    # Edges adjacent to e at u (excluding e itself)
    sum_u = 0.0
    for z in g.neighbors(u):
        if z == v:
            continue
        w_eu = float(g[u][z].get("weight", 1.0))
        if w_eu > 0:
            sum_u += math.sqrt(w_e / w_eu)

    # Edges adjacent to e at v (excluding e itself)
    sum_v = 0.0
    for z in g.neighbors(v):
        if z == u:
            continue
        w_ev = float(g[v][z].get("weight", 1.0))
        if w_ev > 0:
            sum_v += math.sqrt(w_e / w_ev)

    return float(w_e * (w_u * (1.0 - sum_u) + w_v * (1.0 - sum_v)))


def weighted_forman_curvatures(g: nx.Graph) -> dict[tuple[int, int], float]:
    return {(int(u), int(v)): weighted_forman_edge(g, int(u), int(v)) for u, v in g.edges()}
