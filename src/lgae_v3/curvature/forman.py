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
    """Metric–measure Forman curvature (v4.1.1 canonical formula).

    Implements the metric–measure Forman expression using separate vertex
    measure m_1, edge measure m_2 (affinity), and metric ω (length):

        F_ω(e) = m_2(e)/m_1(u) + m_2(e)/m_1(v)
                 - Σ_{e_u~e} [m_2(e_u)/m_1(u)] · [ω(e_u)/ω(e)]
                 - Σ_{e_v~e} [m_2(e_v)/m_1(v)] · [ω(e_v)/ω(e)]

    where:
    - m_1(v) = vertex measure (default: weighted degree = Σ a_{vz})
    - m_2(e) = edge measure = affinity a_e
    - ω(e) = metric length ℓ_e

    This is distinct from the ``weighted_af3_proxy`` heuristic. The formula
    uses explicit metric ratios (ω(e_u)/ω(e)) rather than square-root
    weight ratios, and operates on the separated metric–measure structure.
    """
    if not g.has_edge(u, v):
        raise ValueError(f"({u},{v}) is not an edge")
    a_e = float(g[u][v].get("weight", 1.0))  # edge measure m_2(e)
    if a_e <= 0:
        raise ValueError("edge affinity must be positive for metric-measure Forman")
    # Metric length ω(e) — default to 1/affinity if not present
    omega_e = float(g[u][v].get("length", 1.0 / a_e))
    if omega_e <= 0:
        raise ValueError("edge length must be positive for metric-measure Forman")

    # Vertex measure m_1(v) = weighted degree (sum of affinities)
    m1_u = float(sum(g[u][z].get("weight", 1.0) for z in g.neighbors(u)))
    m1_v = float(sum(g[v][z].get("weight", 1.0) for z in g.neighbors(v)))
    if m1_u <= 0 or m1_v <= 0:
        raise ValueError("vertex measure must be positive for metric-measure Forman")

    # Sum over edges adjacent to e at u (excluding e itself)
    sum_u = 0.0
    for z in g.neighbors(u):
        if z == v:
            continue
        a_eu = float(g[u][z].get("weight", 1.0))  # m_2(e_u)
        omega_eu = float(g[u][z].get("length", 1.0 / a_eu if a_eu > 0 else 1.0))  # ω(e_u)
        if a_eu > 0 and omega_eu > 0:
            sum_u += (a_eu / m1_u) * (omega_eu / omega_e)

    # Sum over edges adjacent to e at v (excluding e itself)
    sum_v = 0.0
    for z in g.neighbors(v):
        if z == u:
            continue
        a_ev = float(g[v][z].get("weight", 1.0))  # m_2(e_v)
        omega_ev = float(g[v][z].get("length", 1.0 / a_ev if a_ev > 0 else 1.0))  # ω(e_v)
        if a_ev > 0 and omega_ev > 0:
            sum_v += (a_ev / m1_v) * (omega_ev / omega_e)

    return float(a_e / m1_u + a_e / m1_v - sum_u - sum_v)


def weighted_forman_curvatures(g: nx.Graph) -> dict[tuple[int, int], float]:
    return {(int(u), int(v)): weighted_forman_edge(g, int(u), int(v)) for u, v in g.edges()}
