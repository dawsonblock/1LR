from __future__ import annotations

import networkx as nx
import numpy as np
from scipy.optimize import linprog

from .ollivier import ollivier_edge


def lly_half_idleness(g: nx.Graph, u: int, v: int) -> float:
    """Exact LLY via the p=1/2 identity when its graph assumptions apply."""
    return 2.0 * ollivier_edge(g, u, v, p=0.5)


def weighted_lly_half_idleness(g: nx.Graph, u: int, v: int) -> float:
    """Weighted LLY via the p=1/2 identity using weighted Ollivier."""
    from .ollivier import weighted_ollivier_edge
    return 2.0 * weighted_ollivier_edge(g, u, v, p=0.5)


def lly_laplacian_lp(g: nx.Graph, x: int, y: int, *, normalized: bool = True) -> float:
    """Limit-free LLY by finite Lipschitz linear programming.

    Convention: Δ=P-I for normalized=True, and f(y)-f(x)=1 on adjacent x~y.
    Then κ_LLY(x,y)=inf[Δf(x)-Δf(y)].
    """
    if not g.has_edge(x, y):
        raise ValueError("reference implementation currently expects adjacent vertices")
    nodes = list(g.nodes())
    idx = {n: i for i, n in enumerate(nodes)}
    n = len(nodes)
    c = np.zeros(n, dtype=float)
    if normalized:
        for z in g.neighbors(x): c[idx[z]] += 1.0 / g.degree[x]
        c[idx[x]] -= 1.0
        for z in g.neighbors(y): c[idx[z]] -= 1.0 / g.degree[y]
        c[idx[y]] += 1.0
    else:
        for z in g.neighbors(x): c[idx[z]] += 1.0
        c[idx[x]] -= float(g.degree[x])
        for z in g.neighbors(y): c[idx[z]] -= 1.0
        c[idx[y]] += float(g.degree[y])

    Aub=[]; bub=[]
    for a,b in g.edges():
        row=np.zeros(n); row[idx[a]]=1; row[idx[b]]=-1
        Aub.append(row); bub.append(1.0)
        Aub.append(-row); bub.append(1.0)
    Aeq=[]; beq=[]
    row=np.zeros(n); row[idx[x]]=1.0
    Aeq.append(row); beq.append(0.0)
    row=np.zeros(n); row[idx[y]]=1.0
    Aeq.append(row); beq.append(1.0)
    res=linprog(c, A_ub=np.asarray(Aub), b_ub=np.asarray(bub), A_eq=np.asarray(Aeq), b_eq=np.asarray(beq), bounds=[(None,None)]*n, method="highs")
    if not res.success:
        raise RuntimeError(f"LLY LP failed: {res.message}")
    return float(res.fun)


def weighted_lly_laplacian_lp(g: nx.Graph, x: int, y: int) -> float:
    """Weighted limit-free LLY by finite Lipschitz linear programming.

    Uses the weighted normalized Laplacian: Δ = I - D_w^{-1} W, where
    D_w is the weighted degree matrix and W is the weighted adjacency.

    The Lipschitz constraint is unit per unit weighted distance:
    |f(a) - f(b)| <= w_{ab} for each edge (a,b).
    The boundary condition is f(y) - f(x) = d(x,y) (the weighted shortest
    path distance, not the direct edge weight, to ensure feasibility when
    the direct edge is not the shortest path).
    The curvature is then κ = (Δf(x) - Δf(y)) / d(x,y).
    """
    if not g.has_edge(x, y):
        raise ValueError("weighted LLY currently expects adjacent vertices")
    w_xy = float(g[x][y].get("weight", 1.0))
    if w_xy <= 0:
        raise ValueError("edge weight must be positive for weighted LLY")

    # Use weighted shortest path distance for boundary condition
    d_xy = float(nx.dijkstra_path_length(g, x, y, weight="weight"))

    nodes = list(g.nodes())
    idx = {n: i for i, n in enumerate(nodes)}
    n = len(nodes)

    # Weighted degree: sum of edge weights
    def wdeg(node):
        return float(sum(g[node][z].get("weight", 1.0) for z in g.neighbors(node)))

    dx = wdeg(x)
    dy = wdeg(y)
    if dx <= 0 or dy <= 0:
        raise ValueError("weighted degree must be positive for weighted LLY")

    # Cost vector: Δf(x) - Δf(y) = (f(x) - P_w f(x)) - (f(y) - P_w f(y))
    c = np.zeros(n, dtype=float)
    for z in g.neighbors(x):
        w = g[x][z].get("weight", 1.0)
        c[idx[z]] += w / dx
    c[idx[x]] -= 1.0
    for z in g.neighbors(y):
        w = g[y][z].get("weight", 1.0)
        c[idx[z]] -= w / dy
    c[idx[y]] += 1.0

    # Lipschitz constraints: |f(a) - f(b)| <= w_{ab} (unit per weighted distance)
    Aub = []
    bub = []
    for a, b in g.edges():
        w = g[a][b].get("weight", 1.0)
        if w <= 0:
            continue
        row = np.zeros(n)
        row[idx[a]] = 1.0
        row[idx[b]] = -1.0
        Aub.append(row)
        bub.append(w)
        Aub.append(-row)
        bub.append(w)

    # Boundary: f(x) = 0, f(y) = d_xy (shortest path distance)
    Aeq = []
    beq = []
    row = np.zeros(n)
    row[idx[x]] = 1.0
    Aeq.append(row)
    beq.append(0.0)
    row = np.zeros(n)
    row[idx[y]] = 1.0
    Aeq.append(row)
    beq.append(d_xy)

    res = linprog(
        c, A_ub=np.asarray(Aub), b_ub=np.asarray(bub),
        A_eq=np.asarray(Aeq), b_eq=np.asarray(beq),
        bounds=[(None, None)] * n, method="highs",
    )
    if not res.success:
        raise RuntimeError(f"weighted LLY LP failed: {res.message}")
    # Normalize by shortest path distance to get curvature
    return float(res.fun) / d_xy


def integral_lly_deficit(curvatures, kappa0: float = 0.0) -> float:
    values = curvatures.values() if isinstance(curvatures, dict) else curvatures
    return float(sum(max(0.0, float(kappa0) - float(k)) for k in values))


def crosscheck_lly(g: nx.Graph, edges=None, atol: float = 1e-7) -> dict[str, object]:
    target=list(g.edges() if edges is None else edges)
    rows=[]; max_err=0.0
    for u,v in target:
        a=lly_laplacian_lp(g,u,v)
        b=lly_half_idleness(g,u,v)
        err=abs(a-b); max_err=max(max_err,err)
        rows.append({"edge":(int(u),int(v)),"laplacian":a,"half_idleness":b,"abs_error":err})
    return {"ok": bool(max_err <= atol), "max_abs_error":max_err, "rows":rows}
