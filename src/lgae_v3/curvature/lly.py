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
    """Weighted limit-free LLY with metric-measure separation.

    Uses the **affinity-based** normalized Laplacian:
        Δ f(x) = Σ_y P_a(x,y) [f(y) - f(x)]
    where P_a(x,y) = a_{xy} / Σ_j a_{xj}.

    The Lipschitz constraint uses **metric length**:
        |f(a) - f(b)| ≤ ℓ_{ab}  for each edge (a,b)

    The boundary condition uses the shortest-path metric:
        f(x) = 0, f(y) = d_ℓ(x,y)

    This cleanly separates the Laplacian (from affinity) from the Lipschitz
    metric (from length), matching the Bai–Huang–Lu–Yau formulation where
    the metric d and the transition rule P are independent.

    κ_LLY(x,y) = inf[Δf(x) - Δf(y)] / d_ℓ(x,y)
    """
    if not g.has_edge(x, y):
        raise ValueError("weighted LLY currently expects adjacent vertices")

    # Ensure all edges have a length attribute (default: 1/weight)
    for a, b in g.edges():
        if "length" not in g[a][b]:
            w = g[a][b].get("weight", 1.0)
            g[a][b]["length"] = 1.0 / w if w > 0 else 1.0

    # Shortest-path distance in the metric d_ℓ
    d_xy = float(nx.dijkstra_path_length(g, x, y, weight="length"))

    nodes = list(g.nodes())
    idx = {n: i for i, n in enumerate(nodes)}
    n = len(nodes)

    # Affinity-based weighted degree: sum of edge affinities
    def adeg(node):
        return float(sum(g[node][z].get("weight", 1.0) for z in g.neighbors(node)))

    dx = adeg(x)
    dy = adeg(y)
    if dx <= 0 or dy <= 0:
        raise ValueError("affinity degree must be positive for weighted LLY")

    # Cost vector: Δf(x) - Δf(y) using affinity-based Laplacian
    # P_a f(x) = Σ_z a_{xz}/dx * f(z)
    # Δf(x) = f(x) - P_a f(x)
    c = np.zeros(n, dtype=float)
    for z in g.neighbors(x):
        a = g[x][z].get("weight", 1.0)  # affinity
        c[idx[z]] += a / dx
    c[idx[x]] -= 1.0
    for z in g.neighbors(y):
        a = g[y][z].get("weight", 1.0)  # affinity
        c[idx[z]] -= a / dy
    c[idx[y]] += 1.0

    # Lipschitz constraints: |f(a) - f(b)| ≤ ℓ_{ab} (metric length)
    Aub = []
    bub = []
    for a, b in g.edges():
        ell = g[a][b].get("length", 1.0 / g[a][b].get("weight", 1.0))
        if ell <= 0:
            continue
        row = np.zeros(n)
        row[idx[a]] = 1.0
        row[idx[b]] = -1.0
        Aub.append(row)
        bub.append(ell)
        Aub.append(-row)
        bub.append(ell)

    # Boundary: f(x) = 0, f(y) = d_ℓ(x,y) (shortest path in metric)
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
