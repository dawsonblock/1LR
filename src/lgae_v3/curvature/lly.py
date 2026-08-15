from __future__ import annotations

import networkx as nx
import numpy as np
from scipy.optimize import linprog

from .ollivier import ollivier_edge


def lly_half_idleness(g: nx.Graph, u: int, v: int) -> float:
    """Exact LLY via the p=1/2 identity when its graph assumptions apply."""
    return 2.0 * ollivier_edge(g, u, v, p=0.5)


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
