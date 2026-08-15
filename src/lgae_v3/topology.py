from __future__ import annotations

import networkx as nx
import numpy as np
import torch

from .types import GraphBuffers, edge_role_from_code


def graphbuffers_to_networkx(graph: GraphBuffers) -> nx.Graph:
    graph.validate()
    g = nx.Graph()
    g.add_nodes_from(range(graph.num_nodes))
    src, dst, w = graph.active()
    lengths = graph.active_length()
    roles = graph.active_roles()
    for u, v, ww, ell, rr in zip(src.tolist(), dst.tolist(), w.tolist(), lengths.tolist(), roles.tolist()):
        g.add_edge(
            int(u), int(v),
            weight=float(ww),  # affinity/conductance
            length=float(ell),  # metric length
            role=edge_role_from_code(int(rr)).value,
        )
    return g


def topology_signature(g: nx.Graph) -> dict[str, float]:
    c = nx.number_connected_components(g)
    n = g.number_of_nodes()
    e = g.number_of_edges()
    beta1 = e - n + c
    return {"nodes": float(n), "edges": float(e), "beta0": float(c), "beta1": float(beta1)}


def topology_drift(a: dict[str, float], b: dict[str, float]) -> float:
    return float(abs(a.get("beta0", 0) - b.get("beta0", 0)) + abs(a.get("beta1", 0) - b.get("beta1", 0)))


def persistent_homology_signature(z: torch.Tensor, maxdim: int = 1) -> dict[str, float] | None:
    """Persistent-homology summary of the latent cloud; None if ripser is unavailable."""
    try:
        from ripser import ripser
    except Exception:
        return None
    arr = z.detach().cpu().float().numpy()
    dgms = ripser(arr, maxdim=maxdim)["dgms"]
    out: dict[str, float] = {}
    for dim, dgm in enumerate(dgms):
        finite = dgm[np.isfinite(dgm[:, 1])] if len(dgm) else dgm
        persistence = (finite[:, 1] - finite[:, 0]) if len(finite) else np.array([], dtype=float)
        out[f"ph{dim}_count"] = float(len(dgm))
        out[f"ph{dim}_total_persistence"] = float(persistence.sum()) if len(persistence) else 0.0
    return out


def persistent_homology_drift(a: dict[str, float] | None, b: dict[str, float] | None) -> float | None:
    if a is None or b is None:
        return None
    keys = sorted(set(a) | set(b))
    return float(sum(abs(float(a.get(k, 0.0)) - float(b.get(k, 0.0))) for k in keys))


def persistent_homology_diagrams(z: torch.Tensor, maxdim: int = 1) -> list[np.ndarray] | None:
    """Return raw persistence diagrams from ripser; None if unavailable.

    Each diagram is an array of shape (n_points, 2) with (birth, death) pairs.
    """
    try:
        from ripser import ripser
    except Exception:
        return None
    arr = z.detach().cpu().float().numpy()
    dgms = ripser(arr, maxdim=maxdim)["dgms"]
    return [np.asarray(d) for d in dgms]


def bottleneck_distance(dgm_a: np.ndarray, dgm_b: np.ndarray) -> float:
    """Compute the bottleneck distance between two persistence diagrams.

    Uses the Hungarian algorithm to find the optimal matching that
    minimizes the maximum L-infinity distance. Each point may be matched
    to a point in the other diagram or to its projection on the diagonal.
    Falls back to a greedy upper bound if scipy is unavailable.
    """
    def _finite(d: np.ndarray) -> np.ndarray:
        if len(d) == 0:
            return d
        return d[np.isfinite(d[:, 1])]

    a = _finite(dgm_a)
    b = _finite(dgm_b)
    if len(a) == 0 and len(b) == 0:
        return 0.0

    def _to_diag(pts: np.ndarray) -> np.ndarray:
        if len(pts) == 0:
            return np.empty((0, 2))
        mid = (pts[:, 0] + pts[:, 1]) / 2.0
        return np.stack([mid, mid], axis=1)

    def _dist(p: np.ndarray, q: np.ndarray) -> float:
        return float(max(abs(p[0] - q[0]), abs(p[1] - q[1])))

    try:
        from scipy.optimize import linear_sum_assignment
        na, nb = len(a), len(b)
        size = na + nb
        cost = np.zeros((size, size))
        for i in range(na):
            for j in range(nb):
                cost[i, j] = _dist(a[i], b[j])
            cost[i, nb + i] = _dist(a[i], _to_diag(a[i:i+1])[0])
        for j in range(nb):
            cost[na + j, j] = _dist(_to_diag(b[j:j+1])[0], b[j])
        row_ind, col_ind = linear_sum_assignment(cost)
        return float(max(cost[r, c] for r, c in zip(row_ind, col_ind)))
    except ImportError:
        if len(a) == 0:
            return float(max(_dist(_to_diag(b[j:j+1])[0], b[j]) for j in range(len(b))))
        if len(b) == 0:
            return float(max(_dist(a[i], _to_diag(a[i:i+1])[0]) for i in range(len(a))))
        used_b: set[int] = set()
        max_dist = 0.0
        for i in range(len(a)):
            best_j, best_d = -1, float("inf")
            for j in range(len(b)):
                if j in used_b:
                    continue
                d = _dist(a[i], b[j])
                if d < best_d:
                    best_d, best_j = d, j
            diag_d = _dist(a[i], _to_diag(a[i:i+1])[0])
            if best_j >= 0 and best_d <= diag_d:
                max_dist = max(max_dist, best_d)
                used_b.add(best_j)
            else:
                max_dist = max(max_dist, diag_d)
        for j in range(len(b)):
            if j not in used_b:
                max_dist = max(max_dist, _dist(_to_diag(b[j:j+1])[0], b[j]))
        return float(max_dist)


def persistent_homology_bottleneck_drift(
    z_a: torch.Tensor, z_b: torch.Tensor, maxdim: int = 1
) -> float | None:
    """Compute the max bottleneck distance across all PH dimensions.

    Returns None if ripser is unavailable.
    """
    dgms_a = persistent_homology_diagrams(z_a, maxdim=maxdim)
    dgms_b = persistent_homology_diagrams(z_b, maxdim=maxdim)
    if dgms_a is None or dgms_b is None:
        return None
    max_bd = 0.0
    for da, db in zip(dgms_a, dgms_b):
        bd = bottleneck_distance(da, db)
        max_bd = max(max_bd, bd)
    return float(max_bd)
