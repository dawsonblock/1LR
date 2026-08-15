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
