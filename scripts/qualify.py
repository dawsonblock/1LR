from __future__ import annotations

import json
import math
import networkx as nx
import numpy as np
import torch

from lgae_v3 import LGAEConfig, LGAEEngine, SOConnectionBank, make_graph_buffers
from lgae_v3.curvature import (
    crosscheck_lly,
    af3_edge,
    weak_entropic_graph,
    bakry_emery_curvature,
    log_sinkhorn_wasserstein,
    validate_reversible_markov,
)
from lgae_v3.curvature.ollivier import _transport_lp
from lgae_v3.mutations import RicciFlowReweight
from lgae_v3.operators import spectral_gap_graphbuffers


def generator(g):
    n = len(g)
    P = torch.zeros(n, n, dtype=torch.float64)
    for u in g:
        for v in g.neighbors(u):
            P[u, v] = 1.0 / g.degree[u]
    return P - torch.eye(n, dtype=torch.float64)


def _safe_json(x):
    if isinstance(x, dict):
        return {str(k): _safe_json(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_safe_json(v) for v in x]
    if isinstance(x, float) and not math.isfinite(x):
        return "Infinity" if x > 0 else ("-Infinity" if x < 0 else "NaN")
    return x


def main():
    graphs = {
        "K2": nx.path_graph(2),
        "P4": nx.path_graph(4),
        "C4": nx.cycle_graph(4),
        "K3": nx.complete_graph(3),
    }
    report = {"schema": "LGAE_V3_QUALIFICATION_V3", "version": "3.2.0", "graphs": {}, "checks": {}, "pass": True}
    for name, g in graphs.items():
        lly = crosscheck_lly(g)
        Q = generator(g)
        row = {
            "lly": lly,
            "af3": {str(e): af3_edge(g, *e) for e in g.edges()},
            "weak_entropic": weak_entropic_graph(g),
            "bakry": [bakry_emery_curvature(Q, i) for i in g],
        }
        report["graphs"][name] = row
        report["pass"] &= bool(lly["ok"])

    p4_expected = [1.0, 0.2928932188134524, 0.2928932188134524, 1.0]
    p4_actual = report["graphs"]["P4"]["bakry"]
    report["checks"]["bakry_p4_schur_oracle"] = all(abs(a - b) < 1e-8 for a, b in zip(p4_actual, p4_expected))
    report["checks"]["bakry_k2_oracle"] = abs(report["graphs"]["K2"]["bakry"][0] - 2.0) < 1e-8
    report["checks"]["entropic_empty_two_hop_is_inf"] = all(math.isinf(v) for v in report["graphs"]["K3"]["weak_entropic"].values())

    # Log-domain Sinkhorn stress qualification against exact LP.
    C = np.array([[0.0, 1000.0], [1000.0, 0.0]], dtype=float)
    a = np.array([0.999, 0.001], dtype=float); b = np.array([0.001, 0.999], dtype=float)
    exact = _transport_lp(C, a, b)
    approx = log_sinkhorn_wasserstein(C, a, b, epsilon=0.005, max_iter=2000, tolerance=1e-11)
    report["checks"]["sinkhorn_log_small_eps_large_diameter"] = math.isfinite(approx) and abs(approx - exact) <= max(2.0, 0.002 * exact)
    report["sinkhorn_stress"] = {"exact": exact, "approx": approx}

    # Reversible normalized Markov measure.
    P = torch.tensor([[0., 1., 0.], [0.5, 0., 0.5], [0., 1., 0.]], dtype=torch.float64)
    m = validate_reversible_markov(P)
    report["checks"]["reversible_volume_measure"] = bool(torch.allclose(m, torch.tensor([0.25, 0.5, 0.25], dtype=torch.float64), atol=1e-10))

    # SO(d) parameterization is invariant by construction.
    bank = SOConnectionBank(4, 3, parameterization="cayley", dtype=torch.float64)
    with torch.no_grad():
        bank.raw_generators.normal_(0, 0.3)
    orth, det = bank.invariant_error()
    report["checks"]["so_connection_invariants"] = float(orth.detach().max()) < 1e-10 and float(det.detach().max()) < 1e-10

    # Sparse LOBPCG must agree with exact spectral gap on a medium cycle.
    cg = nx.cycle_graph(24)
    gb = make_graph_buffers(24, list(cg.edges()), dtype=torch.float64)
    exact_gap, _ = spectral_gap_graphbuffers(gb, solver="exact")
    lob_gap, method = spectral_gap_graphbuffers(gb, solver="lobpcg", lobpcg_min_nodes=6, niter=200, tol=1e-9, seed=7)
    report["checks"]["lobpcg_matches_exact"] = method == "lobpcg" and abs(lob_gap - exact_gap) <= max(1e-5, 0.002 * exact_gap)
    report["spectral"] = {"exact": exact_gap, "lobpcg": lob_gap, "method": method}

    # Log-Ricci update remains positive under extreme curvature.
    rg = make_graph_buffers(3, [(0, 1, 0.5), (1, 2, 0.5)], capacity=4)
    RicciFlowReweight({(0, 1): 1000.0, (1, 2): -1000.0}, dt=1.0, min_weight=1e-3, max_weight=10.0).apply(rg)
    _, _, rw = rg.active()
    report["checks"]["ricci_flow_weight_positivity"] = bool(torch.isfinite(rw).all().item() and (rw >= 1e-3).all().item() and (rw <= 10.0).all().item())

    # Local bridge guard rejects disconnecting surgery before expensive audit.
    cfg = LGAEConfig(); cfg.fiber.d_base = 2; cfg.fiber.d_max = 4
    eng = LGAEEngine(make_graph_buffers(4, [(0, 1), (1, 2), (2, 3)], capacity=8), cfg)
    from lgae_v3.mutations import PruneEdge
    rr = eng.evaluate_and_maybe_commit(PruneEdge(1, 2))
    report["checks"]["local_bridge_gate"] = rr.decision.value == "reject" and rr.before is None

    report["pass"] &= all(bool(v) for v in report["checks"].values())
    print(json.dumps(_safe_json(report), indent=2, default=str, allow_nan=False))
    return 0 if report["pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
