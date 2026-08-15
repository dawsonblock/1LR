from __future__ import annotations

from dataclasses import dataclass
import math
import networkx as nx
import torch
from torch import Tensor

from .config import LGAEConfig
from .curvature import (
    af3_edge,
    degree_weighted_af3_proxy,
    lly_laplacian_lp,
    lly_half_idleness,
    integral_lly_deficit,
    weak_entropic_graph_detailed,
    bakry_emery_curvature,
    sampled_cde_prime_residual,
    multiscale_ollivier_edge,
    normalized_markov_generator,
)
from .metrics import edge_diffusion_metrics
from .fibers import directed_so_matrices
from .operators import (
    actuation_operator,
    actuation_markov_edges,
    diagnostic_diffusion_operator,
    DualOperatorState,
    generator_from_markov,
    sparse_laplacian_step,
    spectral_gap_symmetric,
    spectral_gap_graphbuffers,
    actuation_markov_edges_with_slots,
    sparse_laplacian_step_gauge,
)
from .topology import (
    graphbuffers_to_networkx,
    topology_signature,
    topology_drift,
    persistent_homology_signature,
    persistent_homology_drift,
)
from .types import AuditSnapshot, GraphBuffers, MutationDecision, MutationResult


@dataclass(slots=True)
class FastSignals:
    gamma: Tensor
    radius: Tensor
    local_var: Tensor
    edge_af3: dict[tuple[int, int], float]
    edge_waf3_proxy: dict[tuple[int, int], float]


class GeometryGovernor:
    """Multi-timescale geometry governor with independent actuation/diagnostic operators."""

    def __init__(self, cfg: LGAEConfig) -> None:
        self.cfg = cfg

    def operators(self, graph: GraphBuffers, z: Tensor) -> DualOperatorState:
        pa = actuation_operator(
            graph,
            symmetric=self.cfg.operator.symmetric_actuation,
            self_loop=self.cfg.operator.self_loop,
        )
        pd = diagnostic_diffusion_operator(
            z,
            k=self.cfg.operator.diagnostic_k,
            epsilon_floor=self.cfg.operator.diagnostic_epsilon_floor,
            full_kernel_max_nodes=self.cfg.operator.diagnostic_full_kernel_max_nodes,
        )
        return DualOperatorState(pa, pd)

    def fast_signals(self, graph: GraphBuffers, z: Tensor) -> FastSignals:
        src, dst, pw = actuation_markov_edges(
            graph,
            symmetric=self.cfg.operator.symmetric_actuation,
            self_loop=self.cfg.operator.self_loop,
        )
        m = edge_diffusion_metrics(z, src, dst, pw, graph.num_nodes)
        g = graphbuffers_to_networkx(graph)
        af = {(int(u), int(v)): af3_edge(g, int(u), int(v)) for u, v in g.edges()}
        waf = {(int(u), int(v)): degree_weighted_af3_proxy(g, int(u), int(v)) for u, v in g.edges()}
        return FastSignals(m["gamma"], m["radius"], m["local_var"], af, waf)

    def shadow_rollout(self, graph: GraphBuffers, z: Tensor, gauge_bank=None) -> Tensor:
        out = z.detach().clone()
        steps = int(self.cfg.mutation.shadow_steps)
        if steps <= 0:
            return out
        if gauge_bank is None or self.cfg.fiber.gauge_dim <= 0:
            src, dst, pw = actuation_markov_edges(
                graph, symmetric=self.cfg.operator.symmetric_actuation, self_loop=self.cfg.operator.self_loop,
            )
            for _ in range(steps):
                out = sparse_laplacian_step(
                    out, src, dst, pw, eta=float(self.cfg.mutation.shadow_eta), num_nodes=graph.num_nodes,
                )
                if not bool(torch.isfinite(out).all().item()):
                    raise FloatingPointError("non-finite latent state during shadow rollout")
            return out
        src, dst, pw, slots, reverse = actuation_markov_edges_with_slots(
            graph, symmetric=self.cfg.operator.symmetric_actuation, self_loop=self.cfg.operator.self_loop,
        )
        conn = directed_so_matrices(gauge_bank, slots, reverse)
        for _ in range(steps):
            out = sparse_laplacian_step_gauge(
                out, src, dst, pw, conn, gauge_dim=self.cfg.fiber.gauge_dim,
                eta=float(self.cfg.mutation.shadow_eta), num_nodes=graph.num_nodes,
            )
            if not bool(torch.isfinite(out).all().item()):
                raise FloatingPointError("non-finite latent state during gauge shadow rollout")
        return out

    def audit(self, graph: GraphBuffers, z: Tensor, *, seed: int = 0) -> AuditSnapshot:
        graph.validate()
        if z.ndim != 2 or z.shape[0] != graph.num_nodes:
            raise ValueError("z must have shape [num_nodes, D]")
        if not bool(torch.isfinite(z).all().item()):
            raise ValueError("latent state contains NaN/Inf")

        ops = self.operators(graph, z)
        g = graphbuffers_to_networkx(graph)
        lam, spectral_method = spectral_gap_graphbuffers(
            graph,
            solver=self.cfg.audit.spectral_solver,
            lobpcg_min_nodes=self.cfg.audit.spectral_lobpcg_min_nodes,
            niter=self.cfg.audit.spectral_lobpcg_niter,
            tol=self.cfg.audit.spectral_lobpcg_tol,
            seed=self.cfg.audit.spectral_seed + seed,
        )
        discrepancy = float(ops.discrepancy(self.cfg.operator.operator_discrepancy).item())
        topo = topology_signature(g)
        ph = persistent_homology_signature(z) if self.cfg.audit.persistent_homology_enabled else None
        details: dict = {
            "lly_complete": False,
            "lly_crosscheck_max_error": None,
            "entropic_nodes": 0,
            "entropic_complete": True,
            "entropic_failures": {},
            "bakry_nodes": 0,
            "cde_kind": "sampled_violation",
            "curvature_weight_mode": self.cfg.audit.curvature_weight_mode,
            "diagnostic_support_mode": (
                "full_soft_kernel" if graph.num_nodes <= self.cfg.operator.diagnostic_full_kernel_max_nodes else "topk_support_approximation"
            ),
            "persistent_homology": ph,
            "graph_version": int(graph.version),
            "graph_state_hash": graph.state_hash(),
            "spectral_solver_used": spectral_method,
        }

        edges = list(g.edges())
        edges.sort(key=lambda e: af3_edge(g, *e))

        # Explicit mesoscopic ORC diagnostic on the highest-priority local edges.
        orc_edges = edges[: max(int(self.cfg.audit.orc_top_k), 0)]
        orc_multi: dict[tuple[int, int], dict[int, float]] = {}
        for u, v in orc_edges:
            orc_multi[(int(u), int(v))] = {
                int(r): multiscale_ollivier_edge(
                    g, int(u), int(v), radius=int(r),
                    backend=self.cfg.audit.orc_backend,
                    sinkhorn_epsilon=self.cfg.audit.sinkhorn_epsilon,
                    sinkhorn_max_iter=self.cfg.audit.sinkhorn_max_iter,
                    sinkhorn_tolerance=self.cfg.audit.sinkhorn_tolerance,
                )
                for r in self.cfg.audit.orc_radii
            }
        details["multiscale_orc"] = orc_multi

        max_exact = max(int(self.cfg.audit.exact_lly_top_k), 0)
        target_edges = edges if len(edges) <= max_exact else edges[:max_exact]
        lly: dict[tuple[int, int], float] = {}
        cross_err = 0.0
        role_deficit = 0.0
        for u, v in target_edges:
            a = lly_laplacian_lp(g, int(u), int(v))
            b = lly_half_idleness(g, int(u), int(v))
            lly[(int(u), int(v))] = a
            cross_err = max(cross_err, abs(a - b))
            role = str(g[int(u)][int(v)].get("role", "generic"))
            target = float(self.cfg.audit.role_lly_targets.get(role, self.cfg.audit.role_lly_targets.get("generic", 0.0)))
            role_deficit += max(0.0, target - a)
        details["lly_complete"] = len(target_edges) == len(edges)
        details["lly_crosscheck_max_error"] = cross_err if target_edges else None
        details["lly"] = lly
        details["role_lly_deficit"] = role_deficit if target_edges else None
        deficit = integral_lly_deficit(lly, self.cfg.audit.integral_lly_threshold) if target_edges else None

        fast = self.fast_signals(graph, z)
        order = torch.argsort(fast.gamma, descending=True).tolist()
        ent_nodes = order[: min(self.cfg.audit.entropic_nodes, len(order))]
        ent_detail = weak_entropic_graph_detailed(g, nodes=ent_nodes)
        ent_values = {i: r.value for i, r in ent_detail.items() if r.value is not None}
        ent_fail = {i: {"status": r.status, "message": r.message} for i, r in ent_detail.items() if r.value is None}
        ent_min = min(ent_values.values()) if ent_values else None
        details["entropic_nodes"] = len(ent_detail)
        details["entropic_complete"] = not bool(ent_fail)
        details["entropic_failures"] = ent_fail
        details["entropic"] = ent_values
        details["entropic_status"] = {i: r.status for i, r in ent_detail.items()}

        Q, stationary_measure = normalized_markov_generator(ops.p_diagnostic.to(torch.float64))
        details["bakry_stationary_measure_min"] = float(stationary_measure.min().item())
        details["bakry_generator"] = "reversible_normalized_markov"
        be_nodes = order[: min(self.cfg.audit.bakry_nodes, len(order))]
        be = [bakry_emery_curvature(Q, int(i), dimension=self.cfg.audit.cde_dimension) for i in be_nodes]
        be_min = min(be) if be else None
        details["bakry_nodes"] = len(be)
        details["bakry_values"] = be
        cde_nodes = order[: min(self.cfg.audit.cde_nodes, len(order))]
        cde = (
            sampled_cde_prime_residual(
                Q,
                cde_nodes,
                dimension=self.cfg.audit.cde_dimension,
                samples=self.cfg.audit.cde_samples,
                seed=seed,
            )
            if cde_nodes
            else None
        )
        return AuditSnapshot(
            lambda2=lam,
            operator_discrepancy=discrepancy,
            integral_lly_deficit=deficit,
            weak_entropic_min=ent_min,
            bakry_min=be_min,
            cde_residual=cde,
            topology_signature=topo,
            details=details,
        )

    def _decide_transition(
        self,
        before: AuditSnapshot,
        after: AuditSnapshot,
        *,
        transition_name: str,
        metadata: dict | None = None,
        gauge_bank=None,
    ) -> MutationResult:
        reasons: list[str] = []
        hard_fail = False
        uncertain = False
        a = self.cfg.audit

        if a.min_lambda2 is not None and after.lambda2 < float(a.min_lambda2) - 1e-9:
            reasons.append("spectral_gap_below_min")
            hard_fail = True
        if a.max_operator_discrepancy is not None and after.operator_discrepancy > float(a.max_operator_discrepancy):
            reasons.append("operator_discrepancy_above_max")
            hard_fail = True

        drift = topology_drift(before.topology_signature, after.topology_signature)
        if a.max_topology_drift is not None and drift > float(a.max_topology_drift):
            reasons.append("topology_drift_above_max")
            hard_fail = True
        beta0_inc = int(round(after.topology_signature.get("beta0", 0) - before.topology_signature.get("beta0", 0)))
        if a.preserve_beta0 and beta0_inc > int(a.max_component_increase):
            reasons.append("connected_component_increase")
            hard_fail = True

        if a.max_cde_residual is not None and after.cde_residual is not None and after.cde_residual > float(a.max_cde_residual):
            reasons.append("sampled_cde_residual_above_max")
            hard_fail = True

        if a.entropic_require_success and not bool(after.details.get("entropic_complete", True)):
            reasons.append("weak_entropic_solver_unqualified")
            uncertain = True
        if a.entropic_drop_tolerance is not None:
            if before.weak_entropic_min is None or after.weak_entropic_min is None:
                reasons.append("weak_entropic_comparison_unavailable")
                uncertain = True
            else:
                delta = after.weak_entropic_min - before.weak_entropic_min
                if delta < -float(a.entropic_drop_tolerance):
                    reasons.append("weak_entropic_drop")
                    hard_fail = True

        if after.integral_lly_deficit is not None and a.max_integral_lly_deficit is not None:
            if after.details.get("lly_complete", False):
                if after.integral_lly_deficit > float(a.max_integral_lly_deficit):
                    reasons.append("integral_lly_deficit_above_max")
                    hard_fail = True
            else:
                uncertain = True
                reasons.append("integral_lly_sampled_not_global")
        elif after.integral_lly_deficit is not None and not after.details.get("lly_complete", False):
            # Still disclose that the global deficit was not certified.
            reasons.append("integral_lly_sampled_not_global")
            uncertain = True

        role_def = after.details.get("role_lly_deficit")
        if a.max_role_lly_deficit is not None and role_def is not None:
            if after.details.get("lly_complete", False):
                if float(role_def) > float(a.max_role_lly_deficit):
                    reasons.append("role_conditioned_lly_deficit_above_max")
                    hard_fail = True
            else:
                reasons.append("role_conditioned_lly_sampled_not_global")
                uncertain = True

        cross = after.details.get("lly_crosscheck_max_error")
        if a.require_lly_crosscheck and cross is not None and cross > float(a.max_lly_crosscheck_error):
            reasons.append("lly_exact_paths_disagree")
            hard_fail = True

        ph_drift = persistent_homology_drift(
            before.details.get("persistent_homology"), after.details.get("persistent_homology")
        )
        if a.require_persistent_homology and ph_drift is None:
            reasons.append("persistent_homology_unavailable")
            uncertain = True
        if a.max_ph_drift is not None:
            if ph_drift is None:
                reasons.append("persistent_homology_comparison_unavailable")
                uncertain = True
            elif ph_drift > float(a.max_ph_drift):
                reasons.append("persistent_homology_drift_above_max")
                hard_fail = True

        if hard_fail:
            decision = MutationDecision.REJECT
        elif uncertain and self.cfg.mutation.quarantine_on_uncertainty:
            decision = MutationDecision.QUARANTINE
        else:
            decision = MutationDecision.ACCEPT
        if not reasons:
            reasons = ["all_enabled_constraints_passed"]
        meta = {
            "mutation": transition_name,
            "topology_drift": drift,
            "beta0_increase": beta0_inc,
            "persistent_homology_drift": ph_drift,
            **(metadata or {}),
        }
        return MutationResult(decision, reasons, before=before, after=after, metadata=meta)

    def _local_mutation_gate(self, graph: GraphBuffers, mutation) -> tuple[bool, str | None]:
        """Cheap sub-complex gate before expensive global audits."""
        if not self.cfg.audit.local_disconnect_gate:
            return True, None
        name = getattr(mutation, "name", "")
        if name == "prune_edge" and self.cfg.audit.preserve_beta0:
            u = int(getattr(mutation, "u")); v = int(getattr(mutation, "v"))
            g = graphbuffers_to_networkx(graph)
            if g.has_edge(u, v):
                # Bridge test is O(V+E), far cheaper than curvature/spectral audits.
                if (min(u, v), max(u, v)) in { (min(a,b), max(a,b)) for a,b in nx.bridges(g) }:
                    return False, "local_bridge_prune_would_disconnect"
        return True, None

    def evaluate_mutation(self, graph: GraphBuffers, z: Tensor, mutation, *, seed: int = 0, gauge_bank=None) -> tuple[MutationResult, GraphBuffers]:
        allowed, local_reason = self._local_mutation_gate(graph, mutation)
        if not allowed:
            reasons = [local_reason or "local_mutation_gate_failed"]
            if local_reason == "local_bridge_prune_would_disconnect":
                reasons.append("connected_component_increase")
            return MutationResult(
                MutationDecision.REJECT, reasons,
                metadata={"mutation": getattr(mutation, "name", type(mutation).__name__), "local_gate": True},
            ), graph.clone()
        before = self.audit(graph, z, seed=seed)
        shadow = graph.clone()
        try:
            metadata = mutation.apply(shadow)
            shadow.validate()
            z_shadow = self.shadow_rollout(shadow, z, gauge_bank=gauge_bank)
            after = self.audit(shadow, z_shadow, seed=seed)
        except Exception as exc:
            return (
                MutationResult(
                    MutationDecision.REJECT,
                    [f"mutation_or_shadow_failed:{exc}"],
                    before=before,
                    metadata={"mutation": getattr(mutation, "name", type(mutation).__name__)},
                ),
                shadow,
            )
        metadata = {
            **metadata,
            "base_graph_version": int(graph.version),
            "base_graph_hash": graph.state_hash(),
            "shadow_graph_version": int(shadow.version),
            "shadow_graph_hash": shadow.state_hash(),
            "shadow_steps": int(self.cfg.mutation.shadow_steps),
            "shadow_latent_delta_norm": float(torch.linalg.vector_norm(z_shadow - z).item()),
        }
        result = self._decide_transition(
            before,
            after,
            transition_name=getattr(mutation, "name", type(mutation).__name__),
            metadata=metadata,
        )
        return result, shadow

    def evaluate_latent_transition(
        self,
        graph: GraphBuffers,
        z_before: Tensor,
        z_after: Tensor,
        *,
        name: str = "fiber_mutation",
        seed: int = 0,
        metadata: dict | None = None,
        gauge_bank=None,
    ) -> MutationResult:
        before = self.audit(graph, z_before, seed=seed)
        try:
            z_shadow = self.shadow_rollout(graph, z_after, gauge_bank=gauge_bank)
            after = self.audit(graph, z_shadow, seed=seed)
        except Exception as exc:
            return MutationResult(MutationDecision.REJECT, [f"latent_shadow_failed:{exc}"], before=before, metadata={"mutation": name})
        meta = {
            "base_graph_version": int(graph.version),
            "base_graph_hash": graph.state_hash(),
            "shadow_steps": int(self.cfg.mutation.shadow_steps),
            "direct_latent_delta_norm": float(torch.linalg.vector_norm(z_after - z_before).item()),
            "shadow_latent_delta_norm": float(torch.linalg.vector_norm(z_shadow - z_before).item()),
            **(metadata or {}),
        }
        return self._decide_transition(before, after, transition_name=name, metadata=meta)
