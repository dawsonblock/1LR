from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import random
import networkx as nx
import torch
from torch import Tensor, nn

from .config import LGAEConfig, validate_config
from .fibers import FixedWidthFiberLatent, FiberController, FiberStateSnapshot, SOConnectionBank, directed_so_matrices
from .governor import GeometryGovernor
from .metrics import spawn_score_from_pressure
from .mutations import AddEdge, RicciFlowReweight, MutationCooldownTracker, mutation_from_spec, mutation_to_spec
from .operators import actuation_markov_edges, sparse_laplacian_step, actuation_markov_edges_with_slots, sparse_laplacian_step_gauge
from .topology import graphbuffers_to_networkx
from .types import AuditSnapshot, GraphBuffers, MutationDecision, MutationResult


@dataclass(slots=True)
class QuarantineItem:
    kind: str
    result: MutationResult
    base_graph_version: int
    base_graph_hash: str
    base_fiber_hash: str | None = None
    base_gauge_hash: str | None = None
    mutation_spec: dict | None = None
    shadow_graph: GraphBuffers | None = None
    shadow_fibers: FiberStateSnapshot | None = None
    created_step: int = 0


def _audit_to_dict(x: AuditSnapshot | None):
    if x is None:
        return None
    return {
        "lambda2": x.lambda2,
        "operator_discrepancy": x.operator_discrepancy,
        "integral_lly_deficit": x.integral_lly_deficit,
        "weak_entropic_min": x.weak_entropic_min,
        "bakry_min": x.bakry_min,
        "cde_residual": x.cde_residual,
        "topology_signature": x.topology_signature,
        "details": x.details,
    }


def _audit_from_dict(d):
    return None if d is None else AuditSnapshot(**d)


def _result_to_dict(r: MutationResult) -> dict:
    return {
        "decision": r.decision.value,
        "reasons": list(r.reasons),
        "before": _audit_to_dict(r.before),
        "after": _audit_to_dict(r.after),
        "metadata": dict(r.metadata),
    }


def _result_from_dict(d: dict) -> MutationResult:
    return MutationResult(
        MutationDecision(d["decision"]),
        list(d["reasons"]),
        before=_audit_from_dict(d.get("before")),
        after=_audit_from_dict(d.get("after")),
        metadata=dict(d.get("metadata", {})),
    )


def _fiber_snapshot_to_dict(s: FiberStateSnapshot | None):
    if s is None:
        return None
    return {name: getattr(s, name).detach().cpu() for name in (
        "latent", "gate_logits", "active_mask", "age", "utility_ema", "spawn_counter", "gamma_ema"
    )}


def _fiber_snapshot_from_dict(d, *, device=None) -> FiberStateSnapshot | None:
    if d is None:
        return None
    return FiberStateSnapshot(*(torch.as_tensor(d[name], device=device).clone() for name in (
        "latent", "gate_logits", "active_mask", "age", "utility_ema", "spawn_counter", "gamma_ema"
    )))


class LGAEEngine(nn.Module):
    """LGAE-v3.2 multi-timescale geometric controller.

    Numerical field dynamics use sparse row-stochastic edge reductions. Discrete graph and
    fiber changes are shadow-evaluated by the same governor before commit. Quarantined shadows
    are version/hash bound so stale external approval cannot overwrite newer authoritative state.
    """

    def __init__(self, graph: GraphBuffers, cfg: LGAEConfig | None = None, *, device=None, dtype=torch.float32) -> None:
        super().__init__()
        self.cfg = validate_config(cfg or LGAEConfig())
        torch.manual_seed(self.cfg.seed)
        random.seed(self.cfg.seed)
        graph.validate()
        self.graph = graph
        self.fibers = FixedWidthFiberLatent(graph.num_nodes, self.cfg.fiber, device=device or graph.weight.device, dtype=dtype)
        self.fiber_controller = FiberController(self.fibers)
        self.gauge_connections = (
            SOConnectionBank(
                graph.capacity, self.cfg.fiber.gauge_dim,
                parameterization=self.cfg.fiber.gauge_parameterization,
                device=device or graph.weight.device, dtype=dtype,
            ) if self.cfg.fiber.gauge_dim > 0 else None
        )
        self.governor = GeometryGovernor(self.cfg)
        self.cooldowns = MutationCooldownTracker(self.cfg.mutation.edge_cooldown_steps)
        self.step_index = 0
        self.quarantine: list[QuarantineItem] = []
        self.optimizers: list[Any] = []

    def register_optimizer(self, optimizer: Any) -> None:
        """Register an optimizer so slot resets also clear optimizer moment slices."""
        if optimizer is not None and optimizer not in self.optimizers:
            self.optimizers.append(optimizer)

    def unregister_optimizer(self, optimizer: Any) -> None:
        if optimizer in self.optimizers:
            self.optimizers.remove(optimizer)

    def forward(self) -> Tensor:
        return self.fibers()

    @torch.no_grad()
    def diffuse_(self, eta: float = 0.01, noise: float = 0.0) -> Tensor:
        z = self.fibers()
        if self.gauge_connections is None:
            src, dst, pw = actuation_markov_edges(
                self.graph, symmetric=self.cfg.operator.symmetric_actuation, self_loop=self.cfg.operator.self_loop,
            )
            proposal = sparse_laplacian_step(z, src, dst, pw, eta=float(eta), num_nodes=self.graph.num_nodes)
        else:
            src, dst, pw, slots, reverse = actuation_markov_edges_with_slots(
                self.graph, symmetric=self.cfg.operator.symmetric_actuation, self_loop=self.cfg.operator.self_loop,
            )
            conn = directed_so_matrices(self.gauge_connections, slots, reverse)
            proposal = sparse_laplacian_step_gauge(
                z, src, dst, pw, conn, gauge_dim=self.cfg.fiber.gauge_dim,
                eta=float(eta), num_nodes=self.graph.num_nodes,
            )
        if noise:
            proposal = proposal + float(noise) * torch.randn_like(proposal)
        if not bool(torch.isfinite(proposal).all().item()):
            raise FloatingPointError("diffusion produced non-finite latent state")
        mask = self.fibers.effective_mask()
        active = self.fibers.active_mask
        recovered = proposal / mask.clamp_min(1e-4)
        # Inactive channels are storage capacity, not hidden state. Keeping them zero
        # prevents large latent values from accumulating behind a zero active mask.
        self.fibers.latent.copy_(torch.where(active, recovered, torch.zeros_like(recovered)))
        self.step_index += 1
        if self.gauge_connections is not None and self.cfg.fiber.gauge_retraction_interval > 0:
            if self.step_index % self.cfg.fiber.gauge_retraction_interval == 0:
                self.gauge_connections.retract_raw_()
        return self.fibers()

    @torch.no_grad()
    def fiber_tick(self, residual: Tensor | None = None, uncertainty: Tensor | None = None) -> dict:
        z_before = self.fibers().detach().clone()
        sig = self.governor.fast_signals(self.graph, z_before)
        node_b = torch.zeros(self.graph.num_nodes, device=z_before.device, dtype=z_before.dtype)
        for (u, v), k in sig.edge_waf3_proxy.items():
            pressure = max(0.0, -float(k))
            p = node_b.new_tensor(pressure)
            node_b[u] = torch.maximum(node_b[u], p)
            node_b[v] = torch.maximum(node_b[v], p)
        res = torch.zeros(self.graph.num_nodes, device=z_before.device, dtype=z_before.dtype) if residual is None else (
            residual.square().mean(-1) if residual.ndim > 1 else residual.abs()
        )
        unc = torch.zeros_like(res) if uncertainty is None else uncertainty.to(res)
        score = spawn_score_from_pressure(
            sig.gamma, sig.radius, sig.local_var, node_b, res, unc, self.fibers.capacity.to(z_before.dtype)
        )
        self.fiber_controller.update_gamma_ema(sig.gamma)
        persistent = self.fiber_controller.persistent_candidates(score, sig.gamma)
        nodes = self.fiber_controller.select_birth_nodes(persistent, score)

        # Monitoring state and age are allowed to progress before the transactional mutation.
        self.fiber_controller.age()
        base_snapshot = self.fibers.snapshot()
        base_fiber_hash = self.fibers.state_hash()
        init = self.fiber_controller.residual_scalar_initialization(residual if residual is not None else z_before, nodes)
        births = self.fiber_controller.activate(nodes, init)
        deaths = self.fiber_controller.prune()
        changed = births.count + deaths.count > 0
        result = None

        if changed and self.cfg.fiber.govern_mutations:
            shadow_snapshot = self.fibers.snapshot()
            z_after = self.fibers().detach().clone()
            result = self.governor.evaluate_latent_transition(
                self.graph,
                z_before,
                z_after,
                name="fiber_birth_death",
                seed=self.cfg.seed + self.step_index,
                metadata={"birth_count": births.count, "death_count": deaths.count},
                gauge_bank=self.gauge_connections,
            )
            if result.decision == MutationDecision.REJECT:
                self.fibers.restore(base_snapshot)
            elif result.decision == MutationDecision.QUARANTINE:
                self.fibers.restore(base_snapshot)
                self.quarantine.append(QuarantineItem(
                    kind="fiber",
                    result=result,
                    base_graph_version=int(self.graph.version),
                    base_graph_hash=self.graph.state_hash(),
                    base_fiber_hash=base_fiber_hash,
                    base_gauge_hash=None if self.gauge_connections is None else self.gauge_connections.state_hash(),
                    shadow_fibers=shadow_snapshot,
                    created_step=int(self.step_index),
                ))
        return {
            "score": score,
            "gamma": sig.gamma,
            "bottleneck_pressure": node_b,
            "births": births,
            "deaths": deaths,
            "decision": None if result is None else result.decision,
            "reasons": [] if result is None else result.reasons,
            "capacity": self.fibers.capacity.clone(),
        }

    @torch.no_grad()
    def propose_midpoint_edge(self, node: int | None = None) -> AddEdge | None:
        z = self.fibers()
        sig = self.governor.fast_signals(self.graph, z)
        if node is None:
            node = int(torch.argmax(sig.gamma).item())
        g = graphbuffers_to_networkx(self.graph)
        dist = nx.single_source_shortest_path_length(g, node, cutoff=2)
        targets = [v for v, d in dist.items() if d == 2]
        if not targets:
            return None
        target = max(targets, key=lambda v: float(torch.linalg.vector_norm(z[node] - z[v]).item()))
        return AddEdge(node, int(target), self.cfg.mutation.edge_add_weight)

    @torch.no_grad()
    def evaluate_and_maybe_commit(self, mutation) -> MutationResult:
        allowed, blocked = self.cooldowns.allows(mutation, self.step_index)
        if not allowed:
            return MutationResult(
                MutationDecision.REJECT, ["edge_mutation_cooldown"],
                metadata={"mutation": getattr(mutation, "name", type(mutation).__name__), "blocked_edges": {str(k): v for k, v in blocked.items()}},
            )
        base_hash = self.graph.state_hash()
        base_version = int(self.graph.version)
        result, shadow = self.governor.evaluate_mutation(
            self.graph, self.fibers().detach(), mutation, seed=self.cfg.seed + self.step_index,
            gauge_bank=self.gauge_connections,
        )
        if result.decision == MutationDecision.ACCEPT:
            old_valid = self.graph.valid.clone()
            self.graph = shadow
            self.cooldowns.record(mutation, self.step_index)
            if self.gauge_connections is not None:
                reset = torch.where(old_valid != self.graph.valid)[0]
                self.gauge_connections.reset_slots(reset, optimizers=self.optimizers)
        elif result.decision == MutationDecision.QUARANTINE:
            self.quarantine.append(QuarantineItem(
                kind="graph", result=result, base_graph_version=base_version, base_graph_hash=base_hash,
                base_gauge_hash=None if self.gauge_connections is None else self.gauge_connections.state_hash(),
                mutation_spec=mutation_to_spec(mutation), shadow_graph=shadow, created_step=int(self.step_index),
            ))
        return result

    def propose_ricci_flow(self, curvatures: dict[tuple[int, int], float], *, target_curvature: float | None = None) -> RicciFlowReweight:
        return RicciFlowReweight(
            curvatures=curvatures,
            target_curvature=self.cfg.mutation.ricci_target_curvature if target_curvature is None else float(target_curvature),
            dt=self.cfg.mutation.ricci_flow_dt,
            min_weight=self.cfg.mutation.min_edge_weight,
            max_weight=self.cfg.mutation.max_edge_weight,
        )

    @torch.no_grad()
    def resolve_quarantine(self, index: int = 0, *, accept: bool = False) -> MutationResult:
        item = self.quarantine.pop(index)
        if not accept:
            item.result.metadata["quarantine_resolution"] = "rejected_by_external_authority"
            return item.result

        stale_graph = (
            int(self.graph.version) != int(item.base_graph_version)
            or self.graph.state_hash() != item.base_graph_hash
        )
        stale_fiber = item.base_fiber_hash is not None and self.fibers.state_hash() != item.base_fiber_hash
        stale_gauge = item.base_gauge_hash is not None and (self.gauge_connections is None or self.gauge_connections.state_hash() != item.base_gauge_hash)
        if self.cfg.mutation.require_state_hash_match and (stale_graph or stale_fiber or stale_gauge):
            return MutationResult(
                MutationDecision.REJECT,
                ["stale_quarantine_base"],
                before=item.result.before,
                after=item.result.after,
                metadata={**item.result.metadata, "quarantine_resolution": "rejected_stale", "stale_graph": stale_graph, "stale_fiber": stale_fiber, "stale_gauge": stale_gauge},
            )

        if item.kind == "graph":
            if item.shadow_graph is None:
                raise RuntimeError("corrupt graph quarantine item")
            item.shadow_graph.validate()
            old_valid = self.graph.valid.clone()
            self.graph = item.shadow_graph
            if item.mutation_spec is not None:
                mutation = mutation_from_spec(item.mutation_spec)
                self.cooldowns.record(mutation, self.step_index)
            if self.gauge_connections is not None:
                self.gauge_connections.reset_slots(torch.where(old_valid != self.graph.valid)[0], optimizers=self.optimizers)
        elif item.kind == "fiber":
            if item.shadow_fibers is None:
                raise RuntimeError("corrupt fiber quarantine item")
            self.fibers.restore(item.shadow_fibers)
        else:
            raise RuntimeError(f"unknown quarantine kind: {item.kind}")
        item.result.metadata["quarantine_resolution"] = "accepted_by_external_authority"
        return item.result

    @torch.no_grad()
    def audit(self):
        return self.governor.audit(self.graph, self.fibers().detach(), seed=self.cfg.seed + self.step_index)

    def checkpoint_payload(self) -> dict:
        qrows = []
        for q in self.quarantine:
            qrows.append({
                "kind": q.kind,
                "result": _result_to_dict(q.result),
                "base_graph_version": q.base_graph_version,
                "base_graph_hash": q.base_graph_hash,
                "base_fiber_hash": q.base_fiber_hash,
                "base_gauge_hash": q.base_gauge_hash,
                "mutation_spec": q.mutation_spec,
                "shadow_graph": None if q.shadow_graph is None else q.shadow_graph.to_state_dict(),
                "shadow_fibers": _fiber_snapshot_to_dict(q.shadow_fibers),
                "created_step": q.created_step,
            })
        return {
            "schema": "LGAE_V3_CHECKPOINT_V3",
            "version": "3.2.0",
            "config": asdict(self.cfg),
            "step_index": int(self.step_index),
            "graph": self.graph.to_state_dict(),
            "model_state": {k: v.detach().cpu() for k, v in self.state_dict().items()},
            "cooldowns": self.cooldowns.to_state_dict(),
            "quarantine": qrows,
        }

    def save_checkpoint(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.checkpoint_payload(), p)

    @torch.no_grad()
    def load_checkpoint_(self, path: str | Path, *, map_location=None) -> None:
        payload = torch.load(path, map_location=map_location or self.fibers.latent.device, weights_only=False)
        if payload.get("schema") not in {"LGAE_V3_CHECKPOINT_V2", "LGAE_V3_CHECKPOINT_V3"}:
            raise ValueError("unsupported checkpoint schema")
        graph = GraphBuffers.from_state_dict(payload["graph"], device=self.fibers.latent.device)
        if graph.num_nodes != self.graph.num_nodes:
            raise ValueError("checkpoint node count does not match engine")
        self.graph = graph
        self.load_state_dict(payload["model_state"], strict=True)
        self.step_index = int(payload["step_index"])
        self.cooldowns = MutationCooldownTracker.from_state_dict(payload.get("cooldowns", {"cooldown_steps": self.cfg.mutation.edge_cooldown_steps}))
        self.quarantine.clear()
        for row in payload.get("quarantine", []):
            self.quarantine.append(QuarantineItem(
                kind=row["kind"],
                result=_result_from_dict(row["result"]),
                base_graph_version=int(row["base_graph_version"]),
                base_graph_hash=row["base_graph_hash"],
                base_fiber_hash=row.get("base_fiber_hash"),
                base_gauge_hash=row.get("base_gauge_hash"),
                mutation_spec=row.get("mutation_spec"),
                shadow_graph=None if row.get("shadow_graph") is None else GraphBuffers.from_state_dict(row["shadow_graph"], device=self.fibers.latent.device),
                shadow_fibers=_fiber_snapshot_from_dict(row.get("shadow_fibers"), device=self.fibers.latent.device),
                created_step=int(row.get("created_step", 0)),
            ))
