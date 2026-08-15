from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
import hashlib
import json
import random
import networkx as nx
import torch
from torch import Tensor, nn

from .config import LGAEConfig, config_governance_hash, config_structural_hash, validate_config
from .fibers import FixedWidthFiberLatent, FiberController, FiberStateSnapshot, SOConnectionBank, directed_so_matrices
from .governor import GeometryGovernor
from .metrics import spawn_score_from_pressure
from .mutations import AddEdge, RicciFlowReweight, MutationCooldownTracker, mutation_from_spec, mutation_to_spec
from .operators import actuation_markov_edges, sparse_laplacian_step, actuation_markov_edges_with_slots, sparse_laplacian_step_gauge
from .topology import graphbuffers_to_networkx
from .types import AuditSnapshot, GraphBuffers, MutationDecision, MutationResult
from .version import VERSION as __version__


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
        # Graph is the canonical slot-generation authority. Sync gauge bank
        # generations to match the graph at initialization so there is exactly
        # one source of truth for slot lifecycle identity.
        self._sync_generations()
        self.governor = GeometryGovernor(self.cfg)
        self.cooldowns = MutationCooldownTracker(self.cfg.mutation.edge_cooldown_steps)
        self.step_index = 0
        self.quarantine: list[QuarantineItem] = []
        self.optimizers: list[Any] = []

    def _sync_generations(self) -> None:
        """Copy graph slot_generation into gauge bank (graph is canonical authority)."""
        if self.gauge_connections is not None and self.graph.slot_generation is not None:
            self.gauge_connections.slot_generation.copy_(
                self.graph.slot_generation.to(self.gauge_connections.slot_generation.device)
            )

    def assert_generation_sync(self) -> None:
        """Verify graph and gauge slot generations are identical. Raises on mismatch."""
        if self.gauge_connections is None or self.graph.slot_generation is None:
            return
        g = self.graph.slot_generation.to(self.gauge_connections.slot_generation.device)
        if not bool(torch.equal(g, self.gauge_connections.slot_generation)):
            raise RuntimeError(
                "graph/gauge slot_generation divergence detected: "
                f"graph={g.tolist()}, gauge={self.gauge_connections.slot_generation.tolist()}"
            )

    def authority_hash(self) -> str:
        """Canonical state commitment H(G, g_e, U, F, C_g).

        Combines graph state hash (which now includes slot generations), gauge
        connection hash, fiber state hash, and governance config fingerprint
        into a single SHA-256 commitment. This is the authoritative identity
        that receipts and checkpoints bind to.
        """
        h = hashlib.sha256()
        h.update(self.graph.state_hash().encode())
        h.update(b"|gauge:")
        h.update((self.gauge_connections.state_hash() if self.gauge_connections is not None else "none").encode())
        h.update(b"|fiber:")
        h.update(self.fibers.state_hash().encode())
        h.update(b"|gov:")
        h.update(config_governance_hash(self.cfg).encode())
        return h.hexdigest()

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
        base_gauge_hash = None if self.gauge_connections is None else self.gauge_connections.state_hash()
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
                # Graph is canonical generation authority: sync gauge generations
                # from the graph rather than letting the gauge bank increment
                # independently.
                self.gauge_connections.reset_slots(
                    reset, optimizers=self.optimizers,
                    sync_generation=self.graph.slot_generation,
                )
            self.assert_generation_sync()
            # Bind gauge authority hash into the accepted-mutation receipt metadata.
            result.metadata["base_gauge_hash"] = base_gauge_hash
            result.metadata["authority_hash_after"] = self.authority_hash()
        elif result.decision == MutationDecision.QUARANTINE:
            self.quarantine.append(QuarantineItem(
                kind="graph", result=result, base_graph_version=base_version, base_graph_hash=base_hash,
                base_gauge_hash=base_gauge_hash,
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
            target_field=self.cfg.mutation.ricci_flow_target,
            coupled=self.cfg.mutation.ricci_flow_coupled,
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
                self.gauge_connections.reset_slots(
                    torch.where(old_valid != self.graph.valid)[0],
                    optimizers=self.optimizers,
                    sync_generation=self.graph.slot_generation,
                )
            self.assert_generation_sync()
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
            "schema": "LGAE_V3_CHECKPOINT_V4",
            "version": __version__,
            "version": "3.3.0",
            "config": asdict(self.cfg),
            "config_structural_hash": config_structural_hash(self.cfg),
            "config_governance_hash": config_governance_hash(self.cfg),
            "authority_hash": self.authority_hash(),
            "step_index": int(self.step_index),
            "graph": self.graph.to_state_dict(),
            "model_state": {k: v.detach().cpu() for k, v in self.state_dict().items()},
            "optimizer_state": self._optimizer_state_to_dict(),
            "cooldowns": self.cooldowns.to_state_dict(),
            "quarantine": qrows,
        }

    def _optimizer_state_to_dict(self) -> dict:
        """Serialize registered optimizer state for checkpoint persistence.

        State is indexed by parameter position within the flattened param list
        across all param_groups, so it can be restored into a fresh optimizer
        with the same parameter structure without relying on ``id()``.
        """
        if not self.optimizers:
            return {}
        result: dict[str, Any] = {}
        for i, opt in enumerate(self.optimizers):
            all_params: list[Any] = []
            for pg in opt.param_groups:
                all_params.extend(pg["params"])
            opt_state: dict[str, Any] = {"param_groups": [], "state": []}
            for pg in opt.param_groups:
                opt_state["param_groups"].append({
                    "lr": float(pg.get("lr", 0.0)),
                    "betas": tuple(float(b) for b in pg.get("betas", (0.9, 0.999))),
                    "eps": float(pg.get("eps", 1e-8)),
                    "weight_decay": float(pg.get("weight_decay", 0.0)),
                    "momentum": float(pg.get("momentum", 0.0)),
                })
            for pos, param in enumerate(all_params):
                state = opt.state.get(param)
                if state is None:
                    continue
                state_dict: dict[str, Any] = {"param_pos": pos}
                for key, value in state.items():
                    if isinstance(value, Tensor):
                        state_dict[key] = value.detach().cpu()
                    elif isinstance(value, (int, float)):
                        state_dict[key] = value
                opt_state["state"].append(state_dict)
            result[f"optimizer_{i}"] = opt_state
        return result

    def _restore_optimizer_state(self, opt_state: dict, optimizer: Any) -> None:
        """Restore optimizer state from checkpoint dict using positional matching."""
        if not opt_state or not hasattr(optimizer, "state"):
            return
        all_params: list[Any] = []
        for pg in optimizer.param_groups:
            all_params.extend(pg["params"])
        for stored in opt_state.get("state", []):
            pos = stored.get("param_pos")
            if pos is None or pos >= len(all_params):
                continue
            param = all_params[pos]
            for key, value in stored.items():
                if key == "param_pos":
                    continue
                if isinstance(value, Tensor):
                    optimizer.state[param][key] = value.to(device=param.device, dtype=param.dtype)
                else:
                    optimizer.state[param][key] = value

    def _reset_registered_optimizers(self) -> None:
        """Clear all state for registered optimizers (used when checkpoint lacks optimizer state)."""
        for opt in self.optimizers:
            if hasattr(opt, "state"):
                for param in list(opt.state.keys()):
                    opt.state[param].clear()
                opt.state.clear()

    def save_checkpoint(self, path: str | Path) -> None:
        """Save checkpoint.

        If ``path`` ends with ``.pt``, uses the legacy pickle format (trusted
        local use only). If ``path`` is a directory or ends with ``.ckpt/``,
        uses the safe safetensors + JSON format suitable for untrusted
        interchange.
        """
        p = Path(path)
        if p.suffix == ".pt" or (p.is_file() and not p.is_dir()):
            p.parent.mkdir(parents=True, exist_ok=True)
            torch.save(self.checkpoint_payload(), p)
        else:
            self._save_checkpoint_safe(p)

    def _save_checkpoint_safe(self, dir_path: Path) -> None:
        """Save checkpoint in safetensors + JSON format (no pickle, untrusted-safe)."""
        try:
            from safetensors.torch import save_file
        except ImportError as exc:
            raise ImportError(
                "safetensors is required for safe checkpoint format: pip install safetensors"
            ) from exc
        dir_path.mkdir(parents=True, exist_ok=True)
        payload = self.checkpoint_payload()

        # Collect all tensors into a flat dict for safetensors
        tensors: dict[str, Tensor] = {}
        tensors["graph.src"] = payload["graph"]["src"]
        tensors["graph.dst"] = payload["graph"]["dst"]
        tensors["graph.weight"] = payload["graph"]["weight"]
        tensors["graph.valid"] = payload["graph"]["valid"]
        tensors["graph.valid"] = tensors["graph.valid"].to(torch.uint8)
        if payload["graph"].get("length") is not None:
            tensors["graph.length"] = payload["graph"]["length"]
        if payload["graph"].get("role") is not None:
            tensors["graph.role"] = payload["graph"]["role"]
        if payload["graph"].get("slot_generation") is not None:
            tensors["graph.slot_generation"] = payload["graph"]["slot_generation"]
        for k, v in payload["model_state"].items():
            tensors[f"model.{k}"] = v

        # Serialize optimizer tensor state
        opt_state = payload.get("optimizer_state", {})
        for opt_name, opt_data in opt_state.items():
            for si, st in enumerate(opt_data.get("state", [])):
                for key, value in st.items():
                    if key == "param_id":
                        continue
                    if isinstance(value, Tensor):
                        tensors[f"{opt_name}.state.{si}.{key}"] = value

        # Persist quarantined shadow graph tensors for durable quarantine
        for qi, q in enumerate(payload.get("quarantine", [])):
            sg = q.get("shadow_graph")
            if sg is not None:
                tensors[f"quarantine.{qi}.shadow_graph.src"] = sg["src"]
                tensors[f"quarantine.{qi}.shadow_graph.dst"] = sg["dst"]
                tensors[f"quarantine.{qi}.shadow_graph.weight"] = sg["weight"]
                if sg.get("length") is not None:
                    tensors[f"quarantine.{qi}.shadow_graph.length"] = sg["length"]
                tensors[f"quarantine.{qi}.shadow_graph.valid"] = sg["valid"].to(torch.uint8)
                if sg.get("role") is not None:
                    tensors[f"quarantine.{qi}.shadow_graph.role"] = sg["role"]
                if sg.get("slot_generation") is not None:
                    tensors[f"quarantine.{qi}.shadow_graph.slot_generation"] = sg["slot_generation"]

        save_file(tensors, str(dir_path / "tensors.safetensors"))

        # JSON sidecar with non-tensor data
        graph_json = {
            "num_nodes": payload["graph"]["num_nodes"],
            "version": payload["graph"]["version"],
            "state_hash": payload["graph"]["state_hash"],
            "has_role": payload["graph"].get("role") is not None,
            "has_slot_generation": payload["graph"].get("slot_generation") is not None,
            "has_length": payload["graph"].get("length") is not None,
        }
        controller_json = {
            "step_index": payload["step_index"],
            "cooldowns": payload["cooldowns"],
            "quarantine": [{
                **{k: v for k, v in q.items() if k not in ("shadow_graph", "shadow_fibers")},
                "shadow_graph_meta": None if q.get("shadow_graph") is None else {
                    "num_nodes": q["shadow_graph"]["num_nodes"],
                    "version": q["shadow_graph"]["version"],
                    "state_hash": q["shadow_graph"]["state_hash"],
                    "has_role": q["shadow_graph"].get("role") is not None,
                    "has_slot_generation": q["shadow_graph"].get("slot_generation") is not None,
                    "has_length": q["shadow_graph"].get("length") is not None,
                },
                "shadow_fibers": q.get("shadow_fibers"),
            } for q in payload["quarantine"]],
            "optimizer_meta": {
                name: {"param_groups": od["param_groups"], "state_meta": [
                    {k: v for k, v in s.items() if not isinstance(v, Tensor)} for s in od.get("state", [])
                ]} for name, od in opt_state.items()
            },
        }
        governance_json = {
            "config": payload["config"],
            "config_structural_hash": payload["config_structural_hash"],
            "config_governance_hash": payload["config_governance_hash"],
            "authority_hash": payload["authority_hash"],
        }
        manifest = {
            "schema": "LGAE_V3_SAFE_CHECKPOINT_V2",
            "version": __version__,
            "files": ["tensors.safetensors", "graph.json", "controller.json", "governance.json"],
            "tensor_keys": sorted(tensors.keys()),
        }
        (dir_path / "manifest.json").write_text(json.dumps(manifest, sort_keys=True, indent=2))
        (dir_path / "graph.json").write_text(json.dumps(graph_json, sort_keys=True, indent=2, default=str))
        (dir_path / "controller.json").write_text(json.dumps(controller_json, sort_keys=True, indent=2, default=str))
        (dir_path / "governance.json").write_text(json.dumps(governance_json, sort_keys=True, indent=2, default=str))

    @torch.no_grad()
    def load_checkpoint_(
        self,
        path: str | Path,
        *,
        map_location=None,
        allow_governance_mismatch: bool = False,
        optimizer_load_policy: str = "restore",
    ) -> None:
        """Load checkpoint with authority enforcement.

        Parameters
        ----------
        path
            Path to a ``.pt`` file (legacy pickle, trusted only) or a safe
            checkpoint directory (safetensors + JSON).
        map_location
            Device mapping for tensor restoration.
        allow_governance_mismatch
            If False (default), a governance config mismatch raises
            ``ValueError``. If True, the mismatch is accepted with a warning
            in the returned metadata (explicit migration).
        optimizer_load_policy
            ``"restore"``: restore optimizer state from checkpoint (default).
            ``"reset"``: clear all registered optimizer state.
            ``"reject"``: raise if any optimizer is registered.
        """
        p = Path(path)
        if p.is_dir() or str(path).endswith("/"):
            self._load_checkpoint_safe_(
                p, map_location=map_location,
                allow_governance_mismatch=allow_governance_mismatch,
                optimizer_load_policy=optimizer_load_policy,
            )
        else:
            self._load_checkpoint_legacy_(
                p, map_location=map_location,
                allow_governance_mismatch=allow_governance_mismatch,
                optimizer_load_policy=optimizer_load_policy,
            )

    def _enforce_config_authority(
        self, payload_config: dict, payload_structural: str | None, payload_governance: str | None,
        *, allow_governance_mismatch: bool,
    ) -> None:
        """Enforce structural and governance config fingerprints."""
        from .config import LGAEConfig as _Cfg
        # Reconstruct config from payload for hashing
        saved_cfg = _Cfg()
        from dataclasses import fields as _fields
        for section_name in ("fiber", "operator", "audit", "mutation", "compile"):
            section_data = payload_config.get(section_name, {})
            section = getattr(saved_cfg, section_name)
            for f in _fields(section):
                if f.name in section_data:
                    setattr(section, f.name, section_data[f.name])
        saved_cfg.seed = int(payload_config.get("seed", 0))

        current_structural = config_structural_hash(self.cfg)
        current_governance = config_governance_hash(self.cfg)
        saved_structural = payload_structural or config_structural_hash(saved_cfg)
        saved_governance = payload_governance or config_governance_hash(saved_cfg)

        if saved_structural != current_structural:
            raise ValueError(
                "checkpoint structural config mismatch: checkpoint and engine have "
                "incompatible tensor shapes or dimensions. "
                f"saved={saved_structural[:16]} current={current_structural[:16]}"
            )
        if saved_governance != current_governance and not allow_governance_mismatch:
            raise ValueError(
                "checkpoint governance config mismatch: the audit/mutation policy "
                "differs between checkpoint and engine. Pass "
                "allow_governance_mismatch=True to explicitly migrate. "
                f"saved={saved_governance[:16]} current={current_governance[:16]}"
            )

    def _handle_optimizers_on_load(
        self, payload_optimizer_state: dict | None, *, optimizer_load_policy: str,
    ) -> None:
        """Apply optimizer semantics during checkpoint load."""
        if optimizer_load_policy not in ("restore", "reset", "reject"):
            raise ValueError(f"invalid optimizer_load_policy: {optimizer_load_policy}")
        if optimizer_load_policy == "reject" and self.optimizers:
            raise RuntimeError(
                "checkpoint load rejected because optimizers are registered "
                "(optimizer_load_policy='reject')"
            )
        if not self.optimizers:
            return
        if optimizer_load_policy == "reset":
            self._reset_registered_optimizers()
            return
        # restore
        if payload_optimizer_state:
            for i, opt in enumerate(self.optimizers):
                opt_data = payload_optimizer_state.get(f"optimizer_{i}")
                if opt_data:
                    self._restore_optimizer_state(opt_data, opt)
        else:
            # Checkpoint has no optimizer state. Reset to avoid mixing
            # checkpoint parameters with stale runtime optimizer history.
            self._reset_registered_optimizers()

    def _load_checkpoint_legacy_(
        self, path: Path, *, map_location=None, allow_governance_mismatch: bool, optimizer_load_policy: str,
    ) -> None:
        payload = torch.load(path, map_location=map_location or self.fibers.latent.device, weights_only=False)
        schema = payload.get("schema")
        if schema not in {"LGAE_V3_CHECKPOINT_V2", "LGAE_V3_CHECKPOINT_V3", "LGAE_V3_CHECKPOINT_V4"}:
            raise ValueError("unsupported checkpoint schema")
        # V2/V3 checkpoints predate config authority enforcement; only enforce for V4.
        if schema == "LGAE_V3_CHECKPOINT_V4":
            self._enforce_config_authority(
                payload["config"], payload.get("config_structural_hash"),
                payload.get("config_governance_hash"),
                allow_governance_mismatch=allow_governance_mismatch,
            )
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
        # Sync gauge generations from graph (canonical authority)
        self._sync_generations()
        # Handle optimizer state
        opt_state = payload.get("optimizer_state") if schema == "LGAE_V3_CHECKPOINT_V4" else None
        self._handle_optimizers_on_load(opt_state, optimizer_load_policy=optimizer_load_policy)

    def _load_checkpoint_safe_(
        self, dir_path: Path, *, map_location=None, allow_governance_mismatch: bool, optimizer_load_policy: str,
    ) -> None:
        try:
            from safetensors.torch import load_file as _load_safetensors
        except ImportError as exc:
            raise ImportError(
                "safetensors is required for safe checkpoint format: pip install safetensors"
            ) from exc
        manifest = json.loads((dir_path / "manifest.json").read_text())
        if manifest.get("schema") not in ("LGAE_V3_SAFE_CHECKPOINT_V1", "LGAE_V3_SAFE_CHECKPOINT_V2"):
            raise ValueError("unsupported safe checkpoint schema")
        governance = json.loads((dir_path / "governance.json").read_text())
        self._enforce_config_authority(
            governance["config"], governance.get("config_structural_hash"),
            governance.get("config_governance_hash"),
            allow_governance_mismatch=allow_governance_mismatch,
        )
        graph_meta = json.loads((dir_path / "graph.json").read_text())
        controller = json.loads((dir_path / "controller.json").read_text())
        tensors = _load_safetensors(str(dir_path / "tensors.safetensors"))
        device = map_location or self.fibers.latent.device

        # Reconstruct graph
        graph_payload = {
            "num_nodes": graph_meta["num_nodes"],
            "src": tensors["graph.src"].to(device=device, dtype=torch.long),
            "dst": tensors["graph.dst"].to(device=device, dtype=torch.long),
            "weight": tensors["graph.weight"].to(device=device),
            "valid": tensors["graph.valid"].to(device=device, dtype=torch.bool),
            "length": tensors.get("graph.length"),
            "role": tensors.get("graph.role"),
            "slot_generation": tensors.get("graph.slot_generation"),
            "version": graph_meta["version"],
            "state_hash": graph_meta["state_hash"],
        }
        graph = GraphBuffers.from_state_dict(graph_payload, device=device)
        if graph.num_nodes != self.graph.num_nodes:
            raise ValueError("checkpoint node count does not match engine")
        self.graph = graph

        # Reconstruct model state
        model_state = {}
        prefix = "model."
        for k, v in tensors.items():
            if k.startswith(prefix):
                model_state[k[len(prefix):]] = v.to(device=device)
        self.load_state_dict(model_state, strict=True)
        self.step_index = int(controller["step_index"])
        self.cooldowns = MutationCooldownTracker.from_state_dict(controller.get("cooldowns", {"cooldown_steps": self.cfg.mutation.edge_cooldown_steps}))
        self.quarantine.clear()
        # Reconstruct quarantined items with full shadow graphs from safetensors
        for qi, row in enumerate(controller.get("quarantine", [])):
            sg_meta = row.get("shadow_graph_meta")
            shadow_graph = None
            if sg_meta is not None:
                # Reconstruct shadow graph from safetensors
                sg_prefix = f"quarantine.{qi}.shadow_graph."
                sg_payload = {
                    "num_nodes": sg_meta["num_nodes"],
                    "src": tensors[sg_prefix + "src"].to(device=device, dtype=torch.long),
                    "dst": tensors[sg_prefix + "dst"].to(device=device, dtype=torch.long),
                    "weight": tensors[sg_prefix + "weight"].to(device=device),
                    "valid": tensors[sg_prefix + "valid"].to(device=device, dtype=torch.bool),
                    "length": tensors.get(sg_prefix + "length"),
                    "role": tensors.get(sg_prefix + "role"),
                    "slot_generation": tensors.get(sg_prefix + "slot_generation"),
                    "version": sg_meta["version"],
                    "state_hash": sg_meta["state_hash"],
                }
                shadow_graph = GraphBuffers.from_state_dict(sg_payload, device=device)
            self.quarantine.append(QuarantineItem(
                kind=row["kind"],
                result=_result_from_dict(row["result"]),
                base_graph_version=int(row["base_graph_version"]),
                base_graph_hash=row["base_graph_hash"],
                base_fiber_hash=row.get("base_fiber_hash"),
                base_gauge_hash=row.get("base_gauge_hash"),
                mutation_spec=row.get("mutation_spec"),
                shadow_graph=shadow_graph,
                shadow_fibers=_fiber_snapshot_from_dict(row.get("shadow_fibers"), device=device),
                created_step=int(row.get("created_step", 0)),
            ))
        self._sync_generations()
        # Reconstruct optimizer state from safetensors
        opt_state: dict[str, Any] = {}
        opt_meta = controller.get("optimizer_meta", {})
        for opt_name, opt_data_meta in opt_meta.items():
            opt_state[opt_name] = {"param_groups": opt_data_meta["param_groups"], "state": []}
        # Find optimizer state tensors in the safetensors file
        for opt_name in opt_meta:
            prefix = f"{opt_name}.state."
            state_list: list[dict] = []
            indices: set[int] = set()
            for k in tensors:
                if k.startswith(prefix):
                    parts = k[len(prefix):].split(".", 1)
                    if parts:
                        indices.add(int(parts[0]))
            for si in sorted(indices):
                st: dict[str, Any] = {}
                for k, v in tensors.items():
                    if k.startswith(f"{prefix}{si}."):
                        key = k[len(f"{prefix}{si}."):]
                        st[key] = v.to(device=device)
                # Add non-tensor meta (param_id, step)
                for sm in opt_meta[opt_name].get("state_meta", []):
                    if sm.get("param_id") is not None and not any(
                        k == "param_id" for k in st
                    ):
                        st["param_id"] = sm["param_id"]
                state_list.append(st)
            opt_state[opt_name]["state"] = state_list
        self._handle_optimizers_on_load(opt_state if opt_state else None, optimizer_load_policy=optimizer_load_policy)
