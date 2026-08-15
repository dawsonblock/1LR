from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import yaml


@dataclass(slots=True)
class FiberConfig:
    d_base: int = 32
    d_max: int = 64
    spawn_width: int = 4
    max_births_per_event: int = 8
    max_deaths_per_event: int = 8
    score_threshold: float = 2.0
    gamma_quantile: float = 0.90
    persistence_steps: int = 3
    birth_gate_logit: float = -4.0
    base_gate_logit: float = 2.0
    min_age_for_death: int = 16
    utility_threshold: float = 1e-4
    ema_decay: float = 0.95
    birth_penalty: float = 1e-4
    gate_l1_penalty: float = 1e-5
    inactive_penalty: float = 1e-6
    govern_mutations: bool = True

    # Optional gauge connection on the first gauge_dim latent coordinates.
    # 0 disables parallel-transport connections and preserves v3.1 behavior.
    gauge_dim: int = 0
    gauge_parameterization: str = "cayley"  # cayley | exp
    gauge_retraction_interval: int = 0


@dataclass(slots=True)
class OperatorConfig:
    diagnostic_k: int = 16
    diagnostic_epsilon_floor: float = 1e-4
    diagnostic_full_kernel_max_nodes: int = 512
    self_loop: float = 0.0
    symmetric_actuation: bool = True
    operator_discrepancy: str = "frobenius"


@dataclass(slots=True)
class AuditConfig:
    local_top_k: int = 32
    orc_top_k: int = 4
    orc_radii: list[int] = field(default_factory=lambda: [1, 2])
    orc_backend: str = "sinkhorn_log"  # sinkhorn_log | exact_lp
    sinkhorn_epsilon: float = 0.05
    sinkhorn_max_iter: int = 200
    sinkhorn_tolerance: float = 1e-6
    exact_lly_top_k: int = 8
    entropic_nodes: int = 16
    bakry_nodes: int = 8
    cde_nodes: int = 4
    cde_samples: int = 64
    cde_dimension: float = 16.0
    integral_lly_threshold: float = 0.0

    # Spectral solver: exact for small graphs, sparse LOBPCG above threshold.
    spectral_solver: str = "auto"  # auto | exact | lobpcg
    spectral_lobpcg_min_nodes: int = 256
    spectral_lobpcg_niter: int = 60
    spectral_lobpcg_tol: float = 1e-6
    spectral_seed: int = 0
    local_disconnect_gate: bool = True

    # Explicit safety semantics. None means monitor-only, not a disguised huge threshold.
    max_integral_lly_deficit: float | None = None
    min_lambda2: float | None = 0.0
    max_operator_discrepancy: float | None = None
    max_topology_drift: float | None = 2.0
    max_cde_residual: float | None = None
    entropic_drop_tolerance: float | None = None
    max_role_lly_deficit: float | None = None
    max_ph_drift: float | None = None

    preserve_beta0: bool = True
    max_component_increase: int = 0
    entropic_require_success: bool = True
    require_lly_crosscheck: bool = True
    max_lly_crosscheck_error: float = 1e-6
    persistent_homology_enabled: bool = True
    require_persistent_homology: bool = False
    curvature_weight_mode: str = "unweighted_reference"
    role_lly_targets: dict[str, float] = field(default_factory=lambda: {
        "generic": 0.0,
        "cluster": 0.0,
        "bridge": -1.0,
        "hierarchy": -0.5,
        "causal": -0.5,
        "memory": 0.0,
    })


@dataclass(slots=True)
class MutationConfig:
    mutation_interval: int = 128
    audit_interval: int = 512
    shadow_steps: int = 2
    shadow_eta: float = 0.01
    max_edge_weight: float = 10.0
    min_edge_weight: float = 1e-3
    edge_add_weight: float = 1.0
    quarantine_on_uncertainty: bool = True
    require_state_hash_match: bool = True

    # Multi-horizon shadow certification (v4.1).
    # When non-empty, a mutation must remain admissible across ALL horizons.
    # When empty, falls back to single-horizon shadow_steps.
    shadow_horizons: list[int] = field(default_factory=list)  # e.g. [1,2,4,8,16]

    # Ricci-flow/surgery hardening.
    ricci_flow_dt: float = 0.05
    ricci_target_curvature: float = 0.0
    ricci_flow_target: str = "weight"  # "weight" (affinity) or "length" (metric)
    ricci_flow_coupled: bool = True  # inverse-update the other scalar
    edge_cooldown_steps: int = 20
    add_curvature_threshold: float = -0.20
    deadband: float = 0.05
    prune_curvature_threshold: float = 0.20


@dataclass(slots=True)
class CompileConfig:
    enabled: bool = False
    dynamic: bool | None = False
    mode: str = "default"
    fullgraph: bool = False
    backend: str = "inductor"
    isolate_recompiles: bool = True
    edge_bucket_size: int = 256


@dataclass(slots=True)
class LGAEConfig:
    seed: int = 0
    fiber: FiberConfig = field(default_factory=FiberConfig)
    operator: OperatorConfig = field(default_factory=OperatorConfig)
    audit: AuditConfig = field(default_factory=AuditConfig)
    mutation: MutationConfig = field(default_factory=MutationConfig)
    compile: CompileConfig = field(default_factory=CompileConfig)


def _update_dataclass(obj: Any, values: Mapping[str, Any]) -> Any:
    allowed = {f.name for f in fields(obj)}
    unknown = set(values) - allowed
    if unknown:
        raise ValueError(f"Unknown configuration keys for {type(obj).__name__}: {sorted(unknown)}")
    for key, value in values.items():
        current = getattr(obj, key)
        if hasattr(current, "__dataclass_fields__") and isinstance(value, Mapping):
            _update_dataclass(current, value)
        else:
            setattr(obj, key, value)
    return obj


def validate_config(cfg: LGAEConfig) -> LGAEConfig:
    if not (0 < cfg.fiber.d_base <= cfg.fiber.d_max):
        raise ValueError("Require 0 < d_base <= d_max")
    if not (0.0 < cfg.fiber.gamma_quantile < 1.0):
        raise ValueError("gamma_quantile must lie in (0,1)")
    if cfg.fiber.spawn_width <= 0:
        raise ValueError("spawn_width must be positive")
    if cfg.fiber.gauge_dim < 0 or cfg.fiber.gauge_dim > cfg.fiber.d_max:
        raise ValueError("gauge_dim must lie in [0,d_max]")
    if cfg.fiber.gauge_parameterization not in {"cayley", "exp"}:
        raise ValueError("gauge_parameterization must be 'cayley' or 'exp'")
    if cfg.fiber.gauge_retraction_interval < 0:
        raise ValueError("gauge_retraction_interval cannot be negative")
    if cfg.operator.diagnostic_k <= 0:
        raise ValueError("diagnostic_k must be positive")
    if cfg.operator.diagnostic_full_kernel_max_nodes < 1:
        raise ValueError("diagnostic_full_kernel_max_nodes must be positive")
    if cfg.audit.curvature_weight_mode not in ("unweighted_reference", "weighted"):
        raise ValueError("curvature_weight_mode must be 'unweighted_reference' or 'weighted'")
    if any(int(r) < 1 for r in cfg.audit.orc_radii):
        raise ValueError("orc_radii must contain positive integers")
    if cfg.audit.orc_backend not in {"sinkhorn_log", "exact_lp"}:
        raise ValueError("orc_backend must be 'sinkhorn_log' or 'exact_lp'")
    if cfg.audit.sinkhorn_epsilon <= 0 or cfg.audit.sinkhorn_max_iter <= 0 or cfg.audit.sinkhorn_tolerance <= 0:
        raise ValueError("invalid Sinkhorn configuration")
    if cfg.audit.spectral_solver not in {"auto", "exact", "lobpcg"}:
        raise ValueError("spectral_solver must be auto, exact, or lobpcg")
    if cfg.audit.spectral_lobpcg_min_nodes < 6:
        raise ValueError("spectral_lobpcg_min_nodes must be at least 6 for k=2 LOBPCG")
    if cfg.audit.spectral_lobpcg_niter <= 0 or cfg.audit.spectral_lobpcg_tol <= 0:
        raise ValueError("invalid LOBPCG configuration")
    if cfg.mutation.shadow_steps < 0:
        raise ValueError("shadow_steps cannot be negative")
    if cfg.mutation.shadow_eta < 0:
        raise ValueError("shadow_eta cannot be negative")
    if any(int(h) < 1 for h in cfg.mutation.shadow_horizons):
        raise ValueError("shadow_horizons must contain positive integers")
    if not (0 < cfg.mutation.min_edge_weight <= cfg.mutation.max_edge_weight):
        raise ValueError("edge weight clamp must be positive and ordered")
    if cfg.mutation.ricci_flow_dt <= 0:
        raise ValueError("ricci_flow_dt must be positive")
    if cfg.mutation.ricci_flow_target not in ("weight", "length"):
        raise ValueError("ricci_flow_target must be 'weight' or 'length'")
    if cfg.mutation.edge_cooldown_steps < 0:
        raise ValueError("edge_cooldown_steps cannot be negative")
    if cfg.mutation.deadband < 0:
        raise ValueError("deadband cannot be negative")
    if not (cfg.mutation.add_curvature_threshold < -cfg.mutation.deadband <= 0 <= cfg.mutation.deadband < cfg.mutation.prune_curvature_threshold):
        raise ValueError("surgery thresholds must define a strict add/deadband/prune separation")
    if cfg.compile.edge_bucket_size <= 0:
        raise ValueError("edge_bucket_size must be positive")
    return cfg


def load_config(path: str | Path | None = None, overrides: Mapping[str, Any] | None = None) -> LGAEConfig:
    cfg = LGAEConfig()
    if path is not None:
        with Path(path).open("r", encoding="utf-8") as f:
            payload = yaml.safe_load(f) or {}
        if not isinstance(payload, Mapping):
            raise ValueError("Config root must be a mapping")
        _update_dataclass(cfg, payload)
    if overrides:
        _update_dataclass(cfg, overrides)
    return validate_config(cfg)


# ---------------------------------------------------------------------------
# Config fingerprinting for checkpoint authority.
#
# The configuration is split into two fingerprints:
#   - structural: dimensions and capacities that define the tensor shapes.
#     A structural mismatch makes a checkpoint physically incompatible.
#   - governance: audit thresholds and mutation policies that define semantics.
#     A governance mismatch changes behavior and requires explicit migration.
# ---------------------------------------------------------------------------

_STRUCTURAL_FIELDS = {
    "fiber": ("d_base", "d_max", "spawn_width", "gauge_dim", "gauge_parameterization"),
    "operator": ("diagnostic_full_kernel_max_nodes",),
    "compile": ("edge_bucket_size",),
}

_GOVERNANCE_FIELDS = {
    "fiber": ("govern_mutations", "birth_penalty", "gate_l1_penalty", "inactive_penalty",
              "score_threshold", "gamma_quantile", "persistence_steps", "min_age_for_death",
              "utility_threshold", "ema_decay", "max_births_per_event", "max_deaths_per_event",
              "birth_gate_logit", "base_gate_logit", "gauge_retraction_interval"),
    "operator": ("diagnostic_k", "diagnostic_epsilon_floor", "self_loop", "symmetric_actuation",
                 "operator_discrepancy"),
    "audit": ("local_top_k", "orc_top_k", "orc_radii", "orc_backend", "sinkhorn_epsilon",
              "sinkhorn_max_iter", "sinkhorn_tolerance", "exact_lly_top_k", "entropic_nodes",
              "bakry_nodes", "cde_nodes", "cde_samples", "cde_dimension", "integral_lly_threshold",
              "spectral_solver", "spectral_lobpcg_min_nodes", "spectral_lobpcg_niter",
              "spectral_lobpcg_tol", "spectral_seed", "local_disconnect_gate",
              "max_integral_lly_deficit", "min_lambda2", "max_operator_discrepancy",
              "max_topology_drift", "max_cde_residual", "entropic_drop_tolerance",
              "max_role_lly_deficit", "max_ph_drift", "preserve_beta0", "max_component_increase",
              "entropic_require_success", "require_lly_crosscheck", "max_lly_crosscheck_error",
              "persistent_homology_enabled", "require_persistent_homology",
              "curvature_weight_mode", "role_lly_targets"),
    "mutation": ("mutation_interval", "audit_interval", "shadow_steps", "shadow_eta",
                 "max_edge_weight", "min_edge_weight", "edge_add_weight",
                 "quarantine_on_uncertainty", "require_state_hash_match",
                 "ricci_flow_dt", "ricci_target_curvature", "edge_cooldown_steps",
                 "add_curvature_threshold", "deadband", "prune_curvature_threshold"),
    "compile": ("enabled", "dynamic", "mode", "fullgraph", "backend", "isolate_recompiles"),
}


def _config_fingerprint(cfg: LGAEConfig, sections: dict[str, tuple[str, ...]]) -> str:
    """Compute a SHA-256 fingerprint over selected config fields."""
    payload: dict[str, Any] = {"seed": int(cfg.seed)}
    for section_name, field_names in sections.items():
        section = getattr(cfg, section_name)
        section_dict = asdict(section)
        payload[section_name] = {k: section_dict[k] for k in field_names if k in section_dict}
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def config_structural_hash(cfg: LGAEConfig) -> str:
    """Fingerprint over structural config fields (dimensions, capacities, shapes)."""
    return _config_fingerprint(cfg, _STRUCTURAL_FIELDS)


def config_governance_hash(cfg: LGAEConfig) -> str:
    """Fingerprint over governance config fields (audit thresholds, mutation policies)."""
    return _config_fingerprint(cfg, _GOVERNANCE_FIELDS)
