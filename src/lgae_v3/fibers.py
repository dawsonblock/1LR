from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import hashlib

import torch
from torch import Tensor, nn

from .config import FiberConfig


@dataclass(slots=True)
class FiberStateSnapshot:
    latent: Tensor
    gate_logits: Tensor
    active_mask: Tensor
    age: Tensor
    utility_ema: Tensor
    spawn_counter: Tensor
    gamma_ema: Tensor

    def clone(self) -> "FiberStateSnapshot":
        return FiberStateSnapshot(*(x.clone() for x in (
            self.latent, self.gate_logits, self.active_mask, self.age,
            self.utility_ema, self.spawn_counter, self.gamma_ema,
        )))


class FixedWidthFiberLatent(nn.Module):
    """Static [N,D_max] latent with dynamic per-node active capacity."""

    def __init__(
        self,
        num_nodes: int,
        cfg: FiberConfig,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        self.num_nodes = int(num_nodes)
        self.cfg = cfg
        if not (0 < cfg.d_base <= cfg.d_max):
            raise ValueError("Require 0 < d_base <= d_max")
        self.latent = nn.Parameter(0.02 * torch.randn(num_nodes, cfg.d_max, device=device, dtype=dtype))
        active = torch.zeros(num_nodes, cfg.d_max, dtype=torch.bool, device=device)
        active[:, : cfg.d_base] = True
        self.register_buffer("active_mask", active)
        gate = torch.full((num_nodes, cfg.d_max), cfg.birth_gate_logit, device=device, dtype=dtype)
        gate[:, : cfg.d_base] = cfg.base_gate_logit
        self.gate_logits = nn.Parameter(gate)
        self.register_buffer("age", torch.zeros(num_nodes, cfg.d_max, device=device, dtype=dtype))
        self.register_buffer("utility_ema", torch.zeros(num_nodes, cfg.d_max, device=device, dtype=dtype))
        self.register_buffer("spawn_counter", torch.zeros(num_nodes, dtype=torch.int64, device=device))
        self.register_buffer("gamma_ema", torch.zeros(num_nodes, device=device, dtype=dtype))

    def effective_mask(self) -> Tensor:
        return self.active_mask.to(self.latent.dtype) * torch.sigmoid(self.gate_logits)

    def forward(self) -> Tensor:
        return self.latent * self.effective_mask()

    @property
    def capacity(self) -> Tensor:
        return self.active_mask.sum(dim=-1)

    def regularization(self) -> dict[str, Tensor]:
        active = self.active_mask.to(self.latent.dtype)
        gates = torch.sigmoid(self.gate_logits)
        birth = active[:, self.cfg.d_base :].sum()
        gate_l1 = (active * gates).sum()
        inactive_energy = ((1.0 - active) * self.latent.square()).mean()
        loss = self.cfg.birth_penalty * birth + self.cfg.gate_l1_penalty * gate_l1 + self.cfg.inactive_penalty * inactive_energy
        return {"fiber_loss": loss, "birth_cost": birth, "gate_l1": gate_l1, "inactive_energy": inactive_energy}

    @torch.no_grad()
    def snapshot(self) -> FiberStateSnapshot:
        return FiberStateSnapshot(
            self.latent.detach().clone(),
            self.gate_logits.detach().clone(),
            self.active_mask.clone(),
            self.age.clone(),
            self.utility_ema.clone(),
            self.spawn_counter.clone(),
            self.gamma_ema.clone(),
        )

    @torch.no_grad()
    def restore(self, snap: FiberStateSnapshot) -> None:
        for name, target, source in (
            ("latent", self.latent, snap.latent),
            ("gate_logits", self.gate_logits, snap.gate_logits),
            ("active_mask", self.active_mask, snap.active_mask),
            ("age", self.age, snap.age),
            ("utility_ema", self.utility_ema, snap.utility_ema),
            ("spawn_counter", self.spawn_counter, snap.spawn_counter),
            ("gamma_ema", self.gamma_ema, snap.gamma_ema),
        ):
            if target.shape != source.shape:
                raise ValueError(f"fiber snapshot shape mismatch for {name}")
            target.copy_(source.to(device=target.device, dtype=target.dtype))

    def state_hash(self) -> str:
        h = hashlib.sha256()
        for tensor in (
            self.latent.detach(), self.gate_logits.detach(), self.active_mask,
            self.age, self.utility_ema, self.spawn_counter, self.gamma_ema,
        ):
            x = tensor.detach().cpu().contiguous()
            h.update(str(x.dtype).encode())
            h.update(str(tuple(x.shape)).encode())
            h.update(x.view(torch.uint8).numpy().tobytes())
        return h.hexdigest()


@dataclass(slots=True)
class FiberEvent:
    nodes: Tensor
    channels: Tensor

    @property
    def count(self) -> int:
        return int(self.nodes.numel())


class FiberController:
    def __init__(self, module: FixedWidthFiberLatent) -> None:
        self.module = module
        self.cfg = module.cfg

    @torch.no_grad()
    def update_utility(self, latent_grad: Optional[Tensor]) -> None:
        if latent_grad is None:
            utility = self.module.latent.detach().abs()
        else:
            utility = (latent_grad.detach() * self.module.latent.detach()).abs()
        denom = utility.mean(dim=-1, keepdim=True).clamp_min(1e-8)
        utility = utility / denom
        utility = utility * self.module.active_mask.to(utility.dtype)
        d = self.cfg.ema_decay
        self.module.utility_ema.mul_(d).add_(utility, alpha=1.0 - d)

    @torch.no_grad()
    def update_gamma_ema(self, gamma: Tensor) -> None:
        d = self.cfg.ema_decay
        self.module.gamma_ema.mul_(d).add_(gamma.detach(), alpha=1.0 - d)

    @torch.no_grad()
    def persistent_candidates(self, score: Tensor, gamma: Tensor) -> Tensor:
        threshold = torch.quantile(gamma.detach(), self.cfg.gamma_quantile)
        candidate = (score > self.cfg.score_threshold) & (gamma > threshold)
        self.module.spawn_counter.copy_(
            torch.where(candidate, self.module.spawn_counter + 1, torch.zeros_like(self.module.spawn_counter))
        )
        return self.module.spawn_counter >= self.cfg.persistence_steps

    @torch.no_grad()
    def select_birth_nodes(self, persistent: Tensor, score: Tensor) -> Tensor:
        available = self.module.capacity < self.cfg.d_max
        ids = torch.where(persistent & available)[0]
        k = min(ids.numel(), self.cfg.max_births_per_event)
        if k == 0:
            return ids
        order = torch.topk(score[ids], k=k, largest=True).indices
        return ids[order]

    @torch.no_grad()
    def activate(self, nodes: Tensor, init_values: Tensor | None = None) -> FiberEvent:
        if nodes.numel() == 0:
            empty = nodes.new_empty((0,))
            return FiberEvent(empty, empty)
        selected_nodes: list[Tensor] = []
        selected_channels: list[Tensor] = []
        for row, node in enumerate(nodes.tolist()):
            free = torch.where(~self.module.active_mask[node])[0]
            channels = free[: self.cfg.spawn_width]
            if channels.numel() == 0:
                continue
            self.module.active_mask[node, channels] = True
            self.module.gate_logits[node, channels].fill_(self.cfg.birth_gate_logit)
            values = 0.01 * torch.randn_like(self.module.latent[node, channels]) if init_values is None else init_values[row, : channels.numel()].to(self.module.latent)
            self.module.latent[node, channels].copy_(values)
            self.module.age[node, channels] = 0
            self.module.utility_ema[node, channels] = 0
            selected_nodes.append(torch.full_like(channels, node))
            selected_channels.append(channels)
        if not selected_nodes:
            empty = nodes.new_empty((0,))
            return FiberEvent(empty, empty)
        return FiberEvent(torch.cat(selected_nodes), torch.cat(selected_channels))

    @torch.no_grad()
    def age(self) -> None:
        self.module.age.add_(self.module.active_mask.to(self.module.age.dtype))

    @torch.no_grad()
    def dead_candidates(self) -> tuple[Tensor, Tensor]:
        extra = torch.zeros_like(self.module.active_mask)
        extra[:, self.cfg.d_base :] = True
        dead = self.module.active_mask & extra & (self.module.age >= self.cfg.min_age_for_death) & (self.module.utility_ema <= self.cfg.utility_threshold)
        nodes, channels = torch.where(dead)
        if nodes.numel() > self.cfg.max_deaths_per_event:
            util = self.module.utility_ema[nodes, channels]
            order = torch.argsort(util)[: self.cfg.max_deaths_per_event]
            nodes, channels = nodes[order], channels[order]
        return nodes, channels

    @torch.no_grad()
    def deactivate(self, nodes: Tensor, channels: Tensor) -> FiberEvent:
        if nodes.numel():
            self.module.active_mask[nodes, channels] = False
            self.module.gate_logits[nodes, channels].fill_(self.cfg.birth_gate_logit)
            self.module.latent[nodes, channels].zero_()
            self.module.age[nodes, channels].zero_()
            self.module.utility_ema[nodes, channels].zero_()
        return FiberEvent(nodes, channels)

    @torch.no_grad()
    def prune(self) -> FiberEvent:
        return self.deactivate(*self.dead_candidates())

    @torch.no_grad()
    def residual_scalar_initialization(self, residual: Tensor, nodes: Tensor) -> Tensor:
        if residual.ndim == 1:
            energy = residual.abs()
        else:
            energy = residual.square().mean(dim=-1).sqrt()
        base = energy[nodes].clamp_min(1e-8).unsqueeze(-1)
        signs = torch.where(
            torch.arange(self.cfg.spawn_width, device=nodes.device) % 2 == 0,
            1.0,
            -1.0,
        ).to(base)
        return 0.01 * base * signs.unsqueeze(0)


def skew_symmetric(raw: Tensor) -> Tensor:
    """Map arbitrary matrices to the Lie algebra so(d)."""
    if raw.ndim < 2 or raw.shape[-1] != raw.shape[-2]:
        raise ValueError("raw generator must end in square matrices")
    return 0.5 * (raw - raw.transpose(-1, -2))


def cayley_so(a: Tensor) -> Tensor:
    """Cayley retraction from a skew matrix to SO(d)."""
    if a.ndim < 2 or a.shape[-1] != a.shape[-2]:
        raise ValueError("generator must end in square matrices")
    d = a.shape[-1]
    eye = torch.eye(d, dtype=a.dtype, device=a.device).expand(a.shape[:-2] + (d, d))
    return torch.linalg.solve(eye - 0.5 * a, eye + 0.5 * a)


def project_to_so_d(u: Tensor) -> Tensor:
    """Nearest SVD/polar projection of a batch of square matrices to SO(d)."""
    if u.ndim < 2 or u.shape[-1] != u.shape[-2]:
        raise ValueError("u must end in square matrices")
    left, _, vh = torch.linalg.svd(u)
    r = left @ vh
    det = torch.linalg.det(r)
    diag = torch.ones(u.shape[:-2] + (u.shape[-1],), dtype=u.dtype, device=u.device)
    diag[..., -1] = torch.where(det < 0, -torch.ones_like(det), torch.ones_like(det))
    return left @ torch.diag_embed(diag) @ vh


class SOConnectionBank(nn.Module):
    """Fixed-capacity edge connection bank with exact SO(d) parameterization.

    Parameters are unconstrained, but every matrix exposed by :meth:`matrices` is built
    from the skew-symmetric Lie algebra so(d), so Euclidean optimizer steps cannot leave
    the special-orthogonal group. The bank is indexed by graph buffer slot, allowing
    graph topology to mutate without changing parameter shapes or optimizer state.
    """

    def __init__(
        self,
        edge_capacity: int,
        dim: int,
        *,
        parameterization: str = "cayley",
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        if edge_capacity < 0 or dim <= 0:
            raise ValueError("edge_capacity must be nonnegative and dim positive")
        if parameterization not in {"cayley", "exp"}:
            raise ValueError("parameterization must be 'cayley' or 'exp'")
        self.edge_capacity = int(edge_capacity)
        self.dim = int(dim)
        self.parameterization = parameterization
        self.raw_generators = nn.Parameter(torch.zeros(edge_capacity, dim, dim, device=device, dtype=dtype))
        self.register_buffer("slot_generation", torch.zeros(edge_capacity, dtype=torch.long, device=device))

    def generators(self) -> Tensor:
        return skew_symmetric(self.raw_generators)

    def matrices(self) -> Tensor:
        a = self.generators()
        if self.parameterization == "exp":
            return torch.matrix_exp(a)
        return cayley_so(a)

    def forward(self, slot_ids: Tensor | None = None) -> Tensor:
        mats = self.matrices()
        return mats if slot_ids is None else mats[slot_ids]

    @torch.no_grad()
    def reset_slots(
        self,
        slot_ids: Tensor | list[int] | tuple[int, ...],
        optimizers: Any = None,
        *,
        sync_generation: Tensor | None = None,
    ) -> None:
        """Reset Lie-algebra generators and optimizer state for retired/reassigned slots.

        Parameters
        ----------
        slot_ids
            Edge-slot indices to reset.
        optimizers
            Optimizer(s) whose per-slot state should be cleared. The reset is
            optimizer-generic: every tensor-valued state entry whose leading
            dimension matches ``edge_capacity`` is zeroed for the affected slots.
            Scalar state entries (e.g. ``step`` counters) are preserved.
        sync_generation
            If provided, the gauge bank's ``slot_generation`` is set to match
            this tensor for the affected slots rather than independently
            incremented. This makes the graph the canonical generation
            authority and prevents dual-authority drift.
        """
        if not isinstance(slot_ids, Tensor):
            slot_ids = torch.as_tensor(slot_ids, dtype=torch.long, device=self.raw_generators.device)
        else:
            slot_ids = slot_ids.to(self.raw_generators.device)
        if slot_ids.numel() == 0:
            return

        # 1. Zero out raw parameter generators
        self.raw_generators.index_fill_(0, slot_ids, 0.0)

        # 2. Zero parameter gradients if present
        if self.raw_generators.grad is not None:
            self.raw_generators.grad.index_fill_(0, slot_ids, 0.0)

        # 3. Update slot generation: sync from canonical authority or increment
        if sync_generation is not None:
            self.slot_generation[slot_ids] = sync_generation[slot_ids].to(self.slot_generation.device)
        else:
            self.slot_generation.index_add_(0, slot_ids, torch.ones_like(slot_ids))

        # 4. Clear optimizer state slices for the modified slots.
        # Optimizer-generic: zero every tensor-valued state whose leading
        # dimension matches edge_capacity. Scalar states (step counters, etc.)
        # are explicitly preserved.
        if optimizers is not None:
            opts = [optimizers] if not isinstance(optimizers, (list, tuple, set)) else list(optimizers)
            for opt in opts:
                if not hasattr(opt, "state"):
                    continue
                param_state = opt.state.get(self.raw_generators)
                if param_state is None:
                    continue
                for key, value in list(param_state.items()):
                    if not isinstance(value, Tensor):
                        continue
                    if value.ndim >= 1 and value.shape[0] == self.edge_capacity:
                        value.index_fill_(0, slot_ids.to(value.device), 0.0)

    @torch.no_grad()
    def retract_raw_(self) -> None:
        """Optional numerical cleanup by projecting current matrices and taking log.

        Normal training does not need this because the Lie-algebra parameterization is
        exact. This method is deliberately conservative: it projects to SO(d), then uses
        the skew part of R-I as a small-angle retraction back into parameter space.
        """
        r = project_to_so_d(self.matrices())
        approx = skew_symmetric(r - torch.eye(self.dim, dtype=r.dtype, device=r.device))
        self.raw_generators.copy_(approx)

    def state_hash(self) -> str:
        h = hashlib.sha256()
        x = self.raw_generators.detach().cpu().contiguous()
        g = self.slot_generation.detach().cpu().contiguous()
        h.update(self.parameterization.encode())
        h.update(str(self.dim).encode())
        h.update(str(tuple(x.shape)).encode())
        h.update(x.view(torch.uint8).numpy().tobytes())
        h.update(g.view(torch.uint8).numpy().tobytes())
        return h.hexdigest()

    def invariant_error(self) -> tuple[Tensor, Tensor]:
        r = self.matrices()
        eye = torch.eye(self.dim, dtype=r.dtype, device=r.device)
        orth = torch.linalg.matrix_norm(r.transpose(-1, -2) @ r - eye, ord="fro", dim=(-2, -1))
        det = (torch.linalg.det(r) - 1.0).abs()
        return orth, det

# Compatibility helper kept outside the class body for compile-friendly call sites.


def directed_so_matrices_static(bank: SOConnectionBank, slot_ids: Tensor, reverse: Tensor) -> Tensor:
    """Fixed-shape SO(d) gather for compiled/padded edge kernels.

    ``slot_ids < 0`` denotes synthetic/padded edges and maps to identity. No
    data-dependent Python branch is used, so tensor metadata stays static.
    """
    if slot_ids.ndim != 1 or reverse.ndim != 1 or slot_ids.shape != reverse.shape:
        raise ValueError("slot_ids and reverse must be equal-length vectors")
    if bank.edge_capacity == 0:
        return torch.eye(bank.dim, dtype=bank.raw_generators.dtype, device=bank.raw_generators.device).expand(slot_ids.numel(), bank.dim, bank.dim)
    safe = slot_ids.clamp_min(0)
    mats = bank.matrices()[safe]
    eye = torch.eye(bank.dim, dtype=mats.dtype, device=mats.device).expand_as(mats)
    mats = torch.where((slot_ids >= 0)[:, None, None], mats, eye)
    return torch.where(reverse[:, None, None], mats.transpose(-1, -2), mats)

def directed_so_matrices(bank: SOConnectionBank, slot_ids: Tensor, reverse: Tensor) -> Tensor:
    if slot_ids.ndim != 1 or reverse.ndim != 1 or slot_ids.shape != reverse.shape:
        raise ValueError("slot_ids and reverse must be equal-length vectors")
    out = torch.eye(bank.dim, dtype=bank.raw_generators.dtype, device=bank.raw_generators.device).expand(slot_ids.numel(), bank.dim, bank.dim).clone()
    valid = slot_ids >= 0
    if bool(valid.any().item()):
        mats = bank.matrices()[slot_ids[valid]]
        rev = reverse[valid]
        mats = torch.where(rev[:, None, None], mats.transpose(-1, -2), mats)
        out[valid] = mats
    return out
