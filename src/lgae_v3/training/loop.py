from __future__ import annotations

from typing import Dict, Optional
import torch
from torch import Tensor

from ..evolution import LGAEEngine
from .core import LGAETrainCore


def train_step(
    core: LGAETrainCore,
    engine: LGAEEngine,
    optimizer: torch.optim.Optimizer,
    *,
    target: Tensor,
    src: Tensor,
    dst: Tensor,
    weight: Tensor,
    valid: Tensor,
    bottleneck_pressure: Tensor,
    residual_target: Optional[Tensor] = None,
    uncertainty: Optional[Tensor] = None,
    edge_slot: Optional[Tensor] = None,
    reverse: Optional[Tensor] = None,
    step: int = 0,
    spawn_interval: int | None = None,
) -> Dict[str, object]:
    if core.latent is not engine.fibers:
        raise ValueError("training core and engine must share the same fiber module")
    optimizer.zero_grad(set_to_none=True)
    out = core(
        target=target,
        src=src,
        dst=dst,
        weight=weight,
        valid=valid,
        bottleneck_pressure=bottleneck_pressure,
        residual_target=residual_target,
        uncertainty=uncertainty,
        edge_slot=edge_slot,
        reverse=reverse,
    )
    out["loss"].backward()
    latent_grad = None if engine.fibers.latent.grad is None else engine.fibers.latent.grad.detach().clone()
    optimizer.step()
    engine.fiber_controller.update_utility(latent_grad)

    interval = int(spawn_interval or engine.cfg.mutation.mutation_interval)
    controller_output = None
    if interval > 0 and step % interval == 0:
        # The authoritative engine recomputes its own graph-derived signals and governs the mutation.
        controller_output = engine.fiber_tick(residual=out["residual_error"].detach(), uncertainty=uncertainty)
    return {
        "loss": out["loss"].detach(),
        "task_loss": out["task_loss"].detach(),
        "lap_loss": out["lap_loss"].detach(),
        "fiber_loss": out["fiber_loss"].detach(),
        "spawn_score_mean": out["spawn_score"].detach().mean(),
        "controller": controller_output,
    }
