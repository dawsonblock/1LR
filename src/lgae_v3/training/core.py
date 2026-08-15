from __future__ import annotations

from typing import Dict, Optional
import torch
from torch import Tensor, nn
import torch.nn.functional as F

from ..fibers import FixedWidthFiberLatent, SOConnectionBank, directed_so_matrices_static
from ..metrics import edge_diffusion_metrics, edge_diffusion_metrics_gauge, spawn_score_from_pressure


class LGAETrainCore(nn.Module):
    """Compile-friendly task + sparse diffusion + fiber loss core.

    Discrete mutation decisions are intentionally absent. Inputs use fixed/padded edge
    buffers; ``valid`` masks inactive slots without changing tensor shapes.
    """

    def __init__(
        self,
        latent: FixedWidthFiberLatent,
        decoder: nn.Module,
        *,
        gauge_bank: SOConnectionBank | None = None,
        gauge_dim: int | None = None,
    ) -> None:
        super().__init__()
        self.latent = latent
        self.decoder = decoder
        self.gauge_bank = gauge_bank
        self.gauge_dim = int(gauge_dim if gauge_dim is not None else (0 if gauge_bank is None else gauge_bank.dim))
        if self.gauge_bank is not None and self.gauge_dim != self.gauge_bank.dim:
            raise ValueError("gauge_dim must equal gauge_bank.dim")

    def forward(
        self,
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
    ) -> Dict[str, Tensor]:
        z = self.latent()
        prediction = self.decoder(z)
        task_loss = F.mse_loss(prediction, target)
        eff_weight = weight.to(z.dtype) * valid.to(z.dtype)
        if self.gauge_bank is None:
            metrics = edge_diffusion_metrics(
                z, src, dst, eff_weight, z.shape[0], normalize_mass=True, validate_weights=False
            )
        else:
            if edge_slot is None or reverse is None:
                raise ValueError("gauge-aware training requires edge_slot and reverse buffers")
            connection = directed_so_matrices_static(self.gauge_bank, edge_slot, reverse)
            metrics = edge_diffusion_metrics_gauge(
                z, src, dst, eff_weight, connection, self.gauge_dim, z.shape[0],
                normalize_mass=True, validate_weights=False,
            )
        residual_error = z.square().mean(dim=-1) if residual_target is None else (z - residual_target).square().mean(dim=-1)
        uncertainty = torch.zeros_like(metrics["gamma"]) if uncertainty is None else uncertainty.to(z)
        spawn = spawn_score_from_pressure(
            metrics["gamma"], metrics["radius"], metrics["local_var"], bottleneck_pressure.to(z),
            residual_error, uncertainty, self.latent.capacity.to(z.dtype),
        )
        lap_loss = metrics["gamma"].mean()
        fiber_loss = self.latent.regularization()["fiber_loss"]
        loss = task_loss + 1e-3 * lap_loss + fiber_loss
        return {
            "loss": loss,
            "task_loss": task_loss,
            "lap_loss": lap_loss,
            "fiber_loss": fiber_loss,
            "z": z,
            "gamma": metrics["gamma"],
            "radius": metrics["radius"],
            "local_var": metrics["local_var"],
            "spawn_score": spawn,
            "residual_error": residual_error,
        }
