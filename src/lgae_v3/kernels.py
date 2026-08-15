from __future__ import annotations

import torch
from torch import Tensor, nn

from .metrics import gamma_vector, diffusion_radius, local_variance, edge_diffusion_metrics
from .operators import positive_laplacian_from_markov, sparse_laplacian_step


class FieldKernel(nn.Module):
    """Dense reference fixed-shape kernel for small graphs."""
    def forward(self, z: Tensor, p: Tensor, eta: Tensor | float):
        l = positive_laplacian_from_markov(p)
        z_next = z - eta * (l @ z)
        return z_next, gamma_vector(z, p), diffusion_radius(z, p), local_variance(z, p)


class SparseFieldKernel(nn.Module):
    """Compile-friendly O(E D) field kernel over directed row-stochastic edges."""
    def forward(self, z: Tensor, src: Tensor, dst: Tensor, pweight: Tensor, eta: Tensor | float):
        z_next = sparse_laplacian_step(z, src, dst, pweight, eta=eta, num_nodes=z.shape[0])
        m = edge_diffusion_metrics(z, src, dst, pweight, z.shape[0], validate_weights=False)
        return z_next, m["gamma"], m["radius"], m["local_var"]
