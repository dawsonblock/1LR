# v3.2 hardening notes

## 1. Gauge invariants

The build no longer permits a trainable connection matrix to drift out of the special-orthogonal group. Optimizers act on raw generators; only their skew-symmetric part enters Cayley/exponential mapping. Tests perform real Adam steps and assert both `U^T U = I` and `det(U)=+1`.

## 2. Optimal transport

`curvature/ollivier.py` contains two backends:

- `exact_lp`: reference/qualification path;
- `sinkhorn_log`: log-scaling entropic approximation with cost normalization, finite-input validation, convergence tolerance, and original-metric cost evaluation.

This removes probability-space scaling underflow at small regularization.

## 3. Reversible Gamma calculus

The Bakry–Émery implementation keeps the v3.1 Schur-complement fix. v3.2 adds validation that the supplied Markov kernel is row stochastic and reversible under a positive stationary measure before constructing `Delta=P-I`.

## 4. Flow positivity + surgery hysteresis

`RicciFlowReweight` performs multiplicative updates in log geometry and clamps only after the positive exponential update. `MutationCooldownTracker` stores canonical undirected edge keys and is checkpointed. Threshold helpers separate addition, deadband, and pruning regions.

## 5. Spectral scaling

`operators.spectral_gap_graphbuffers()` forms a sparse symmetric normalized Laplacian and uses LOBPCG above the configured graph-size threshold. Isolated vertices return a zero gap immediately. The mutation governor also has an `O(V+E)` bridge check before global curvature/spectral work.

## 6. Compile stability

Graph mutation remains eager. Training kernels consume fixed-size padded/bucketed edge buffers. `refresh_padded_markov_edges_()` updates source, destination, weights, and validity masks in place and raises when a bucket is exhausted instead of silently reallocating.

## 7. Final release-hardening additions

The release pass tightened the numerical boundaries further:

- Sinkhorn removes exact zero-mass rows/columns before iteration and checks recovered coupling marginals before accepting a result.
- Reversible stationary measures are reconstructed from detailed-balance ratios instead of a dense eigenvector solve.
- `normalized_markov_generator()` re-normalizes float32 kernels in float64 and clears the final row-sum ulp from the diagonal before Gamma calculus.
- Inactive fixed-width fiber channels are zeroed after diffusion, preventing large dormant values from accumulating behind an inactive mask.


## 8. Differentiable gauge training path

`LGAETrainCore` optionally owns the same fixed-capacity `SOConnectionBank` used by the engine.
`padded_markov_edges_with_slots()` supplies fixed-shape edge-slot and orientation buffers.
During the Laplacian loss, neighbor features are parallel transported into each source frame
before Γ/radius/variance are computed. This makes connection generators receive ordinary
autograd gradients while every exposed transport matrix remains exactly in `SO(d)` by
construction. Discrete graph surgery remains eager and outside the compiled training core.
