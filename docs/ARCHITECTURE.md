# LGAE-v3.2 architecture

LGAE-v3.2 is a multi-timescale **geometry governor**. Curvature is a sensor family; graph/fiber/gauge changes are actuators; only a transaction that survives shadow evolution and independent audits becomes authoritative.

## State and authority split

- `P^A`: weighted actuation/transport Markov operator.
- `P^D`: independently reconstructed diagnostic diffusion operator.
- `Z[N,D_max]`: fixed-width latent state with controller-owned active mask and trainable gates.
- `U_e in SO(d_g)`: optional edge-slot gauge connection generated from `A_e in so(d_g)`.
- fixed-capacity graph buffers: topology changes values/masks without ordinary compiled-kernel shape changes.
- authoritative graph version + SHA-256 state hash; fiber and gauge state hashes protect quarantines.

## Gauge layer

For each edge slot,

`A_e = 0.5 * (R_e - R_e^T)`

and either

`U_e = exp(A_e)`

or

`U_e = (I - A_e/2)^(-1) (I + A_e/2)`.

Both keep `U_e` in `SO(d)` up to floating-point error. Reverse transport uses `U_e^T`. The graph edge capacity and gauge parameter capacity are identical, so edge surgery does not replace parameters or optimizer state.

## Audit ladder

1. Sparse Gamma energy / diffusion radius / residuals.
2. AF3 and degree-weighted AF3 proxy candidate generation.
3. Multiscale Ollivier using exact LP or stabilized log-domain Sinkhorn; dual exact LLY.
4. Weak entropic curvature.
5. Integral/role-conditioned LLY and Schur-complement Bakry–Émery.
6. Sampled CDE', topology, persistent homology, and spectral certification.

Bakry–Émery uses `Delta = P-I` for a reversible row-stochastic Markov kernel rather than a raw combinatorial `D-A` generator.

## Ricci flow and surgery

Metric reweighting is multiplicative in log-space:

`w_e' = clamp(w_e * exp(-dt*(kappa_e-kappa_target)), w_min, w_max)`.

This prevents non-positive weights. An edge cooldown map and separated curvature thresholds provide hysteresis so a recently modified edge cannot immediately oscillate across a surgery boundary.

## Spectral safety

Before expensive audits, a local bridge gate rejects a protected edge deletion that would disconnect the graph. Spectral gap then uses:

- exact symmetric eigensolve for small graphs;
- sparse `torch.lobpcg` for larger normalized Laplacians;
- explicit zero gap for isolated vertices;
- fail-closed behavior on iterative-solver failure when no safe fallback is configured.

## Compilation boundary

Compiled region:

- latent/fiber arithmetic;
- fixed-capacity edge reductions;
- task/reconstruction losses;
- sparse field dynamics.

Eager region:

- graph surgery;
- Sinkhorn/LP/LLY/entropic solvers;
- spectral/topological governance;
- birth/death selection and quarantine handling.

`padded_markov_edges()` creates a bucketed static buffer and `refresh_padded_markov_edges_()` updates values in place after graph changes without changing metadata.
