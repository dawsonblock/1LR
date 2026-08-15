# LGAE-v3.2 Geometric Hardening Build Report

Build: `3.2.0`

Base: `lgae_v3_merged_hardened` v3.1.0.

## Implemented roadmap

1. **Fiber bundle / gauge group**
   - fixed-capacity `SOConnectionBank` indexed by graph slots;
   - skew-symmetric Lie-algebra parameterization in `so(d)`;
   - Cayley and matrix-exponential maps into `SO(d)`;
   - SVD/polar `project_to_so_d` utility;
   - reverse-edge inverse transport and slot reset on topology retirement/reuse;
   - gauge state included in checkpoint and stale-quarantine identity checks;
   - gauge-covariant sparse training metrics provide integrated optimizer gradients to connection generators.

2. **Discrete curvature stability**
   - log-domain Sinkhorn W1 backend, with exact LP retained as the qualification oracle;
   - exact removal of zero-mass transport support;
   - marginal-residual convergence verification rather than scaling-variable-only convergence;
   - Bakry-Émery/CDE use reversible normalized Markov generators and stationary volume measure;
   - stationary measure reconstructed from detailed-balance ratios without a dense eigensolve;
   - float64 row renormalization before forming the Markov generator;
   - retained corrected Gamma-nullspace Schur complement.

3. **Dynamic Ricci flow / surgery**
   - positive log-conformal `RicciFlowReweight` update;
   - configurable `[w_min,w_max]` clamp;
   - per-edge cooldown state;
   - separate add/deadband/prune thresholds;
   - cooldown state is checkpointed and restored.

4. **Governor scaling / disconnected states**
   - sparse symmetric-normalized Laplacian with LOBPCG above configurable graph size;
   - exact eigensolve below threshold;
   - isolated vertices produce a zero spectral gap rather than NaN propagation;
   - local bridge gate rejects protected disconnecting prunes before expensive global audits;
   - graph and fiber changes remain transactional with shadow rollout and rollback/quarantine.

5. **PyTorch compilation & slot lifecycle**
   - discrete surgery, Sinkhorn, and governor decisions remain outside compiled kernels;
   - bucketed fixed-capacity edge buffers;
   - in-place edge-buffer refresh after topology changes;
   - static-width fiber representation retained;
   - inactive fiber storage is forced to zero after diffusion so dormant channels cannot hide large latent values;
   - monotonic slot generations (g_e) and optimizer-aware slot resets (m=0, v=0) prevent momentum leakage across slot retirements/reuses.

## Qualification

- Pytest collection: **72 tests**.
- Full test suite: **72/72 passed**.
- `scripts/qualify.py`: **PASS**.
- Editable install with `--no-build-isolation`: **PASS**.
- Installed CLI version/import check: **PASS** (`3.2.0`).
- CPU Inductor fixed-shape compile smoke (`N=32,D=4,E=64`): **PASS**.
- End-to-end CLI demo: **PASS**; candidate mutation reaches a governed `quarantine` decision rather than numerical failure.

Qualification includes:
- `SO(d)` orthogonality and determinant invariants after real Adam optimizer steps;
- SVD projection of a reflection back to `SO(d)`;
- small-epsilon / large-diameter log-Sinkhorn stress against exact LP;
- exact zero-mass transport support;
- fail-closed Sinkhorn nonconvergence;
- reversible stationary-volume measure oracles;
- generator row-sum roundoff regression;
- corrected Bakry P4/K2 Schur-complement oracles;
- dual-path exact LLY agreement;
- sparse LOBPCG vs exact cycle spectral gap;
- isolated-vertex spectral handling;
- extreme-curvature positive Ricci-flow weights;
- cooldown/deadband surgery behavior;
- local bridge-gate rejection;
- fixed-capacity compile-buffer refresh across graph mutation;
- dormant-fiber latent-state suppression;
- integrated gauge-training gradient and post-step SO(d) invariant check.

## Boundaries

This is a research-grade controller, not a formal safety proof. Sinkhorn remains an entropically regularized approximation, CDE' is sampled, large diagnostic feature clouds still need an ANN backend, and the reference LLY/entropic curvature stack remains unweighted unless a weighted backend is explicitly added. Sparse LOBPCG is used only as a read-only safety diagnostic; PyTorch documents that sparse `torch.lobpcg` does not support backward gradients.
