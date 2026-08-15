# LGAE-v5.1.0 Deep Audit: Sparse Scaling, Float64 Discrepancy, ANN Index Build Report

Build: `5.1.0`

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
   - monotonic slot generations (g_e) and optimizer-generic slot resets prevent momentum leakage across slot retirements/reuses.

6. **v3.3 Authority and persistence hardening**
   - canonical authority hash `H(G, g_e, U, F, C_g)` binds graph, gauge, fiber, and governance config;
   - slot_generation is cryptographically committed in the graph state hash (schema V3);
   - graph is the canonical generation authority; gauge bank syncs from graph at init, commit, and checkpoint boundaries;
   - `assert_generation_sync()` verifies graph/gauge generation equality;
   - checkpoint config enforcement: structural mismatch fails immediately, governance mismatch requires explicit `allow_governance_mismatch=True`;
   - optimizer checkpoint semantics: `optimizer_load_policy` supports `"restore"`, `"reset"`, `"reject"`;
   - safe checkpoint format: `safetensors + JSON` directory (no pickle, untrusted-safe);
   - optimizer-generic slot reset: clears all tensor-valued state matching edge capacity (Adam, AdamW, SGD, Adagrad, RMSProp, etc.);
   - hash-chained receipts with `H_i = SHA256(H_{i-1} || R_i)` and `verify_receipt_chain()`;
   - receipts bind gauge authority hash at transaction boundary;
   - exact manifest coverage with `scripts/generate_manifest.py --check`.

7. **v4.0 Sparse weighted geometry**
   - `SparseDualOperatorState` replaces dense `N×N` dual operators with `O(Nk)` edge lists;
   - `diagnostic_diffusion_edges()` uses k-NN without materializing full pairwise distance matrix;
   - `sparse_operator_discrepancy()` computes Frobenius/L1 discrepancy on union of supports;
   - chunked k-NN for very large N (>4096) to bound peak memory;
   - weighted Ollivier: edge-weight-proportional lazy measure + Dijkstra shortest-path cost;
   - weighted LLY: weighted normalized Laplacian + shortest-path-distance boundary conditions;
   - weighted AF3: weighted degree replaces unweighted degree;
   - `curvature_weight_mode='weighted'` accepted in config validation;
   - governor audit and fast signals dispatch to weighted backends when configured.

8. **v4.1 Metric–measure separation + multi-horizon certification**
   - `GraphBuffers` carries independent `weight` (affinity) and `length` (metric) tensors;
   - default inverse relationship `length = 1/weight` when only one scalar provided;
   - `make_graph_buffers` accepts `(u,v,a,ell)` 4-tuples for explicit metric-measure;
   - state hash schema V4 includes both `weight` and `length`;
   - checkpoint roundtrips both fields; backward compat with old checkpoints (derives length);
   - mutations (AddEdge/ReweightEdge/PruneEdge/RicciFlow) update both fields;
   - `graphbuffers_to_networkx` stores both `weight` and `length` attributes;
   - weighted ORC: ground cost from `length`, measures from `P(affinity)`;
   - weighted LLY: Lipschitz from `length`, Laplacian from `P(affinity)`;
   - `weighted_forman_edge`: literature-faithful formula with sqrt weight ratios;
   - `weighted_af3_proxy`: clearly labeled proxy, not canonical Forman;
   - `ricci_flow_target`: "weight" or "length", with optional coupling;
   - `shadow_horizons = [1,2,4,8,16]`: mutation must be admissible across ALL horizons;
   - scalability claims corrected: bounded-memory exact k-NN, not sub-quadratic ANN.

## Qualification

- Pytest collection: **492 tests**.
- Full test suite: **492/492 passed**.
- `scripts/qualify.py`: **PASS**.
- Editable install with `--no-build-isolation`: **PASS**.
- Installed CLI version/import check: **PASS** (`5.1.0`).
- CPU Inductor fixed-shape compile smoke (`N=32,D=4,E=64`): **PASS**.
- End-to-end CLI demo: **PASS**; candidate mutation reaches a governed `quarantine` decision rather than numerical failure.
- Manifest verification: **PASS** (`scripts/generate_manifest.py --check`).
- N=2500 sparse governor audit: **PASS** (3.2s, no global dense allocation).

### v5.1.0 additions

- **Dynamic gauge connections** (`dynamic_gauge.py`): Context-conditioned SO(d) transport `U_ij = exp(skew(f_θ(z_i, z_j, c_t)))`
- **Multi-timescale adaptation** (`timescales.py`): Fast/medium/slow timescale separation prevents mutual drift
- **Sheaf-adjacency diffusion** (`sheaf_diffusion.py`): Sheaf-adjacency + normalization + gating vs pure Laplacian
- **ANN-backed neighbor index** (`ann_index.py`): FAISS or numpy HNSW fallback with exact reranking pipeline
- **Causal edge semantics** (`causal_edges.py`): Association vs causal edges, do-interventions, counterfactuals
- **Hypergraph** (`hypergraph.py`): Higher-order relationships via hyperedges, clique/star expansion

### v5.0.0 additions

- **Learned structural executive** (`executive.py`): Proposal model with bilevel objective
- **Long-term credit assignment** (`credit.py`): Discounted returns at horizons {16, 100, 1000}
- **Calibrated uncertainty** (`uncertainty.py`): Ensemble epistemic UQ + LCB acceptance gate
- **Stability/plasticity** (`consolidation.py`): Capacity budget, fiber lifecycle, probation gate
- **Benchmark harness** (`benchmark/`): 6 synthetic tasks with known-optimal mutations
- **Counterfactual engine** (`counterfactual.py`): Candidate comparison with NO_OP baseline
- **Closed loop** (`structural_loop.py`): observe→predict→counterfactual→certify→train

### v4.1.3 additions

- **Analytic vertex selection**: union of transport pressure, LLY, discrepancy, touched nodes
- **Local neighborhood cap**: max_local_nodes=256, radius=1 → N=2500 audit 73s→3.2s
- **Float64 discrepancy**: 16 parameterized cases at 1e-10 tolerance
- **Duplicate edge coalescing**: deliberately duplicated COO edges verified
- **v4 checkpoint length mandatory**: schema versioning with legacy migration
- **Stale quarantine detection**: base_graph_hash check after restart
- **Parameterized governance hash**: 90+ fields tested for hash sensitivity
- **StructuralMutation protocol**: touched_region() on all mutation classes
- **Neighbor index abstraction**: NeighborIndex protocol + ExactChunkedKNN + recall_at_k
- **Forman reference tests**: K2, path, star, tree, uniform reduction
- **Multi-horizon combinations**: all 9 decision aggregation patterns

### v4.1.1 fixes

- **P0-1**: Sparse governor works for N>2048 (local BE/CDE extraction)
- **P0-2**: Sparse discrepancy coalesces duplicate COO edges via `torch.sparse_coo_tensor.coalesce()`
- **P0-3**: Safe checkpoint persists `length` tensor alongside `weight`
- **P0-4**: Governance hash includes `shadow_horizons`, `ricci_flow_target`, `ricci_flow_coupled`
- **P0-5**: RicciFlow serialization preserves `target_field` and `coupled`
- **P1-1**: Multi-horizon uses max severity aggregation (QUARANTINE propagates)
- **P1-2**: Safe checkpoint persists complete shadow graphs for durable quarantine
- **P1-3**: Weighted Forman uses metric-measure formula (m₁, m₂, ω)
- **P2-1**: Single `version.py` module for all version identity

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
