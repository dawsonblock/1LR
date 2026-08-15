<div align="center">

# LGAE-v4.0 / 1LR: Governed Adaptive Geometry Engine

**A Multi-Timescale Geometric Controller for Self-Evolving Graph and Fiber-Bundle Latent Spaces**

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.6+](https://img.shields.io/badge/PyTorch-2.6+-ee4c2c.svg)](https://pytorch.org/)
[![CI Status](https://github.com/dawsonblock/1LR/actions/workflows/ci.yml/badge.svg)](https://github.com/dawsonblock/1LR/actions)
[![Tests](https://img.shields.io/badge/tests-112%2F112%20passing-brightgreen.svg)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Gauge: SO(d)](https://img.shields.io/badge/gauge-SO(d)%20Invariance-purple.svg)]()

</div>

---

## Overview

**LGAE-v4.0 (`1LR`)** is a hardened geometric deep learning engine and dynamical controller. It operates over graph-structured data and continuous fiber bundles, combining continuous field diffusion, Lie-algebra gauge connections, discrete Ricci-flow surgery, and multi-operator curvature diagnostics.

### v4.0 — Sparse Weighted Geometry

This release closes the scalability and weighted-geometry gaps identified in the v3.3 audit:

- **Sparse dual operators**: `SparseDualOperatorState` replaces the dense `N×N` actuation and diagnostic diffusion operators with `O(Nk)` edge-list representations. The diagnostic diffusion uses k-NN without materializing the full pairwise distance matrix. Operator discrepancy is computed on the union of supports.
- **Weighted curvature backends**: `curvature_weight_mode='weighted'` is now supported. Weighted Ollivier uses edge-weight-proportional lazy measures and Dijkstra shortest-path costs. Weighted LLY uses the weighted normalized Laplacian with shortest-path-distance boundary conditions. Weighted AF3 uses weighted degree instead of unweighted degree.

### v3.3 — Authority and Persistence Hardening

This release closes the state-authority gap identified in the v3.2 audit:

- **Canonical authority hash** `H(G, g_e, U, F, C_g)` binds graph, gauge, fiber, and governance config into a single SHA-256 commitment.
- **Slot-generation cryptographic binding**: `slot_generation` is now included in the graph state hash, preventing ABA-style slot reuse from going undetected.
- **Graph/gauge generation synchronization**: the graph is the canonical generation authority; gauge bank generations sync from the graph at init, commit, and checkpoint boundaries.
- **Checkpoint config enforcement**: structural config mismatch fails immediately; governance mismatch requires explicit `allow_governance_mismatch=True` migration flag.
- **Optimizer checkpoint semantics**: `optimizer_load_policy` supports `"restore"`, `"reset"`, and `"reject"` — no more silent mixing of checkpoint parameters with stale optimizer history.
- **Safe checkpoint format**: `safetensors + JSON` directory format for untrusted interchange (no pickle deserialization).
- **Optimizer-generic slot reset**: clears all tensor-valued optimizer state matching edge capacity, not just Adam-specific keys (handles Adagrad, RMSProp, etc.).
- **Hash-chained receipts**: tamper-evident ledger with `H_i = SHA256(H_{i-1} || R_i)` and `verify_receipt_chain()`.
- **Receipts bind gauge authority**: accepted-mutation receipts now include `base_gauge_hash` and `authority_hash_after`.
- **Exact manifest coverage**: `scripts/generate_manifest.py` with `--check` mode; `.gitignore` explicitly declared as excluded.

### Core Architecture Principle
> **Field dynamics are sparse and compiled; discrete evolution is transactional and eager; curvature diagnoses rather than directly dictates topology.**

```
       +-------------------------------------------------------------+
       |                  Continuous Field Dynamics                  |
       |  - Latent states z in R^{N x D} with dynamic fiber gating   |
       |  - Sparse row-stochastic Markov diffusion (O(E x D))        |
       |  - SO(d) gauge parallel transport: U_e in SO(d_g)           |
       +------------------------------+------------------------------+
                                      |
                         Fast Geometric Signals (Gamma, r, Var)
                                      v
       +-------------------------------------------------------------+
       |               Transaction & Shadow Evaluation               |
       |  - Eager shadow rollout with dual actuation/diagnostics     |
       |  - Log-conformal Ricci flow: w' = clamp(w * exp(-dt * dk))  |
       |  - Graph surgery: add, reweight, prune with hysteresis      |
       +------------------------------+------------------------------+
                                      |
                      Curvature & Topological Audits
                                      v
       +-------------------------------------------------------------+
       |                   Authoritative Governor                    |
       |  - Exact LLY & Log-Sinkhorn Ollivier (W1 optimal transport) |
       |  - Reversible Bakry-Émery CD(K, N) with Schur complements   |
       |  - Sparse LOBPCG spectral certificate & beta_0 protection   |
       |  - Accept, Reject, or Quarantine with SHA-256 state locks   |
       +-------------------------------------------------------------+
```

---

## Key Features & Hardening

### 1. $\mathrm{SO}(d)$ Gauge Connection Bank (`SOConnectionBank`)
* **Lie-Algebra Parameterization**: Generator parameters live in unconstrained space $R_e$, strictly mapped through the skew-symmetric algebra $\mathfrak{so}(d)$ via $A_e = \frac{1}{2}(R_e - R_e^T)$ and mapped to $\mathrm{SO}(d)$ via Cayley retraction or Matrix Exponential ($\exp(A_e)$).
* **Guaranteed Invariance**: Connections strictly satisfy $U_e^T U_e = I$ and $\det(U_e) = +1$ to machine precision across arbitrary Euclidean optimizer steps.
* **Slot Generation Lifecycle ($g_e$)**: Monotonic generation counters $(g_e \leftarrow g_e + 1)$ track slot allocation and retirement. Generations are cryptographically committed in the graph state hash and synchronized between graph and gauge authorities.
* **Optimizer Momentum Isolation**: When an edge slot is retired or reused, all tensor-valued optimizer state slices whose leading dimension matches edge capacity are zeroed (optimizer-generic: handles Adam, AdamW, SGD, Adagrad, RMSProp, etc.). Scalar state (step counters) is preserved.

### 2. Stable Optimal Transport (Log-Sinkhorn Ollivier Curvature)
* **Log-Domain Scaling**: Eliminates probability-space underflow at small entropic regularization $\epsilon$.
* **Zero-Mass Pruning**: Exact support removal for unvisited states.
* **Marginal-Residual Certification**: Convergence validated against recovered coupling marginals rather than dual scaling differences alone.
* **Exact Ground-Truth Oracle**: High-precision linear programming (`exact_lp`) retained for qualification checks.

### 3. Reversible $\Gamma$-Calculus & Bakry–Émery ($CD(K, N)$)
* **Continuous-Time Reversible Markov Generators**: $\Delta = P - I$ formed with detailed-balance volume measure reconstruction.
* **Float64 Conditioning**: Precision row re-normalization and diagonal ULP cancellation.
* **$\Gamma$-Nullspace Schur Complement**: Eliminates uncoupled higher-hop coordinates ($B_{\text{eff}} = B_{pp} - B_{pn} B_{nn}^+ B_{np}$), preventing false-positive curvature anomalies.

### 4. Log-Conformal Ricci Flow & Surgery Hysteresis
* **Weight Positivity**: Multiplicative updates $w \leftarrow \text{clamp}(w \cdot \exp(-\Delta t(\kappa - \kappa^*)), w_{\min}, w_{\max})$ guarantee weights never cross zero.
* **Anti-Thrashing Cooldown**: Canonical edge cooldown tracker separates addition, deadband, and pruning regions.
* **$O(V+E)$ Bridge Filter**: Rejects disconnecting edge removals before triggering expensive global audits.

### 5. `torch.compile` Compatibility & Predictability
* **Fixed-Shape Buffer Bucketing**: `GraphBuffers` round capacity to fixed-size buckets with in-place value refresh (`refresh_padded_markov_edges_`).
* **Dormant Fiber Channel Suppression**: Inactive latent coordinates are zeroed post-diffusion to prevent hidden energy accumulation.

---

## Installation

### Prerequisites
- Python 3.11+
- PyTorch 2.6+
- NumPy, SciPy, NetworkX, PyYAML

```bash
# Clone the repository
git clone https://github.com/dawsonblock/1LR.git
cd 1LR

# Install in editable mode with development dependencies
python -m pip install -e '.[dev]' --no-build-isolation
```

---

## Quickstart & Code Examples

### 1. Gauge Parallel Transport on Fiber Bundles

```python
import torch
from lgae_v3 import LGAEConfig, LGAEEngine, make_bucketed_graph_buffers

# Configure fiber dimensions and gauge group
cfg = LGAEConfig()
cfg.fiber.d_base = 8
cfg.fiber.d_max = 16
cfg.fiber.gauge_dim = 8
cfg.fiber.gauge_parameterization = "cayley"  # 'cayley' or 'exp'

# Initialize bucketed graph buffers
edges = [(0, 1), (1, 2), (2, 3), (3, 0)]
graph = make_bucketed_graph_buffers(num_nodes=4, edges=edges, bucket_size=256)

# Create engine and execute gauge-covariant diffusion
engine = LGAEEngine(graph, cfg)
z_next = engine.diffuse_(eta=0.01)

# Verify SO(d) invariants
orth_err, det_err = engine.gauge_connections.invariant_error()
print(f"Max orthogonality error: {orth_err.max():.2e}")
print(f"Max determinant error:   {det_err.max():.2e}")
```

### 2. Differentiable Training Core with Optimizer Isolation

```python
import torch
from torch import nn
from lgae_v3 import LGAEConfig, LGAEEngine, LGAETrainCore
from lgae_v3.training import padded_markov_edges_with_slots, train_step

cfg = LGAEConfig()
cfg.fiber.d_base = 4
cfg.fiber.d_max = 8
cfg.fiber.gauge_dim = 4

graph = make_bucketed_graph_buffers(4, [(0, 1), (1, 2), (2, 3)], bucket_size=32)
engine = LGAEEngine(graph, cfg)

# Setup train core with shared gauge connections
decoder = nn.Linear(8, 2)
core = LGAETrainCore(engine.fibers, decoder, gauge_bank=engine.gauge_connections, gauge_dim=4)
optimizer = torch.optim.AdamW(core.parameters(), lr=1e-3)

# Padded fixed-shape buffers for torch.compile stability
src, dst, w, valid, slot, reverse = padded_markov_edges_with_slots(graph, max_edges=32)
target = torch.randn(4, 2)
pressure = torch.zeros(4)

# Execute one step (automatically registers optimizer for slot lifecycle management)
metrics = train_step(
    core, engine, optimizer,
    target=target, src=src, dst=dst, weight=w, valid=valid,
    bottleneck_pressure=pressure, edge_slot=slot, reverse=reverse,
    step=0, spawn_interval=50
)
print("Step Loss:", metrics["loss"].item())
```

### 3. Curvature Auditing & Governed Surgery

```python
from lgae_v3.mutations import AddEdge, PruneEdge, ReweightEdge

# Propose an edge addition
mutation = engine.propose_midpoint_edge()

# Shadow-evaluate and govern transaction
result = engine.evaluate_and_maybe_commit(mutation)
print("Decision:", result.decision.value)  # 'accept', 'reject', or 'quarantine'
print("Reasons:", result.reasons)
print("Authority hash after:", result.metadata.get("authority_hash_after"))
```

### 4. Checkpoint Authority & Safe Persistence

```python
# Save in safe (safetensors + JSON) format for untrusted interchange
engine.save_checkpoint("checkpoint_dir/")

# Save in legacy pickle format (trusted local use only)
engine.save_checkpoint("checkpoint.pt")

# Load with config authority enforcement
engine2.load_checkpoint_("checkpoint_dir/")

# Load with explicit governance migration
engine2.load_checkpoint_(
    "checkpoint_dir/",
    allow_governance_mismatch=True,
    optimizer_load_policy="restore",  # "restore" | "reset" | "reject"
)

# Verify canonical authority hash
print("Authority:", engine2.authority_hash())
engine2.assert_generation_sync()  # raises on graph/gauge generation divergence
```

### 5. Hash-Chained Receipt Ledger

```python
from lgae_v3.receipts import mutation_receipt, append_receipt, verify_receipt_chain

# Create and append a chained receipt
receipt = mutation_receipt(
    result,
    authority_state_hash_before=engine.authority_hash(),
    gauge_authority_hash=engine.gauge_connections.state_hash(),
)
append_receipt("ledger.jsonl", receipt)

# Verify the entire chain is tamper-evident
is_valid, errors = verify_receipt_chain("ledger.jsonl")
assert is_valid
```

---

## CLI Utilities

```bash
# Run qualification suite across all geometric and numerical oracles
python scripts/qualify.py

# Run full test suite with zero warnings
pytest -v -W error

# Run self-evolving graph demo
lgae-v3 demo --nodes 10 --steps 4

# Cross-validate exact LLY curvature paths
lgae-v3 qualify-lly --graph cycle --nodes 6

# Generate or verify the SHA-256 integrity manifest
python scripts/generate_manifest.py           # write manifest
python scripts/generate_manifest.py --check   # verify manifest
```

---

## Mathematical Oracles & Qualification Matrix

| Metric / Oracle | Graph / Test Case | Theoretical Target | LGAE-v3.2 Result | Status |
| :--- | :--- | :--- | :--- | :--- |
| **Bakry–Émery $K_\infty$** | Path $P_4$ (interior) | $1 - \frac{\sqrt{2}}{2} \approx 0.292893$ | `0.2928932188` | **PASS** |
| **Bakry–Émery $K_\infty$** | Path $P_4$ (endpoints) | $1.0$ | `1.0000000000` | **PASS** |
| **Bakry–Émery $K_\infty$** | Complete $K_2$ | $2.0$ | `2.0000000000` | **PASS** |
| **Exact LLY Agreement** | $K_2, P_4, C_4, K_3$ | $\kappa_{\text{LP}} = 2\kappa_{1/2}$ | Max error: `0.0` | **PASS** |
| **Weak Entropic Curvature** | $K_3$ (empty 2-hop shell) | $+\infty$ | `Infinity` | **PASS** |
| **Log-Sinkhorn vs LP** | Large metric / small $\epsilon$ | $998.0$ | `997.999999999` | **PASS** |
| **$SO(d)$ Invariance** | Post-Adam steps | $\|U^T U - I\|_F < 10^{-10}$ | `Pass` | **PASS** |
| **Sparse LOBPCG Spectral Gap** | Cycle $C_{24}$ | Matches exact $\lambda_2$ | `0.03407417` | **PASS** |

---

## Repository Structure

```
.
├── .github/
│   └── workflows/
│       └── ci.yml            # Multi-version (Py 3.11 & 3.12) GitHub Actions CI
├── configs/
│   └── default.yaml          # Default engine and audit configurations
├── docs/
│   ├── ARCHITECTURE.md       # Full system architecture and state split
│   ├── MATHEMATICS.md        # Complete mathematical formulations and proofs
│   ├── V32_HARDENING.md      # Detailed v3.2 stability and gauge hardening notes
│   └── READING_LIST.md       # Theoretical background and references
├── examples/
│   └── run_lgae_v3.py        # End-to-end execution example
├── scripts/
│   ├── benchmark_compile.py  # torch.compile benchmark (eager/static/dynamic)
│   ├── benchmark_memory.py   # Memory footprint and step latency profiling
│   └── qualify.py            # Geometric qualification suite
├── src/lgae_v3/
│   ├── core/                 # Compatibility layer and engine entrypoints
│   ├── curvature/            # Bakry-Émery, CDE', Entropic, Forman, LLY, Ollivier
│   ├── training/             # LGAETrainCore, padded buffers, and train loops
│   ├── compile_utils.py      # Torch compile utilities
│   ├── config.py             # Strongly typed dataclass configurations
│   ├── evolution.py          # Authoritative LGAEEngine
│   ├── fibers.py             # FixedWidthFiberLatent & SOConnectionBank
│   ├── governor.py           # GeometryGovernor & transition audits
│   ├── metrics.py            # Gauge-covariant sparse diffusion metrics
│   ├── mutations.py          # Log-conformal Ricci flow & graph surgeries
│   ├── operators.py          # Actuation & diagnostic Markov operators
│   ├── receipts.py           # Cryptographic receipt logging
│   └── topology.py           # NetworkX conversion, Betti numbers & PH
└── tests/                    # 18 test modules with 112 verified unit/regression tests
```

---

## License

This project is licensed under the [MIT License](LICENSE).
