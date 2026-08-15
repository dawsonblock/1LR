<div align="center">

# LGAE-v3.2: 1LR (Laplacian Geometric Adaptive Evolution)

**A Multi-Timescale Geometric Controller for Self-Evolving Graph and Fiber-Bundle Latent Spaces**

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.6+](https://img.shields.io/badge/PyTorch-2.6+-ee4c2c.svg)](https://pytorch.org/)
[![Tests](https://img.shields.io/badge/tests-69%2F69%20passing-brightgreen.svg)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Gauge: SO(d)](https://img.shields.io/badge/gauge-SO(d)%20Invariance-purple.svg)]()

</div>

---

## Overview

**LGAE-v3.2 (`1LR`)** is a hardened geometric deep learning framework and dynamical controller. It operates over graph-structured data and continuous fiber bundles, combining continuous field diffusion, Lie-algebra gauge connections, discrete Ricci-flow surgery, and rigorous multi-operator curvature audits.

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

## Key Features & Hardening (v3.2)

- **$\mathrm{SO}(d)$ Gauge Connection Bank (`SOConnectionBank`)**:
  - Parameters live in Lie algebra $\mathfrak{so}(d)$ via skew-symmetric mapping $A_e = \frac{1}{2}(R_e - R_e^T)$.
  - Exact Cayley or Matrix Exponential ($\exp(A_e)$) retractions ensure every connection strictly satisfies $U_e^T U_e = I$ and $\det(U_e) = +1$ across arbitrary Adam/SGD steps.
  - Reverse transport uses $U_e^T = U_e^{-1}$.
  - Fixed-capacity buffer slot indexing prevents tensor reallocation during topology mutations.

- **Stable Optimal Transport (Log-Sinkhorn Ollivier Curvature)**:
  - Log-domain scaling eliminates underflow at low entropic regularization.
  - Zero-mass support rows/columns are cleanly removed.
  - Convergence certified against recovered coupling marginal residuals.
  - Reference linear programming (`exact_lp`) retained as exact ground-truth oracle.

- **Reversible $\Gamma$-Calculus & Bakry–Émery ($CD(K, N)$)**:
  - Reversible row-stochastic Markov generators $\Delta = P - I$ with detailed-balance volume measure reconstruction.
  - Float64 row renormalization and diagonal ULP cancellation.
  - Full Schur complement $B_{\text{eff}} = B_{pp} - B_{pn} B_{nn}^+ B_{np}$ eliminates $\Gamma$-null directions without false positives.

- **Log-Conformal Ricci Flow & Surgery Hysteresis**:
  - Multiplicative exponential updates $w \leftarrow \text{clamp}(w \cdot \exp(-\Delta t(\kappa - \kappa^*)), w_{\min}, w_{\max})$ guarantee weight positivity.
  - Per-edge cooldown tracker and distinct add/deadband/prune thresholds prevent edge flapping.
  - $O(V+E)$ bridge filter immediately blocks disconnecting deletions.

- **`torch.compile` Compatibility & Buffer Management**:
  - Bucketed fixed-capacity graph buffers (`GraphBuffers`) with in-place value refresh (`refresh_padded_markov_edges_`).
  - Dormant fiber channels zeroed out to prevent hidden latent energy buildup.

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

# Install in editable mode with dev dependencies
python -m pip install -e '.[dev]' --no-build-isolation
```

---

## Quickstart & Examples

### 1. Basic Engine & Gauge Parallel Transport

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

# Create engine and perform gauge-covariant diffusion
engine = LGAEEngine(graph, cfg)
z_next = engine.diffuse_(eta=0.01)

# Verify SO(d) invariants
orth_err, det_err = engine.gauge_connections.invariant_error()
print(f"Max orthogonality error: {orth_err.max():.2e}")
print(f"Max determinant error:   {det_err.max():.2e}")
```

### 2. Differentiable Training Core

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

# Execute one step
metrics = train_step(
    core, engine, optimizer,
    target=target, src=src, dst=dst, weight=w, valid=valid,
    bottleneck_pressure=pressure, edge_slot=slot, reverse=reverse,
    step=0, spawn_interval=50
)
print("Loss:", metrics["loss"].item())
```

### 3. Curvature Auditing & Mutation Governance

```python
from lgae_v3.mutations import AddEdge, PruneEdge, ReweightEdge

# Propose an edge addition
mutation = engine.propose_midpoint_edge()

# Shadow-evaluate and govern transaction
result = engine.evaluate_and_maybe_commit(mutation)
print("Decision:", result.decision.value)  # 'accept', 'reject', or 'quarantine'
print("Reasons:", result.reasons)
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
└── tests/                    # 69 test modules covering 100% of functional paths
```

---

## License

This project is licensed under the [MIT License](LICENSE).
