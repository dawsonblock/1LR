# Prioritized Reading List — Discrete Curvature, Ricci Flow & Geometric Controllers

Ordered for building and hardening an LGAE-v3-style geometric controller. Each entry lists the most important theorems / results for implementation.

---

## Tier 0 — Must-read before coding (2025–2026 corrections)

### 1. Chen, Deng, Zheng, Chen — “Why Discrete Curvature Fails to Fully Capture Over-squashing in GNNs” (ICLR 2026)
- **Key result:** High negative curvature is *sufficient but not necessary* for over-squashing. Ollivier–Ricci misses ~30–40% of severely squashed edges.
- **Practical output:** Weighted Augmented Forman-3 (WAF3) improves detection; approximate WAF3 processes multi-million-edge graphs in tens of seconds.
- **Implication for controller:** Use WAF3 (or strong Augmented Forman) as the continuous candidate generator; never treat positive local ORC as a safety certificate.

### 2. Caich & Abbahaddou — Local–Global Geometric Insights via Entropic Curvature (arXiv:2607.22381)
- **Key result:** Graph *entropic* curvature (Lott–Sturm–Villani style displacement convexity of entropy on \(W_1\) geodesics). Weak proxy \(\kappa_w\) computable in \(O(k_{\mathrm{iter}} d_{\max}^2)\) per node.
- **Expansion paradox:** Sparsity + strong spectral expansion + positive entropic curvature cannot coexist at large scale. Oversmoothing and oversquashing sit at opposite ends of one spectrum.
- **Midpoint-Completion Rewiring:** Adding two-hop midpoints cannot decrease \(\kappa_w\) at the rewired node (monotonicity theorem).
- **Implication:** Local ORC/Forman cannot certify long-range transport. Use \(\kappa_w\) (or equivalent) as a global mutation constraint.

### 3. Ramos Olivé — Integral Ricci Curvature for Graphs (arXiv:2502.16465, Electron. J. Combin. 2025)
- **Key result:** Integral LLY quantity \(I_{\kappa_0}\) measuring how much curvature sits below a threshold. Proves Bonnet–Myers, Moore, and Lichnerowicz-type estimates **without** requiring the graph to be positively curved everywhere.
- **Implication:** Correct global certificate for mixed-curvature evolving graphs. Prefer \(I_{\kappa_0}\) over pointwise positivity constraints.

### 4. Reddy — Diffusion Operator Geometry of Feedforward Representations (arXiv:2605.01107)
- **Key result:** Gaussian-kernel diffusion operators on feature clouds are locally Lipschitz; hard \(k\)-NN adjacency is discontinuous at neighbor-order ties. On CIFAR ResNet snapshots the diffusion class chain moved far less under perturbation than a matched directed \(k\)-NN graph.
- **Implication:** Neural-geometry diagnostics (Hehl-style \(\rho(x)\), curvature gap) should be computed on soft diffusion operators, not hard \(k\)-NN graphs of embeddings.

---

## Tier 1 — Core geometric foundations

### 5. Hehl, von Renesse, Weber — Neural Feature Geometry Evolves as Discrete Ricci Flow (ICML 2026 Spotlight, arXiv:2509.22362)
- Feature-space geometric graphs evolve analogously to discrete Ricci flow across >20 000 networks.
- Class separability tracks community formation.
- Local Ricci evolution coefficients \(\rho(x)\) and curvature gap \(\Delta\mathcal{O}\).
- Practical outputs: geometry-informed early stopping and depth selection.
- **Use as:** local field diagnostic on soft diffusion operators (not hard k-NN).

### 6. Baptista et al. — Deep Learning as Ricci Flow (Sci. Rep. 2024, arXiv:2404.14265)
- Global “Ricci network flow” strength correlates with accuracy across 1 500+ classifiers.
- **Use as:** global training-health diagnostic (complementary to Hehl’s local field).

### 7. Topping, Di Giovanni, Chamberlain, Dong, Bronstein — Understanding over-squashing and bottlenecks on graphs via curvature (ICLR 2022, arXiv:2111.14522)
- Introduces Balanced Forman curvature.
- Proves negatively curved edges are responsible for over-squashing *in their setting*.
- Stochastic Discrete Ricci Flow (SDRF) rewiring.
- **Still foundational**, but later work shows negative curvature is not necessary for all oversquashing.

### 8. Münch & Wojciechowski — Ollivier Ricci curvature for general graph Laplacians (Adv. Math. 2019, arXiv:1712.00875)
- Limit-free characterization of (a form of) Ollivier/LLY via the Laplacian.
- Lower bound on Ollivier curvature ⇔ Lipschitz decay of heat-equation solutions (discrete Renesse–Sturm).
- Laplacian comparison, non-explosion, diameter bounds.
- **Preferred LLY implementation base.**

### 9. Lin, Lu, Yau — Ricci Curvature of Graphs (Tohoku Math. J. 2011)
- Original Lin–Lu–Yau definition as limit of Ollivier-type curvature with idleness.

### 10. Bauer, Horn, Lin, Lippner, Mangoubi, Yau — Li-Yau inequality on graphs (J. Diff. Geom. 2015, arXiv:1306.2561)
- Introduces CDE / CDE′ conditions that restore Li–Yau gradient estimates on graphs.
- Harnack inequalities, heat-kernel bounds, polynomial volume growth under non-negative curvature, strengthened Buser-type inequality.
- **Must implement \(\widetilde{\Gamma}_2\), not ordinary \(\Gamma_2\).**

### 11. Cushing, Liu, Peyerimhoff — Bakry–Émery curvature functions of graphs (arXiv:1606.01496)
- Curvature function \(\mathcal{K}_{G,x}(\mathcal{N})\); increasing/concave behaviour; local matrix / SDP characterization.
- Implement via local curvature matrix + eigenproblem.

---

## Tier 2 — Supporting theory & directed / higher-order extensions

### 12. Bai, Cheng, Hua / Bai — Discrete Einstein metrics on trees and unicyclic graphs (arXiv:2604.22449, arXiv:2607.14748)
- Existence/uniqueness of LLY Einstein metrics on trees via Perron eigenvector of an edge-indexed Ricci matrix.
- On unicyclic graphs: non-uniqueness or non-existence possible (e.g. triangle + pendant leaf has none).
- **Warning:** “Flow to constant LLY curvature” is not always well-posed.

### 13. Hehl — A condition for non-negative Lin-Lu-Yau curvature (arXiv:2502.03896)
- Degree thresholds guaranteeing non-negative LLY on finite graphs.

### 14. Directed LLY & Ricci flow
- Lin–Lu–Yau of digraphs via optimal transport couplings (arXiv:2606.16530).
- Core subgraphs of directed graphs via discrete Ricci curvature flow (arXiv:2512.07899).

### 15. Discrete scalar curvature as weighted sum of Ollivier–Ricci (arXiv:2510.04936)
- Continuum limit results linking discrete ORC to classical scalar curvature.

### 16. Samal, Jost et al. — Bakry–Émery-Ricci as network geometry measure (arXiv:2402.06616)
- Empirical comparison of BE, Ollivier, Forman on synthetic and real networks; related but non-identical geometry. Supports keeping BE as a distinct vertex-analytic layer.

### 17. Forman graph-to-hypergraph lift for long-range oversquashing (arXiv:2508.11390)

### 18. Survey / roadmap
- A roadmap for curvature-based geometric data analysis and learning (arXiv:2510.22599, 2025).
- Melanie Weber, SIAM News 2025 — Discrete Curvature and Applications in Graph Machine Learning.

---

## Classical anchors (always keep)

| Paper | Role |
|-------|------|
| Ollivier 2009 | Optimal-transport Ricci on metric spaces / Markov chains |
| Forman 2003 | Combinatorial Ricci via discrete Bochner–Weitzenböck |
| Lin–Lu–Yau 2011 | Parameter-free limit form of Ollivier on graphs |
| Münch–Wojciechowski 2017/2019 | Limit-free Laplacian characterization + heat-flow contraction |
| Bauer et al. 2015 | CDE′ + discrete Li–Yau |
| Cushing–Liu–Peyerimhoff | Bakry–Émery curvature functions on graphs |

---

## Implementation priority for LGAE-v3

1. Soft diffusion operator + \(\Gamma_Z\) transport pressure  
2. WAF3 / Augmented Forman candidate generation  
3. Limit-free LLY (Münch–Wojciechowski) on critical edges  
4. Weak entropic \(\kappa_w\) + integral LLY \(I_{\kappa_0}\) as mutation constraints  
5. Bakry–Émery matrix + sampled CDE′ after large mutations  
6. Hehl-style local Ricci coefficients on soft feature diffusion operators  
7. Persistent homology as orthogonal topological guard  

Do not implement “maximize all curvatures” as the objective. Implement **constrained multi-objective evolution** with role-aware prescribed curvature targets.
