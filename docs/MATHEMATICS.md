# Mathematical conventions

## Operators

`P` is row-stochastic. Positive Laplacian: `L = I-P`. Γ-calculus generator: `Δ = P-I`.

For vector latent state `z_i`:

`Γ_Z(i) = 1/2 sum_j P_ij ||z_j-z_i||^2`.

The sparse implementation evaluates this over directed row-stochastic edges, avoiding an `N×N×D` intermediate.

## AF3

For an unweighted undirected edge `(u,v)`:

`AF3 = 4 - deg(u) - deg(v) + 3*T(u,v)`.

The degree-weighted implementation is deliberately named a proxy rather than paper-exact WAF3.

## Ollivier and LLY

`κ_p(x,y)=1-W1(μ_x^p, μ_y^p)/d(x,y)`.

The reference LLY path solves the finite Lipschitz LP

`κ_LLY = inf_{Lip(f)<=1, f(y)-f(x)=1} [Δf(x)-Δf(y)]`.

An independent exact path uses `κ_LLY = 2 κ_{1/2}` where the theorem assumptions apply. Qualification tests require the two paths to agree on supported graphs.

The multiscale ORC diagnostic replaces one-hop lazy measures with uniform closed-ball measures at configured radii. It is explicitly a mesoscopic diagnostic, not a claim that every such choice is the unique continuum-limit construction.

## Integral and role-conditioned LLY

`I_{κ0} = sum_e max(0, κ0 - κ_LLY(e))`.

It is a deficit: **lower is better**.

For an edge role `r`, the controller can also accumulate

`I_role = sum_e max(0, κ*(role(e)) - κ_LLY(e))`.

This permits intentionally negative-curvature bridges while using different targets for clusters or memory edges.

## Bakry–Émery

For fixed vertex `x`, CD(K,N) is a generalized quadratic-form inequality. Γ has a nullspace containing variables outside the one-step neighborhood. Those variables cannot simply be discarded because Γ2 couples them to neighbor variables.

The implementation therefore partitions the numerator quadratic form into Γ-positive and Γ-null variables and minimizes over the null variables by a Schur complement:

`B_eff = B_pp - B_pn B_nn^+ B_np`.

It then solves the generalized eigenvalue problem `B_eff v = K A_eff v`. If a Γ-null direction makes the numerator unbounded below, curvature is reported as `-inf`.

Regression oracles include `K_inf=2` on K2, `K_inf=1` at a P4 endpoint, and `K_inf≈0.2928932188` at a P4 interior vertex under the normalized random-walk generator.

## Weak entropic curvature

The SLSQP routine maximizes the two-hop `H_L` objective. Solver failure returns an unqualified result; it does not substitute the uniform feasible point because that would overestimate `-2 log H*` and could make unsafe geometry look better.

If `S_2(z)` is empty, the implementation preserves the algorithmic value `κ_w(z)=+∞`.

## CDE′

The build evaluates a sampled violation statistic of

`Γ~2(f) >= (1/N) f^2 (Δ log f)^2 + K Γ(f)`,

where `Γ~2 = Γ2 - Γ(f, Γ(f)/f)`.

This Monte-Carlo routine is explicitly not advertised as an exact universal CDE′ proof.

## Gauge connections

For each fixed graph-buffer edge slot, an unconstrained raw matrix `R_e` is reduced to

`A_e = (R_e - R_e^T)/2 in so(d)`.

The connection is either

`U_e = exp(A_e)`

or the Cayley map

`U_e = (I - A_e/2)^(-1)(I + A_e/2)`.

For real skew-symmetric `A_e`, both are special-orthogonal; the reverse undirected direction uses `U_e^T`.

## Stabilized Sinkhorn W1

The approximate Ollivier backend works with

`log K_ij = -C_ij / epsilon`

and alternating log scalings

`log u = log a - logsumexp(log K + log v)`

`log v = log b - logsumexp(log K + log u)`.

The optimization cost can be normalized for conditioning, while the final plan is evaluated against the original metric cost. Exact linear programming remains the reference path.

## Log-conformal Ricci update

The edge metric update is integrated in log weight:

`d log(w_e)/dt = -(kappa_e-kappa_target)`

so one step is

`w_e' = clamp(w_e exp(-dt(kappa_e-kappa_target)), w_min, w_max)`.

The exponential update is positive before clamping and therefore cannot cross through zero as a linear Euler weight update can.

## Spectral gap

For weighted undirected graph adjacency `A` and weighted degree `D`, the safety gap is the second eigenvalue of

`L_sym = I - D^(-1/2) A D^(-1/2)`.

Small graphs use exact symmetric eigendecomposition. Larger graphs can use sparse LOBPCG for the two smallest eigenpairs. Isolated vertices are treated as zero connectivity rather than passed through an invalid normalization.
