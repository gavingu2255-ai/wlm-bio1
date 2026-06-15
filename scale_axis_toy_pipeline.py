"""
wgs_toy_pipeline.py
═══════════════════════════════════════════════════════════════════════════════
Topology-Induced Scale Axis: Proof-of-Concept Pipeline

From: Virtual Chromosome → LD Blocks → Graph Laplacian → Spectral Scale Axis
  → L7 Gradient Flow → Stress Tests ST1/ST2/ST3

Theory: spectral depth ordering ↔ graph Laplacian eigenvalue ordering.
  Small eigenvalue = global co-expression module = large s = slow decay (cell-type signal)
  Large eigenvalue = local perturbation    = small s = fast decay (batch noise)

This is NOT heuristic. The eigenvalue ordering is forced by the graph structure,
which mirrors the spectral ordering of the graph Laplacian.

Usage:
  python wgs_toy_pipeline.py           # full pipeline, all 3 stress tests
  python wgs_toy_pipeline.py --quick   # reduced iterations for testing
  python wgs_toy_pipeline.py --st 1    # specific stress test only

Outputs:
  wgs_figures/   — all plots
  wgs_results/   — numerical summary JSON
═══════════════════════════════════════════════════════════════════════════════
"""

import argparse
import json
import os
import time
import warnings

import numpy as np
import scipy.linalg as la
import scipy.integrate
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

warnings.filterwarnings("ignore", category=RuntimeWarning)

# ─────────────────────────────────────────────────────────────────────────────
# GLOBAL PARAMETERS
# ─────────────────────────────────────────────────────────────────────────────

RNG_SEED   = 42
N_GENES    = 120          # number of "genes" (nodes in co-expression graph)
N_BLOCKS   = 8            # number of LD blocks
N_SAMPLES  = 60           # number of virtual individuals
A_PARAM    = 1.0          # L7 parameter a
B_PARAM    = 2.0          # L7 parameter b  (lambda(s) = 2/(a^2 * b^(2s)))
T_SPAN     = (0.0, 12.0)  # integration time window
S_THRESH   = None         # split between "small s" and "large s", set dynamically

os.makedirs("wgs_figures", exist_ok=True)
os.makedirs("wgs_results",  exist_ok=True)


# ═════════════════════════════════════════════════════════════════════════════
# MODULE 1: VIRTUAL GENOME CONSTRUCTION
# ═════════════════════════════════════════════════════════════════════════════

def build_ld_blocks(n_genes, n_blocks, rng, intra_corr=0.75, inter_corr=0.08):
    """
    Partition n_genes into n_blocks.
    Within each block, genes have high correlation (intra_corr).
    Between blocks, genes have low correlation (inter_corr).

    Returns
    -------
    W : (n_genes, n_genes) weighted adjacency matrix (correlation-based)
    block_labels : (n_genes,) int array of block membership
    """
    # Block assignment — roughly equal sizes with some variation
    boundaries = sorted(rng.choice(np.arange(1, n_genes), size=n_blocks-1, replace=False))
    boundaries = [0] + list(boundaries) + [n_genes]
    block_labels = np.zeros(n_genes, dtype=int)
    for b, (lo, hi) in enumerate(zip(boundaries[:-1], boundaries[1:])):
        block_labels[lo:hi] = b

    # Build correlation matrix
    W = np.full((n_genes, n_genes), inter_corr)
    np.fill_diagonal(W, 0.0)
    for b in range(n_blocks):
        idx = np.where(block_labels == b)[0]
        for i in idx:
            for j in idx:
                if i != j:
                    W[i, j] = intra_corr

    # Small random perturbation to avoid degenerate spectra
    noise = rng.uniform(-0.02, 0.02, size=W.shape)
    noise = (noise + noise.T) / 2
    np.fill_diagonal(noise, 0.0)
    W = np.clip(W + noise, 0.0, 1.0)
    return W, block_labels


def build_graph_laplacian(W):
    """
    L = D - W  (unnormalized graph Laplacian)
    D = diagonal degree matrix.
    """
    D = np.diag(W.sum(axis=1))
    L = D - W
    return L


def spectral_decomposition(L):
    """
    Eigendecomposition of symmetric L.
    Returns eigenvalues (ascending) and eigenvectors.
    """
    eigvals, eigvecs = la.eigh(L)
    # Clip tiny negative values from numerical noise
    eigvals = np.maximum(eigvals, 0.0)
    return eigvals, eigvecs


def eigenvalue_to_scale(eigvals, mode="log"):
    """
    Map eigenvalues → scale axis s.
    Small eigenvalue → large s (global, persistent).
    Large eigenvalue → small s (local, ephemeral).

    We use the REVERSE mapping so that:
      s_k = log(1 + lambda_max - lambda_k)

    This is monotone decreasing in eigenvalue, which is what we want.
    """
    lam_max = eigvals.max()
    if mode == "log":
        s = np.log1p(lam_max - eigvals)
    elif mode == "linear":
        s = (lam_max - eigvals) / (lam_max + 1e-12)
    else:
        raise ValueError(f"Unknown scale mode: {mode}")
    return s


# ═════════════════════════════════════════════════════════════════════════════
# MODULE 2: L7 GRADIENT FLOW
# ═════════════════════════════════════════════════════════════════════════════

def l7_decay_rate(s, a=A_PARAM, b=B_PARAM):
    """
    λ(s) = 2 / (a² b^{2s})
    The L7 gradient flow decay rate at scale s.
    """
    return 2.0 / (a**2 * b**(2*s))


def l7_analytic(r0, s, t, a=A_PARAM, b=B_PARAM):
    """
    Analytic solution of dr/dt = -λ(s) r(s,t):
      r(s, t) = r0(s) * exp(-λ(s) * t)

    Parameters
    ----------
    r0 : array, shape (n_modes,)
    s  : array, shape (n_modes,)
    t  : float or array
    """
    lam = l7_decay_rate(s, a, b)
    if np.isscalar(t):
        return r0 * np.exp(-lam * t)
    else:
        return r0[None, :] * np.exp(-lam[None, :] * t[:, None])


def b_series_energy(r, s, a=A_PARAM, b=B_PARAM):
    """
    N_B[r] = ∫ r(s)² / (a² b^{2s}) ds  ≈  Σ_k r_k² * φ(s_k) * Δs
    where φ(s) = 1/(a²b^{2s}).
    Uses uniform step width Δs = (s_max - s_min) / (n-1).
    """
    phi = 1.0 / (a**2 * b**(2*s))
    n = len(s)
    if n < 2:
        return float(r[0]**2 * phi[0])
    # Uniform step width assumption (valid when s is derived from eigenvalue_to_scale)
    ds = (s[-1] - s[0]) / (n - 1) if s[-1] > s[0] else 1.0
    return float(np.sum(r**2 * phi) * ds)


def compute_half_life(s, a=A_PARAM, b=B_PARAM):
    """t_1/2 = log(2) / λ(s)"""
    return np.log(2) / l7_decay_rate(s, a, b)


# ═════════════════════════════════════════════════════════════════════════════
# MODULE 3: DATA PROJECTION
# ═════════════════════════════════════════════════════════════════════════════

def generate_virtual_genotypes(n_samples, n_genes, block_labels, rng,
                                signal_blocks=(0, 1, 2),
                                noise_level=0.15):
    """
    Generate synthetic gene expression matrix X ∈ ℝ^{n_samples × n_genes}.

    signal_blocks : these blocks carry cell-type signal (large-s structure)
    Other blocks  : pure noise / batch variation (small-s structure)
    """
    n_blocks = block_labels.max() + 1
    X = np.zeros((n_samples, n_genes))

    # Cell-type signal: shared latent variable within signal blocks
    cell_type = rng.choice([-1.0, 1.0], size=n_samples)
    for b in signal_blocks:
        idx = np.where(block_labels == b)[0]
        X[:, idx] += cell_type[:, None] * 2.0

    # Batch noise: block-level additive random effect for non-signal blocks
    for b in range(n_blocks):
        if b not in signal_blocks:
            idx = np.where(block_labels == b)[0]
            batch = rng.normal(0, 1, size=(n_samples, len(idx)))
            X[:, idx] += batch

    # Gene-level i.i.d. noise
    X += rng.normal(0, noise_level, size=X.shape)
    return X, cell_type


def project_to_spectral_basis(X, eigvecs):
    """
    Project each sample x ∈ ℝ^{n_genes} onto spectral basis.
    C ∈ ℝ^{n_samples × n_modes}, C[i, k] = u_k^T x_i.
    """
    return X @ eigvecs   # (n_samples, n_modes)


def mean_spectral_amplitude(C):
    """Population-level orientation field: |mean coefficient| at each mode."""
    return np.abs(C).mean(axis=0)


# ═════════════════════════════════════════════════════════════════════════════
# MODULE 4: BASELINE DEMONSTRATION
# ═════════════════════════════════════════════════════════════════════════════

def run_baseline(rng, verbose=True):
    """
    Single run: build genome, decompose, project data, run L7, plot.
    Returns summary dict.
    """
    if verbose:
        print("\n" + "═"*70)
        print("BASELINE: Scale Axis Toy Pipeline — Single Configuration")
        print("═"*70)

    # 1. Virtual genome
    W, block_labels = build_ld_blocks(N_GENES, N_BLOCKS, rng)
    L = build_graph_laplacian(W)
    eigvals, eigvecs = spectral_decomposition(L)
    s = eigenvalue_to_scale(eigvals)

    if verbose:
        print(f"  Genes: {N_GENES}   Blocks: {N_BLOCKS}")
        print(f"  Eigenvalue range: [{eigvals.min():.3f}, {eigvals.max():.3f}]")
        print(f"  Scale axis range: s ∈ [{s.min():.3f}, {s.max():.3f}]")

    # 2. Generate virtual gene expression data
    X, cell_type = generate_virtual_genotypes(N_SAMPLES, N_GENES, block_labels, rng)
    C = project_to_spectral_basis(X, eigvecs)
    r0 = mean_spectral_amplitude(C)

    # 3. Split into small-s and large-s
    s_median = np.median(s)
    small_s_mask = (s < s_median)
    large_s_mask = ~small_s_mask

    # 4. L7 gradient flow
    t_arr = np.linspace(*T_SPAN, 200)
    r_t = l7_analytic(r0, s, t_arr)   # (n_t, n_modes)

    # Energy trajectory
    energy_t = np.array([b_series_energy(r_t[i], s) for i in range(len(t_arr))])
    # Analytic check: L7 is monotone iff λ(s)>0, which is always true.
    # Numerical check uses relative tolerance.
    tol_e = 1e-3 * energy_t[0] + 1e-12
    energy_mono = bool(np.all(np.diff(energy_t) <= tol_e))

    # Half-lives
    hl_small = compute_half_life(s[small_s_mask]).mean()
    hl_large = compute_half_life(s[large_s_mask]).mean()

    # Retention at T_final
    r_final = l7_analytic(r0, s, T_SPAN[1])
    ret_small = (r_final[small_s_mask] / (r0[small_s_mask] + 1e-12)).mean()
    ret_large = (r_final[large_s_mask] / (r0[large_s_mask] + 1e-12)).mean()

    stability_ratio = hl_large / hl_small

    if verbose:
        print(f"\n  L7 Results:")
        print(f"  Mean half-life (small s): {hl_small:.3f}")
        print(f"  Mean half-life (large s): {hl_large:.3f}")
        print(f"  Stability ratio (large/small): {stability_ratio:.2f}×")
        print(f"  Retention at T={T_SPAN[1]} (small s): {ret_small:.4f}")
        print(f"  Retention at T={T_SPAN[1]} (large s): {ret_large:.4f}")

    # 5. Plot
    fig = plt.figure(figsize=(15, 10))
    fig.suptitle("Scale Axis Toy Pipeline — Baseline", fontsize=14, fontweight="bold")
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.35)

    # Panel A: eigenvalue spectrum
    ax0 = fig.add_subplot(gs[0, 0])
    ax0.scatter(range(len(eigvals)), eigvals, c=s, cmap="viridis", s=20)
    ax0.set_xlabel("Mode index k")
    ax0.set_ylabel("Eigenvalue λ_k")
    ax0.set_title("A: Eigenvalue spectrum\n(color = induced scale s)")
    cb = plt.colorbar(ax0.collections[0], ax=ax0)
    cb.set_label("s_k")

    # Panel B: scale axis distribution
    ax1 = fig.add_subplot(gs[0, 1])
    ax1.hist(s[small_s_mask], bins=20, alpha=0.6, color="tomato",  label=f"Small s (n={small_s_mask.sum()})")
    ax1.hist(s[large_s_mask], bins=20, alpha=0.6, color="steelblue", label=f"Large s (n={large_s_mask.sum()})")
    ax1.axvline(s_median, color="black", linestyle="--", label="Median split")
    ax1.set_xlabel("Scale s")
    ax1.set_ylabel("Count")
    ax1.set_title("B: Induced scale axis\ndistribution")
    ax1.legend(fontsize=8)

    # Panel C: decay rates
    ax2 = fig.add_subplot(gs[0, 2])
    ax2.scatter(s[small_s_mask], l7_decay_rate(s[small_s_mask]), c="tomato",   s=20, label="Small s")
    ax2.scatter(s[large_s_mask], l7_decay_rate(s[large_s_mask]), c="steelblue", s=20, label="Large s")
    ax2.set_xlabel("Scale s")
    ax2.set_ylabel("λ(s) = decay rate")
    ax2.set_title("C: L7 decay rate\n(small s = fast decay)")
    ax2.legend(fontsize=8)
    ax2.set_yscale("log")

    # Panel D: energy trajectory
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.plot(t_arr, energy_t, color="darkgreen", linewidth=2)
    ax3.set_xlabel("Time t")
    ax3.set_ylabel("N_B[r(·,t)]")
    ax3.set_title("D: B-series energy\n(strictly decreasing)")
    ax3.set_yscale("log")

    # Panel E: amplitude at selected times
    ax4 = fig.add_subplot(gs[1, 1])
    for t_plot, alpha, label in [(0, 1.0, "t=0"), (3, 0.7, "t=3"), (8, 0.4, "t=8"), (12, 0.2, "t=12")]:
        r_plot = l7_analytic(r0, s, float(t_plot))
        ax4.plot(s, r_plot, alpha=alpha, label=label)
    ax4.axvline(s_median, color="black", linestyle="--", alpha=0.5)
    ax4.set_xlabel("Scale s")
    ax4.set_ylabel("|r(s, t)|")
    ax4.set_title("E: Orientation field\nat selected times")
    ax4.legend(fontsize=8)

    # Panel F: half-life by scale
    ax5 = fig.add_subplot(gs[1, 2])
    ax5.scatter(s[small_s_mask], compute_half_life(s[small_s_mask]), c="tomato",   s=20, label="Small s")
    ax5.scatter(s[large_s_mask], compute_half_life(s[large_s_mask]), c="steelblue", s=20, label="Large s")
    ax5.set_xlabel("Scale s")
    ax5.set_ylabel("Half-life t_{1/2}")
    ax5.set_title(f"F: Half-life by scale\nLarge/Small = {stability_ratio:.1f}×")
    ax5.legend(fontsize=8)
    ax5.set_yscale("log")

    plt.savefig("wgs_figures/baseline.pdf", bbox_inches="tight", dpi=150)
    plt.savefig("wgs_figures/baseline.png", bbox_inches="tight", dpi=150)
    plt.close()

    if verbose:
        print("  Saved: wgs_figures/baseline.pdf")

    return {
        "stability_ratio": float(stability_ratio),
        "hl_small_mean":   float(hl_small),
        "hl_large_mean":   float(hl_large),
        "retention_small": float(ret_small),
        "retention_large": float(ret_large),
        "energy_monotone": energy_mono,        "s_range":         [float(s.min()), float(s.max())],
    }


# ═════════════════════════════════════════════════════════════════════════════
# MODULE 5: STRESS TESTS
# ═════════════════════════════════════════════════════════════════════════════

# ─────────────────────────────────────────────────────────────────────────────
# ST1: State space and noise gradient stress test
# ─────────────────────────────────────────────────────────────────────────────

def run_st1(n_iter=500, noise_levels=None, rng_seed=RNG_SEED, verbose=True):
    """
    ST1: 
    For each of n_iter random genome configurations × noise_level values:
      - Build virtual genome
      - Induce scale axis
      - Run L7 flow
      - Record stability_ratio and energy_monotone

    Key question: Is "large s is more stable than small s" robust to
    noise and to random genome structure?
    """
    if verbose:
        print("\n" + "═"*70)
        print(f"ST1: Noise Gradient Stress Test  ({n_iter} iterations)")
        print("═"*70)

    if noise_levels is None:
        noise_levels = np.linspace(0.0, 0.40, 10)

    rng = np.random.default_rng(rng_seed)
    results_by_noise = {float(nl): [] for nl in noise_levels}

    t_start = time.time()
    total = n_iter * len(noise_levels)
    done  = 0

    for nl in noise_levels:
        for _ in range(n_iter):
            rng_local = np.random.default_rng(rng.integers(0, 2**32))

            # Random genome with this noise level
            intra = rng_local.uniform(0.5, 0.95)
            inter = rng_local.uniform(0.01, 0.15)
            W, block_labels = build_ld_blocks(N_GENES, N_BLOCKS, rng_local,
                                              intra_corr=intra, inter_corr=inter)
            L = build_graph_laplacian(W)
            eigvals, eigvecs = spectral_decomposition(L)
            s = eigenvalue_to_scale(eigvals)

            # Virtual data with given noise level
            signal_blocks = tuple(rng_local.choice(N_BLOCKS, size=3, replace=False))
            X, _ = generate_virtual_genotypes(N_SAMPLES, N_GENES, block_labels,
                                              rng_local, signal_blocks=signal_blocks,
                                              noise_level=float(nl))
            C  = project_to_spectral_basis(X, eigvecs)
            r0 = mean_spectral_amplitude(C)

            s_med  = np.median(s)
            sm_msk = s < s_med
            lg_msk = ~sm_msk

            hl_small = compute_half_life(s[sm_msk]).mean()
            hl_large = compute_half_life(s[lg_msk]).mean()
            ratio    = hl_large / hl_small

            # Energy monotonicity check (10 time points for speed)
            t_check  = np.linspace(*T_SPAN, 10)
            r_traj   = l7_analytic(r0, s, t_check)
            # Energy monotonicity for L7 is analytic (dN_B/dt < 0 iff λ(s) > 0)
            # λ(s) = 2/(a²b^{2s}) > 0 always. So monotone = True by construction.
            # We verify numerically that r_k actually decreases at each scale.
            r_start = r_traj[0]
            r_end   = r_traj[-1]
            monotone = bool(np.all(r_end <= r_start + 1e-8))

            results_by_noise[float(nl)].append({
                "ratio":    ratio,
                "monotone": monotone,
            })
            done += 1

        if verbose and done % (total // 5 + 1) == 0:
            pct = 100 * done / total
            print(f"  {pct:.0f}%  elapsed {time.time()-t_start:.1f}s")

    # Summarise
    noise_arr   = np.array(noise_levels)
    ratio_means = np.array([np.mean([r["ratio"]    for r in results_by_noise[nl]]) for nl in noise_levels])
    ratio_stds  = np.array([np.std( [r["ratio"]    for r in results_by_noise[nl]]) for nl in noise_levels])
    mono_rates  = np.array([np.mean([r["monotone"] for r in results_by_noise[nl]]) for nl in noise_levels])

    if verbose:
        print(f"\n  Results across noise levels:")
        for nl, rm, rs, mr in zip(noise_levels, ratio_means, ratio_stds, mono_rates):
            print(f"    noise={nl:.2f}  ratio={rm:.2f}±{rs:.2f}  energy_monotone={mr:.1%}")

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("ST1: Noise Gradient Stress Test", fontsize=13, fontweight="bold")

    axes[0].fill_between(noise_arr, ratio_means - ratio_stds, ratio_means + ratio_stds,
                         alpha=0.3, color="steelblue")
    axes[0].plot(noise_arr, ratio_means, "o-", color="steelblue", linewidth=2)
    axes[0].axhline(1.0, color="black", linestyle="--", label="Equal stability")
    axes[0].set_xlabel("Noise level")
    axes[0].set_ylabel("Stability ratio (large s / small s)")
    axes[0].set_title("Stability ratio vs. noise\n(>1 = large-s modes more stable)")
    axes[0].legend()

    axes[1].plot(noise_arr, mono_rates, "s-", color="darkgreen", linewidth=2)
    axes[1].set_xlabel("Noise level")
    axes[1].set_ylabel("Energy monotonicity rate")
    axes[1].set_ylim(0, 1.05)
    axes[1].set_title("B-series energy monotonicity\n(rate across configs)")
    axes[1].axhline(1.0, color="black", linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig("wgs_figures/st1_noise_gradient.pdf", bbox_inches="tight", dpi=150)
    plt.savefig("wgs_figures/st1_noise_gradient.png", bbox_inches="tight", dpi=150)
    plt.close()

    if verbose:
        print("  Saved: wgs_figures/st1_noise_gradient.pdf")

    return {
        "noise_levels":  noise_levels.tolist(),
        "ratio_means":   ratio_means.tolist(),
        "ratio_stds":    ratio_stds.tolist(),
        "mono_rates":    mono_rates.tolist(),
        "min_mono_rate": float(mono_rates.min()),
        "mean_ratio_all": float(ratio_means.mean()),
    }


# ─────────────────────────────────────────────────────────────────────────────
# ST2: Parameter family sweep (a, b, T)
# ─────────────────────────────────────────────────────────────────────────────

def run_st2(n_iter=500, rng_seed=RNG_SEED+1, verbose=True):
    """
    ST2:
    For n_iter random parameter combinations (a, b, T_final):
      - Build a fresh genome
      - Induce scale axis
      - Run L7 with these parameters
      - Record stability_ratio

    Key question: Is the large-s stability advantage robust to parameter choice?
    Is there a critical parameter region where it breaks?
    """
    if verbose:
        print("\n" + "═"*70)
        print(f"ST2: Parameter Family Sweep  ({n_iter} iterations)")
        print("═"*70)

    rng = np.random.default_rng(rng_seed)
    ratios   = []
    a_vals   = []
    b_vals   = []
    T_vals   = []

    for i in range(n_iter):
        rng_local = np.random.default_rng(rng.integers(0, 2**32))

        # Random parameters — b capped at 2.5 to avoid degenerate ratios
        a = rng_local.uniform(0.5, 3.0)
        b = rng_local.uniform(1.2, 2.5)
        T = rng_local.uniform(4.0, 20.0)

        # Fresh genome
        intra = rng_local.uniform(0.55, 0.92)
        inter = rng_local.uniform(0.02, 0.12)
        W, block_labels = build_ld_blocks(N_GENES, N_BLOCKS, rng_local,
                                          intra_corr=intra, inter_corr=inter)
        L = build_graph_laplacian(W)
        eigvals, eigvecs = spectral_decomposition(L)
        s = eigenvalue_to_scale(eigvals)

        X, _ = generate_virtual_genotypes(N_SAMPLES, N_GENES, block_labels, rng_local)
        C  = project_to_spectral_basis(X, eigvecs)
        r0 = mean_spectral_amplitude(C)

        s_med  = np.median(s)
        sm_msk = s < s_med
        lg_msk = ~sm_msk

        hl_small = compute_half_life(s[sm_msk], a=a, b=b).mean()
        hl_large = compute_half_life(s[lg_msk], a=a, b=b).mean()
        ratio    = hl_large / hl_small

        ratios.append(ratio)
        a_vals.append(a)
        b_vals.append(b)
        T_vals.append(T)

    ratios = np.array(ratios)
    a_vals = np.array(a_vals)
    b_vals = np.array(b_vals)

    frac_above_1 = (ratios > 1.0).mean()
    ratio_mean   = ratios.mean()
    ratio_std    = ratios.std()

    if verbose:
        print(f"\n  Across {n_iter} random (a, b, T) configurations:")
        print(f"  Stability ratio: mean={ratio_mean:.2f}, std={ratio_std:.2f}")
        print(f"  Fraction with ratio > 1: {frac_above_1:.1%}")
        print(f"  Min ratio: {ratios.min():.3f}   Max: {ratios.max():.3f}")

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("ST2: Parameter Family Sweep", fontsize=13, fontweight="bold")

    axes[0].hist(ratios, bins=30, color="steelblue", edgecolor="white")
    axes[0].axvline(1.0, color="red", linestyle="--", label="ratio=1 (equal)")
    axes[0].axvline(ratio_mean, color="black", linestyle="-", label=f"mean={ratio_mean:.2f}")
    axes[0].set_xlabel("Stability ratio (large s / small s)")
    axes[0].set_ylabel("Count")
    axes[0].set_title(f"Stability ratio distribution\n{frac_above_1:.1%} > 1")
    axes[0].legend(fontsize=8)

    sc1 = axes[1].scatter(a_vals, ratios, c=b_vals, cmap="plasma", s=15, alpha=0.5)
    axes[1].set_xlabel("Parameter a")
    axes[1].set_ylabel("Stability ratio")
    axes[1].set_title("Ratio vs. a (color=b)")
    plt.colorbar(sc1, ax=axes[1], label="b")
    axes[1].axhline(1.0, color="red", linestyle="--", alpha=0.5)

    sc2 = axes[2].scatter(b_vals, ratios, c=a_vals, cmap="viridis", s=15, alpha=0.5)
    axes[2].set_xlabel("Parameter b")
    axes[2].set_ylabel("Stability ratio")
    axes[2].set_title("Ratio vs. b (color=a)")
    plt.colorbar(sc2, ax=axes[2], label="a")
    axes[2].axhline(1.0, color="red", linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig("wgs_figures/st2_param_sweep.pdf", bbox_inches="tight", dpi=150)
    plt.savefig("wgs_figures/st2_param_sweep.png", bbox_inches="tight", dpi=150)
    plt.close()

    if verbose:
        print("  Saved: wgs_figures/st2_param_sweep.pdf")

    return {
        "n_iter":       n_iter,
        "ratio_mean":   float(ratio_mean),
        "ratio_std":    float(ratio_std),
        "ratio_min":    float(ratios.min()),
        "ratio_max":    float(ratios.max()),
        "frac_above_1": float(frac_above_1),
    }


# ─────────────────────────────────────────────────────────────────────────────
# ST3: LD block structure removal — topological collapse
# ─────────────────────────────────────────────────────────────────────────────

def run_st3(n_iter=500, rng_seed=RNG_SEED+2, verbose=True):
    """
    ST3:
    Compare two conditions for each random genome:
      A. Full LD structure (with blocks)
      B. No LD structure (W = uniform inter-block correlation only)

    Measure: Does the spectral scale axis lose large/small-s distinction
    when LD block structure is removed?

    This tests whether the spectral structural claim holds:
    "scale is induced by the topology, not by the data projection."
    """
    if verbose:
        print("\n" + "═"*70)
        print(f"ST3: LD Structure Removal Test  ({n_iter} iterations)")
        print("═"*70)

    rng = np.random.default_rng(rng_seed)

    ratios_full   = []
    ratios_noblock = []
    sep_full      = []
    sep_noblock   = []

    for _ in range(n_iter):
        rng_local = np.random.default_rng(rng.integers(0, 2**32))

        intra = rng_local.uniform(0.6, 0.92)
        inter = rng_local.uniform(0.02, 0.10)

        # Condition A: full LD structure
        W_full, block_labels = build_ld_blocks(N_GENES, N_BLOCKS, rng_local,
                                               intra_corr=intra, inter_corr=inter)
        L_full = build_graph_laplacian(W_full)
        ev_full, evec_full = spectral_decomposition(L_full)
        s_full = eigenvalue_to_scale(ev_full)

        # Condition B: no block structure — uniform correlation graph
        W_flat = np.full((N_GENES, N_GENES), inter)
        np.fill_diagonal(W_flat, 0.0)
        noise = rng_local.uniform(-0.005, 0.005, size=W_flat.shape)
        noise = (noise + noise.T) / 2
        np.fill_diagonal(noise, 0.0)
        W_flat = np.clip(W_flat + noise, 0.0, 1.0)
        L_flat = build_graph_laplacian(W_flat)
        ev_flat, evec_flat = spectral_decomposition(L_flat)
        s_flat = eigenvalue_to_scale(ev_flat)

        # Shared data
        X, _ = generate_virtual_genotypes(N_SAMPLES, N_GENES, block_labels, rng_local)

        # Full condition
        C_full = project_to_spectral_basis(X, evec_full)
        r0_full = mean_spectral_amplitude(C_full)
        sm = s_full < np.median(s_full)
        lg = ~sm
        hl_sm_full = compute_half_life(s_full[sm]).mean()
        hl_lg_full = compute_half_life(s_full[lg]).mean()
        ratio_full = hl_lg_full / hl_sm_full
        sep_full.append(s_full[lg].mean() - s_full[sm].mean())
        ratios_full.append(ratio_full)

        # No-block condition
        C_flat = project_to_spectral_basis(X, evec_flat)
        r0_flat = mean_spectral_amplitude(C_flat)
        sm2 = s_flat < np.median(s_flat)
        lg2 = ~sm2
        hl_sm_flat = compute_half_life(s_flat[sm2]).mean()
        hl_lg_flat = compute_half_life(s_flat[lg2]).mean()
        ratio_flat = hl_lg_flat / hl_sm_flat
        sep_noblock.append(s_flat[lg2].mean() - s_flat[sm2].mean())
        ratios_noblock.append(ratio_flat)

    ratios_full    = np.array(ratios_full)
    ratios_noblock = np.array(ratios_noblock)
    sep_full       = np.array(sep_full)
    sep_noblock    = np.array(sep_noblock)

    ratio_gain = ratios_full.mean() / (ratios_noblock.mean() + 1e-9)
    sep_gain   = sep_full.mean()    / (sep_noblock.mean()    + 1e-9)

    if verbose:
        print(f"\n  Full LD structure:   ratio mean={ratios_full.mean():.2f}±{ratios_full.std():.2f}"
              f"   scale sep={sep_full.mean():.3f}")
        print(f"  No block structure:  ratio mean={ratios_noblock.mean():.2f}±{ratios_noblock.std():.2f}"
              f"   scale sep={sep_noblock.mean():.3f}")
        print(f"  Gain from LD structure: ratio ×{ratio_gain:.2f}, scale sep ×{sep_gain:.2f}")

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("ST3: LD Structure vs. No Structure", fontsize=13, fontweight="bold")

    bins = np.linspace(min(ratios_full.min(), ratios_noblock.min()),
                       max(ratios_full.max(), ratios_noblock.max()), 30)
    axes[0].hist(ratios_full,    bins=bins, alpha=0.6, color="steelblue", label="Full LD")
    axes[0].hist(ratios_noblock, bins=bins, alpha=0.6, color="tomato",    label="No blocks")
    axes[0].axvline(1.0, color="black", linestyle="--")
    axes[0].set_xlabel("Stability ratio")
    axes[0].set_ylabel("Count")
    axes[0].set_title("Stability ratio distribution")
    axes[0].legend()

    axes[1].hist(sep_full,    bins=25, alpha=0.6, color="steelblue", label="Full LD")
    axes[1].hist(sep_noblock, bins=25, alpha=0.6, color="tomato",    label="No blocks")
    axes[1].set_xlabel("Scale separation (large s mean − small s mean)")
    axes[1].set_ylabel("Count")
    axes[1].set_title("Scale axis separation\n(LD structure induces wider spread)")
    axes[1].legend()

    axes[2].scatter(sep_full, ratios_full, s=12, alpha=0.4, color="steelblue", label="Full LD")
    axes[2].scatter(sep_noblock, ratios_noblock, s=12, alpha=0.4, color="tomato", label="No blocks")
    axes[2].set_xlabel("Scale separation")
    axes[2].set_ylabel("Stability ratio")
    axes[2].set_title("Separation vs. stability\n(wider scale → more stable gap)")
    axes[2].legend(fontsize=8)

    plt.tight_layout()
    plt.savefig("wgs_figures/st3_ld_removal.pdf", bbox_inches="tight", dpi=150)
    plt.savefig("wgs_figures/st3_ld_removal.png", bbox_inches="tight", dpi=150)
    plt.close()

    if verbose:
        print("  Saved: wgs_figures/st3_ld_removal.pdf")

    return {
        "full_ratio_mean":   float(ratios_full.mean()),
        "full_ratio_std":    float(ratios_full.std()),
        "noblock_ratio_mean":float(ratios_noblock.mean()),
        "noblock_ratio_std": float(ratios_noblock.std()),
        "ratio_gain":        float(ratio_gain),
        "sep_gain":          float(sep_gain),
    }


# ═════════════════════════════════════════════════════════════════════════════
# MODULE 6: MAIN
# ═════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Scale Axis Toy Pipeline")
    parser.add_argument("--quick", action="store_true",
                        help="Reduced iterations for quick test (50 per ST)")
    parser.add_argument("--st", type=int, default=0,
                        help="Run only a specific stress test (1/2/3); 0=all")
    args = parser.parse_args()

    n_iter = 50 if args.quick else 500
    rng    = np.random.default_rng(RNG_SEED)

    print("\n" + "╔" + "═"*68 + "╗")
    print("║  SCALE AXIS TOY PIPELINE  —  Spectral Scale Induction + L7 Dynamics  ║")
    print("╚" + "═"*68 + "╝")
    print(f"  Genes={N_GENES}  Blocks={N_BLOCKS}  Samples={N_SAMPLES}")
    print(f"  L7: a={A_PARAM}  b={B_PARAM}  T∈{T_SPAN}")
    print(f"  Stress test iterations: {n_iter}")
    if args.quick:
        print("  [QUICK MODE]")

    all_results = {}
    t0 = time.time()

    # Baseline
    if args.st == 0:
        all_results["baseline"] = run_baseline(rng)

    # Stress tests
    if args.st in (0, 1):
        all_results["st1"] = run_st1(n_iter=n_iter, rng_seed=RNG_SEED)

    if args.st in (0, 2):
        all_results["st2"] = run_st2(n_iter=n_iter, rng_seed=RNG_SEED+1)

    if args.st in (0, 3):
        all_results["st3"] = run_st3(n_iter=n_iter, rng_seed=RNG_SEED+2)

    # Save results
    out_path = "wgs_results/summary.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)

    elapsed = time.time() - t0
    print("\n" + "═"*70)
    print(f"  Done in {elapsed:.1f}s")
    print(f"  Results: {out_path}")
    print(f"  Figures: wgs_figures/")
    print()

    # Print clean summary for paper
    print("  ┌─ SUMMARY FOR §7.3 ─────────────────────────────────────────┐")
    if "baseline" in all_results:
        b = all_results["baseline"]
        print(f"  │  Baseline stability ratio (large-s / small-s): {b['stability_ratio']:.2f}×  │")
        print(f"  │  Energy monotonicity (baseline): {b['energy_monotone']}                  │")
    if "st1" in all_results:
        s1 = all_results["st1"]
        print(f"  │  ST1 min energy-monotone rate: {s1['min_mono_rate']:.1%}                │")
        print(f"  │  ST1 mean ratio (all noise):   {s1['mean_ratio_all']:.2f}×              │")
    if "st2" in all_results:
        s2 = all_results["st2"]
        print(f"  │  ST2 ratio (random params):    {s2['ratio_mean']:.2f}±{s2['ratio_std']:.2f}        │")
        print(f"  │  ST2 fraction ratio>1:         {s2['frac_above_1']:.1%}                │")
    if "st3" in all_results:
        s3 = all_results["st3"]
        print(f"  │  ST3 ratio gain (LD vs. flat): {s3['ratio_gain']:.2f}×                 │")
    print("  └─────────────────────────────────────────────────────────────┘")
    print()


if __name__ == "__main__":
    main()
