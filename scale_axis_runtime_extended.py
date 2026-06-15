"""
scale_axis_runtime_extended.py
═══════════════════════════════════════════════════════════════════════════════
Extended Runtime Scaling Benchmark — 8 hour version
RTX 5060 Ti / modern CPU target

Tests across 8 dataset sizes from 1k to 500k cells:
  - Our method: O(g³) eigendecomposition + O(N·g) gradient flow
  - ComBat:     O(N·g) linear model fitting
  - Harmony:    O(N·k·iter) iterative KMeans alignment

Key question: at what cell count does Harmony become significantly slower?
Expected crossover: ~20k-50k cells

Output:
  runtime_results/runtime_extended.json
  runtime_figures/runtime_scaling.pdf

Usage:
  python scale_axis_runtime_extended.py
  python scale_axis_runtime_extended.py --h5ad Immune_ALL_human.h5ad
═══════════════════════════════════════════════════════════════════════════════
"""

import argparse
import json
import logging
import os
import sys
import time
import threading
import warnings

import numpy as np
import scipy.linalg as la
import scipy.stats as stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
os.makedirs("runtime_results", exist_ok=True)
os.makedirs("runtime_figures", exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# Logger
# ─────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(message)s",
    handlers=[
        logging.FileHandler("runtime_results/runtime_progress.log",
                            encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ]
)
log = logging.getLogger("runtime")


# ─────────────────────────────────────────────────────────────────────────────
# Data generation
# ─────────────────────────────────────────────────────────────────────────────
def make_data(n_cells, n_genes, n_batches=5, seed=42):
    """Synthetic multi-batch scRNA-seq data."""
    rng = np.random.default_rng(seed)
    n_types = 8
    celltype = rng.choice(n_types, size=n_cells)
    batch_labels = np.array([f"study_{i % n_batches}" for i in range(n_cells)])

    X = rng.normal(0, 0.3, (n_cells, n_genes))
    n_block = n_genes // n_types
    for ct in range(n_types):
        mask = celltype == ct
        g0 = ct * n_block
        g1 = (ct+1) * n_block if ct < n_types-1 else n_genes
        X[mask, g0:g1] += rng.normal(2.0, 0.2, (mask.sum(), g1-g0))

    # Batch effects
    for bi in range(n_batches):
        bm = batch_labels == f"study_{bi}"
        X[bm] *= (1.0 + bi * 0.15)
        if bi > 0:
            X[bm, :n_genes//4] += bi * 0.4

    X = np.log1p(np.maximum(X, 0))
    gene_names = [f"gene_{i}" for i in range(n_genes)]
    return X, celltype, batch_labels, gene_names


def subsample_real(h5ad_path, n_cells, n_genes, seed=42):
    """Subsample real data to target size."""
    import scanpy as sc
    adata = sc.read_h5ad(h5ad_path)
    X_full = adata.X.toarray() if hasattr(adata.X, 'toarray') else np.array(adata.X)
    X_full = np.log1p(np.maximum(X_full, 0))

    rng = np.random.default_rng(seed)
    n_c = min(n_cells, X_full.shape[0])
    cidx = rng.choice(X_full.shape[0], n_c, replace=False)
    X = X_full[cidx]

    # HVG selection
    if X.shape[1] > n_genes:
        var = X.var(axis=0)
        gidx = np.argsort(var)[-n_genes:]
        X = X[:, gidx]

    # Batch labels
    batch_col = None
    for col in adata.obs.columns:
        if any(k in col.lower() for k in ['batch','study','lab','protocol']):
            batch_col = col
            break
    if batch_col:
        batch_labels = np.array(adata.obs[batch_col])[cidx]
    else:
        batch_labels = np.array([f"b{i%5}" for i in range(len(cidx))])

    gene_names = [f"gene_{i}" for i in range(X.shape[1])]
    return X, batch_labels, gene_names


# ─────────────────────────────────────────────────────────────────────────────
# Methods
# ─────────────────────────────────────────────────────────────────────────────
def time_our_method(X, n_runs=3):
    """Our method: graph + eigendecomposition + gradient flow."""
    n_genes = X.shape[1]
    matrix_mb = (n_genes * n_genes * 4) / 1e6

    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()

        # Graph
        W = np.corrcoef(X.T)
        np.fill_diagonal(W, 0.0)
        W = np.maximum(W, 0.0)

        # Laplacian eigendecomposition  O(g^3)
        D = np.diag(W.sum(axis=1))
        L = D - W
        eigvals, eigvecs = la.eigh(L)
        eigvals = np.maximum(eigvals, 0.0)

        # Scale axis
        s = np.log1p(eigvals.max() - eigvals)

        # Gradient flow — analytic solution O(N*g)
        lam = 2.0 / (1.0**2 * 2.0**(2*s))
        decay = np.exp(-lam * 12.0)
        _ = (X @ eigvecs * decay[None, :]) @ eigvecs.T

        times.append(time.perf_counter() - t0)

    return {
        'time_s':       float(np.median(times)),
        'time_std':     float(np.std(times)),
        'matrix_mb':    matrix_mb,
        'n_runs':       n_runs,
        'requires_labels': False,
        'iterative':    False,
        'analytic':     True,
    }


def time_combat(X, batch_labels, n_runs=3, timeout=600):
    """ComBat: linear model batch correction."""
    try:
        import scanpy as sc
        import anndata as ad
        times = []
        for run_i in range(n_runs):
            t0 = time.perf_counter()
            adata = ad.AnnData(X=X.copy())
            adata.obs['batch'] = batch_labels
            sc.pp.combat(adata, key='batch')
            elapsed = time.perf_counter() - t0
            times.append(elapsed)
            if elapsed > timeout:
                log.info(f"    ComBat timeout at {elapsed:.0f}s after {run_i+1} run(s)")
                break
        return {
            'time_s':          float(np.median(times)),
            'time_std':        float(np.std(times)) if len(times) > 1 else 0.0,
            'n_runs':          len(times),
            'requires_labels': True,
            'iterative':       False,
            'analytic':        False,
            'timed_out':       any(t > timeout for t in times),
        }
    except Exception as e:
        return {'error': str(e), 'requires_labels': True}


def time_harmony(X, batch_labels, n_runs=3, timeout=600):
    """Harmony: iterative KMeans alignment in PCA space."""
    try:
        import harmonypy as hm
        import pandas as pd

        # PCA first (fixed cost)
        Xc = X - X.mean(axis=0)
        n_pc = min(30, X.shape[1]-1, X.shape[0]-1)
        try:
            _, _, Vt = la.svd(Xc, full_matrices=False)
            pca = Xc @ Vt[:n_pc].T
        except Exception:
            pca = Xc[:, :n_pc]

        times = []
        for run_i in range(n_runs):
            t0 = time.perf_counter()
            meta = pd.DataFrame({'batch': batch_labels})
            hm.run_harmony(pca, meta, vars_use=['batch'],
                           max_iter_harmony=20, verbose=False)
            elapsed = time.perf_counter() - t0
            times.append(elapsed)
            if elapsed > timeout:
                log.info(f"    Harmony timeout at {elapsed:.0f}s after {run_i+1} run(s)")
                break

        return {
            'time_s':          float(np.median(times)),
            'time_std':        float(np.std(times)) if len(times) > 1 else 0.0,
            'n_runs':          len(times),
            'requires_labels': True,
            'iterative':       True,
            'analytic':        False,
            'timed_out':       any(t > timeout for t in times),
        }
    except Exception as e:
        return {'error': str(e), 'requires_labels': True}


# ─────────────────────────────────────────────────────────────────────────────
# Benchmark runner
# ─────────────────────────────────────────────────────────────────────────────
def run_one_size(n_cells, n_genes, h5ad_path=None, n_runs=3):
    log.info(f"\n  [{n_cells:,} cells x {n_genes:,} genes]")

    if h5ad_path and os.path.exists(h5ad_path):
        X, batch_labels, gene_names = subsample_real(h5ad_path, n_cells, n_genes)
        log.info(f"    Real data: {X.shape[0]} cells x {X.shape[1]} genes")
    else:
        X, _, batch_labels, gene_names = make_data(n_cells, n_genes)

    results = {'n_cells': n_cells, 'n_genes': n_genes,
               'actual_cells': X.shape[0], 'actual_genes': X.shape[1]}

    # Our method
    log.info(f"    Our method...", )
    r = time_our_method(X, n_runs=n_runs)
    results['ScaleAxis'] = r
    log.info(f"      {r['time_s']:.3f}s ± {r['time_std']:.3f}s  "
             f"matrix={r['matrix_mb']:.1f}MB  no labels")

    # ComBat
    log.info(f"    ComBat...")
    r = time_combat(X, batch_labels, n_runs=n_runs)
    results['ComBat'] = r
    if 'error' not in r:
        log.info(f"      {r['time_s']:.3f}s ± {r.get('time_std',0):.3f}s  "
                 f"uses batch labels")
    else:
        log.info(f"      error: {r['error']}")

    # Harmony
    log.info(f"    Harmony...")
    r = time_harmony(X, batch_labels, n_runs=n_runs)
    results['Harmony'] = r
    if 'error' not in r:
        log.info(f"      {r['time_s']:.3f}s ± {r.get('time_std',0):.3f}s  "
                 f"iterative KMeans  uses batch labels")
    else:
        log.info(f"      error: {r['error']}")

    # Speedup ratios
    our_t = results['ScaleAxis']['time_s']
    for m in ['ComBat', 'Harmony']:
        if 'time_s' in results.get(m, {}):
            ratio = results[m]['time_s'] / our_t
            log.info(f"      {m}: {ratio:.2f}x vs our method")

    return results


def plot_scaling(all_results):
    sizes = [(r['actual_cells'], r['actual_genes']) for r in all_results.values()]
    cell_counts = [s[0] for s in sizes]

    methods = {
        'ScaleAxis': ('Scale Axis + Gradient Flow (ours, no labels)',
                      'steelblue', 'o-', 2.5),
        'ComBat':    ('ComBat (uses batch labels)',
                      'tomato',    's--', 1.8),
        'Harmony':   ('Harmony (iterative KMeans, uses batch labels)',
                      'seagreen',  '^--', 1.8),
    }

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(
        "Runtime Scaling: Topology-Induced Scale Axis vs. Existing Methods\n"
        "Key: our method scales with gene count (O(g³)), "
        "not cell count (O(N))",
        fontsize=11, fontweight='bold'
    )

    # Panel 1: absolute runtime
    for method, (label, color, style, lw) in methods.items():
        xs, ys, yerrs = [], [], []
        for r in all_results.values():
            if method in r and 'time_s' in r[method]:
                xs.append(r['actual_cells'])
                ys.append(r[method]['time_s'])
                yerrs.append(r[method].get('time_std', 0))
        if xs:
            axes[0].errorbar(xs, ys, yerr=yerrs,
                             fmt=style, color=color, label=label,
                             linewidth=lw, markersize=8, capsize=4)

    axes[0].set_xlabel("Number of cells (N)")
    axes[0].set_ylabel("Runtime (seconds, median of 3 runs)")
    axes[0].set_title("Absolute runtime vs. cell count")
    axes[0].legend(fontsize=8, loc='upper left')
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xscale('log')
    axes[0].set_yscale('log')

    # Panel 2: slowdown ratio vs our method
    for method, (label, color, style, lw) in methods.items():
        if method == 'ScaleAxis':
            continue
        xs, ys = [], []
        for r in all_results.values():
            our_t = r['ScaleAxis']['time_s']
            if method in r and 'time_s' in r[method]:
                xs.append(r['actual_cells'])
                ys.append(r[method]['time_s'] / our_t)
        if xs:
            axes[1].plot(xs, ys, style, color=color, label=label,
                         linewidth=lw, markersize=8)

    axes[1].axhline(1, color='steelblue', linestyle='-',
                    label='Scale Axis (baseline = 1x)', linewidth=2)
    axes[1].set_xlabel("Number of cells (N)")
    axes[1].set_ylabel("Runtime ratio (vs. Scale Axis)")
    axes[1].set_title("Relative runtime (our method = 1.0x)")
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)
    axes[1].set_xscale('log')

    plt.tight_layout()
    plt.savefig("runtime_figures/runtime_scaling.pdf", dpi=150, bbox_inches='tight')
    plt.savefig("runtime_figures/runtime_scaling.png", dpi=150, bbox_inches='tight')
    plt.close()
    log.info("Figure saved: runtime_figures/runtime_scaling.pdf")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Extended Runtime Benchmark")
    parser.add_argument('--h5ad', type=str, default=None)
    parser.add_argument('--n_runs', type=int, default=3)
    args = parser.parse_args()

    log.info("=" * 60)
    log.info("EXTENDED RUNTIME SCALING BENCHMARK")
    log.info("Target: 8 hours  |  8 dataset sizes")
    log.info("=" * 60)

    # 8 sizes: small to very large
    # n_genes fixed at HVG=2000 for large sizes (realistic)
    sizes = [
        ('s1',   1000,   500),
        ('s2',   3000,   800),
        ('s3',   5000,  1000),
        ('s4',  10000,  1500),
        ('s5',  20000,  2000),
        ('s6',  50000,  2000),
        ('s7', 100000,  2000),
        ('s8', 200000,  2000),
    ]

    all_results = {}
    t_total = time.time()

    for label, n_cells, n_genes in sizes:
        t0 = time.time()
        log.info(f"\n{'='*60}")
        log.info(f"SIZE {label}: {n_cells:,} cells x {n_genes:,} genes")

        try:
            r = run_one_size(n_cells, n_genes,
                             h5ad_path=args.h5ad,
                             n_runs=args.n_runs)
            all_results[label] = r
        except Exception as e:
            log.info(f"  ERROR: {e}")
            all_results[label] = {'error': str(e),
                                   'n_cells': n_cells, 'n_genes': n_genes}

        elapsed = time.time() - t0
        total_elapsed = time.time() - t_total
        log.info(f"  Size done in {elapsed:.0f}s  "
                 f"(total: {total_elapsed/3600:.2f}h)")

        # Save intermediate results
        with open("runtime_results/runtime_extended.json", 'w') as f:
            json.dump(all_results, f, indent=2, default=str)

        # Plot after each size
        try:
            plot_scaling({k: v for k, v in all_results.items()
                          if 'ScaleAxis' in v})
        except Exception:
            pass

    # Final summary
    log.info("\n" + "=" * 60)
    log.info("FINAL SUMMARY FOR PAPER:")
    log.info("=" * 60)
    log.info(f"{'Cells':>8}  {'Genes':>6}  "
             f"{'Ours':>8}  {'ComBat':>8}  {'Harmony':>8}  "
             f"{'CB ratio':>9}  {'Harm ratio':>10}")
    log.info("-" * 70)

    for label, n_cells, n_genes in sizes:
        r = all_results.get(label, {})
        if 'ScaleAxis' not in r:
            continue
        our_t = r['ScaleAxis']['time_s']
        cb_t  = r.get('ComBat',  {}).get('time_s', float('nan'))
        hm_t  = r.get('Harmony', {}).get('time_s', float('nan'))
        cb_r  = cb_t / our_t if not np.isnan(cb_t) else float('nan')
        hm_r  = hm_t / our_t if not np.isnan(hm_t) else float('nan')

        log.info(f"{r['actual_cells']:>8,}  {r['actual_genes']:>6,}  "
                 f"{our_t:>8.2f}s  "
                 f"{cb_t:>8.2f}s  "
                 f"{hm_t:>8.2f}s  "
                 f"{cb_r:>9.1f}x  "
                 f"{hm_r:>10.1f}x")

    total_h = (time.time() - t_total) / 3600
    log.info(f"\nTotal elapsed: {total_h:.2f} hours")
    log.info("Results: runtime_results/runtime_extended.json")


if __name__ == "__main__":
    main()
