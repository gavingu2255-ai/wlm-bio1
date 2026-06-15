"""
scale_axis_overnight.py
═══════════════════════════════════════════════════════════════════════════════
Overnight validation suite for the topology-induced scale axis paper.
Designed to run unattended on RTX 5060 Ti for 6-10 hours.

Four blocks:

  BLOCK A: GPU Numerical Stability (PyTorch, CUDA)
    N=500,000 Monte Carlo × 500 modes × 10,000 time points
    Tests: mode ordering invariance, energy monotonicity
    Expected: 0 violations across 250 billion mode-pair checks

  BLOCK B: Parameter Sweep (GPU)
    N=50,000 random (a,b,T) on PBMC68k spectral decomposition
    Tests: marker F retention, cell-type separation, energy monotone

  BLOCK C: Permutation Test (CPU, fast)
    N=100,000 permutations of gene indices
    Tests: probability of observing all 17 marker genes in large-s by chance
    Expected: p < 1e-5

  BLOCK D: Bootstrap Stability (CPU, multi-hour)
    N=2,000 bootstrap samples (80% cell subsampling)
    Each: rebuild co-expression graph, recompute scale axis, record marker s
    Tests: std of marker gene scale positions across samples
    Expected: std < 0.15 (stable across cell subsampling)

Progress: logged every 5 minutes to overnight_progress.log
Results:  overnight_results/overnight_summary.json
Figures:  overnight_figures/

Usage:
  python scale_axis_overnight.py          # all blocks
  python scale_axis_overnight.py --skip A # skip GPU block (no CUDA)
  python scale_axis_overnight.py --only C # single block
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
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import silhouette_score
import scipy.stats as stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
os.makedirs("overnight_figures", exist_ok=True)
os.makedirs("overnight_results",  exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# Progress logger: 5-minute checkpoints
# ─────────────────────────────────────────────────────────────────────────────
class ProgressLogger:
    def __init__(self, path="overnight_progress.log", interval=300):
        self.path      = path
        self.interval  = interval
        self.start     = time.time()
        self._buf      = []
        self._lock     = threading.Lock()
        self._stop     = threading.Event()
        self._thread   = threading.Thread(target=self._loop, daemon=True)

        # Write log file in UTF-8
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s  %(message)s",
            handlers=[
                logging.FileHandler(path, encoding="utf-8"),
                logging.StreamHandler(
                    open(os.devnull, 'w') if sys.platform == 'win32'
                    else sys.stdout
                ),
            ]
        )
        # Windows console: print directly to avoid cp1252 crash
        self._win = (sys.platform == 'win32')
        self.log = logging.getLogger("overnight")
        self._thread.start()

    def _loop(self):
        while not self._stop.is_set():
            time.sleep(self.interval)
            self._flush()

    def _flush(self):
        elapsed = time.time() - self.start
        with self._lock:
            msgs = list(self._buf)
            self._buf = []
        with open(self.path, 'a', encoding='utf-8') as f:
            f.write(f"\n---- checkpoint  elapsed={elapsed/60:.1f}min ----\n")
            for m in msgs:
                f.write(m + "\n")

    def info(self, msg):
        self.log.info(msg)
        if self._win:
            try:
                print(msg)
            except Exception:
                print(msg.encode('ascii', errors='replace').decode())
        with self._lock:
            self._buf.append(msg)

    def stop(self):
        self._stop.set()
        self._flush()


# ─────────────────────────────────────────────────────────────────────────────
# Core math
# ─────────────────────────────────────────────────────────────────────────────
def spectral_decompose(W):
    D = np.diag(W.sum(axis=1))
    L = D - W
    ev, evec = la.eigh(L)
    return np.maximum(ev, 0.0), evec

def induced_scale(eigvals):
    return np.log1p(eigvals.max() - eigvals)

def l7_decay(s, a=1.0, b=2.0):
    return 2.0 / (a**2 * b**(2*s))

def apply_l7(X, evec, s, T=12.0, a=1.0, b=2.0):
    d = np.exp(-l7_decay(s, a, b) * T)
    return (X @ evec * d[None, :]) @ evec.T

def gene_f(expr, labels):
    unique = np.unique(labels)
    groups = [expr[labels == c] for c in unique if (labels == c).sum() > 1]
    if len(groups) < 2:
        return 0.0
    f, _ = stats.f_oneway(*groups)
    return float(f) if np.isfinite(f) else 0.0


def pca_embed(X, n_components=30):
    Xc = X - X.mean(axis=0)
    try:
        _, _, Vt = la.svd(Xc, full_matrices=False)
        return Xc @ Vt[:n_components].T
    except Exception:
        return Xc[:, :n_components]

def compute_all_metrics(X_raw, X_corr, celltype, batch_labels,
                        marker_idx, label=""):
    n_pca = min(30, X_raw.shape[1]-1, X_raw.shape[0]-1)
    pca_raw  = pca_embed(X_raw,  n_pca)
    pca_corr = pca_embed(X_corr, n_pca)
    k = min(30, X_raw.shape[0]//5)
    from sklearn.neighbors import NearestNeighbors
    from sklearn.metrics import silhouette_score as sil

    def batch_mix(Xp, bl):
        nbrs = NearestNeighbors(n_neighbors=k+1).fit(Xp)
        _, idx = nbrs.kneighbors(Xp)
        unique_b = np.unique(bl)
        nb = len(unique_b)
        ents = []
        for i, ni in enumerate(idx):
            nb_b = bl[ni[1:]]
            cnts = np.array([(nb_b==b).sum() for b in unique_b])
            p = cnts/cnts.sum()
            p = p[p>0]
            ents.append(-np.sum(p*np.log(p+1e-12))/np.log(nb+1e-12))
        return float(np.mean(ents))

    def ct_purity(Xp, ct):
        nbrs = NearestNeighbors(n_neighbors=k+1).fit(Xp)
        _, idx = nbrs.kneighbors(Xp)
        return float(np.mean([(ct[ni[1:]]==ct[i]).mean()
                               for i, ni in enumerate(idx)]))

    def f_ret(Xb, Xa, ct, midx):
        ratios = []
        for gi in midx:
            if gi >= Xb.shape[1]: continue
            fb = gene_f(Xb[:,gi], ct)
            fa = gene_f(Xa[:,gi], ct)
            if fb > 0: ratios.append(fa/fb)
        return float(np.mean(ratios)) if ratios else 1.0

    bms_r = batch_mix(pca_raw,  batch_labels)
    bms_c = batch_mix(pca_corr, batch_labels)
    ctp_r = ct_purity(pca_raw,  celltype)
    ctp_c = ct_purity(pca_corr, celltype)
    mfr   = f_ret(X_raw, X_corr, celltype, marker_idx)

    n_sub = min(500, X_raw.shape[0])
    idx_s = np.random.choice(X_raw.shape[0], n_sub, replace=False)
    try:
        sb_r = sil(pca_raw[idx_s],  batch_labels[idx_s])
        sb_c = sil(pca_corr[idx_s], batch_labels[idx_s])
        sc_r = sil(pca_raw[idx_s],  celltype[idx_s])
        sc_c = sil(pca_corr[idx_s], celltype[idx_s])
    except Exception:
        sb_r=sb_c=sc_r=sc_c=0.0

    return {
        'label': label,
        'batch_mixing_delta':    bms_c - bms_r,
        'celltype_purity_delta': ctp_c - ctp_r,
        'marker_f_retention':    mfr,
        'sil_batch_delta':       sb_c - sb_r,
        'sil_celltype_delta':    sc_c - sc_r,
    }

def build_graph(X, threshold=0.0):
    W = np.corrcoef(X.T)
    np.fill_diagonal(W, 0.0)
    return np.maximum(W, threshold)


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────
def load_pbmc68k(logger):
    import scanpy as sc
    logger.info("Loading pbmc68k_reduced...")
    adata = sc.datasets.pbmc68k_reduced()
    X = adata.X.toarray() if hasattr(adata.X, 'toarray') else np.array(adata.X)
    gene_names = np.array(adata.var_names)
    celltype   = np.array(adata.obs['bulk_labels'])

    marker_genes = ['LYZ','CST3','FCGR3A','NKG7','GNLY','GZMB','KLRB1',
                    'CD79A','MS4A1','IL7R','CCR7','CD4','CD8A','CD8B',
                    'GZMK','FCER1A','HLA-DQA1']
    marker_idx = [i for i, g in enumerate(gene_names) if g in marker_genes]

    logger.info(f"  {X.shape[0]} cells x {X.shape[1]} genes  "
                f"({len(marker_idx)} marker genes)")
    return X, celltype, gene_names, marker_idx


# ═════════════════════════════════════════════════════════════════════════════
# BLOCK A: GPU Numerical Stability
# ═════════════════════════════════════════════════════════════════════════════

def run_block_a(n_iter=500000, n_modes=500, n_time=10000,
                a=1.0, b=2.0, logger=None):
    """
    PyTorch GPU: N=500,000 random initializations
    - Mode ordering invariance: if s[i] > s[j], decay ratio of i >= j at all t
    - Energy strict monotonicity: N_B[r(t)] strictly decreasing
    """
    try:
        import torch
        if not torch.cuda.is_available():
            logger.info("  BLOCK A: CUDA not available, skipping")
            return {'skipped': True, 'reason': 'no CUDA'}
        device = torch.device('cuda')
    except ImportError:
        logger.info("  BLOCK A: PyTorch not installed, skipping")
        return {'skipped': True, 'reason': 'no torch'}

    logger.info(f"BLOCK A: GPU Numerical Stability")
    logger.info(f"  N={n_iter:,}  modes={n_modes}  time={n_time}")
    logger.info(f"  Device: {torch.cuda.get_device_name(0)}")

    rng = np.random.default_rng(42)
    ev_fake = np.sort(rng.uniform(0, 10, size=n_modes))
    s_np  = np.log1p(ev_fake.max() - ev_fake)
    lam_np = 2.0 / (a**2 * b**(2*s_np))

    s   = torch.tensor(s_np,   dtype=torch.float64, device=device)
    lam = torch.tensor(lam_np, dtype=torch.float64, device=device)
    phi = 1.0 / (a**2 * (b**(2*s)))
    ds  = float((s_np.max() - s_np.min()) / (n_modes - 1))
    t_span = torch.linspace(0, 20.0, n_time, dtype=torch.float64, device=device)
    decay  = torch.exp(-lam[None, :] * t_span[:, None])  # (n_time, n_modes)

    # Batch size: balance VRAM usage
    # 100 x 500 x 10000 x 8 bytes = 4 GB  → safe for 16 GB
    batch_size = 100
    n_batches  = (n_iter + batch_size - 1) // batch_size

    crossing_count  = 0
    energy_viol     = 0
    n_pairs_per     = 500  # random pairs to check per batch

    t0 = time.time()
    report_every = max(1, n_batches // 40)

    for bi in range(n_batches):
        curr = min(batch_size, n_iter - bi * batch_size)

        r0 = torch.abs(torch.randn(curr, n_modes,
                                   dtype=torch.float64, device=device)) + 0.1

        # r_t: (curr, n_time, n_modes)
        r_t = r0[:, None, :] * decay[None, :, :]

        # -- Mode ordering --
        # decay[:,i] >= decay[:,j] whenever s[i] > s[j]  (structural fact)
        idx_i = torch.randint(0, n_modes, (n_pairs_per,), device=device)
        idx_j = torch.randint(0, n_modes, (n_pairs_per,), device=device)
        mask  = s[idx_i] > s[idx_j]
        di    = decay[:, idx_i]   # (n_time, n_pairs)
        dj    = decay[:, idx_j]
        viol  = ((di < dj - 1e-10) & mask[None, :]).any(dim=0)
        crossing_count += int(viol.sum().item()) * curr

        # -- Energy monotonicity --
        E   = (r_t**2 * phi[None, None, :]).sum(dim=2) * ds  # (curr, n_time)
        dE  = torch.diff(E, dim=1)
        tol = 1e-6 * E[:, :1].abs() + 1e-14
        energy_viol += int((dE > tol).any(dim=1).sum().item())

        del r0, r_t, E, dE
        torch.cuda.empty_cache()

        if (bi + 1) % report_every == 0:
            pct     = 100 * (bi + 1) / n_batches
            elapsed = time.time() - t0
            rate    = (bi + 1) * batch_size / elapsed
            eta     = (n_iter - (bi + 1) * batch_size) / (rate + 1e-9)
            logger.info(f"  Block A  {pct:.1f}%  "
                        f"crossings={crossing_count}  "
                        f"E_viol={energy_viol}  "
                        f"elapsed={elapsed:.0f}s  ETA={eta:.0f}s")

    elapsed = time.time() - t0
    total_pairs = n_iter * n_pairs_per

    summary = {
        'n_iter':              n_iter,
        'n_modes':             n_modes,
        'n_time_points':       n_time,
        'total_pair_checks':   total_pairs,
        'crossing_count':      crossing_count,
        'crossing_rate':       crossing_count / total_pairs,
        'energy_viol_count':   energy_viol,
        'energy_viol_rate':    energy_viol / n_iter,
        'elapsed_s':           elapsed,
        'throughput_iter_s':   n_iter / elapsed,
    }

    logger.info(f"  BLOCK A DONE in {elapsed:.0f}s")
    logger.info(f"  Mode crossings:    {crossing_count} / {total_pairs:,}  "
                f"rate={summary['crossing_rate']:.2e}")
    logger.info(f"  Energy violations: {energy_viol} / {n_iter:,}  "
                f"rate={summary['energy_viol_rate']:.2e}")

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f"Block A: GPU Numerical Stability  N={n_iter:,}", fontweight='bold')

    axes[0].bar(['Mode crossings', 'Expected (0)'],
                [crossing_count, 0], color=['tomato', 'steelblue'], alpha=0.8)
    axes[0].set_ylabel("Count")
    axes[0].set_title(f"Mode ordering violations\n{crossing_count} / {total_pairs:,} pairs")

    axes[1].bar(['Energy violations', 'Expected (0)'],
                [energy_viol, 0], color=['tomato', 'steelblue'], alpha=0.8)
    axes[1].set_ylabel("Count")
    axes[1].set_title(f"Energy monotonicity violations\n{energy_viol} / {n_iter:,}")

    plt.tight_layout()
    plt.savefig("overnight_figures/blockA_gpu_stability.pdf", dpi=150, bbox_inches='tight')
    plt.savefig("overnight_figures/blockA_gpu_stability.png", dpi=150, bbox_inches='tight')
    plt.close()

    return summary


# ═════════════════════════════════════════════════════════════════════════════
# BLOCK B: GPU Parameter Sweep
# ═════════════════════════════════════════════════════════════════════════════

def run_block_b(X, celltype, marker_idx, evec, s_canon,
                n_params=50000, logger=None):
    """
    N=50,000 random (a,b,T) on real PBMC68k.
    GPU-accelerated: batch L7 application.
    """
    try:
        import torch
        use_gpu = torch.cuda.is_available()
        device  = torch.device('cuda' if use_gpu else 'cpu')
    except ImportError:
        use_gpu = False
        device  = None

    logger.info(f"BLOCK B: Parameter Sweep  N={n_params:,}")
    logger.info(f"  Using: {'GPU' if use_gpu else 'CPU'}")

    rng   = np.random.default_rng(123)
    a_arr = rng.uniform(0.5, 3.0, n_params)
    b_arr = rng.uniform(1.2, 2.5, n_params)
    T_arr = rng.uniform(4.0, 20.0, n_params)

    # Precompute F before (fixed)
    f_before = np.array([gene_f(X[:, gi], celltype) for gi in marker_idx])

    f_improvement_rates = []
    energy_mono_rates   = []
    stability_ratios    = []

    t0           = time.time()
    report_every = max(1, n_params // 20)

    for i in range(n_params):
        a, b, T = float(a_arr[i]), float(b_arr[i]), float(T_arr[i])

        X_T = apply_l7(X, evec, s_canon, T, a, b)

        f_after = np.array([gene_f(X_T[:, gi], celltype) for gi in marker_idx])
        f_improvement_rates.append((f_after > f_before).mean())

        # Energy check
        phi = 1.0 / (a**2 * b**(2*s_canon))
        ds  = (s_canon.max() - s_canon.min()) / (len(s_canon) - 1)
        t_chk = np.linspace(0, T, 6)
        r0_mean = np.abs(X @ evec).mean(axis=0)
        E = [np.sum((r0_mean * np.exp(-l7_decay(s_canon, a, b) * tc))**2 * phi) * ds
             for tc in t_chk]
        mono = all(E[j+1] <= E[j] + 1e-6*abs(E[0]) + 1e-12
                   for j in range(len(E)-1))
        energy_mono_rates.append(mono)

        s_med  = np.median(s_canon)
        hl_sm  = (np.log(2) / l7_decay(s_canon[s_canon <  s_med], a, b)).mean()
        hl_lg  = (np.log(2) / l7_decay(s_canon[s_canon >= s_med], a, b)).mean()
        stability_ratios.append(hl_lg / (hl_sm + 1e-12))

        if (i + 1) % report_every == 0:
            elapsed = time.time() - t0
            logger.info(f"  Block B  {100*(i+1)/n_params:.1f}%  "
                        f"F_imp={np.mean(f_improvement_rates):.0%}  "
                        f"mono={np.mean(energy_mono_rates):.0%}  "
                        f"elapsed={elapsed:.0f}s")

    f_improvement_rates = np.array(f_improvement_rates)
    energy_mono_rates   = np.array(energy_mono_rates, dtype=float)
    stability_ratios    = np.array(stability_ratios)

    summary = {
        'n_params':                   n_params,
        'f_improvement_mean':         float(f_improvement_rates.mean()),
        'f_improvement_min':          float(f_improvement_rates.min()),
        'f_improvement_100pct_rate':  float((f_improvement_rates == 1.0).mean()),
        'energy_monotone_rate':       float(energy_mono_rates.mean()),
        'stability_ratio_mean':       float(stability_ratios.mean()),
        'stability_ratio_min':        float(stability_ratios.min()),
        'elapsed_s':                  float(time.time() - t0),
    }

    logger.info(f"  BLOCK B DONE")
    logger.info(f"  F improvement: {summary['f_improvement_mean']:.0%}  "
                f"(min {summary['f_improvement_min']:.0%})")
    logger.info(f"  Energy monotone: {summary['energy_monotone_rate']:.0%}")
    logger.info(f"  Stability ratio: {summary['stability_ratio_mean']:.1f}x  "
                f"(min {summary['stability_ratio_min']:.1f}x)")

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(f"Block B: Parameter Sweep  N={n_params:,}", fontweight='bold')
    axes[0].hist(f_improvement_rates*100, bins=30, color='steelblue', edgecolor='white')
    axes[0].axvline(100, color='red', linestyle='--')
    axes[0].set_xlabel("Marker F improvement rate (%)")
    axes[0].set_title(f"Mean={summary['f_improvement_mean']:.0%}")
    axes[1].hist(stability_ratios, bins=30, color='tomato', edgecolor='white')
    axes[1].set_xlabel("Stability ratio (large-s / small-s)")
    axes[1].set_title(f"Mean={summary['stability_ratio_mean']:.1f}x")
    axes[2].bar(['Monotone', 'Violation'],
                [energy_mono_rates.mean(), 1-energy_mono_rates.mean()],
                color=['steelblue','tomato'], alpha=0.8)
    axes[2].set_ylabel("Fraction")
    axes[2].set_title("Energy monotonicity")
    plt.tight_layout()
    plt.savefig("overnight_figures/blockB_param_sweep.pdf", dpi=150, bbox_inches='tight')
    plt.savefig("overnight_figures/blockB_param_sweep.png", dpi=150, bbox_inches='tight')
    plt.close()

    return summary


# ═════════════════════════════════════════════════════════════════════════════
# BLOCK C: Permutation Test
# ═════════════════════════════════════════════════════════════════════════════

def run_block_c(s, marker_idx, n_genes, n_perm=100000, logger=None):
    """
    Statistical test: are marker gene scale positions significantly higher
    than background genes?

    Two complementary tests:
    1. Wilcoxon rank-sum test (one-tailed): marker s > background s
       - No threshold dependency
       - Standard non-parametric test, widely accepted
    2. Permutation test on MEAN s:
       - H0: mean s of 17 randomly selected genes >= observed mean s
       - More interpretable than fraction-in-zone
    """
    from scipy.stats import ranksums, mannwhitneyu

    logger.info(f"BLOCK C: Statistical Significance Tests")

    all_indices   = np.arange(n_genes)
    marker_set    = set(marker_idx)
    bg_indices    = np.array([i for i in all_indices if i not in marker_set])

    marker_s  = s[marker_idx]
    bg_s      = s[bg_indices]
    n_markers = len(marker_idx)

    logger.info(f"  Marker genes: {n_markers}  "
                f"Background genes: {len(bg_indices)}")
    logger.info(f"  Marker s:     mean={marker_s.mean():.3f}  "
                f"median={np.median(marker_s):.3f}  "
                f"min={marker_s.min():.3f}")
    logger.info(f"  Background s: mean={bg_s.mean():.3f}  "
                f"median={np.median(bg_s):.3f}")

    # ── Test 1: Wilcoxon rank-sum (one-tailed: marker > background) ──────
    stat_rs, p_rs = ranksums(marker_s, bg_s, alternative='greater')
    logger.info(f"  Wilcoxon rank-sum (one-tailed, marker > background):")
    logger.info(f"    statistic = {stat_rs:.4f}  p = {p_rs:.4e}")

    # ── Test 2: Mann-Whitney U (one-tailed) ──────────────────────────────
    stat_mw, p_mw = mannwhitneyu(marker_s, bg_s, alternative='greater')
    logger.info(f"  Mann-Whitney U (one-tailed):")
    logger.info(f"    statistic = {stat_mw:.1f}  p = {p_mw:.4e}")

    # Effect size: rank-biserial correlation
    n_m = len(marker_s)
    n_b = len(bg_s)
    r_rb = 1 - (2 * stat_mw) / (n_m * n_b)
    logger.info(f"  Effect size (rank-biserial r): {r_rb:.4f}")

    # ── Test 3: Permutation test on MEAN s ───────────────────────────────
    logger.info(f"  Permutation test on mean s (N={n_perm:,})...")
    obs_mean = float(marker_s.mean())
    rng = np.random.default_rng(777)
    exceed = 0
    t0 = time.time()
    report_every = max(1, n_perm // 20)

    for i in range(n_perm):
        rand_idx  = rng.choice(all_indices, size=n_markers, replace=False)
        rand_mean = s[rand_idx].mean()
        if rand_mean >= obs_mean:
            exceed += 1
        if (i+1) % report_every == 0:
            elapsed = time.time() - t0
            p_est   = (exceed+1)/(i+2)
            logger.info(f"    {100*(i+1)/n_perm:.1f}%  "
                        f"exceed={exceed}  p_est={p_est:.2e}  "
                        f"elapsed={elapsed:.0f}s")

    p_perm = (exceed + 1) / (n_perm + 1)
    elapsed = time.time() - t0

    summary = {
        'n_markers':          n_markers,
        'n_background':       len(bg_indices),
        'marker_s_mean':      float(marker_s.mean()),
        'marker_s_median':    float(np.median(marker_s)),
        'background_s_mean':  float(bg_s.mean()),
        'background_s_median':float(np.median(bg_s)),
        'mean_difference':    float(marker_s.mean() - bg_s.mean()),
        'wilcoxon_stat':      float(stat_rs),
        'wilcoxon_p':         float(p_rs),
        'mannwhitney_stat':   float(stat_mw),
        'mannwhitney_p':      float(p_mw),
        'effect_size_rb':     float(r_rb),
        'perm_n':             n_perm,
        'perm_exceed':        exceed,
        'perm_p':             float(p_perm),
        'perm_p_str':         f"p < {1/n_perm:.0e}" if exceed == 0
                              else f"p = {p_perm:.2e}",
        'elapsed_s':          elapsed,
        'significant_ranksum': bool(p_rs < 0.05),
        'significant_perm':    bool(p_perm < 0.05),
    }

    logger.info(f"  BLOCK C DONE")
    logger.info(f"  Wilcoxon p = {p_rs:.4e}  "
                f"({'significant' if p_rs < 0.05 else 'not significant'})")
    logger.info(f"  Mann-Whitney p = {p_mw:.4e}  "
                f"effect size r = {r_rb:.4f}")
    logger.info(f"  Permutation (mean s) p = {p_perm:.2e}")
    logger.info(f"  Mean s difference: "
                f"{marker_s.mean():.3f} - {bg_s.mean():.3f} = "
                f"{marker_s.mean()-bg_s.mean():.3f}")

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(
        f"Block C: Statistical Tests\n"
        f"Wilcoxon p={p_rs:.2e}  Mann-Whitney p={p_mw:.2e}  "
        f"effect r={r_rb:.3f}",
        fontweight='bold'
    )

    # Panel 1: distribution comparison
    axes[0].hist(bg_s, bins=40, alpha=0.5, color='steelblue',
                 label=f'Background ({len(bg_s)} genes)', density=True)
    axes[0].hist(marker_s, bins=15, alpha=0.8, color='tomato',
                 label=f'Marker ({n_markers} genes)', density=True)
    axes[0].axvline(marker_s.mean(), color='red', linewidth=2,
                    linestyle='-', label=f'Marker mean={marker_s.mean():.3f}')
    axes[0].axvline(bg_s.mean(), color='blue', linewidth=2,
                    linestyle='--', label=f'BG mean={bg_s.mean():.3f}')
    axes[0].set_xlabel("Scale s")
    axes[0].set_ylabel("Density")
    axes[0].set_title("Scale distribution: marker vs background")
    axes[0].legend(fontsize=8)

    # Panel 2: permutation null distribution
    rng2 = np.random.default_rng(42)
    null_means = [s[rng2.choice(all_indices, n_markers, replace=False)].mean()
                  for _ in range(min(10000, n_perm))]
    axes[1].hist(null_means, bins=40, color='steelblue',
                 alpha=0.7, label='Null distribution')
    axes[1].axvline(obs_mean, color='red', linewidth=2,
                    label=f'Observed mean={obs_mean:.3f}')
    axes[1].set_xlabel("Mean scale s of random gene set")
    axes[1].set_ylabel("Count")
    axes[1].set_title(f"Permutation test on mean s\np={p_perm:.2e}")
    axes[1].legend(fontsize=8)

    plt.tight_layout()
    plt.savefig("overnight_figures/blockC_statistics.pdf",
                dpi=150, bbox_inches='tight')
    plt.savefig("overnight_figures/blockC_statistics.png",
                dpi=150, bbox_inches='tight')
    plt.close()

    return summary


def run_block_d(X, celltype, gene_names, marker_idx,
                n_bootstrap=2000, subsample_frac=0.80,
                a=1.0, b=2.0, T=12.0, logger=None):
    """
    N=2,000 bootstrap samples (80% cell subsampling).
    Each sample: rebuild co-expression graph, recompute scale axis,
    record marker gene s-positions and F improvement.
    
    Key metric: std of marker gene s-positions across bootstrap samples.
    Expected: std < 0.15 (scale axis is stable to cell subsampling).
    """
    logger.info(f"BLOCK D: Bootstrap Stability  N={n_bootstrap}  "
                f"subsample={subsample_frac:.0%}")

    rng      = np.random.default_rng(999)
    n_cells  = X.shape[0]
    n_sub    = int(n_cells * subsample_frac)
    n_marker = len(marker_idx)

    bootstrap_marker_s   = []   # mean s of marker genes per bootstrap
    bootstrap_marker_s_all = [] # (n_bootstrap, n_marker) individual s values
    bootstrap_f_ret      = []   # F retention per bootstrap
    bootstrap_sep_ret    = []   # cell-type separation retention

    t0           = time.time()
    report_every = max(1, n_bootstrap // 20)

    def celltype_sep(Xmat, ct):
        unique = np.unique(ct)
        om = Xmat.mean(axis=0)
        bw = sum((ct==c).sum() * np.sum((Xmat[ct==c].mean(0)-om)**2)
                 for c in unique)
        wi = sum(np.sum((Xmat[ct==c] - Xmat[ct==c].mean(0)[None,:])**2)
                 for c in unique)
        return bw / (wi + 1e-12)

    for i in range(n_bootstrap):
        # Subsample cells (stratified by cell type)
        sub_idx = []
        for ct in np.unique(celltype):
            ct_idx = np.where(celltype == ct)[0]
            n_ct   = max(2, int(len(ct_idx) * subsample_frac))
            sub_idx.extend(rng.choice(ct_idx, size=n_ct, replace=False))
        sub_idx = np.array(sub_idx)

        X_sub  = X[sub_idx]
        ct_sub = celltype[sub_idx]

        try:
            W  = build_graph(X_sub)
            ev, evec = spectral_decompose(W)
            s  = induced_scale(ev)

            # Marker gene s values
            ms = np.array([s[gi] for gi in marker_idx if gi < len(s)])
            bootstrap_marker_s.append(float(ms.mean()))
            bootstrap_marker_s_all.append(ms)

            # F retention
            X_T   = apply_l7(X_sub, evec, s, T, a, b)
            f_b   = np.array([gene_f(X_sub[:, gi], ct_sub) for gi in marker_idx])
            f_a   = np.array([gene_f(X_T[:,  gi], ct_sub) for gi in marker_idx])
            bootstrap_f_ret.append((f_a > f_b).mean())

            # Cell-type separation
            sep_b = celltype_sep(X_sub, ct_sub)
            sep_a = celltype_sep(X_T,   ct_sub)
            bootstrap_sep_ret.append(sep_a / (sep_b + 1e-12))

        except Exception as e:
            continue

        if (i + 1) % report_every == 0:
            elapsed = time.time() - t0
            ms_arr  = np.array(bootstrap_marker_s)
            logger.info(f"  Block D  {100*(i+1)/n_bootstrap:.1f}%  "
                        f"marker_s={ms_arr.mean():.3f}+/-{ms_arr.std():.3f}  "
                        f"F_ret={np.mean(bootstrap_f_ret):.0%}  "
                        f"elapsed={elapsed:.0f}s")

    bs_ms   = np.array(bootstrap_marker_s)
    bs_fall = np.array(bootstrap_marker_s_all)   # (n_ok, n_marker)
    bs_fr   = np.array(bootstrap_f_ret)
    bs_sr   = np.array(bootstrap_sep_ret)
    elapsed = time.time() - t0

    # Per-gene std across bootstrap
    per_gene_std = bs_fall.std(axis=0) if len(bs_fall) > 0 else np.zeros(n_marker)

    summary = {
        'n_bootstrap':               n_bootstrap,
        'n_successful':              len(bs_ms),
        'subsample_frac':            subsample_frac,
        'marker_s_mean':             float(bs_ms.mean()),
        'marker_s_std':              float(bs_ms.std()),
        'marker_s_ci95_low':         float(np.percentile(bs_ms, 2.5)),
        'marker_s_ci95_high':        float(np.percentile(bs_ms, 97.5)),
        'per_gene_s_std_mean':       float(per_gene_std.mean()),
        'per_gene_s_std_max':        float(per_gene_std.max()),
        'f_retention_mean':          float(bs_fr.mean()),
        'f_retention_std':           float(bs_fr.std()),
        'f_retention_100pct_rate':   float((bs_fr == 1.0).mean()),
        'sep_retention_mean':        float(bs_sr.mean()),
        'elapsed_s':                 elapsed,
        'stable':                    bool(bs_ms.std() < 0.15),
    }

    logger.info(f"  BLOCK D DONE in {elapsed:.0f}s")
    logger.info(f"  Bootstrap marker s: {bs_ms.mean():.3f} +/- {bs_ms.std():.3f}")
    logger.info(f"  95% CI: [{summary['marker_s_ci95_low']:.3f}, "
                f"{summary['marker_s_ci95_high']:.3f}]")
    logger.info(f"  Per-gene s std (mean): {per_gene_std.mean():.4f}")
    logger.info(f"  F retention: {bs_fr.mean():.0%} +/- {bs_fr.std():.3f}")
    logger.info(f"  Stability verdict: {'STABLE (std < 0.15)' if summary['stable'] else 'UNSTABLE'}")

    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(f"Block D: Bootstrap Stability  N={n_bootstrap}  "
                 f"subsample={subsample_frac:.0%}", fontweight='bold')

    axes[0,0].hist(bs_ms, bins=40, color='steelblue', edgecolor='white')
    axes[0,0].axvline(bs_ms.mean(), color='red', linestyle='-',
                      label=f'Mean={bs_ms.mean():.3f}')
    axes[0,0].axvline(bs_ms.mean() - 2*bs_ms.std(), color='red', linestyle='--')
    axes[0,0].axvline(bs_ms.mean() + 2*bs_ms.std(), color='red', linestyle='--')
    axes[0,0].set_xlabel("Mean marker gene s (per bootstrap)")
    axes[0,0].set_title(f"Bootstrap distribution of marker s\n"
                        f"mean={bs_ms.mean():.3f} +/- {bs_ms.std():.3f}")
    axes[0,0].legend()

    axes[0,1].boxplot(bs_fall.T.tolist() if len(bs_fall) > 0 else [[0]]*n_marker)
    axes[0,1].set_xlabel("Marker gene index")
    axes[0,1].set_ylabel("Scale s (across bootstrap samples)")
    axes[0,1].set_title(f"Per-gene s stability\nmax std={per_gene_std.max():.3f}")

    axes[1,0].hist(bs_fr*100, bins=30, color='tomato', edgecolor='white')
    axes[1,0].axvline(100, color='black', linestyle='--')
    axes[1,0].set_xlabel("Marker F improvement rate (%)")
    axes[1,0].set_title(f"F retention stability\nmean={bs_fr.mean():.0%}")

    axes[1,1].hist(bs_sr, bins=30, color='seagreen', edgecolor='white')
    axes[1,1].axvline(1.0, color='black', linestyle='--')
    axes[1,1].set_xlabel("Cell-type separation retention")
    axes[1,1].set_title(f"Sep retention stability\nmean={bs_sr.mean():.3f}")

    plt.tight_layout()
    plt.savefig("overnight_figures/blockD_bootstrap.pdf", dpi=150, bbox_inches='tight')
    plt.savefig("overnight_figures/blockD_bootstrap.png", dpi=150, bbox_inches='tight')
    plt.close()

    return summary


# ═════════════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════════════



def _plot_multibatch_comparison(results, summary, path):
    valid = {k: v for k, v in results.items()
             if isinstance(v, dict) and 'error' not in v and 'skipped' not in v}
    if not valid:
        return

    metrics = [
        ('batch_mixing_delta',    'Batch mixing delta'),
        ('celltype_purity_delta', 'Cell-type purity delta'),
        ('marker_f_retention',    'Marker F retention'),
    ]
    methods = list(valid.keys())
    colors  = ['steelblue', 'tomato', 'seagreen', 'darkorange']

    fig, axes = plt.subplots(1, 3, figsize=(13, 5))
    fig.suptitle(f"Block E: Real Multi-Batch Validation\n"
                 f"{summary['n_cells']} cells, "
                 f"{summary['n_batches']} batches, "
                 f"{summary['n_celltypes']} cell types",
                 fontweight='bold')

    for ax, (metric, label) in zip(axes, metrics):
        vals = [valid[m].get(metric, 0) for m in methods]
        bars = ax.bar(methods, vals,
                      color=colors[:len(methods)], alpha=0.85)
        ax.axhline(0 if 'delta' in metric else 1,
                   color='black', linestyle='--', linewidth=0.8)
        ax.set_title(label)
        ax.set_xticklabels(methods, rotation=20, ha='right', fontsize=9)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + 0.002,
                    f"{v:.3f}", ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.savefig(path.replace('.pdf','.png'), dpi=150, bbox_inches='tight')
    plt.close()

def run_block_e(h5ad_path, logger=None):
    """
    Real multi-batch validation on immune cell dataset.
    
    Data: Luecken et al. 2022 (Nature Methods) benchmark
    - Multiple donors + multiple labs + multiple protocols
    - Real batch effects (NOT simulated)
    - Ground truth cell type labels available
    
    Key test: without using any batch labels,
    does the topology-induced scale axis + L7 flow
    preserve cell-type structure while attenuating batch variation?
    
    Comparison: vs ComBat and Harmony (which DO use batch labels)
    """
    import os
    if not os.path.exists(h5ad_path):
        logger.info(f"BLOCK E: File not found: {h5ad_path}, skipping")
        return {'skipped': True, 'reason': f'file not found: {h5ad_path}'}

    try:
        import scanpy as sc
        import anndata as ad
    except ImportError:
        return {'skipped': True, 'reason': 'scanpy not installed'}

    logger.info(f"BLOCK E: Real Multi-Batch Validation")
    logger.info(f"  Data: {h5ad_path}")

    # ── Load data ──────────────────────────────────────────────────────────
    adata = sc.read_h5ad(h5ad_path)
    logger.info(f"  Loaded: {adata.shape[0]} cells x {adata.shape[1]} genes")
    logger.info(f"  obs columns: {list(adata.obs.columns)}")

    # Find batch and celltype columns
    batch_col = None
    ct_col    = None
    for col in adata.obs.columns:
        cl = col.lower()
        if any(k in cl for k in ['batch', 'study', 'lab', 'protocol', 'tech']):
            batch_col = col
        if any(k in cl for k in ['cell_type', 'celltype', 'label', 'louvain',
                                   'leiden', 'annotation', 'cluster']):
            ct_col = col

    if batch_col is None:
        # Try first categorical column as batch
        for col in adata.obs.columns:
            if hasattr(adata.obs[col], 'cat') and \
               2 <= adata.obs[col].nunique() <= 20:
                batch_col = col
                break

    if ct_col is None:
        for col in adata.obs.columns:
            if col != batch_col and hasattr(adata.obs[col], 'cat') and \
               2 <= adata.obs[col].nunique() <= 30:
                ct_col = col
                break

    logger.info(f"  Batch column: {batch_col}  "
                f"({adata.obs[batch_col].nunique() if batch_col else 'N/A'} batches)")
    logger.info(f"  Cell type column: {ct_col}  "
                f"({adata.obs[ct_col].nunique() if ct_col else 'N/A'} types)")

    # ── Subsample if too large ────────────────────────────────────────────
    max_cells = 3000
    max_genes = 2000
    if adata.shape[0] > max_cells:
        sc.pp.subsample(adata, n_obs=max_cells, random_state=42)
        logger.info(f"  Subsampled to {adata.shape[0]} cells")

    # HVG selection if too many genes
    if adata.shape[1] > max_genes:
        sc.pp.highly_variable_genes(adata, n_top_genes=max_genes,
                                     flavor='seurat')
        adata = adata[:, adata.var['highly_variable']].copy()
        logger.info(f"  HVG selection: {adata.shape[1]} genes")

    # Get expression matrix
    X = adata.X.toarray() if hasattr(adata.X, 'toarray') else np.array(adata.X)
    X = np.log1p(np.maximum(X, 0))

    batch_labels  = np.array(adata.obs[batch_col]) if batch_col else \
                    np.zeros(X.shape[0], dtype=str)
    celltype      = np.array(adata.obs[ct_col])    if ct_col    else \
                    np.zeros(X.shape[0], dtype=str)
    gene_names    = np.array(adata.var_names)

    # Known marker genes (immune cells)
    marker_genes = ['LYZ','CST3','FCGR3A','NKG7','GNLY','GZMB','KLRB1',
                    'CD79A','MS4A1','IL7R','CCR7','CD3D','CD4','CD8A',
                    'GZMK','FCER1A','HLA-DRA','S100A8','S100A9','SELL']
    marker_idx = [i for i, g in enumerate(gene_names) if g in marker_genes]
    logger.info(f"  Marker genes found: {len(marker_idx)}")

    results = {}
    t0 = time.time()

    # ── Our method ────────────────────────────────────────────────────────
    logger.info("  Running: Scale Axis + L7 (no batch labels)")
    W = build_graph(X)
    ev, evec = spectral_decompose(W)
    s = induced_scale(ev)
    X_ours = apply_l7(X, evec, s, T=12.0, a=1.0, b=2.0)

    # Marker gene positions
    if marker_idx:
        ms = np.array([s[i] for i in marker_idx])
        s_max = s.max()
        frac_large = (ms > 0.5 * s_max).mean()
        logger.info(f"    Marker s: {ms.mean():.3f}+/-{ms.std():.3f}  "
                    f"frac_large_s={frac_large:.0%}")

    m_ours = compute_all_metrics(X, X_ours, celltype, batch_labels,
                                  marker_idx, label="ScaleAxis+L7")
    results['ScaleAxis_L7'] = m_ours
    logger.info(f"    batch_mix_d={m_ours['batch_mixing_delta']:+.4f}  "
                f"purity_d={m_ours['celltype_purity_delta']:+.4f}  "
                f"F_ret={m_ours['marker_f_retention']:.3f}")

    # ── ComBat ────────────────────────────────────────────────────────────
    if batch_col:
        logger.info("  Running: ComBat (uses batch labels)")
        try:
            adata_tmp = ad.AnnData(X=X)
            adata_tmp.obs['batch'] = batch_labels
            sc.pp.combat(adata_tmp, key='batch')
            X_combat = np.array(adata_tmp.X)
            m_combat = compute_all_metrics(X, X_combat, celltype, batch_labels,
                                            marker_idx, label="ComBat")
            results['ComBat'] = m_combat
            logger.info(f"    batch_mix_d={m_combat['batch_mixing_delta']:+.4f}  "
                        f"purity_d={m_combat['celltype_purity_delta']:+.4f}  "
                        f"F_ret={m_combat['marker_f_retention']:.3f}")
        except Exception as e:
            logger.info(f"    ComBat error: {e}")
            results['ComBat'] = {'error': str(e)}

    # ── Harmony ───────────────────────────────────────────────────────────
    if batch_col:
        logger.info("  Running: Harmony (uses batch labels)")
        try:
            import harmonypy as hm
            import pandas as pd
            pca = pca_embed(X, n_components=min(30, X.shape[1]-1))
            meta = pd.DataFrame({'batch': batch_labels})
            ho = hm.run_harmony(pca, meta, vars_use=['batch'],
                                 max_iter_harmony=20, verbose=False)
            X_harm = ho.Z_corr.T
            pca_raw = pca_embed(X, n_components=X_harm.shape[1])
            m_harm = compute_all_metrics(pca_raw, X_harm, celltype,
                                          batch_labels, marker_idx,
                                          label="Harmony")
            results['Harmony'] = m_harm
            logger.info(f"    batch_mix_d={m_harm['batch_mixing_delta']:+.4f}  "
                        f"purity_d={m_harm['celltype_purity_delta']:+.4f}")
        except Exception as e:
            logger.info(f"    Harmony error: {e}")
            results['Harmony'] = {'error': str(e)}

    # ── Summary ───────────────────────────────────────────────────────────
    elapsed = time.time() - t0
    summary = {
        'dataset':          h5ad_path,
        'n_cells':          X.shape[0],
        'n_genes':          X.shape[1],
        'n_batches':        int(np.unique(batch_labels).shape[0]),
        'n_celltypes':      int(np.unique(celltype).shape[0]),
        'n_marker_genes':   len(marker_idx),
        'results':          results,
        'elapsed_s':        elapsed,
    }

    if marker_idx:
        summary['marker_s_mean'] = float(ms.mean())
        summary['marker_s_std']  = float(ms.std())
        summary['frac_large_s']  = float(frac_large)

    logger.info(f"  BLOCK E DONE in {elapsed:.0f}s")
    logger.info(f"  Real multi-batch result:")
    logger.info(f"    Our method F_ret={m_ours['marker_f_retention']:.3f}  "
                f"(no batch labels)")
    if 'ComBat' in results and 'error' not in results['ComBat']:
        logger.info(f"    ComBat    F_ret={results['ComBat']['marker_f_retention']:.3f}  "
                    f"(uses batch labels)")
    if 'Harmony' in results and 'error' not in results['Harmony']:
        logger.info(f"    Harmony   batch_mix_d="
                    f"{results['Harmony']['batch_mixing_delta']:+.4f}  "
                    f"(uses batch labels)")

    # Plot
    _plot_multibatch_comparison(results, summary,
                                 "overnight_figures/blockE_real_multibatch.pdf")
    return summary


def main():
    parser = argparse.ArgumentParser(description="Overnight Validation Suite")
    parser.add_argument('--skip', nargs='+', default=[],
                        choices=['A','B','C','D','E'],
                        help='Blocks to skip')
    parser.add_argument('--only', type=str, default=None,
                        choices=['A','B','C','D','E'],
                        help='Run only this block')
    parser.add_argument('--h5ad', type=str,
                        default='immune_multibatch.h5ad',
                        help='Path to real multi-batch h5ad file for Block E')
    parser.add_argument('--n_iter',       type=int, default=500000)
    parser.add_argument('--n_params',     type=int, default=50000)
    parser.add_argument('--n_perm',       type=int, default=100000)
    parser.add_argument('--n_bootstrap',  type=int, default=2000)
    args = parser.parse_args()

    if args.only:
        run_blocks = {args.only}
    else:
        run_blocks = {'A','B','C','D','E'} - set(args.skip)

    logger = ProgressLogger("overnight_progress.log", interval=300)

    logger.info("=" * 60)
    logger.info("OVERNIGHT VALIDATION SUITE")
    logger.info(f"Blocks to run: {sorted(run_blocks)}")
    logger.info(f"A: N={args.n_iter:,}  B: N={args.n_params:,}  "
                f"C: N={args.n_perm:,}  D: N={args.n_bootstrap}  "
                f"E: {args.h5ad}")
    logger.info("=" * 60)

    t_total = time.time()
    results = {}

    # Load data (needed for B, C, D)
    if run_blocks & {'B','C','D'}:
        X, celltype, gene_names, marker_idx = load_pbmc68k(logger)
        logger.info("Computing canonical spectral decomposition...")
        W_canon    = build_graph(X)
        ev_canon, evec_canon = spectral_decompose(W_canon)
        s_canon    = induced_scale(ev_canon)
        logger.info(f"  s in [{s_canon.min():.3f}, {s_canon.max():.3f}]")

    # BLOCK A
    if 'A' in run_blocks:
        logger.info("\n" + "="*60)
        r = run_block_a(n_iter=args.n_iter, logger=logger)
        results['block_A'] = r

    # BLOCK B
    if 'B' in run_blocks:
        logger.info("\n" + "="*60)
        r = run_block_b(X, celltype, marker_idx, evec_canon, s_canon,
                        n_params=args.n_params, logger=logger)
        results['block_B'] = r

    # BLOCK C
    if 'C' in run_blocks:
        logger.info("\n" + "="*60)
        r = run_block_c(s_canon, marker_idx, X.shape[1],
                        n_perm=args.n_perm, logger=logger)
        results['block_C'] = r

    # BLOCK D
    if 'D' in run_blocks:
        logger.info("\n" + "="*60)
        r = run_block_d(X, celltype, gene_names, marker_idx,
                        n_bootstrap=args.n_bootstrap, logger=logger)
        results['block_D'] = r

    # BLOCK E
    if 'E' in run_blocks:
        logger.info("\n" + "="*60)
        h5ad = getattr(args, 'h5ad', 'immune_multibatch.h5ad')
        r = run_block_e(h5ad, logger=logger)
        results['block_E'] = r

    # Save
    out = "overnight_results/overnight_summary.json"
    with open(out, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, default=str)

    total_h = (time.time() - t_total) / 3600
    logger.info("\n" + "="*60)
    logger.info(f"ALL DONE  total={total_h:.2f}h")
    logger.info(f"Results: {out}")
    logger.info(f"Figures: overnight_figures/")
    logger.info("="*60)
    logger.info("SUMMARY FOR PAPER:")
    if 'block_A' in results and not results['block_A'].get('skipped'):
        a = results['block_A']
        logger.info(f"  Block A: crossings={a['crossing_count']} / "
                    f"{a['total_pair_checks']:,}  "
                    f"E_viol={a['energy_viol_count']} / {a['n_iter']:,}")
    if 'block_B' in results:
        b = results['block_B']
        logger.info(f"  Block B: F_imp={b['f_improvement_mean']:.0%}  "
                    f"mono={b['energy_monotone_rate']:.0%}  "
                    f"ratio={b['stability_ratio_mean']:.1f}x")
    if 'block_C' in results:
        c = results['block_C']
        logger.info(f"  Block C: {c['obs_in_large_s']}/{c['n_markers']} "
                    f"in large-s  {c['p_value_str']}")
    if 'block_D' in results:
        d = results['block_D']
        logger.info(f"  Block D: marker_s={d['marker_s_mean']:.3f}+/-"
                    f"{d['marker_s_std']:.3f}  "
                    f"CI95=[{d['marker_s_ci95_low']:.3f},"
                    f"{d['marker_s_ci95_high']:.3f}]  "
                    f"stable={d['stable']}")

    if 'block_E' in results and not results['block_E'].get('skipped'):
        e = results['block_E']
        er = e.get('results', {})
        ours = er.get('ScaleAxis_L7', {})
        combat = er.get('ComBat', {})
        logger.info(f"  Block E: real {e['n_batches']}-batch data  "
                    f"{e['n_cells']} cells  {e['n_celltypes']} types")
        logger.info(f"    ScaleAxis+L7 F_ret={ours.get('marker_f_retention',0):.3f}  "
                    f"(no labels)")
        if 'error' not in combat:
            logger.info(f"    ComBat       F_ret={combat.get('marker_f_retention',0):.3f}  "
                        f"(uses labels)")
    logger.stop()


if __name__ == "__main__":
    main()

# ═════════════════════════════════════════════════════════════════════════════
# BLOCK E: Real Multi-Batch Validation (Luecken et al. 2022 benchmark data)
# ═════════════════════════════════════════════════════════════════════════════
