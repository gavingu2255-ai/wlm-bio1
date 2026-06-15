"""
scale_axis_benchmark.py
═══════════════════════════════════════════════════════════════════════════════
Multi-Batch Benchmark: Topology-Induced Scale Axis vs. ComBat / Harmony / scVI

Four modules:

  MODULE 1: Multi-batch data loading
    - PBMC68k (bulk_labels, single source, baseline)
    - Simulated multi-batch from PBMC68k (split into 3 artificial batches
      with different library-size offsets and chemistry shifts)
    - If available: real multi-batch via scib benchmark datasets

  MODULE 2: Baseline methods comparison
    - ComBat   (via pycombat or scanpy's pp.combat)
    - Harmony  (via harmonypy)
    - scVI     (skip if not installed, flag in results)
    - Our method: topology-induced scale axis + L7 flow

  MODULE 3: Graph construction robustness
    - Pearson (threshold 0.0, 0.1, 0.2, 0.3)
    - Thresholded Pearson (p50, p75, p90)
    - kNN graph (k=5, 10, 20)
    - Mutual information proxy
    - Metric: stability of marker gene s-positions + F improvement rate

  MODULE 4: Quantitative metrics
    - Batch mixing score (entropy of batch labels in kNN neighborhood)
    - Cell-type purity (fraction of same-type cells in kNN neighborhood)
    - Marker gene retention (F-stat ratio after/before correction)
    - Silhouette score (cell-type vs batch)

Progress: logged to benchmark_progress.log every 5 minutes.

Usage:
  python scale_axis_benchmark.py              # full run
  python scale_axis_benchmark.py --quick      # reduced N for testing
  python scale_axis_benchmark.py --module 2   # single module

Requirements:
  pip install scanpy anndata numpy scipy matplotlib scikit-learn tqdm
  pip install harmonypy          # for Harmony comparison
  pip install pycombat           # for ComBat comparison (optional)
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
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import silhouette_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

warnings.filterwarnings("ignore")
os.makedirs("benchmark_figures", exist_ok=True)
os.makedirs("benchmark_results",  exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# Progress logger: writes to file every 5 minutes
# ─────────────────────────────────────────────────────────────────────────────
class ProgressLogger:
    def __init__(self, log_path="benchmark_progress.log", interval=300):
        self.log_path = log_path
        self.interval = interval  # seconds
        self.messages = []
        self.start_time = time.time()
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._writer_loop, daemon=True)
        self._thread.start()

        # Also set up Python logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s  %(message)s',
            handlers=[
                logging.FileHandler(log_path),
                logging.StreamHandler(sys.stdout),
            ]
        )
        self.log = logging.getLogger("benchmark")

    def _writer_loop(self):
        while not self._stop.is_set():
            time.sleep(self.interval)
            self._flush()

    def _flush(self):
        elapsed = time.time() - self.start_time
        with self._lock:
            msgs = list(self.messages)
            self.messages = []

        with open(self.log_path, 'a') as f:
            f.write(f"\n──── Progress checkpoint  elapsed={elapsed/60:.1f}min ────\n")
            for m in msgs:
                f.write(m + "\n")

    def info(self, msg):
        self.log.info(msg)
        with self._lock:
            self.messages.append(msg)

    def stop(self):
        self._stop.set()
        self._flush()


# ─────────────────────────────────────────────────────────────────────────────
# Core spectral functions
# ─────────────────────────────────────────────────────────────────────────────
def build_laplacian(W):
    D = np.diag(W.sum(axis=1))
    return D - W

def spectral_decompose(L):
    eigvals, eigvecs = la.eigh(L)
    return np.maximum(eigvals, 0.0), eigvecs

def induced_scale(eigvals):
    return np.log1p(eigvals.max() - eigvals)

def l7_decay(s, a=1.0, b=2.0):
    return 2.0 / (a**2 * b**(2*s))

def apply_l7(X, eigvecs, s, T=12.0, a=1.0, b=2.0):
    decay = np.exp(-l7_decay(s, a, b) * T)
    C = X @ eigvecs
    return (C * decay[None, :]) @ eigvecs.T

# ─────────────────────────────────────────────────────────────────────────────
# Graph construction methods
# ─────────────────────────────────────────────────────────────────────────────
def graph_pearson(X, threshold=0.0):
    W = np.corrcoef(X.T)
    np.fill_diagonal(W, 0.0)
    return np.maximum(W, threshold)

def graph_thresholded(X, percentile=75):
    W = np.corrcoef(X.T)
    np.fill_diagonal(W, 0.0)
    thresh = np.percentile(np.abs(W[W != 0]), percentile)
    out = W.copy()
    out[np.abs(out) < thresh] = 0.0
    return np.maximum(out, 0.0)

def graph_knn(X, k=10):
    """kNN graph in gene space (genes as data points, cells as features)"""
    try:
        nbrs = NearestNeighbors(n_neighbors=k+1, metric='cosine').fit(X.T)
        distances, indices = nbrs.kneighbors(X.T)
        n = X.shape[1]
        W = np.zeros((n, n))
        for i, (dists, idxs) in enumerate(zip(distances, indices)):
            for d, j in zip(dists[1:], idxs[1:]):
                W[i, j] = 1.0 - d
                W[j, i] = 1.0 - d
        return np.maximum(W, 0.0)
    except Exception:
        return graph_thresholded(X, percentile=80)

def graph_mi_proxy(X):
    """Mutual information proxy via rank correlation"""
    from scipy.stats import spearmanr
    n_genes = X.shape[1]
    # Spearman rank correlation as MI proxy (monotonic relationships)
    rho, _ = spearmanr(X)
    if np.isscalar(rho):
        return np.zeros((n_genes, n_genes))
    np.fill_diagonal(rho, 0.0)
    return np.maximum(rho, 0.0)

GRAPH_METHODS = {
    'pearson_0.0':  lambda X: graph_pearson(X, 0.0),
    'pearson_0.1':  lambda X: graph_pearson(X, 0.1),
    'pearson_0.2':  lambda X: graph_pearson(X, 0.2),
    'pearson_0.3':  lambda X: graph_pearson(X, 0.3),
    'thresh_p50':   lambda X: graph_thresholded(X, 50),
    'thresh_p75':   lambda X: graph_thresholded(X, 75),
    'thresh_p90':   lambda X: graph_thresholded(X, 90),
    'knn_5':        lambda X: graph_knn(X, 5),
    'knn_10':       lambda X: graph_knn(X, 10),
    'knn_20':       lambda X: graph_knn(X, 20),
    'mi_spearman':  lambda X: graph_mi_proxy(X),
}

# ─────────────────────────────────────────────────────────────────────────────
# Quantitative metrics
# ─────────────────────────────────────────────────────────────────────────────
def batch_mixing_score(X_pca, batch_labels, k=30):
    """
    For each cell, compute entropy of batch labels in its k-NN neighborhood.
    Higher = better mixing (batch effect removed).
    Uses PCA embedding for efficiency.
    """
    nbrs = NearestNeighbors(n_neighbors=k+1).fit(X_pca)
    _, indices = nbrs.kneighbors(X_pca)
    unique_batches = np.unique(batch_labels)
    n_batches = len(unique_batches)

    entropies = []
    for i, nbr_idx in enumerate(indices):
        nbr_batches = batch_labels[nbr_idx[1:]]  # exclude self
        counts = np.array([(nbr_batches == b).sum() for b in unique_batches])
        probs = counts / counts.sum()
        probs = probs[probs > 0]
        entropy = -np.sum(probs * np.log(probs + 1e-12))
        # Normalize by max entropy
        max_entropy = np.log(n_batches)
        entropies.append(entropy / (max_entropy + 1e-12))

    return float(np.mean(entropies))

def celltype_purity(X_pca, celltype_labels, k=30):
    """
    For each cell, fraction of k-NN neighbors with same cell type.
    Higher = better cell-type preservation.
    """
    nbrs = NearestNeighbors(n_neighbors=k+1).fit(X_pca)
    _, indices = nbrs.kneighbors(X_pca)

    purities = []
    for i, nbr_idx in enumerate(indices):
        nbr_types = celltype_labels[nbr_idx[1:]]
        same = (nbr_types == celltype_labels[i]).mean()
        purities.append(same)

    return float(np.mean(purities))

def marker_f_retention(X_before, X_after, celltype, marker_indices):
    """Mean ratio of F-stat after/before for marker genes."""
    ratios = []
    for gi in marker_indices:
        if gi >= X_before.shape[1]:
            continue
        f_b = gene_f_stat(X_before[:, gi], celltype)
        f_a = gene_f_stat(X_after[:,  gi], celltype)
        if f_b > 0:
            ratios.append(f_a / f_b)
    return float(np.mean(ratios)) if ratios else 1.0

def gene_f_stat(expr, celltype):
    unique = np.unique(celltype)
    groups = [expr[celltype == c] for c in unique if (celltype == c).sum() > 1]
    if len(groups) < 2:
        return 0.0
    f, _ = stats.f_oneway(*groups)
    return float(f) if np.isfinite(f) else 0.0

def pca_embed(X, n_components=30):
    """Fast PCA via SVD"""
    Xc = X - X.mean(axis=0)
    try:
        _, _, Vt = la.svd(Xc, full_matrices=False)
        return Xc @ Vt[:n_components].T
    except Exception:
        return Xc[:, :n_components]

def compute_all_metrics(X_raw, X_corrected, celltype, batch_labels,
                        marker_indices, label=""):
    """Compute full metric set for one correction result."""
    n_pca = min(30, X_raw.shape[1]-1, X_raw.shape[0]-1)

    # PCA embeddings
    pca_raw  = pca_embed(X_raw,       n_pca)
    pca_corr = pca_embed(X_corrected, n_pca)

    k = min(30, X_raw.shape[0]//5)

    bms_raw  = batch_mixing_score(pca_raw,  batch_labels, k=k)
    bms_corr = batch_mixing_score(pca_corr, batch_labels, k=k)

    ctp_raw  = celltype_purity(pca_raw,  celltype, k=k)
    ctp_corr = celltype_purity(pca_corr, celltype, k=k)

    mfr = marker_f_retention(X_raw, X_corrected, celltype, marker_indices)

    # Silhouette scores (subsample for speed)
    n_sub = min(500, X_raw.shape[0])
    idx = np.random.choice(X_raw.shape[0], n_sub, replace=False)
    try:
        sil_batch_raw  = silhouette_score(pca_raw[idx],  batch_labels[idx])
        sil_batch_corr = silhouette_score(pca_corr[idx], batch_labels[idx])
        sil_ct_raw     = silhouette_score(pca_raw[idx],  celltype[idx])
        sil_ct_corr    = silhouette_score(pca_corr[idx], celltype[idx])
    except Exception:
        sil_batch_raw = sil_batch_corr = sil_ct_raw = sil_ct_corr = 0.0

    return {
        'label':              label,
        'batch_mixing_raw':   bms_raw,
        'batch_mixing_corr':  bms_corr,
        'batch_mixing_delta': bms_corr - bms_raw,
        'celltype_purity_raw':   ctp_raw,
        'celltype_purity_corr':  ctp_corr,
        'celltype_purity_delta': ctp_corr - ctp_raw,
        'marker_f_retention':    mfr,
        'sil_batch_raw':         sil_batch_raw,
        'sil_batch_corr':        sil_batch_corr,
        'sil_batch_delta':       sil_batch_corr - sil_batch_raw,
        'sil_celltype_raw':      sil_ct_raw,
        'sil_celltype_corr':     sil_ct_corr,
        'sil_celltype_delta':    sil_ct_corr - sil_ct_raw,
    }


# ═════════════════════════════════════════════════════════════════════════════
# MODULE 1: Multi-batch data loading
# ═════════════════════════════════════════════════════════════════════════════

def load_multibatch_data(logger):
    """
    Loads PBMC68k and creates three artificial batches with realistic
    batch effects (library size, chemistry shift, dropout).

    Returns: X, celltype, batch_labels, gene_names, marker_indices
    """
    import scanpy as sc

    logger.info("MODULE 1: Loading and constructing multi-batch data")

    adata = sc.datasets.pbmc68k_reduced()
    X_base = adata.X.toarray() if hasattr(adata.X, 'toarray') else np.array(adata.X)
    gene_names  = np.array(adata.var_names)
    celltype    = np.array(adata.obs['bulk_labels'])

    logger.info(f"  Base data: {X_base.shape[0]} cells × {X_base.shape[1]} genes")
    logger.info(f"  Cell types: {np.unique(celltype)}")

    rng = np.random.default_rng(42)
    n_cells, n_genes = X_base.shape

    # Split into 3 batches (roughly equal, stratified by cell type)
    batch_labels = np.empty(n_cells, dtype=object)
    unique_ct = np.unique(celltype)
    for ct in unique_ct:
        ct_idx = np.where(celltype == ct)[0]
        rng.shuffle(ct_idx)
        thirds = np.array_split(ct_idx, 3)
        for bi, idxs in enumerate(thirds):
            batch_labels[idxs] = f"batch_{bi+1}"

    X_batched = X_base.copy()

    # Batch 2: library size × 1.3 (all genes, multiplicative)
    b2 = batch_labels == "batch_2"
    X_batched[b2] *= 1.3

    # Batch 3: library size × 0.8 + chemistry shift on first 20% of genes
    b3 = batch_labels == "batch_3"
    X_batched[b3] *= 0.8
    n_chem = int(0.20 * n_genes)
    X_batched[b3, :n_chem] += 0.6

    # All batches: random dropout (gene-specific, batch-specific)
    for bi, bname in enumerate(["batch_1", "batch_2", "batch_3"]):
        mask = batch_labels == bname
        dropout_rate = [0.05, 0.12, 0.08][bi]
        dropout_mask = rng.random(size=(mask.sum(), n_genes)) < dropout_rate
        X_batched[mask] *= (1 - dropout_mask.astype(float))

    X_batched = np.log1p(np.maximum(X_batched, 0))

    logger.info(f"  Created 3 batches:")
    for bname in ["batch_1", "batch_2", "batch_3"]:
        n = (batch_labels == bname).sum()
        logger.info(f"    {bname}: {n} cells")

    # Marker genes
    marker_genes = ['LYZ','CST3','FCGR3A','NKG7','GNLY','GZMB','KLRB1',
                    'CD79A','MS4A1','IL7R','CCR7','CD4','CD8A','CD8B',
                    'GZMK','FCER1A','HLA-DQA1']
    marker_indices = [i for i,g in enumerate(gene_names) if g in marker_genes]
    logger.info(f"  Marker genes found: {len(marker_indices)}")

    return X_batched, celltype, batch_labels, gene_names, marker_indices


# ═════════════════════════════════════════════════════════════════════════════
# MODULE 2: Baseline methods comparison
# ═════════════════════════════════════════════════════════════════════════════

def apply_our_method(X, celltype, batch_labels, gene_names, marker_indices,
                     graph_fn=None, T=12.0, a=1.0, b=2.0, logger=None):
    """Apply topology-induced scale axis + L7 flow"""
    if graph_fn is None:
        graph_fn = lambda X: graph_pearson(X, 0.0)

    W = graph_fn(X)
    L = build_laplacian(W)
    eigvals, eigvecs = spectral_decompose(L)
    s = induced_scale(eigvals)
    X_corr = apply_l7(X, eigvecs, s, T=T, a=a, b=b)
    return X_corr, s, eigvecs

def apply_combat(X, batch_labels, logger=None):
    """Apply ComBat batch correction"""
    try:
        import scanpy as sc
        import anndata as ad
        adata = ad.AnnData(X=X)
        adata.obs['batch'] = batch_labels
        sc.pp.combat(adata, key='batch')
        result = np.array(adata.X)
        if logger:
            logger.info("  ComBat: applied successfully")
        return result, "ComBat"
    except ImportError:
        if logger:
            logger.info("  ComBat: scanpy not available, skipping")
        return None, "ComBat (unavailable)"
    except Exception as e:
        if logger:
            logger.info(f"  ComBat: error — {e}")
        return None, f"ComBat (error: {e})"

def apply_harmony(X, batch_labels, logger=None):
    """Apply Harmony batch correction (on PCA embeddings)"""
    try:
        import harmonypy as hm
        import pandas as pd
        pca = pca_embed(X, n_components=min(30, X.shape[1]-1))
        meta = pd.DataFrame({'batch': batch_labels})
        ho = hm.run_harmony(pca, meta, vars_use=['batch'],
                             max_iter_harmony=20, verbose=False)
        # Harmony corrects PCA; reconstruct approximate corrected expression
        # We report metrics in PCA space for Harmony
        result_pca = ho.Z_corr.T  # (n_cells, n_pcs)
        if logger:
            logger.info(f"  Harmony: applied successfully (PCA space, {result_pca.shape[1]} PCs)")
        return result_pca, "Harmony"
    except ImportError:
        if logger:
            logger.info("  Harmony: harmonypy not installed. pip install harmonypy")
        return None, "Harmony (unavailable)"
    except Exception as e:
        if logger:
            logger.info(f"  Harmony: error — {e}")
        return None, f"Harmony (error)"

def apply_scvi(X, batch_labels, gene_names, logger=None):
    """Apply scVI (if installed)"""
    try:
        import scvi
        import anndata as ad
        import pandas as pd
        adata = ad.AnnData(X=np.expm1(X).astype(int))  # scVI needs raw counts
        adata.obs['batch'] = batch_labels
        adata.var_names = pd.Index(gene_names)
        scvi.model.SCVI.setup_anndata(adata, batch_key='batch')
        model = scvi.model.SCVI(adata, n_latent=20, n_layers=2)
        model.train(max_epochs=50, progress_bar=False)
        latent = model.get_latent_representation()
        if logger:
            logger.info(f"  scVI: applied successfully (latent dim={latent.shape[1]})")
        return latent, "scVI"
    except ImportError:
        if logger:
            logger.info("  scVI: not installed. pip install scvi-tools")
        return None, "scVI (unavailable)"
    except Exception as e:
        if logger:
            logger.info(f"  scVI: error — {e}")
        return None, "scVI (error)"

def run_module2(X, celltype, batch_labels, gene_names, marker_indices, logger):
    """Compare all methods on multi-batch data"""
    logger.info("\nMODULE 2: Baseline Methods Comparison")

    all_metrics = {}
    t0 = time.time()

    # ── Our method ────────────────────────────────────────────────────────
    logger.info("  Running: Topology-Induced Scale Axis + L7")
    X_ours, s_ours, _ = apply_our_method(X, celltype, batch_labels,
                                          gene_names, marker_indices,
                                          logger=logger)
    m_ours = compute_all_metrics(X, X_ours, celltype, batch_labels,
                                  marker_indices, label="ScaleAxis+L7")
    all_metrics['ScaleAxis_L7'] = m_ours
    logger.info(f"    batch_mixing_delta={m_ours['batch_mixing_delta']:+.4f}  "
                f"purity_delta={m_ours['celltype_purity_delta']:+.4f}  "
                f"marker_F_retention={m_ours['marker_f_retention']:.3f}")

    # ── ComBat ────────────────────────────────────────────────────────────
    logger.info("  Running: ComBat")
    X_combat, label_combat = apply_combat(X, batch_labels, logger)
    if X_combat is not None:
        m_combat = compute_all_metrics(X, X_combat, celltype, batch_labels,
                                       marker_indices, label=label_combat)
        all_metrics['ComBat'] = m_combat
        logger.info(f"    batch_mixing_delta={m_combat['batch_mixing_delta']:+.4f}  "
                    f"purity_delta={m_combat['celltype_purity_delta']:+.4f}  "
                    f"marker_F_retention={m_combat['marker_f_retention']:.3f}")
    else:
        all_metrics['ComBat'] = {'label': label_combat, 'unavailable': True}

    # ── Harmony ───────────────────────────────────────────────────────────
    logger.info("  Running: Harmony")
    X_harmony, label_harmony = apply_harmony(X, batch_labels, logger)
    if X_harmony is not None:
        # Harmony works in PCA space; compute metrics there
        pca_raw = pca_embed(X, n_components=X_harmony.shape[1])
        m_harmony = compute_all_metrics(pca_raw, X_harmony, celltype,
                                         batch_labels, marker_indices,
                                         label=label_harmony)
        all_metrics['Harmony'] = m_harmony
        logger.info(f"    batch_mixing_delta={m_harmony['batch_mixing_delta']:+.4f}  "
                    f"purity_delta={m_harmony['celltype_purity_delta']:+.4f}")
    else:
        all_metrics['Harmony'] = {'label': label_harmony, 'unavailable': True}

    # ── scVI ──────────────────────────────────────────────────────────────
    logger.info("  Running: scVI")
    X_scvi, label_scvi = apply_scvi(X, batch_labels, gene_names, logger)
    if X_scvi is not None:
        pca_raw = pca_embed(X, n_components=X_scvi.shape[1])
        m_scvi = compute_all_metrics(pca_raw, X_scvi, celltype,
                                      batch_labels, marker_indices,
                                      label=label_scvi)
        all_metrics['scVI'] = m_scvi
        logger.info(f"    batch_mixing_delta={m_scvi['batch_mixing_delta']:+.4f}  "
                    f"purity_delta={m_scvi['celltype_purity_delta']:+.4f}")
    else:
        all_metrics['scVI'] = {'label': label_scvi, 'unavailable': True}

    elapsed = time.time() - t0
    logger.info(f"  Module 2 done in {elapsed:.0f}s")

    # ── Plot comparison ────────────────────────────────────────────────────
    _plot_comparison(all_metrics, "benchmark_figures/module2_comparison.pdf")

    return all_metrics


def _plot_comparison(all_metrics, path):
    methods = [k for k, v in all_metrics.items() if 'unavailable' not in v]
    if not methods:
        return

    metrics_to_plot = [
        ('batch_mixing_delta',    'Batch mixing Δ\n(higher = more mixing)'),
        ('celltype_purity_delta', 'Cell-type purity Δ\n(higher = better)'),
        ('marker_f_retention',    'Marker F retention\n(>1 = improved)'),
        ('sil_batch_delta',       'Silhouette (batch) Δ\n(lower = less batch)'),
        ('sil_celltype_delta',    'Silhouette (cell type) Δ\n(higher = better)'),
    ]

    fig, axes = plt.subplots(1, len(metrics_to_plot), figsize=(4*len(metrics_to_plot), 5))
    fig.suptitle("Method Comparison: Topology-Induced Scale Axis vs. Baselines",
                 fontsize=12, fontweight='bold')

    colors = ['steelblue', 'tomato', 'seagreen', 'darkorange']

    for ax, (metric, label) in zip(axes, metrics_to_plot):
        values = [all_metrics[m].get(metric, 0) for m in methods]
        bars = ax.bar(methods, values,
                      color=colors[:len(methods)], alpha=0.85, edgecolor='white')
        ax.axhline(0, color='black', linewidth=0.8, linestyle='--')
        ax.set_ylabel(label, fontsize=9)
        ax.set_xticklabels(methods, rotation=30, ha='right', fontsize=8)
        # Value labels
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + 0.001,
                    f"{val:.3f}", ha='center', va='bottom', fontsize=7)

    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.savefig(path.replace('.pdf', '.png'), dpi=150, bbox_inches='tight')
    plt.close()


# ═════════════════════════════════════════════════════════════════════════════
# MODULE 3: Graph construction robustness
# ═════════════════════════════════════════════════════════════════════════════

def run_module3(X, celltype, batch_labels, gene_names, marker_indices,
                n_synthetic=300, logger=None):
    """
    For each graph construction method:
      1. Compute scale axis
      2. Apply L7
      3. Measure metrics
    
    On both real data (PBMC68k multi-batch) and synthetic configs.
    """
    logger.info("\nMODULE 3: Graph Construction Robustness")
    logger.info(f"  {len(GRAPH_METHODS)} graph methods × real data + {n_synthetic} synthetic configs")

    results = {}
    t0 = time.time()

    for gname, gfn in GRAPH_METHODS.items():
        logger.info(f"  Graph: {gname}")
        t1 = time.time()

        # ── Real data ──────────────────────────────────────────────────
        try:
            W = gfn(X)
            L = build_laplacian(W)
            ev, evec = spectral_decompose(L)
            s = induced_scale(ev)
            X_T = apply_l7(X, evec, s)

            marker_s = np.array([s[i] for i in marker_indices if i < len(s)])
            s_max = s.max()

            m_real = compute_all_metrics(X, X_T, celltype, batch_labels,
                                          marker_indices, label=gname)
            m_real['marker_s_mean']   = float(marker_s.mean())
            m_real['marker_s_std']    = float(marker_s.std())
            m_real['marker_s_min']    = float(marker_s.min())
            m_real['frac_top50pct_s'] = float((marker_s > 0.5*s_max).mean())

            logger.info(f"    Real: marker_s={marker_s.mean():.3f}±{marker_s.std():.3f}  "
                        f"F_ret={m_real['marker_f_retention']:.3f}  "
                        f"mix_Δ={m_real['batch_mixing_delta']:+.4f}")
        except Exception as e:
            logger.info(f"    Real: ERROR {e}")
            m_real = {'error': str(e)}

        # ── Synthetic ─────────────────────────────────────────────────
        rng = np.random.default_rng(42)
        syn_results = []
        n_genes = X.shape[1]

        for _ in range(n_synthetic):
            n_cells_s = 150
            n_blocks  = rng.integers(4, 10)
            block_sz  = n_genes // n_blocks
            X_syn  = np.zeros((n_cells_s, n_genes))
            ct_syn = rng.choice(4, size=n_cells_s)

            for bi in range(n_blocks):
                g_start = bi * block_sz
                g_end   = (bi+1) * block_sz if bi < n_blocks-1 else n_genes
                latent  = rng.choice([-1,1], size=n_cells_s) * 2.0
                if bi < n_blocks // 3:
                    X_syn[:, g_start:g_end] = (latent[:,None] +
                        rng.normal(0, 0.2, (n_cells_s, g_end-g_start)))
                else:
                    X_syn[:, g_start:g_end] = rng.normal(0, 1, (n_cells_s, g_end-g_start))

            bat_syn = rng.choice(['b1','b2'], size=n_cells_s)
            # Add batch effect
            X_syn[bat_syn=='b2'] *= 1.2
            X_syn[bat_syn=='b2', :n_genes//5] += 0.5

            try:
                W_s = gfn(X_syn)
                L_s = build_laplacian(W_s)
                ev_s, evec_s = spectral_decompose(L_s)
                s_s = induced_scale(ev_s)
                X_sT = apply_l7(X_syn, evec_s, s_s)

                # Top-10 most cell-type-discriminative genes = "markers"
                f_vals = [gene_f_stat(X_syn[:,g], ct_syn) for g in range(n_genes)]
                top_m  = np.argsort(f_vals)[-10:]
                m_s_vals = s_s[top_m]

                bms_b = batch_mixing_score(pca_embed(X_syn,  min(10,n_genes-1)),
                                           bat_syn, k=15)
                bms_a = batch_mixing_score(pca_embed(X_sT, min(10,n_genes-1)),
                                           bat_syn, k=15)
                ctp_b = celltype_purity(pca_embed(X_syn,  min(10,n_genes-1)),
                                        ct_syn, k=15)
                ctp_a = celltype_purity(pca_embed(X_sT, min(10,n_genes-1)),
                                        ct_syn, k=15)

                syn_results.append({
                    'marker_s_mean':      float(m_s_vals.mean()),
                    'batch_mixing_delta': bms_a - bms_b,
                    'purity_delta':       ctp_a - ctp_b,
                })
            except Exception:
                continue

        m_syn = {}
        if syn_results:
            m_syn = {
                'n_configs':             len(syn_results),
                'marker_s_mean':         float(np.mean([r['marker_s_mean'] for r in syn_results])),
                'marker_s_std':          float(np.std( [r['marker_s_mean'] for r in syn_results])),
                'batch_mixing_delta_mean': float(np.mean([r['batch_mixing_delta'] for r in syn_results])),
                'purity_delta_mean':       float(np.mean([r['purity_delta']       for r in syn_results])),
            }
            logger.info(f"    Syn ({len(syn_results)} ok): "
                        f"marker_s={m_syn['marker_s_mean']:.3f}±{m_syn['marker_s_std']:.3f}  "
                        f"mix_Δ={m_syn['batch_mixing_delta_mean']:+.4f}  "
                        f"purity_Δ={m_syn['purity_delta_mean']:+.4f}")

        results[gname] = {'real': m_real, 'synthetic': m_syn}
        logger.info(f"    {gname} done in {time.time()-t1:.0f}s")

    # ── Summary ────────────────────────────────────────────────────────────
    valid_real = {k: v['real'] for k, v in results.items() if 'error' not in v['real']}
    marker_s_means = [v['marker_s_mean'] for v in valid_real.values()
                      if 'marker_s_mean' in v]
    f_rets  = [v['marker_f_retention'] for v in valid_real.values()
               if 'marker_f_retention' in v]

    summary = {
        'n_graphs':                     len(GRAPH_METHODS),
        'n_valid':                       len(valid_real),
        'marker_s_mean_across_graphs':  float(np.mean(marker_s_means)) if marker_s_means else 0,
        'marker_s_std_across_graphs':   float(np.std(marker_s_means))  if marker_s_means else 0,
        'f_retention_mean':             float(np.mean(f_rets)) if f_rets else 0,
        'f_retention_min':              float(np.min(f_rets))  if f_rets else 0,
    }

    logger.info(f"\n  MODULE 3 SUMMARY:")
    logger.info(f"  Graphs tested: {summary['n_graphs']} ({summary['n_valid']} valid)")
    logger.info(f"  Marker s across graphs: "
                f"{summary['marker_s_mean_across_graphs']:.3f} ± "
                f"{summary['marker_s_std_across_graphs']:.3f}")
    logger.info(f"  F retention: {summary['f_retention_mean']:.3f} "
                f"(min: {summary['f_retention_min']:.3f})")
    logger.info(f"  Elapsed: {time.time()-t0:.0f}s")

    # ── Plot ───────────────────────────────────────────────────────────────
    gnames = list(valid_real.keys())
    ms_m   = [valid_real[g].get('marker_s_mean', 0) for g in gnames]
    ms_s   = [valid_real[g].get('marker_s_std',  0) for g in gnames]
    fret   = [valid_real[g].get('marker_f_retention', 1) for g in gnames]
    bmix   = [valid_real[g].get('batch_mixing_delta', 0) for g in gnames]
    cpur   = [valid_real[g].get('celltype_purity_delta', 0) for g in gnames]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Module 3: Graph Construction Robustness", fontsize=13, fontweight='bold')

    axes[0,0].barh(gnames, ms_m, xerr=ms_s, color='steelblue', alpha=0.8, capsize=3)
    axes[0,0].set_xlabel("Marker gene mean scale s")
    axes[0,0].set_title(f"Marker s position\nmean±std = "
                        f"{summary['marker_s_mean_across_graphs']:.3f}±"
                        f"{summary['marker_s_std_across_graphs']:.3f}")

    axes[0,1].barh(gnames, fret, color='tomato', alpha=0.8)
    axes[0,1].axvline(1.0, color='black', linestyle='--')
    axes[0,1].set_xlabel("Marker F retention (after/before L7)")
    axes[0,1].set_title(f"F retention: {summary['f_retention_mean']:.3f} "
                        f"(min: {summary['f_retention_min']:.3f})")

    axes[1,0].barh(gnames, bmix, color='seagreen', alpha=0.8)
    axes[1,0].axvline(0, color='black', linestyle='--')
    axes[1,0].set_xlabel("Batch mixing Δ (after−before L7)")
    axes[1,0].set_title("Batch mixing improvement")

    axes[1,1].barh(gnames, cpur, color='darkorange', alpha=0.8)
    axes[1,1].axvline(0, color='black', linestyle='--')
    axes[1,1].set_xlabel("Cell-type purity Δ (after−before L7)")
    axes[1,1].set_title("Cell-type purity preservation")

    plt.tight_layout()
    plt.savefig("benchmark_figures/module3_graph_robustness.pdf", dpi=150, bbox_inches='tight')
    plt.savefig("benchmark_figures/module3_graph_robustness.png", dpi=150, bbox_inches='tight')
    plt.close()

    return results, summary


# ═════════════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Scale Axis Benchmark")
    parser.add_argument('--quick',  action='store_true',
                        help='Quick mode (reduced N for testing)')
    parser.add_argument('--module', type=int, default=0,
                        help='Run single module (1/2/3); 0=all')
    parser.add_argument('--n_syn',  type=int, default=None,
                        help='Synthetic configs per graph (Module 3)')
    args = parser.parse_args()

    n_syn = args.n_syn if args.n_syn else (30 if args.quick else 300)

    logger = ProgressLogger("benchmark_progress.log", interval=300)

    logger.info("╔══════════════════════════════════════════════════════════╗")
    logger.info("║  SCALE AXIS MULTI-BATCH BENCHMARK                       ║")
    logger.info("╚══════════════════════════════════════════════════════════╝")
    logger.info(f"  Quick mode: {args.quick}")
    logger.info(f"  Module: {'all' if args.module == 0 else args.module}")
    logger.info(f"  Synthetic configs per graph: {n_syn}")

    t_total = time.time()
    all_results = {}

    # MODULE 1: Load data
    if args.module in (0, 1, 2, 3):
        X, celltype, batch_labels, gene_names, marker_indices = \
            load_multibatch_data(logger)

    # MODULE 2: Comparison
    if args.module in (0, 2):
        m2 = run_module2(X, celltype, batch_labels, gene_names,
                         marker_indices, logger)
        all_results['module2'] = m2

    # MODULE 3: Graph robustness
    if args.module in (0, 3):
        m3_results, m3_summary = run_module3(
            X, celltype, batch_labels, gene_names, marker_indices,
            n_synthetic=n_syn, logger=logger)
        all_results['module3'] = {
            'per_graph': m3_results,
            'summary':   m3_summary,
        }

    # Save
    out_path = "benchmark_results/benchmark_summary.json"
    with open(out_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)

    total_h = (time.time() - t_total) / 3600
    logger.info(f"\nTotal elapsed: {total_h:.2f} hours")
    logger.info(f"Results: {out_path}")
    logger.info(f"Figures: benchmark_figures/")

    logger.info("\n┌─ FINAL SUMMARY FOR PAPER ──────────────────────────────┐")
    if 'module2' in all_results:
        ours = all_results['module2'].get('ScaleAxis_L7', {})
        logger.info(f"│  Our method vs raw:")
        logger.info(f"│    batch_mixing Δ:  {ours.get('batch_mixing_delta', 0):+.4f}")
        logger.info(f"│    purity Δ:        {ours.get('celltype_purity_delta', 0):+.4f}")
        logger.info(f"│    marker F ret:    {ours.get('marker_f_retention', 0):.4f}")
    if 'module3' in all_results:
        s3 = all_results['module3']['summary']
        logger.info(f"│  Graph robustness ({s3['n_graphs']} methods):")
        logger.info(f"│    marker s: {s3['marker_s_mean_across_graphs']:.3f} ± "
                    f"{s3['marker_s_std_across_graphs']:.3f}")
        logger.info(f"│    F ret:   {s3['f_retention_mean']:.3f} "
                    f"(min {s3['f_retention_min']:.3f})")
    logger.info("└────────────────────────────────────────────────────────┘")

    logger.stop()


if __name__ == "__main__":
    main()
