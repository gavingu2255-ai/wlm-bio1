"""
scale_axis_validate_tissue.py
===========================================================================
Topology-Induced Scale Axis -- Cross-Tissue Generalization Validation
(Tabula Sapiens: Liver / Lung)

This is an adaptation of pbmc68k_validate.py to test whether the scale
axis construction and its structural predictions generalize beyond PBMC
data to non-immune, solid-tissue single-cell datasets.

Data: Tabula Sapiens "named" h5ad files downloaded from figshare
      (TS_Liver.h5ad, TS_Lung.h5ad), NOT the CZ CELLxGENE hash-named
      versions -- those use Ensembl IDs and ontology term IDs instead
      of human-readable gene symbols and cell type names.

Key field mapping (Tabula Sapiens "named" version):
    obs['cell_ontology_class']  -> cell type label (replaces bulk_labels)
    obs['method']               -> assay platform (10x 3' v3 / smartseq2)
    var['gene_symbol']          -> gene symbol (replaces var_names)

Usage:
    python scale_axis_validate_tissue.py --file TS_Liver.h5ad --tissue liver
    python scale_axis_validate_tissue.py --file TS_Lung.h5ad --tissue lung
===========================================================================
"""

import argparse
import json
import os
import time
import warnings

import numpy as np
import scipy.stats as stats
import scipy.linalg as la
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch

warnings.filterwarnings("ignore")

# ===========================================================================
# 0. CLI arguments
# ===========================================================================
parser = argparse.ArgumentParser(description="Scale axis validation on a Tabula Sapiens tissue h5ad")
parser.add_argument("--file", type=str, required=True,
                     help="Path to h5ad file, e.g. TS_Liver.h5ad")
parser.add_argument("--tissue", type=str, required=True,
                     help="Short tissue label used in output filenames, e.g. liver / lung")
parser.add_argument("--n_hvg", type=int, default=2000,
                     help="Number of highly variable genes to use (default 2000)")
parser.add_argument("--max_cells", type=int, default=8000,
                     help="Subsample to at most this many cells for tractable eigendecomposition "
                          "(default 8000; set higher if you have the RAM/time budget)")
parser.add_argument("--min_cells_per_type", type=int, default=20,
                     help="Drop cell types with fewer than this many cells (default 20)")
args = parser.parse_args()

TISSUE = args.tissue
os.makedirs("real_figures", exist_ok=True)
os.makedirs("real_results", exist_ok=True)

A_PARAM, B_PARAM, T_FINAL = 1.0, 2.0, 12.0


def l7_decay(s):
    return 2.0 / (A_PARAM**2 * B_PARAM**(2 * s))


def half_life(s):
    return np.log(2) / l7_decay(s)


def apply_l7(X, eigvecs, s, T):
    decay = np.exp(-l7_decay(s) * T)
    C = X @ eigvecs
    return (C * decay[None, :]) @ eigvecs.T


# ===========================================================================
# 1. Marker gene sets by tissue
#    These are canonical, widely-cited marker genes for major cell types
#    in each tissue. Genes not present in the dataset are silently skipped
#    (checked at runtime against gene_symbol).
# ===========================================================================
TISSUE_MARKERS = {
    "liver": {
        "Hepatocyte":        ["ALB", "APOA1", "APOB", "TTR", "HNF4A"],
        "Kupffer cell":      ["CD68", "MARCO", "CD163", "VSIG4"],
        "Hepatic stellate":  ["PDGFRB", "DCN", "COL1A1", "ACTA2"],
        "Endothelial (sinusoid)": ["PECAM1", "STAB2", "LYVE1", "CLEC4G"],
        "Cholangiocyte":     ["KRT19", "KRT7", "EPCAM", "SOX9"],
        "T cell":            ["CD3D", "CD3E", "CD2", "TRAC"],
        "NK cell":           ["NKG7", "GNLY", "KLRD1"],
        "B cell":            ["CD79A", "MS4A1", "CD19"],
    },
    "lung": {
        "AT2 (alveolar type 2)": ["SFTPC", "SFTPB", "SFTPA1", "NAPSA"],
        "AT1 (alveolar type 1)": ["AGER", "PDPN", "CAV1"],
        "Ciliated cell":     ["FOXJ1", "TPPP3", "PIFO"],
        "Club/secretory":    ["SCGB1A1", "SCGB3A1"],
        "Endothelial":       ["PECAM1", "VWF", "CLDN5"],
        "Fibroblast":        ["COL1A1", "COL1A2", "PDGFRA"],
        "Macrophage":        ["MARCO", "MRC1", "CD68"],
        "T cell":            ["CD3D", "CD3E", "TRAC"],
        "NK cell":           ["NKG7", "GNLY", "KLRD1"],
        "B cell":            ["CD79A", "MS4A1", "CD19"],
    },
}

if TISSUE not in TISSUE_MARKERS:
    raise ValueError(f"No marker set defined for tissue '{TISSUE}'. "
                      f"Available: {list(TISSUE_MARKERS.keys())}. "
                      f"Add a new entry to TISSUE_MARKERS to support another tissue.")

markers_by_type = TISSUE_MARKERS[TISSUE]

# ===========================================================================
# 2. Load data
# ===========================================================================
print(f"Loading {args.file} ...")
import scanpy as sc

adata = sc.read_h5ad(args.file)
print(f"  Raw shape: {adata.shape}")

# --- Field mapping for Tabula Sapiens "named" h5ad ---
if "cell_ontology_class" not in adata.obs.columns:
    raise KeyError("Expected obs['cell_ontology_class'] not found. "
                   "Is this the figshare 'named' TS h5ad, not the CZ CELLxGENE hash version? "
                   f"Available obs columns: {adata.obs.columns.tolist()}")
if "gene_symbol" not in adata.var.columns:
    raise KeyError("Expected var['gene_symbol'] not found. "
                   f"Available var columns: {adata.var.columns.tolist()}")

adata.obs["cell_type_label"] = adata.obs["cell_ontology_class"].astype(str)
gene_symbols_full = adata.var["gene_symbol"].astype(str).values

# --- Drop rare cell types (need enough cells per group for ANOVA to be meaningful) ---
vc = adata.obs["cell_type_label"].value_counts()
keep_types = vc[vc >= args.min_cells_per_type].index.tolist()
n_dropped_types = len(vc) - len(keep_types)
if n_dropped_types > 0:
    print(f"  Dropping {n_dropped_types} cell type(s) with < {args.min_cells_per_type} cells")
adata = adata[adata.obs["cell_type_label"].isin(keep_types)].copy()

# --- Subsample cells if needed (eigendecomposition of gene-gene matrix does not
#     scale with cell count directly, but downstream ANOVA / memory does) ---
if adata.n_obs > args.max_cells:
    print(f"  Subsampling {adata.n_obs} -> {args.max_cells} cells")
    sc.pp.subsample(adata, n_obs=args.max_cells, random_state=0)

# --- Highly variable gene selection (mirrors HVG preprocessing used for PBMC runs) ---
print(f"  Selecting top {args.n_hvg} highly variable genes...")
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)
sc.pp.highly_variable_genes(adata, n_top_genes=min(args.n_hvg, adata.n_vars))
adata = adata[:, adata.var["highly_variable"]].copy()

X = adata.X.toarray() if hasattr(adata.X, "toarray") else np.array(adata.X)
gene_names = adata.var["gene_symbol"].astype(str).values
celltype = np.array(adata.obs["cell_type_label"])
method = np.array(adata.obs["method"]) if "method" in adata.obs.columns else None

n_cells, n_genes = X.shape
unique_ct, ct_counts = np.unique(celltype, return_counts=True)

print(f"  Final: {n_cells} cells x {n_genes} genes")
print(f"  Cell types ({len(unique_ct)}):")
for ct, cnt in zip(unique_ct, ct_counts):
    print(f"    {ct}: {cnt} cells")
if method is not None:
    print(f"  Assay methods present: {np.unique(method).tolist()}")

# ===========================================================================
# 3. Co-expression graph -> scale axis
# ===========================================================================
print("\nBuilding co-expression graph...")
corr = np.corrcoef(X.T)
np.fill_diagonal(corr, 0.0)
W = np.maximum(corr, 0.0)
D = np.diag(W.sum(axis=1))
L = D - W

print("Spectral decomposition...")
t0 = time.time()
eigvals, eigvecs = la.eigh(L)
eigvals = np.maximum(eigvals, 0.0)
print(f"  Done in {time.time()-t0:.1f}s")

lam_max = eigvals.max()
s = np.log1p(lam_max - eigvals)

print(f"  lambda in [{eigvals.min():.3f}, {eigvals.max():.3f}]")
print(f"  s in [{s.min():.3f}, {s.max():.3f}]")

n_modes = n_genes
n3 = n_modes // 3
low_lam = slice(0, n3)
mid_lam = slice(n3, 2 * n3)
high_lam = slice(2 * n3, n_modes)

hl_small = half_life(s[high_lam]).mean()
hl_large = half_life(s[low_lam]).mean()
hl_ratio = hl_large / hl_small
print(f"  Half-life ratio (large-s / small-s): {hl_ratio:.1f}x")

# ===========================================================================
# 4. Cell-type signal vs eigenvalue index (mirrors PBMC68k Section 5.1)
# ===========================================================================
print("\nCell-type signal by spectral mode...")

cell_scores = X @ eigvecs

f_celltype = np.zeros(n_modes)
intra_var = np.zeros(n_modes)
inter_var = np.zeros(n_modes)

for k in range(n_modes):
    scores_k = cell_scores[:, k]
    groups = [scores_k[celltype == ct] for ct in unique_ct
              if (celltype == ct).sum() > 1]
    if len(groups) > 1:
        f_val, _ = stats.f_oneway(*groups)
        f_celltype[k] = f_val if np.isfinite(f_val) else 0.0

    intra_var[k] = np.mean([np.var(scores_k[celltype == ct])
                             for ct in unique_ct if (celltype == ct).sum() > 1])
    means = [scores_k[celltype == ct].mean()
             for ct in unique_ct if (celltype == ct).sum() > 1]
    inter_var[k] = np.var(means) if len(means) > 1 else 0.0

f_low = f_celltype[low_lam].mean()
f_mid = f_celltype[mid_lam].mean()
f_high = f_celltype[high_lam].mean()

intra_low = intra_var[low_lam].mean()
intra_high = intra_var[high_lam].mean()
inter_low = inter_var[low_lam].mean()
inter_high = inter_var[high_lam].mean()

ct_f_ratio = f_low / (f_high + 1e-6)
noise_ratio = intra_high / (intra_low + 1e-6)

confirmed_signal = (ct_f_ratio > 1.0 and noise_ratio > 1.0)

print(f"  F_celltype: low-lam={f_low:.2f}, mid-lam={f_mid:.2f}, high-lam={f_high:.2f}")
print(f"  Intra-var:  low-lam={intra_low:.4f}, high-lam={intra_high:.4f}")
print(f"  Cell-type F enrichment in low-lam: {ct_f_ratio:.2f}x")
print(f"  Intra-noise enrichment in high-lam: {noise_ratio:.2f}x")
print(f"  Signal structure: {'CONFIRMED' if confirmed_signal else 'NOT CONFIRMED'}")

# ===========================================================================
# 5. Marker gene scale positions
# ===========================================================================
print("\n  Marker gene positions on topology-induced scale axis:")
marker_results = {}
for ct_name, gene_list in markers_by_type.items():
    ct_markers = []
    for gname in gene_list:
        matches = np.where(gene_names == gname)[0]
        if len(matches) == 0:
            continue
        gi = matches[0]
        s_gene = s[gi]
        hl_g = half_life(s_gene)

        expr = X[:, gi]
        groups = [expr[celltype == ct] for ct in unique_ct
                  if (celltype == ct).sum() > 1]
        f_g, _ = stats.f_oneway(*groups)

        ct_markers.append({
            "gene": gname, "gene_idx": int(gi),
            "s": float(s_gene), "lambda": float(eigvals[gi]),
            "half_life": float(hl_g),
            "F_celltype": float(f_g) if np.isfinite(f_g) else 0.0,
        })
        print(f"    {gname:12s} ({ct_name:25s}): "
              f"s={s_gene:.3f}, lambda={eigvals[gi]:.3f}, "
              f"t1/2={hl_g:.2f}, F={f_g:.1f}")
    marker_results[ct_name] = ct_markers

n_markers_found = sum(len(v) for v in marker_results.values())
n_markers_total = sum(len(v) for v in markers_by_type.values())
print(f"\n  Found {n_markers_found} / {n_markers_total} marker genes in the HVG set.")

# ===========================================================================
# 6. Gradient flow: cell-type separation over time (mirrors Section 5.2)
# ===========================================================================
print("\nGradient flow -- cell-type separation over time...")


def celltype_separation(Xmat, labels):
    unique_l = np.unique(labels)
    overall_mean = Xmat.mean(axis=0)
    between, within = 0.0, 0.0
    for l in unique_l:
        mask = labels == l
        n_l = mask.sum()
        l_mean = Xmat[mask].mean(axis=0)
        between += n_l * np.sum((l_mean - overall_mean) ** 2)
        within += np.sum((Xmat[mask] - l_mean[None, :]) ** 2)
    return between / (within + 1e-12)


t_values = np.array([0, 1, 2, 3, 5, 7, 10, 12])
sep_scores = []
energy_scores = []

phi = 1.0 / (A_PARAM**2 * B_PARAM**(2 * s))
ds = (s.max() - s.min()) / (len(s) - 1)

for T in t_values:
    X_T = apply_l7(X, eigvecs, s, T)
    sep = celltype_separation(X_T, celltype)
    sep_scores.append(sep)
    r0_T = np.abs(X_T @ eigvecs).mean(axis=0)
    energy = np.sum(r0_T**2 * phi) * ds
    energy_scores.append(energy)

sep_scores = np.array(sep_scores)
energy_scores = np.array(energy_scores)

sep_retention = sep_scores[-1] / (sep_scores[0] + 1e-12)
energy_drop = 1.0 - energy_scores[-1] / (energy_scores[0] + 1e-12)
energy_mono = bool(np.all(np.diff(energy_scores) <= 1e-6 * energy_scores[0] + 1e-12))

print("\n  Marker gene F_celltype before/after gradient flow:")
X_T12 = apply_l7(X, eigvecs, s, T_FINAL)
marker_l7 = {}
for ct_name, ct_markers in marker_results.items():
    for m in ct_markers:
        gi = m["gene_idx"]
        expr_before = X[:, gi]
        expr_after = X_T12[:, gi]
        groups_b = [expr_before[celltype == ct] for ct in unique_ct
                    if (celltype == ct).sum() > 1]
        groups_a = [expr_after[celltype == ct] for ct in unique_ct
                    if (celltype == ct).sum() > 1]
        f_b, _ = stats.f_oneway(*groups_b)
        f_a, _ = stats.f_oneway(*groups_a)
        marker_l7[m["gene"]] = {
            "F_before": float(f_b) if np.isfinite(f_b) else 0.0,
            "F_after": float(f_a) if np.isfinite(f_a) else 0.0,
            "s": m["s"],
        }
        print(f"    {m['gene']:12s}: F {f_b:.1f} -> {f_a:.1f}  "
              f"({'UP' if f_a > f_b else 'DOWN'})  s={m['s']:.3f}")

n_marker_up = sum(1 for v in marker_l7.values() if v["F_after"] >= v["F_before"])
confirmed_flow = (sep_retention > 0.5 and energy_mono)

print(f"\n  Cell-type separation: {sep_scores[0]:.4f} -> {sep_scores[-1]:.4f} "
      f"(retention {sep_retention*100:.0f}%)")
print(f"  Energy drop: {energy_drop*100:.0f}%  monotone: {energy_mono}")
print(f"  Marker genes with F retained/increased: {n_marker_up}/{len(marker_l7)}")
print(f"  Flow preservation: {'CONFIRMED' if confirmed_flow else 'NOT CONFIRMED'}")

# ===========================================================================
# 7. Figures
# ===========================================================================
print("\nGenerating figures...")


def smooth(arr, w=15):
    w = min(w, len(arr) - 1) if len(arr) > 1 else 1
    w = max(w, 1)
    return np.convolve(arr, np.ones(w) / w, mode="same")


fig = plt.figure(figsize=(18, 11))
fig.suptitle(
    f"Topology-Induced Scale Axis -- {TISSUE.capitalize()} Validation (Tabula Sapiens)\n"
    f"({n_cells} cells, {n_genes} genes, {len(unique_ct)} real cell types via cell_ontology_class)",
    fontsize=13, fontweight="bold"
)
gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

ax = fig.add_subplot(gs[0, 0])
f_sm = smooth(f_celltype, w=20)
col = np.where(np.arange(n_modes) < n3, 0,
      np.where(np.arange(n_modes) < 2 * n3, 1, 2))
cmap = ["steelblue", "gray", "tomato"]
for ci, c in enumerate(cmap):
    mask = col == ci
    ax.scatter(np.where(mask)[0], f_celltype[mask], c=c, s=4, alpha=0.3)
ax.plot(range(n_modes), f_sm, "k-", linewidth=1.8, label="Smoothed")
ax.axvspan(0, n3, alpha=0.10, color="steelblue")
ax.axvspan(2 * n3, n_modes, alpha=0.10, color="tomato")
ax.set_xlabel("Eigenvector index k  (ascending lambda_k)")
ax.set_ylabel("ANOVA F-statistic\n(cell type, real labels)")
ax.set_title(f"A: Cell-type F by spectral mode\n"
             f"Low-lam(large s)={f_low:.1f}  High-lam(small s)={f_high:.1f}  "
             f"ratio={ct_f_ratio:.2f}x")

ax = fig.add_subplot(gs[0, 1])
zones = ["Low-lam\n(large s)", "Mid-lam", "High-lam\n(small s)"]
intra_z = [intra_var[low_lam].mean(), intra_var[mid_lam].mean(), intra_var[high_lam].mean()]
inter_z = [inter_var[low_lam].mean(), inter_var[mid_lam].mean(), inter_var[high_lam].mean()]
x = np.arange(3)
w = 0.35
ax.bar(x - w / 2, inter_z, w, color="steelblue", alpha=0.8, label="Inter-type variance")
ax.bar(x + w / 2, intra_z, w, color="tomato", alpha=0.8, label="Intra-type variance")
ax.set_xticks(x)
ax.set_xticklabels(zones, fontsize=9)
ax.set_ylabel("Variance")
ax.set_title(f"B: Signal decomposition by lambda zone\n"
             f"Intra-noise ratio (high/low lam): {noise_ratio:.2f}x")
ax.legend(fontsize=8)

ax = fig.add_subplot(gs[0, 2])
all_m = [(m["gene"], m) for ct, mlist in marker_results.items() for m in mlist]
if all_m:
    ct_color_map = {ct: plt.cm.tab10(i / max(len(markers_by_type), 1))
                     for i, ct in enumerate(markers_by_type)}
    for gname, m in all_m:
        for ct, mlist in marker_results.items():
            if any(mm["gene"] == gname for mm in mlist):
                c = ct_color_map.get(ct, "gray")
                break
        ax.scatter(m["s"], m["half_life"], color=c, s=80,
                   zorder=5, edgecolors="black", linewidths=0.4)
        ax.annotate(gname, (m["s"], m["half_life"]),
                    textcoords="offset points", xytext=(4, 2), fontsize=7)
ax.set_xlabel("Scale s  (large s = global co-expression = slow decay)")
ax.set_ylabel("Half-life t1/2")
ax.set_yscale("log")
ax.set_title(f"C: Known {TISSUE} marker genes\non topology-induced scale axis")
patches = [Patch(color=ct_color_map.get(ct, "gray"), label=ct[:22])
           for ct in sorted(markers_by_type)] if all_m else []
if patches:
    ax.legend(handles=patches, fontsize=6, loc="upper left", ncol=1)

ax = fig.add_subplot(gs[1, 0])
ax2 = ax.twinx()
l1, = ax.plot(t_values, sep_scores, "o-", color="steelblue", linewidth=2,
              label="Cell-type separation")
l2, = ax2.plot(t_values, energy_scores, "s--", color="tomato", linewidth=2,
               label="B-series energy")
ax.set_xlabel("Gradient flow time T")
ax.set_ylabel("Cell-type separation score", color="steelblue")
ax2.set_ylabel("B-series energy N_B[r]", color="tomato")
ax.set_title(f"D: Flow timeline\nSep retention={sep_retention*100:.0f}%  "
             f"Energy down {energy_drop*100:.0f}%  monotone={energy_mono}")
ax.legend([l1, l2], [l1.get_label(), l2.get_label()], fontsize=8, loc="center right")

ax = fig.add_subplot(gs[1, 1])
if marker_l7:
    gnames_sorted = sorted(marker_l7.keys(),
                            key=lambda g: marker_l7[g]["s"], reverse=True)[:12]
    f_bef = [marker_l7[g]["F_before"] for g in gnames_sorted]
    f_aft = [marker_l7[g]["F_after"] for g in gnames_sorted]
    y = np.arange(len(gnames_sorted))
    ax.barh(y - 0.18, f_bef, 0.35, color="steelblue", alpha=0.7, label="Before flow")
    ax.barh(y + 0.18, f_aft, 0.35, color="tomato", alpha=0.8, label="After flow")
    ax.set_yticks(y)
    ax.set_yticklabels(gnames_sorted, fontsize=8)
    ax.set_xlabel("F_celltype (ANOVA)")
    ax.set_title("E: Marker gene cell-type\nseparation before/after flow")
    ax.legend(fontsize=8)

ax = fig.add_subplot(gs[1, 2])
hl_all = half_life(s)
sc_plot = ax.scatter(eigvals, s, c=hl_all, cmap="viridis", s=6, alpha=0.5)
sm = plt.cm.ScalarMappable(cmap="viridis",
                            norm=plt.Normalize(hl_all.min(), hl_all.max()))
plt.colorbar(sm, ax=ax, label="Half-life t1/2")
ax.set_xlabel("Eigenvalue lambda_k")
ax.set_ylabel("Scale s_k")
ax.set_title("F: topology-induced scale axis structure\n"
              "lambda -> s (monotone by construction)")
ax.axvline(eigvals[n3], color="steelblue", linestyle="--", alpha=0.5, linewidth=1)
ax.axvline(eigvals[2 * n3], color="tomato", linestyle="--", alpha=0.5, linewidth=1)

fig_path_pdf = f"real_figures/fig_{TISSUE}_validation.pdf"
fig_path_png = f"real_figures/fig_{TISSUE}_validation.png"
plt.savefig(fig_path_pdf, bbox_inches="tight", dpi=150)
plt.savefig(fig_path_png, bbox_inches="tight", dpi=150)
plt.close()
print(f"  Saved: {fig_path_pdf}")

# ===========================================================================
# 8. Save results
# ===========================================================================
results = {
    "data": f"{args.file} -- tissue={TISSUE} ({n_cells} cells, {n_genes} genes, "
            f"{len(unique_ct)} cell types)",
    "scale_axis": {
        "lambda_range": [float(eigvals.min()), float(eigvals.max())],
        "s_range": [float(s.min()), float(s.max())],
        "hl_ratio": float(hl_ratio),
    },
    "signal_structure": {
        "F_celltype_low_lam": float(f_low),
        "F_celltype_high_lam": float(f_high),
        "ct_F_ratio": float(ct_f_ratio),
        "intra_noise_ratio": float(noise_ratio),
        "confirmed": bool(confirmed_signal),
    },
    "flow_preservation": {
        "sep_T0": float(sep_scores[0]),
        "sep_T12": float(sep_scores[-1]),
        "sep_retention": float(sep_retention),
        "energy_drop_pct": float(energy_drop * 100),
        "energy_monotone": energy_mono,
        "n_marker_up": int(n_marker_up),
        "n_marker_total": int(len(marker_l7)),
        "confirmed": bool(confirmed_flow),
    },
    "marker_genes_found": f"{n_markers_found}/{n_markers_total}",
    "marker_genes_flow": marker_l7,
}

results_path = f"real_results/{TISSUE}_results.json"
with open(results_path, "w") as f:
    json.dump(results, f, indent=2)

print(f"\n{'='*65}")
print(f"  {TISSUE.upper()} VALIDATION SUMMARY (Tabula Sapiens)")
print(f"{'='*65}")
print(f"\n  Scale: lambda in [{eigvals.min():.2f},{eigvals.max():.2f}]  "
      f"s in [{s.min():.2f},{s.max():.2f}]  t1/2 ratio {hl_ratio:.1f}x")
print(f"\n  Cell-type spectral structure  [{'OK' if confirmed_signal else 'FAIL'}]")
print(f"    F_celltype: low-lam={f_low:.2f}  high-lam={f_high:.2f}  "
      f"ratio={ct_f_ratio:.2f}x")
print(f"    Intra-noise ratio (high/low lam): {noise_ratio:.2f}x")
print(f"\n  Gradient flow preservation  [{'OK' if confirmed_flow else 'FAIL'}]")
print(f"    Cell-type separation: {sep_scores[0]:.4f} -> {sep_scores[-1]:.4f} "
      f"(retention {sep_retention*100:.0f}%)")
print(f"    Energy drop: {energy_drop*100:.0f}%  monotone: {energy_mono}")
print(f"    Marker genes found: {n_markers_found}/{n_markers_total}")
print(f"    Marker genes with retained/improved F: {n_marker_up}/{len(marker_l7)}")
print(f"\n  Results: {results_path}")
print(f"  Figure:  {fig_path_pdf}\n")
