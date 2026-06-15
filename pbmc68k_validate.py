"""
wgs_pbmc68k_validate.py
═══════════════════════════════════════════════════════════════════════
Topology-Induced Scale Axis — PBMC68k Validation (真实cell type标签)

数据：pbmc68k_reduced (700 cells × 765 genes, bulk_labels = 真实细胞类型)

验证两件事：
  §5.1  低λ模式是否捕获cell-type信号（F_celltype在低λ区更高）？
        高λ模式是否是局部噪声（intra-type variance在高λ区更高）？

  §5.2  L7流后，已知marker基因的cell-type分离度是否保留或增强？
        marker基因是否落在大s区（谱深度对应）？

Usage:
    python wgs_pbmc68k_validate.py
═══════════════════════════════════════════════════════════════════════
"""

import json, os, time, warnings
import numpy as np
import scipy.stats as stats
import scipy.linalg as la
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

warnings.filterwarnings("ignore")
os.makedirs("real_figures", exist_ok=True)
os.makedirs("real_results",  exist_ok=True)

A_PARAM, B_PARAM, T_FINAL = 1.0, 2.0, 12.0

def l7_decay(s):
    return 2.0 / (A_PARAM**2 * B_PARAM**(2*s))

def half_life(s):
    return np.log(2) / l7_decay(s)

def apply_l7(X, eigvecs, s, T):
    decay = np.exp(-l7_decay(s) * T)
    C = X @ eigvecs
    return (C * decay[None, :]) @ eigvecs.T

# ═════════════════════════════════════════════════════════════════════
# 1. 加载数据
# ═════════════════════════════════════════════════════════════════════
print("Loading pbmc68k_reduced...")
import scanpy as sc
adata = sc.datasets.pbmc68k_reduced()

# 表达矩阵
X = adata.X.toarray() if hasattr(adata.X, 'toarray') else np.array(adata.X)
gene_names  = np.array(adata.var_names)
celltype    = np.array(adata.obs['bulk_labels'])

n_cells, n_genes = X.shape
unique_ct, ct_counts = np.unique(celltype, return_counts=True)

print(f"  {n_cells} cells × {n_genes} genes")
print(f"  Cell types ({len(unique_ct)}):")
for ct, cnt in zip(unique_ct, ct_counts):
    print(f"    {ct}: {cnt} cells")

# ═════════════════════════════════════════════════════════════════════
# 2. Co-expression graph → scale axis
# ═════════════════════════════════════════════════════════════════════
print("\nBuilding co-expression graph...")
corr = np.corrcoef(X.T)                  # (n_genes, n_genes)
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
s = np.log1p(lam_max - eigvals)          # topology-induced scale axis

print(f"  λ ∈ [{eigvals.min():.3f}, {eigvals.max():.3f}]")
print(f"  s ∈ [{s.min():.3f}, {s.max():.3f}]")

n_modes = n_genes
n3 = n_modes // 3
low_lam  = slice(0,    n3)
mid_lam  = slice(n3,   2*n3)
high_lam = slice(2*n3, n_modes)

hl_small = half_life(s[high_lam]).mean()   # 高λ → 小s → 短半衰期
hl_large = half_life(s[low_lam]).mean()    # 低λ → 大s → 长半衰期
hl_ratio = hl_large / hl_small
print(f"  Half-life ratio (large-s / small-s): {hl_ratio:.1f}×")

# ═════════════════════════════════════════════════════════════════════
# 3. §5.1  Cell-type signal vs eigenvalue index
#    用真实bulk_labels做ANOVA
# ═════════════════════════════════════════════════════════════════════
print("\n§5.1  Cell-type signal by spectral mode...")

cell_scores = X @ eigvecs    # (n_cells, n_modes)

f_celltype = np.zeros(n_modes)
intra_var  = np.zeros(n_modes)
inter_var  = np.zeros(n_modes)

for k in range(n_modes):
    scores_k = cell_scores[:, k]
    groups   = [scores_k[celltype == ct] for ct in unique_ct
                if (celltype == ct).sum() > 1]
    if len(groups) > 1:
        f_val, _ = stats.f_oneway(*groups)
        f_celltype[k] = f_val if np.isfinite(f_val) else 0.0

    intra_var[k] = np.mean([np.var(scores_k[celltype == ct])
                             for ct in unique_ct if (celltype == ct).sum() > 1])
    means = [scores_k[celltype == ct].mean()
             for ct in unique_ct if (celltype == ct).sum() > 1]
    inter_var[k] = np.var(means) if len(means) > 1 else 0.0

f_low  = f_celltype[low_lam].mean()
f_mid  = f_celltype[mid_lam].mean()
f_high = f_celltype[high_lam].mean()

intra_low  = intra_var[low_lam].mean()
intra_high = intra_var[high_lam].mean()
inter_low  = inter_var[low_lam].mean()
inter_high = inter_var[high_lam].mean()

# 核心预测：低λ（大s）→ 高F_celltype；高λ（小s）→ 高intra-var
ct_f_ratio    = f_low   / (f_high   + 1e-6)   # 预测 > 1
noise_ratio   = intra_high / (intra_low + 1e-6) # 预测 > 1

confirmed_51 = (ct_f_ratio > 1.0 and noise_ratio > 1.0)

print(f"  F_celltype: low-λ={f_low:.2f}, mid-λ={f_mid:.2f}, high-λ={f_high:.2f}")
print(f"  Intra-var:  low-λ={intra_low:.4f}, high-λ={intra_high:.4f}")
print(f"  Cell-type F enrichment in low-λ: {ct_f_ratio:.2f}×")
print(f"  Intra-noise enrichment in high-λ: {noise_ratio:.2f}×")
print(f"  §5.1: {'✓ CONFIRMED' if confirmed_51 else '✗ NOT CONFIRMED'}")

# ═════════════════════════════════════════════════════════════════════
# 4. Marker gene scale positions
#    已知PBMC marker基因 → 检查它们的s值
# ═════════════════════════════════════════════════════════════════════
# 这些marker基因对应pbmc68k_reduced里的细胞类型
pbmc_markers = {
    'CD14+ Monocyte':  ['LYZ', 'CD14', 'CST3', 'FCGR3A'],
    'CD56+ NK':        ['NKG7', 'GNLY', 'GZMB', 'KLRB1'],
    'CD19+ B':         ['CD79A', 'MS4A1', 'CD19'],
    'CD4+/CD25 T Reg': ['IL2RA', 'FOXP3', 'CTLA4'],
    'CD4+ T Helper2':  ['IL7R', 'CCR7', 'CD4'],
    'CD8+ Cytotoxic T':['CD8A', 'CD8B', 'GZMK'],
    'Dendritic':       ['FCER1A', 'HLA-DQA1', 'CLEC4C'],
}

print("\n  Marker gene positions on topology-induced scale axis:")
marker_results = {}
for ct_name, gene_list in pbmc_markers.items():
    ct_markers = []
    for gname in gene_list:
        matches = np.where(gene_names == gname)[0]
        if len(matches) == 0:
            continue
        gi     = matches[0]
        s_gene = s[gi]
        hl_g   = half_life(s_gene)

        # F_celltype for this gene directly
        expr = X[:, gi]
        groups = [expr[celltype == ct] for ct in unique_ct
                  if (celltype == ct).sum() > 1]
        f_g, _ = stats.f_oneway(*groups)

        ct_markers.append({
            'gene': gname, 'gene_idx': int(gi),
            's': float(s_gene), 'lambda': float(eigvals[gi]),
            'half_life': float(hl_g),
            'F_celltype': float(f_g) if np.isfinite(f_g) else 0.0,
        })
        print(f"    {gname:12s} ({ct_name:20s}): "
              f"s={s_gene:.3f}, λ={eigvals[gi]:.3f}, "
              f"t½={hl_g:.2f}, F={f_g:.1f}")
    marker_results[ct_name] = ct_markers

# ═════════════════════════════════════════════════════════════════════
# 5. §5.2  L7流：cell-type分离度随时间演化
# ═════════════════════════════════════════════════════════════════════
print("\n§5.2  L7 flow — cell-type separation over time...")

def celltype_separation(Xmat, labels):
    """Between-type / within-type variance ratio."""
    unique_l = np.unique(labels)
    overall_mean = Xmat.mean(axis=0)
    between, within = 0.0, 0.0
    for l in unique_l:
        mask = labels == l
        n_l  = mask.sum()
        l_mean = Xmat[mask].mean(axis=0)
        between += n_l * np.sum((l_mean - overall_mean)**2)
        within  += np.sum((Xmat[mask] - l_mean[None,:])**2)
    return between / (within + 1e-12)

t_values = np.array([0, 1, 2, 3, 5, 7, 10, 12])
sep_scores    = []
energy_scores = []

phi = 1.0 / (A_PARAM**2 * B_PARAM**(2*s))
ds  = (s.max() - s.min()) / (len(s) - 1)

for T in t_values:
    X_T  = apply_l7(X, eigvecs, s, T)
    sep  = celltype_separation(X_T, celltype)
    sep_scores.append(sep)
    r0_T   = np.abs(X_T @ eigvecs).mean(axis=0)
    energy = np.sum(r0_T**2 * phi) * ds
    energy_scores.append(energy)

sep_scores    = np.array(sep_scores)
energy_scores = np.array(energy_scores)

sep_retention = sep_scores[-1] / (sep_scores[0] + 1e-12)
energy_drop   = 1.0 - energy_scores[-1] / (energy_scores[0] + 1e-12)
energy_mono   = bool(np.all(np.diff(energy_scores) <= 1e-6 * energy_scores[0] + 1e-12))

# L7流前后marker基因的F_celltype变化
print("\n  Marker gene F_celltype before/after L7:")
X_T12 = apply_l7(X, eigvecs, s, T_FINAL)
marker_l7 = {}
for ct_name, ct_markers in marker_results.items():
    for m in ct_markers:
        gi   = m['gene_idx']
        expr_before = X[:, gi]
        expr_after  = X_T12[:, gi]
        groups_b = [expr_before[celltype == ct] for ct in unique_ct
                    if (celltype == ct).sum() > 1]
        groups_a = [expr_after[celltype == ct]  for ct in unique_ct
                    if (celltype == ct).sum() > 1]
        f_b, _ = stats.f_oneway(*groups_b)
        f_a, _ = stats.f_oneway(*groups_a)
        marker_l7[m['gene']] = {
            'F_before': float(f_b) if np.isfinite(f_b) else 0.0,
            'F_after':  float(f_a) if np.isfinite(f_a) else 0.0,
            's': m['s'],
        }
        print(f"    {m['gene']:12s}: F {f_b:.1f} → {f_a:.1f}  "
              f"({'↑' if f_a > f_b else '↓'})  s={m['s']:.3f}")

confirmed_52 = (sep_retention > 0.5 and energy_mono)
print(f"\n  Cell-type separation: {sep_scores[0]:.4f} → {sep_scores[-1]:.4f} "
      f"(retention {sep_retention*100:.0f}%)")
print(f"  B-series energy drop: {energy_drop*100:.0f}%  monotone: {energy_mono}")
print(f"  §5.2: {'✓ CONFIRMED' if confirmed_52 else '✗ NOT CONFIRMED'}")

# ═════════════════════════════════════════════════════════════════════
# 6. Figures
# ═════════════════════════════════════════════════════════════════════
print("\nGenerating figures...")

def smooth(arr, w=15):
    return np.convolve(arr, np.ones(w)/w, mode='same')

fig = plt.figure(figsize=(18, 11))
fig.suptitle(
    "Topology-Induced Scale Axis — PBMC68k Validation\n"
    f"(700 cells, {n_genes} genes, {len(unique_ct)} real cell types via bulk_labels)",
    fontsize=13, fontweight="bold"
)
gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

# Panel A: F_celltype vs eigenvalue index
ax = fig.add_subplot(gs[0, 0])
f_sm = smooth(f_celltype, w=20)
col = np.where(np.arange(n_modes) < n3, 0,
      np.where(np.arange(n_modes) < 2*n3, 1, 2))
cmap = ['steelblue', 'gray', 'tomato']
for ci, c in enumerate(cmap):
    mask = col == ci
    ax.scatter(np.where(mask)[0], f_celltype[mask], c=c, s=4, alpha=0.3)
ax.plot(range(n_modes), f_sm, 'k-', linewidth=1.8, label='Smoothed')
ax.axvspan(0,    n3,          alpha=0.10, color='steelblue')
ax.axvspan(2*n3, n_modes,     alpha=0.10, color='tomato')
ax.set_xlabel("Eigenvector index k  (ascending λ_k)")
ax.set_ylabel("ANOVA F-statistic\n(cell type, real labels)")
ax.set_title(f"A: Cell-type F by spectral mode\n"
             f"Low-λ(large s)={f_low:.1f}  High-λ(small s)={f_high:.1f}  "
             f"ratio={ct_f_ratio:.2f}×")
ax.text(n3*0.3, max(f_celltype)*0.8, "large s\n(global co-expression (large s))", color='steelblue',
        fontsize=8, ha='center')
ax.text(n_modes*0.88, max(f_celltype)*0.8, "small s\n(shallow)", color='tomato',
        fontsize=8, ha='center')

# Panel B: intra/inter variance by zone
ax = fig.add_subplot(gs[0, 1])
zones  = ['Low-λ\n(large s)', 'Mid-λ', 'High-λ\n(small s)']
intra_z = [intra_var[low_lam].mean(), intra_var[mid_lam].mean(), intra_var[high_lam].mean()]
inter_z = [inter_var[low_lam].mean(), inter_var[mid_lam].mean(), inter_var[high_lam].mean()]
x = np.arange(3); w = 0.35
ax.bar(x-w/2, inter_z, w, color='steelblue', alpha=0.8, label='Inter-type variance')
ax.bar(x+w/2, intra_z, w, color='tomato',    alpha=0.8, label='Intra-type variance')
ax.set_xticks(x); ax.set_xticklabels(zones, fontsize=9)
ax.set_ylabel("Variance")
ax.set_title(f"B: Signal decomposition by λ zone\n"
             f"Intra-noise ratio (high/low λ): {noise_ratio:.2f}×")
ax.legend(fontsize=8)

# Panel C: marker genes on scale axis
ax = fig.add_subplot(gs[0, 2])
all_m = [(m["gene"], m) for ct, mlist in marker_results.items()
         for m in mlist]
if all_m:
    ct_color_map = {ct: plt.cm.tab10(i/len(unique_ct))
                    for i, ct in enumerate(unique_ct)}
    for gname, m in all_m:
        # find ct
        for ct, mlist in marker_results.items():
            if any(mm['gene'] == gname for mm in mlist):
                c = ct_color_map.get(ct, 'gray')
                break
        ax.scatter(m['s'], m['half_life'], color=c, s=80,
                   zorder=5, edgecolors='black', linewidths=0.4)
        ax.annotate(gname, (m['s'], m['half_life']),
                    textcoords="offset points", xytext=(4, 2), fontsize=7)
ax.set_xlabel("Scale s  (large s = global co-expression (large s) = slow decay)")
ax.set_ylabel("Half-life t½")
ax.set_yscale('log')
ax.set_title("C: Known marker genes\non topology-induced scale axis")
# legend
from matplotlib.patches import Patch
patches = [Patch(color=ct_color_map.get(ct,'gray'), label=ct[:18])
           for ct in sorted(ct_color_map)]
ax.legend(handles=patches, fontsize=6, loc='upper left', ncol=1)

# Panel D: L7 timeline
ax = fig.add_subplot(gs[1, 0])
ax2 = ax.twinx()
l1, = ax.plot(t_values, sep_scores,    'o-', color='steelblue', linewidth=2,
              label='Cell-type separation')
l2, = ax2.plot(t_values, energy_scores, 's--', color='tomato',  linewidth=2,
               label='B-series energy')
ax.set_xlabel("L7 flow time T")
ax.set_ylabel("Cell-type separation score", color='steelblue')
ax2.set_ylabel("B-series energy N_B[r]", color='tomato')
ax.set_title(f"D: L7 timeline\nSep retention={sep_retention*100:.0f}%  "
             f"Energy↓{energy_drop*100:.0f}%  monotone={energy_mono}")
ax.legend([l1,l2], [l1.get_label(), l2.get_label()], fontsize=8, loc='center right')

# Panel E: F_celltype before/after L7 for marker genes
ax = fig.add_subplot(gs[1, 1])
if marker_l7:
    gnames_sorted = sorted(marker_l7.keys(),
                           key=lambda g: marker_l7[g]['s'], reverse=True)[:12]
    f_bef = [marker_l7[g]['F_before'] for g in gnames_sorted]
    f_aft = [marker_l7[g]['F_after']  for g in gnames_sorted]
    y = np.arange(len(gnames_sorted))
    ax.barh(y-0.18, f_bef, 0.35, color='steelblue', alpha=0.7, label='Before L7')
    ax.barh(y+0.18, f_aft, 0.35, color='tomato',    alpha=0.8, label='After L7')
    ax.set_yticks(y); ax.set_yticklabels(gnames_sorted, fontsize=8)
    ax.set_xlabel("F_celltype (ANOVA)")
    ax.set_title("E: Marker gene cell-type\nseparation before/after L7")
    ax.legend(fontsize=8)

# Panel F: scale axis structure overview
ax = fig.add_subplot(gs[1, 2])
hl_all = half_life(s)
ax.scatter(eigvals, s, c=hl_all, cmap='viridis', s=6, alpha=0.5)
sm = plt.cm.ScalarMappable(cmap='viridis',
     norm=plt.Normalize(hl_all.min(), hl_all.max()))
plt.colorbar(sm, ax=ax, label='Half-life t½')
ax.set_xlabel("Eigenvalue λ_k")
ax.set_ylabel("Scale s_k")
ax.set_title(f"F: topology-induced scale axis structure\n"
             f"λ→s (monotone by construction)")
ax.axvline(eigvals[n3],   color='steelblue', linestyle='--', alpha=0.5, linewidth=1)
ax.axvline(eigvals[2*n3], color='tomato',    linestyle='--', alpha=0.5, linewidth=1)

plt.savefig("real_figures/fig_pbmc68k_validation.pdf", bbox_inches="tight", dpi=150)
plt.savefig("real_figures/fig_pbmc68k_validation.png", bbox_inches="tight", dpi=150)
plt.close()
print("  Saved: real_figures/fig_pbmc68k_validation.pdf")

# ═════════════════════════════════════════════════════════════════════
# 7. 保存结果
# ═════════════════════════════════════════════════════════════════════
results = {
    "data": f"pbmc68k_reduced ({n_cells} cells, {n_genes} genes, "
            f"{len(unique_ct)} cell types)",
    "scale_axis": {
        "lambda_range": [float(eigvals.min()), float(eigvals.max())],
        "s_range":      [float(s.min()), float(s.max())],
        "hl_ratio":     float(hl_ratio),
    },
    "section_5_1": {
        "F_celltype_low_lam":  float(f_low),
        "F_celltype_high_lam": float(f_high),
        "ct_F_ratio":          float(ct_f_ratio),
        "intra_noise_ratio":   float(noise_ratio),
        "confirmed":           bool(confirmed_51),
    },
    "section_5_2": {
        "sep_T0":          float(sep_scores[0]),
        "sep_T12":         float(sep_scores[-1]),
        "sep_retention":   float(sep_retention),
        "energy_drop_pct": float(energy_drop * 100),
        "energy_monotone": energy_mono,
        "confirmed":       bool(confirmed_52),
    },
    "marker_genes_l7": marker_l7,
}

with open("real_results/pbmc68k_results.json", "w") as f:
    json.dump(results, f, indent=2)

print(f"\n{'═'*65}")
print("  PBMC68k VALIDATION SUMMARY")
print(f"{'═'*65}")
print(f"\n  Scale: λ∈[{eigvals.min():.2f},{eigvals.max():.2f}]  "
      f"s∈[{s.min():.2f},{s.max():.2f}]  t½ ratio {hl_ratio:.1f}×")
print(f"\n  §5.1  Cell-type spectral structure  [{'✓' if confirmed_51 else '✗'}]")
print(f"    F_celltype: low-λ={f_low:.2f}  high-λ={f_high:.2f}  "
      f"ratio={ct_f_ratio:.2f}×")
print(f"    Intra-noise ratio (high/low λ): {noise_ratio:.2f}×")
print(f"\n  §5.2  L7 flow preservation  [{'✓' if confirmed_52 else '✗'}]")
print(f"    Cell-type separation: {sep_scores[0]:.4f} → {sep_scores[-1]:.4f} "
      f"(retention {sep_retention*100:.0f}%)")
print(f"    Energy drop: {energy_drop*100:.0f}%  monotone: {energy_mono}")
print(f"\n  Results: real_results/pbmc68k_results.json")
print(f"  Figure:  real_figures/fig_pbmc68k_validation.pdf\n")
