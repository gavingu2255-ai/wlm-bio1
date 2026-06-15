"""
wgs_pbmc_validate.py
═══════════════════════════════════════════════════════════════════════
Topology-Induced Scale Axis — PBMC3k Validation

读取 wgs_pbmc_prep.py 生成的 .npy 文件，运行两个验证：

§5.1  Spectral structure of co-expression topology:
      低λ模式是否捕获全局共表达结构（cluster间差异）？
      高λ模式是否是局部扰动（cluster内噪声）？

§5.2  L7流下cell-type信号保留：
      L7流是否在减少高频成分的同时，保留低λ模式的cluster分离度？

注：PBMC3k是单批次数据，这里用cluster结构验证谱轴的结构性质，
    而非batch correction。batch correction验证需要多批次数据集
    （见论文§5.1 future work）。

Usage:
    python wgs_pbmc_validate.py
═══════════════════════════════════════════════════════════════════════
"""

import json
import os
import time
import numpy as np
import scipy.stats as stats
import scipy.linalg as la
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

os.makedirs("real_figures", exist_ok=True)
os.makedirs("real_results",  exist_ok=True)

# ─────────────────────────────────────────────────────────────────────
# L7 参数
# ─────────────────────────────────────────────────────────────────────
A_PARAM = 1.0
B_PARAM = 2.0
T_FINAL = 12.0

def l7_decay(s, a=A_PARAM, b=B_PARAM):
    return 2.0 / (a**2 * b**(2*s))

def half_life(s, a=A_PARAM, b=B_PARAM):
    return np.log(2) / l7_decay(s, a, b)


# ═════════════════════════════════════════════════════════════════════
# 加载数据
# ═════════════════════════════════════════════════════════════════════

print("Loading saved PBMC data...")
X          = np.load('pbmc_X.npy')
eigvals    = np.load('pbmc_eigvals.npy')
eigvecs    = np.load('pbmc_eigvecs.npy')
s          = np.load('pbmc_s.npy')
cluster    = np.load('pbmc_cluster.npy', allow_pickle=True)
gene_names = np.load('pbmc_gene_names.npy', allow_pickle=True)

with open('pbmc_markers.json') as f:
    markers = json.load(f)

n_cells, n_genes = X.shape
print(f"  {n_cells} cells × {n_genes} HVGs")
print(f"  λ ∈ [{eigvals.min():.3f}, {eigvals.max():.3f}]")
print(f"  s ∈ [{s.min():.3f}, {s.max():.3f}]")
print(f"  Clusters: {np.unique(cluster)}")

# ─────────────────────────────────────────────────────────────────────
# 半衰期统计
# ─────────────────────────────────────────────────────────────────────
s_median   = np.median(s)
small_s    = s < s_median   # 高λ，快速衰减
large_s    = ~small_s       # 低λ，慢速保留

hl_small   = half_life(s[small_s]).mean()
hl_large   = half_life(s[large_s]).mean()
hl_ratio   = hl_large / hl_small

print(f"\n  Half-life (small-s modes): {hl_small:.3f}")
print(f"  Half-life (large-s modes): {hl_large:.3f}")
print(f"  Ratio: {hl_ratio:.1f}×")


# ═════════════════════════════════════════════════════════════════════
# §5.1  谱结构验证
# 预测：低λ模式 → 大s → 捕获全局cluster结构（高F_cluster）
#       高λ模式 → 小s → 局部噪声（低F_cluster，高内方差）
# ═════════════════════════════════════════════════════════════════════

print("\n§5.1  Spectral Structure Validation...")

# 细胞在每个特征向量上的投影得分
# u_k是基因空间向量，cell score = X @ u_k
cell_scores = X @ eigvecs   # (n_cells, n_genes)

clusters_unique = np.unique(cluster)
n_modes = n_genes

f_cluster    = np.zeros(n_modes)
intra_var    = np.zeros(n_modes)
inter_var    = np.zeros(n_modes)

for k in range(n_modes):
    scores_k = cell_scores[:, k]
    groups   = [scores_k[cluster == c] for c in clusters_unique
                if (cluster == c).sum() > 1]
    if len(groups) > 1:
        f_val, _ = stats.f_oneway(*groups)
        f_cluster[k] = f_val if np.isfinite(f_val) else 0.0

    # 组内方差（平均）= 局部噪声指标
    intra = np.mean([np.var(scores_k[cluster == c])
                     for c in clusters_unique if (cluster == c).sum() > 1])
    intra_var[k] = intra

    # 组间方差 = 全局结构指标
    means = [scores_k[cluster == c].mean()
             for c in clusters_unique if (cluster == c).sum() > 1]
    inter_var[k] = np.var(means)

# 分三段：低λ / 中λ / 高λ
n3 = n_modes // 3
low_lam  = slice(0,    n3)
mid_lam  = slice(n3,   2*n3)
high_lam = slice(2*n3, n_modes)

f_low  = f_cluster[low_lam].mean()
f_mid  = f_cluster[mid_lam].mean()
f_high = f_cluster[high_lam].mean()

intra_low  = intra_var[low_lam].mean()
intra_high = intra_var[high_lam].mean()
inter_low  = inter_var[low_lam].mean()
inter_high = inter_var[high_lam].mean()

cluster_f_ratio   = f_low / (f_high + 1e-6)       # 预测 > 1
intra_noise_ratio = intra_high / (intra_low + 1e-6) # 预测 > 1（高λ噪声更大）

print(f"  F_cluster:  low-λ={f_low:.2f}, mid-λ={f_mid:.2f}, high-λ={f_high:.2f}")
print(f"  Intra-var:  low-λ={intra_low:.4f}, high-λ={intra_high:.4f}")
print(f"  Inter-var:  low-λ={inter_low:.4f}, high-λ={inter_high:.4f}")
print(f"  Cluster-F enrichment in low-λ: {cluster_f_ratio:.2f}×")
print(f"  Intra-var enrichment in high-λ: {intra_noise_ratio:.2f}×")

confirmed_51 = (cluster_f_ratio > 1.0 and intra_noise_ratio > 1.0)
print(f"  §5.1 prediction: {'✓ CONFIRMED' if confirmed_51 else '✗ NOT CONFIRMED'}")


# ═════════════════════════════════════════════════════════════════════
# §5.2  L7流下cluster分离度保留
# L7流对小s（高λ）快速衰减，对大s（低λ）慢速保留
# 预测：L7流后，低λ主导的cluster结构得到保留
#       cluster分离度（inter/intra ratio）在L7后不下降
# ═════════════════════════════════════════════════════════════════════

print("\n§5.2  L7 Flow — Cluster Separation Preservation...")

def cluster_separation(Xmat, cluster_labels):
    """
    Between-cluster variance / within-cluster variance ratio。
    越高 = cluster越分明 = cell-type信号越强。
    """
    unique_c = np.unique(cluster_labels)
    overall_mean = Xmat.mean(axis=0)
    # Between-cluster variance (weighted by cluster size)
    between = 0.0
    within  = 0.0
    for c in unique_c:
        mask = cluster_labels == c
        n_c  = mask.sum()
        c_mean = Xmat[mask].mean(axis=0)
        between += n_c * np.sum((c_mean - overall_mean)**2)
        within  += np.sum((Xmat[mask] - c_mean[None,:])**2)
    return between / (within + 1e-12)

def apply_l7(X, eigvecs, s, T, a=A_PARAM, b=B_PARAM):
    lam  = l7_decay(s, a, b)
    decay = np.exp(-lam * T)
    C = X @ eigvecs
    C_evolved = C * decay[None, :]
    return C_evolved @ eigvecs.T

# 测量多个时间点
t_values = np.array([0, 1, 2, 3, 5, 8, 10, 12])
sep_scores    = []
energy_scores = []

phi = 1.0 / (A_PARAM**2 * B_PARAM**(2*s))
ds  = (s.max() - s.min()) / (len(s) - 1) if len(s) > 1 else 1.0

for T in t_values:
    X_T = apply_l7(X, eigvecs, s, T)

    # Cluster separation score
    sep = cluster_separation(X_T, cluster)
    sep_scores.append(sep)

    # B-series energy (discrete approximation)
    r0_T = np.abs(X_T @ eigvecs).mean(axis=0)  # mean amplitude per mode
    energy = np.sum(r0_T**2 * phi) * ds
    energy_scores.append(energy)

sep_scores    = np.array(sep_scores)
energy_scores = np.array(energy_scores)

# Retention: separation at T=12 vs T=0
sep_retention = sep_scores[-1] / (sep_scores[0] + 1e-12)
energy_drop   = 1.0 - energy_scores[-1] / (energy_scores[0] + 1e-12)
energy_mono   = bool(np.all(np.diff(energy_scores) <= 1e-6 * energy_scores[0] + 1e-12))

print(f"  Cluster separation at T=0:  {sep_scores[0]:.4f}")
print(f"  Cluster separation at T=12: {sep_scores[-1]:.4f}")
print(f"  Retention: {sep_retention*100:.1f}%")
print(f"  B-series energy drop: {energy_drop*100:.1f}%")
print(f"  Energy monotone: {energy_mono}")

confirmed_52 = (sep_retention > 0.5)
print(f"  §5.2 prediction: {'✓ CONFIRMED' if confirmed_52 else '✗ NOT CONFIRMED'}")


# ═════════════════════════════════════════════════════════════════════
# LYZ / GNLY / GZMB: 验证可用的marker基因
# ═════════════════════════════════════════════════════════════════════

available_markers = {'LYZ': 'Monocyte', 'GNLY': 'NK_cell', 'GZMB': 'NK_cell'}
marker_stats = {}

print("\n  Available marker genes in HVG:")
for gname, ctype in available_markers.items():
    matches = np.where(gene_names == gname)[0]
    if len(matches) == 0:
        continue
    gi   = matches[0]
    expr = X[:, gi]
    expr_T12 = apply_l7(X, eigvecs, s, 12.0)[:, gi]

    # variance explained by cluster before/after
    groups_before = [expr[cluster == c]   for c in clusters_unique if (cluster==c).sum()>1]
    groups_after  = [expr_T12[cluster==c] for c in clusters_unique if (cluster==c).sum()>1]
    f_before, _ = stats.f_oneway(*groups_before)
    f_after,  _ = stats.f_oneway(*groups_after)

    # scale position of this gene
    gene_idx = gi  # gene index = eigenvector index (gene space)
    s_gene   = s[gi]
    hl_gene  = half_life(s_gene)

    marker_stats[gname] = {
        'cell_type':    ctype,
        'scale_s':      float(s_gene),
        'half_life':    float(hl_gene),
        'F_cluster_before': float(f_before),
        'F_cluster_after':  float(f_after),
    }
    print(f"  {gname} ({ctype}): s={s_gene:.3f}, t½={hl_gene:.2f}, "
          f"F_cluster: {f_before:.1f} → {f_after:.1f}")


# ═════════════════════════════════════════════════════════════════════
# FIGURES
# ═════════════════════════════════════════════════════════════════════

print("\nGenerating figures...")

# ── Smooth helper ──────────────────────────────────────────────────
def smooth(arr, w=15):
    return np.convolve(arr, np.ones(w)/w, mode='same')

fig = plt.figure(figsize=(18, 11))
fig.suptitle(
    "Topology-Induced Scale Axis — PBMC 3k Validation\n"
    "(2700 cells, 500 HVGs, KMeans 8 clusters)",
    fontsize=13, fontweight="bold"
)
gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.42, wspace=0.35)


# Panel A: F_cluster vs eigenvalue index
ax = fig.add_subplot(gs[0, 0])
f_sm = smooth(f_cluster, w=20)
colors = np.where(np.arange(n_modes) < n3, 'steelblue',
         np.where(np.arange(n_modes) < 2*n3, 'gray', 'tomato'))
ax.scatter(range(n_modes), f_cluster, c=colors, s=4, alpha=0.4)
ax.plot(range(n_modes), f_sm, 'k-', linewidth=1.5, label='Smoothed')
ax.axvspan(0,    n3,          alpha=0.10, color='steelblue')
ax.axvspan(2*n3, n_modes,     alpha=0.10, color='tomato')
ax.set_xlabel("Eigenvector index k  (ascending λ)")
ax.set_ylabel("ANOVA F-statistic (cluster)")
ax.set_title(f"A: Cluster-F by spectral mode\n"
             f"Low-λ={f_low:.1f}, High-λ={f_high:.1f} "
             f"(ratio {cluster_f_ratio:.1f}×)")
ax.text(n3*0.3, ax.get_ylim()[1]*0.85, "large s\n(cell-type)", color='steelblue',
        fontsize=8, ha='center')
ax.text(n_modes*0.9, ax.get_ylim()[1]*0.85, "small s\n(noise)", color='tomato',
        fontsize=8, ha='center')


# Panel B: intra-var vs inter-var by zone
ax = fig.add_subplot(gs[0, 1])
zones = ['Low-λ\n(large s)', 'Mid-λ', 'High-λ\n(small s)']
intra_z = [intra_var[low_lam].mean(), intra_var[mid_lam].mean(), intra_var[high_lam].mean()]
inter_z = [inter_var[low_lam].mean(), inter_var[mid_lam].mean(), inter_var[high_lam].mean()]
x = np.arange(3)
w = 0.35
b1 = ax.bar(x - w/2, inter_z, w, color='steelblue', alpha=0.8, label='Inter-cluster var')
b2 = ax.bar(x + w/2, intra_z, w, color='tomato',    alpha=0.8, label='Intra-cluster var')
ax.set_xticks(x)
ax.set_xticklabels(zones, fontsize=9)
ax.set_ylabel("Variance")
ax.set_title(f"B: Inter vs. Intra variance by zone\n"
             f"Intra-noise ratio (high/low λ): {intra_noise_ratio:.2f}×")
ax.legend(fontsize=8)


# Panel C: scale axis (eigenvalue → s → half-life)
ax = fig.add_subplot(gs[0, 2])
hl = half_life(s)
sc1 = ax.scatter(eigvals, hl, c=s, cmap='viridis', s=8, alpha=0.6)
plt.colorbar(sc1, ax=ax, label='Scale s')
ax.set_xlabel("Eigenvalue λ")
ax.set_ylabel("Half-life t½  (log scale)")
ax.set_yscale('log')
ax.set_title(f"C: topology-induced scale axis\nλ→s→t½  (ratio {hl_ratio:.0f}×)")
ax.axvline(eigvals[n3],   color='steelblue', linestyle='--', alpha=0.6, linewidth=1)
ax.axvline(eigvals[2*n3], color='tomato',    linestyle='--', alpha=0.6, linewidth=1)


# Panel D: L7 timeline — cluster separation
ax = fig.add_subplot(gs[1, 0])
ax2 = ax.twinx()
l1, = ax.plot(t_values, sep_scores,    'o-', color='steelblue', linewidth=2,
              label='Cluster separation')
l2, = ax2.plot(t_values, energy_scores, 's--', color='tomato', linewidth=2,
               label='B-series energy')
ax.set_xlabel("L7 flow time T")
ax.set_ylabel("Cluster separation score", color='steelblue')
ax2.set_ylabel("B-series energy N_B[r]", color='tomato')
ax.set_title(f"D: L7 timeline\nSep retention={sep_retention*100:.0f}%  "
             f"Energy drop={energy_drop*100:.0f}%")
lines = [l1, l2]
ax.legend(lines, [l.get_label() for l in lines], fontsize=8, loc='center right')


# Panel E: before/after L7 — per-cluster spread
ax = fig.add_subplot(gs[1, 1])
X_T12 = apply_l7(X, eigvecs, s, T_FINAL)
# Plot first two PCA dimensions for visual
from numpy.linalg import svd as npsvd
_, _, Vt = npsvd(X - X.mean(0), full_matrices=False)
pc1 = (X - X.mean(0)) @ Vt[0]
pc2 = (X - X.mean(0)) @ Vt[1]
pc1_T = (X_T12 - X_T12.mean(0)) @ Vt[0]
pc2_T = (X_T12 - X_T12.mean(0)) @ Vt[1]

cluster_colors = plt.cm.tab10(np.array(cluster.astype(int)) / 8)
ax.scatter(pc1,   pc2,   c=cluster_colors, s=4, alpha=0.4, label='Before L7')
ax.scatter(pc1_T, pc2_T, c=cluster_colors, s=4, alpha=0.8, marker='^',
           label=f'After L7 (T={T_FINAL})')
ax.set_xlabel("PC1")
ax.set_ylabel("PC2")
ax.set_title("E: PCA projection\nbefore (circles) vs after (triangles) L7")
ax.legend(fontsize=7, markerscale=2)


# Panel F: marker gene scale positions
ax = fig.add_subplot(gs[1, 2])
if marker_stats:
    gnames  = list(marker_stats.keys())
    s_vals  = [marker_stats[g]['scale_s']  for g in gnames]
    hl_vals = [marker_stats[g]['half_life'] for g in gnames]
    f_vals  = [marker_stats[g]['F_cluster_before'] for g in gnames]

    sc2 = ax.scatter(s_vals, hl_vals, c=f_vals, cmap='YlOrRd', s=120,
                     zorder=5, edgecolors='black', linewidths=0.5)
    plt.colorbar(sc2, ax=ax, label='F_cluster (before L7)')
    for i, g in enumerate(gnames):
        ax.annotate(g, (s_vals[i], hl_vals[i]),
                    textcoords="offset points", xytext=(5, 3), fontsize=9)
    ax.set_xlabel("Scale s")
    ax.set_ylabel("Half-life t½")
    ax.set_yscale('log')
    ax.set_title("F: Marker gene positions\non topology-induced scale axis")
else:
    ax.text(0.5, 0.5, "No marker genes\nin HVG set",
            ha='center', va='center', transform=ax.transAxes, fontsize=11)
    ax.set_title("F: Marker gene positions")

plt.savefig("real_figures/fig_pbmc_validation.pdf", bbox_inches="tight", dpi=150)
plt.savefig("real_figures/fig_pbmc_validation.png", bbox_inches="tight", dpi=150)
plt.close()
print("  Saved: real_figures/fig_pbmc_validation.pdf")
print("  Saved: real_figures/fig_pbmc_validation.png")


# ═════════════════════════════════════════════════════════════════════
# 保存结果
# ═════════════════════════════════════════════════════════════════════

results = {
    "data": "PBMC 3k (2700 cells, 500 HVGs, KMeans 8 clusters)",
    "scale_axis": {
        "lambda_range": [float(eigvals.min()), float(eigvals.max())],
        "s_range":      [float(s.min()),       float(s.max())],
        "hl_small_s":   float(hl_small),
        "hl_large_s":   float(hl_large),
        "hl_ratio":     float(hl_ratio),
    },
    "section_5_1": {
        "F_cluster_low_lam":    float(f_low),
        "F_cluster_high_lam":   float(f_high),
        "cluster_F_ratio":      float(cluster_f_ratio),
        "intra_noise_ratio":    float(intra_noise_ratio),
        "confirmed":            bool(confirmed_51),
    },
    "section_5_2": {
        "sep_T0":       float(sep_scores[0]),
        "sep_T12":      float(sep_scores[-1]),
        "sep_retention": float(sep_retention),
        "energy_drop_pct": float(energy_drop * 100),
        "energy_monotone": energy_mono,
        "confirmed":    bool(confirmed_52),
    },
    "marker_genes": marker_stats,
}

with open("real_results/pbmc_results.json", "w") as f:
    json.dump(results, f, indent=2)

print("\n" + "═"*65)
print("  PBMC VALIDATION SUMMARY")
print("═"*65)
print(f"\n  Scale axis: λ∈[{eigvals.min():.2f},{eigvals.max():.2f}]  "
      f"s∈[{s.min():.2f},{s.max():.2f}]  half-life ratio {hl_ratio:.0f}×")
print(f"\n  §5.1  Spectral structure  [{'✓' if confirmed_51 else '✗'}]")
print(f"    Cluster-F: low-λ={f_low:.2f}, high-λ={f_high:.2f}  "
      f"(enrichment {cluster_f_ratio:.2f}×)")
print(f"    Intra-noise: high-λ/low-λ = {intra_noise_ratio:.2f}×")
print(f"\n  §5.2  L7 flow preservation  [{'✓' if confirmed_52 else '✗'}]")
print(f"    Cluster separation: {sep_scores[0]:.4f} → {sep_scores[-1]:.4f}  "
      f"(retention {sep_retention*100:.0f}%)")
print(f"    B-series energy drop: {energy_drop*100:.0f}%  monotone: {energy_mono}")
print(f"\n  Saved: real_results/pbmc_results.json")
print(f"  Figure: real_figures/fig_pbmc_validation.pdf")
print()
