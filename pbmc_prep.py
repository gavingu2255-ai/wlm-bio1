"""
wgs_pbmc_prep.py
把PBMC3k预处理并保存为.npy文件，供后续pipeline使用。
不依赖leidenalg。
"""

import scanpy as sc
import numpy as np
from scipy.linalg import eigh
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')

print("Loading PBMC3k...")
adata = sc.datasets.pbmc3k()

# 标准预处理
sc.pp.filter_cells(adata, min_genes=200)
sc.pp.filter_genes(adata, min_cells=3)
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)
sc.pp.highly_variable_genes(adata, n_top_genes=500)
adata_hvg = adata[:, adata.var['highly_variable']].copy()

X = adata_hvg.X.toarray() if hasattr(adata_hvg.X, 'toarray') else np.array(adata_hvg.X)
gene_names = np.array(adata_hvg.var_names)
print(f"HVG matrix: {X.shape}")

# 谱分解
print("Building co-expression graph...")
corr = np.corrcoef(X.T)
np.fill_diagonal(corr, 0.0)
W = np.maximum(corr, 0.0)
D = np.diag(W.sum(axis=1))
L = D - W

print("Spectral decomposition...")
eigvals, eigvecs = eigh(L)
eigvals = np.maximum(eigvals, 0.0)
lam_max = eigvals.max()
s = np.log1p(lam_max - eigvals)
print(f"  lambda range: [{eigvals.min():.3f}, {eigvals.max():.3f}]")
print(f"  s range:      [{s.min():.3f}, {s.max():.3f}]")

# 用PCA + KMeans代替leiden，得到cell type近似标签
print("Clustering cells (PCA + KMeans, 8 clusters)...")
sc.pp.pca(adata_hvg, n_comps=20)
pca_coords = adata_hvg.obsm['X_pca']
km = KMeans(n_clusters=8, random_state=42, n_init=10)
cluster = km.fit_predict(pca_coords).astype(str)
print(f"  Clusters: {np.unique(cluster)}")

# 模拟batch：把cluster 0,1,2,3的细胞当batch_A，其余当batch_B
# 这模拟"不同实验批次测了不同细胞子集"的情形
batch = np.array(['batch_A' if c in ['0','1','2','3'] else 'batch_B'
                  for c in cluster])
print(f"  batch_A: {(batch=='batch_A').sum()}  batch_B: {(batch=='batch_B').sum()}")

# 已知PBMC marker genes（验证§5.2用）
# 这些基因若在HVG里则有效，若不在则自动跳过
pbmc_markers = {
    'T_cell':   ['CD3D', 'CD3E', 'IL7R', 'CCR7', 'CD4'],
    'B_cell':   ['CD79A', 'MS4A1', 'CD79B', 'CD19'],
    'Monocyte': ['CD14', 'LYZ', 'CST3', 'FCGR3A', 'MS4A7'],
    'NK_cell':  ['NKG7', 'GNLY', 'KLRB1', 'GZMB'],
}

# 检查哪些marker基因实际在HVG里
print("\nMarker genes found in HVG set:")
found_markers = {}
for ct, genes in pbmc_markers.items():
    found = [g for g in genes if g in gene_names]
    found_markers[ct] = found
    print(f"  {ct}: {found} ({len(found)}/{len(genes)})")

# 保存
print("\nSaving...")
np.save('pbmc_X.npy',          X)
np.save('pbmc_eigvals.npy',     eigvals)
np.save('pbmc_eigvecs.npy',     eigvecs)
np.save('pbmc_s.npy',           s)
np.save('pbmc_batch.npy',       batch)
np.save('pbmc_cluster.npy',     cluster)
np.save('pbmc_gene_names.npy',  gene_names)

import json
with open('pbmc_markers.json', 'w') as f:
    json.dump(found_markers, f, indent=2)

print("Done. Files saved:")
print("  pbmc_X.npy, pbmc_eigvals.npy, pbmc_eigvecs.npy")
print("  pbmc_s.npy, pbmc_batch.npy, pbmc_cluster.npy")
print("  pbmc_gene_names.npy, pbmc_markers.json")
