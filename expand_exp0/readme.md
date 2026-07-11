# Topology-Induced Scale Axis for scRNA-seq

Code for: "A Topology-Induced Scale Axis for Multi-Scale Annealing in
scRNA-seq Co-Expression Networks"

**Author:** Wujie Gu (g@wujielanguagemodel.com)
**Preprint:** osf.io/rq5vj
**Website:** wujielanguagemodel.com
**Submitted to:** Bioinformatics (Manuscript ID BIOINF-2026-2076)

---

## Update: Cross-tissue generalization validation (added post-submission)

The original submission validated the scale axis construction on PBMC
(peripheral blood) data only. To test whether the structural predictions
generalize beyond immune cells to solid-tissue biology, two additional
non-immune tissues (Liver, Lung) were validated using two independently
processed versions of the same public dataset (Tabula Sapiens).

**Result: all four independent runs confirm both structural predictions
with zero exceptions among 52 tested marker genes.**

| Dataset | Source | Cells used | Cell types | F enrichment (low/high λ) | Markers found | Markers F↑ | Energy monotone |
|---|---|---|---|---|---|---|---|
| Liver | figshare (Tabula Sapiens) | 5,007 | 13 | 4.81× | 4/31 | 4/4 | True |
| Lung | figshare (Tabula Sapiens) | 8,000 (subsampled) | 33 | 5.93× | 10/30 | 10/10 | True |
| Liver | CZ CELLxGENE | 8,000 (subsampled) | 23 | 6.27× | 15/31 | 15/15 | True |
| Lung | CZ CELLxGENE | 8,000 (subsampled) | 30 | 13.52× | 23/30 | 23/23 | True |

Both data sources mix 10x 3' v3 and Smart-seq2 platforms, so these results
also provide implicit evidence that the effect is not platform-specific.

Full run logs and interpretation: see `cross_tissue_validation_summary.md`.

---

## Files and Figure Mapping

### Core pipeline

| File | Paper section | Figures produced |
|---|---|---|
| `scale_axis_toy_pipeline.py` | §4 Numerical Verification | Fig 1 (Baseline: 6-panel eigenvalue/scale/decay/energy/field/half-life), Fig 2 (ST1: Noise Gradient), Fig 3 (ST2: Parameter Sweep), Fig 4 (ST3: Topology Removal) |
| `pbmc_prep.py` | Preprocessing only | None (produces .npy files for pbmc_validate_3k.py) |
| `pbmc68k_validate.py` | §5.1–5.2 PBMC68k | Fig 6 (PBMC68k 6-panel: spectral F, signal decomposition, marker positions, L7 timeline, F before/after, scale axis structure) |
| `pbmc_validate_3k.py` | §5.1–5.2 PBMC3k | Fig 7 (PBMC3k 6-panel: cluster-F, variance, scale axis, L7 timeline, PCA before/after, marker positions) |

### Benchmarking

| File | Paper section | Figures produced |
|---|---|---|
| `scale_axis_overnight.py` | §4.6, §5 | Fig 5 (Block B: Parameter sweep N=50,000), Fig 8 (Block D: Bootstrap stability N=2,000), Fig 9 (Block E: Multi-study Immune_ALL validation). Also runs Block A GPU stability (0-violation result, no figure in paper) and Block C permutation test (reported in text only). |
| `scale_axis_benchmark.py` | §5.3 | Fig 10 (Graph construction robustness: 11 methods × PBMC68k, marker s position + F-retention + batch mixing + cell-type purity) |
| `scale_axis_runtime_extended.py` | §6 Computational complexity | Fig 11 (Runtime comparison: Scale Axis vs. ComBat vs. Harmony, 1k–30k cells, log-scale + relative slowdown) |

### Cross-tissue generalization (post-submission addition)

| File | Purpose | Data source |
|---|---|---|
| `scale_axis_validate_tissue.py` | Validates scale axis + gradient flow on non-immune tissues (Liver, Lung) | figshare "named" Tabula Sapiens h5ad files (`TS_Liver.h5ad`, `TS_Lung.h5ad`). Uses `obs['cell_ontology_class']`, `var['gene_symbol']`, `obs['method']`. |
| `scale_axis_validate_cellxgene.py` | Same validation, independent data source | CZ CELLxGENE hash-named h5ad files (same tissues, larger cell counts, different preprocessing pipeline). Uses `obs['cell_type']`, `var['feature_name']`, `obs['assay']`. |

Both scripts share identical statistical logic (co-expression graph
construction, spectral decomposition, F-statistic by mode, gradient flow,
marker gene tracking) and produce directly comparable output — only the
field-name mapping differs, because the two data sources use different
`obs`/`var` column naming conventions for the same underlying biology.

### Deprecated Files (not needed)

**`real_data_pipeline.py`**
Early development script. Functionality is fully superseded by
`pbmc68k_validate.py` and `pbmc_validate_3k.py`. Contains a `--synthetic`
fallback mode used during development when real data was unavailable. No
figures in the paper correspond to this script. Safe to delete.

**`scale_axis_runtime.py`**
Earlier runtime benchmark covering only 3 dataset sizes (500 / 2,000 /
5,000 cells). Superseded by `scale_axis_runtime_extended.py`, which covers
8 sizes (1k–30k cells) and produces Fig 11. Safe to delete.

---

## Requirements

```
numpy scipy scikit-learn scanpy harmonypy matplotlib
```

Optional (for `scale_axis_overnight.py` Block A GPU):
```
torch  (CUDA-enabled)
```

---

## Usage

```bash
# §4 Synthetic stress tests → Fig 1-4
python scale_axis_toy_pipeline.py

# PBMC3k preprocessing (run before pbmc_validate_3k.py)
python pbmc_prep.py

# §5 PBMC68k validation → Fig 6
python pbmc68k_validate.py

# §5 PBMC3k validation → Fig 7
python pbmc_validate_3k.py

# §4 / §5 Overnight GPU suite → Fig 5, 8, 9
python scale_axis_overnight.py --h5ad Immune_ALL_human.h5ad

# §5.3 Graph construction robustness → Fig 10
python scale_axis_benchmark.py

# §6 Runtime comparison → Fig 11
python scale_axis_runtime_extended.py

# Cross-tissue generalization (post-submission) → Liver / Lung, figshare source
python scale_axis_validate_tissue.py --file TS_Liver.h5ad --tissue liver
python scale_axis_validate_tissue.py --file TS_Lung.h5ad --tissue lung

# Cross-tissue generalization (post-submission) → Liver / Lung, CZ CELLxGENE source
python scale_axis_validate_cellxgene.py --file a4b3a49e-062b-4e3f-8915-02f40607651f.h5ad --tissue liver
python scale_axis_validate_cellxgene.py --file fddf43a3-b7c8-49e9-ac0a-5e24b390849f.h5ad --tissue lung
```

---

## Data

- **PBMC68k:** `scanpy.datasets.pbmc68k_reduced()` (auto-download)
- **PBMC3k:** `scanpy.datasets.pbmc3k()` (auto-download)
- **Immune_ALL:** https://figshare.com/articles/dataset/12420968
  (`Immune_ALL_human.h5ad`, required for Block E in `scale_axis_overnight.py`)
- **Tabula Sapiens Liver / Lung (figshare, "named" version):**
  https://tabula-sapiens-portal.ds.czbiohub.org/
- **Tabula Sapiens Liver / Lung (CZ CELLxGENE version):**
  https://cellxgene.cziscience.com/collections/e5f58829-1a66-40b5-a624-9046778e74f5

---

## License

MIT with non-commercial restriction — see `LICENSE.txt`. Academic and
research use is permitted; commercial use requires prior written permission.
Please cite the associated paper (see header of this README) if you use
this code or its outputs.
