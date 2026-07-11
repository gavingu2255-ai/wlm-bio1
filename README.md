# Topology-Induced Scale Axis for scRNA-seq (Project Gold Pan)

Code for: "A Topology-Induced Scale Axis for Multi-Scale Annealing in scRNA-seq Co-Expression Networks"

**Author:** Wujie Gu (g@wujielanguagemodel.com)  
**Preprint:** osf.io/rq5vj  
**Website:** wujielanguagemodel.com

---

## Files and Figure Mapping

### Core pipeline

| File | Paper section | Figures produced |
|------|--------------|-----------------|
| `scale_axis_toy_pipeline.py` | §4 Numerical Verification | **Fig 1** (Baseline: 6-panel eigenvalue/scale/decay/energy/field/half-life), **Fig 2** (ST1: Noise Gradient), **Fig 3** (ST2: Parameter Sweep), **Fig 4** (ST3: Topology Removal) |
| `pbmc_prep.py` | Preprocessing only | None (produces `.npy` files for `pbmc_validate_3k.py`) |
| `pbmc68k_validate.py` | §5.1–5.2 PBMC68k | **Fig 6** (PBMC68k 6-panel: spectral F, signal decomposition, marker positions, L7 timeline, F before/after, scale axis structure) |
| `pbmc_validate_3k.py` | §5.1–5.2 PBMC3k | **Fig 7** (PBMC3k 6-panel: cluster-F, variance, scale axis, L7 timeline, PCA before/after, marker positions) |

### Benchmarking

| File | Paper section | Figures produced |
|------|--------------|-----------------|
| `scale_axis_overnight.py` | §4.6, §5 | **Fig 5** (Block B: Parameter sweep N=50,000), **Fig 8** (Block D: Bootstrap stability N=2,000), **Fig 9** (Block E: Multi-study Immune_ALL validation). Also runs Block A GPU stability (0-violation result, no figure in paper) and Block C permutation test (reported in text only). |
| `scale_axis_benchmark.py` | §5.3 | **Fig 10** (Graph construction robustness: 11 methods × PBMC68k, marker s position + F-retention + batch mixing + cell-type purity) |
| `scale_axis_runtime_extended.py` | §6 Computational complexity | **Fig 11** (Runtime comparison: Scale Axis vs. ComBat vs. Harmony, 1k–30k cells, log-scale + relative slowdown) |

---

## Deprecated Files (not needed)

### `real_data_pipeline.py`
Early development script. Functionality is fully superseded by `pbmc68k_validate.py` and `pbmc_validate_3k.py`. Contains a `--synthetic` fallback mode used during development when real data was unavailable. No figures in the paper correspond to this script. **Safe to delete.**

### `scale_axis_runtime.py`
Earlier runtime benchmark covering only 3 dataset sizes (500 / 2,000 / 5,000 cells). Superseded by `scale_axis_runtime_extended.py`, which covers 8 sizes (1k–30k cells) and produces Fig 11. **Safe to delete.**

---

## Requirements

```
numpy scipy scikit-learn scanpy harmonypy matplotlib
```

Optional (for `scale_axis_overnight.py` Block A GPU):
```
torch  (CUDA-enabled)
```

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
```

## Data

- **PBMC68k:** `scanpy.datasets.pbmc68k_reduced()` (auto-download)
- **PBMC3k:** `scanpy.datasets.pbmc3k()` (auto-download)
- **Immune_ALL:** https://figshare.com/articles/dataset/12420968 (`Immune_ALL_human.h5ad`, required for Block E in `scale_axis_overnight.py`)
