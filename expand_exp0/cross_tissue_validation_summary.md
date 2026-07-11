# Cross-Tissue Generalization Validation — Summary

**Status:** All four files (figshare "named" + CZ CELLxGENE "hash" versions,
Liver and Lung) completed.

---

## Files downloaded (4 total)

| Label | File | Source | Status |
|---|---|---|---|
| site1_Liver_named | `TS_Liver.h5ad` | figshare (Tabula Sapiens, named fields) | ✅ Run — results below |
| site1_Lung_named | `TS_Lung.h5ad` | figshare (Tabula Sapiens, named fields) | ✅ Run — results below |
| site2_Liver_hash | `a4b3a49e-062b-4e3f-8915-02f40607651f.h5ad` | CZ CELLxGENE | ✅ **Run — results below** |
| site2_Lung_hash | `fddf43a3-b7c8-49e9-ac0a-5e24b390849f.h5ad` | CZ CELLxGENE | ✅ **Run — results below** |

All four runs used the two sibling scripts (`scale_axis_validate_tissue.py`
for the figshare "named" versions, `scale_axis_validate_cellxgene.py` for the
CZ CELLxGENE versions), which read different obs/var field names but produce
directly comparable output.

---

## Results so far

### Liver (`TS_Liver.h5ad`, 5,007 cells → 5,007 cells, 2,000 HVGs, 13 cell types)

- Scale axis: λ ∈ [0.00, 169.58], s ∈ [0.00, 5.14]
- **F enrichment (low-λ / high-λ): 4.81×**
- Intra-noise enrichment (high-λ): 1.31×
- Marker genes found: 4/31 (ALB, HNF4A, STAB2, CLEC4G)
- **All 4 markers: F increased after gradient flow**
- Cell-type separation retention: 107%
- Energy drop: 64%, **monotone: True**
- Assay methods present: 10X, Smart-seq2 (mixed)

### Lung (`TS_Lung.h5ad`, 35,682 cells → subsampled to 8,000, 2,000 HVGs, 33 cell types)

- Scale axis: λ ∈ [0.00, 171.40], s ∈ [0.00, 5.15]
- **F enrichment (low-λ / high-λ): 5.93×** (higher than Liver, and higher
  than any ratio reported in the main PBMC68k paper)
- Intra-noise enrichment (high-λ): 2.15×
- Marker genes found: 10/30 (SFTPC, SFTPB, SFTPA1, NAPSA, TPPP3, SCGB1A1,
  SCGB3A1, CLDN5, MARCO, MRC1)
- **All 10 markers: F increased after gradient flow**
- Cell-type separation retention: 101%
- Energy drop: 38%, **monotone: True**
- Assay methods present: 10X, Smart-seq2 (mixed)

---

### Liver — CZ CELLxGENE (`a4b3a49e-062b-4e3f-8915-02f40607651f.h5ad`, 22,214 cells → subsampled to 8,000, 2,000 HVGs, 23 cell types)

Verbatim run output:

```
Raw shape: (22214, 60606)
Dropping 1 cell type(s) with < 20 cells
Subsampling 22206 -> 8000 cells
Selecting top 2000 highly variable genes...
Final: 8000 cells x 2000 genes
Cell types (23): B cell 48, CD4-positive alpha-beta T cell 102, CD8-positive
  alpha-beta T cell 468, T cell 17, classical monocyte 121, endothelial cell 630,
  erythrocyte 379, fibroblast 58, hematopoietic precursor cell 18, hepatic
  stellate cell 81, hepatocyte 2654, intermediate monocyte 209, intrahepatic
  cholangiocyte 222, macrophage 1153, mast cell 10, mature NK T cell 309,
  monocyte 606, myeloid cell 23, myeloid dendritic cell 22, natural killer
  cell 260, neutrophil 273, non-classical monocyte 41, plasma cell 296
Assay platforms present: ["10x 3' v3", 'Smart-seq2']

lambda in [0.000, 112.183]   s in [0.000, 4.729]
Half-life ratio (large-s / small-s): 2.6x

F_celltype: low-lam=565.13, mid-lam=318.09, high-lam=90.15
Intra-var:  low-lam=0.0235, high-lam=0.2462
Cell-type F enrichment in low-lam: 6.27x
Intra-noise enrichment in high-lam: 10.49x
Signal structure: CONFIRMED

Marker gene positions (15/31 found):
  ALB (Hepatocyte): s=4.571, F=118.0
  APOA1 (Hepatocyte): s=4.646, F=151.7
  TTR (Hepatocyte): s=4.645, F=183.8
  MARCO (Kupffer cell): s=4.686, F=477.2
  CD163 (Kupffer cell): s=4.520, F=1295.1
  VSIG4 (Kupffer cell): s=4.593, F=740.2
  DCN (Hepatic stellate): s=4.689, F=186.9
  ACTA2 (Hepatic stellate): s=4.660, F=240.3
  STAB2 (Endothelial sinusoid): s=4.621, F=435.3
  LYVE1 (Endothelial sinusoid): s=4.625, F=489.3
  KRT7 (Cholangiocyte): s=4.622, F=1097.1
  CD3D (T cell): s=4.555, F=461.4
  TRAC (T cell): s=3.699, F=265.7
  NKG7 (NK cell): s=4.663, F=656.1
  GNLY (NK cell): s=4.652, F=217.2

Marker gene F before -> after gradient flow (all 15 UP):
  ALB 118.0->135.9, APOA1 151.7->191.0, TTR 183.8->322.0, MARCO 477.2->542.9,
  CD163 1295.1->1618.9, VSIG4 740.2->876.6, DCN 186.9->190.8, ACTA2 240.3->243.8,
  STAB2 435.3->482.2, LYVE1 489.3->548.3, KRT7 1097.1->1117.9, CD3D 461.4->478.7,
  TRAC 265.7->279.3, NKG7 656.1->694.7, GNLY 217.2->227.7

Cell-type separation: 0.4882 -> 0.5176 (retention 106%)
Energy drop: 50%  monotone: True
Marker genes with F retained/increased: 15/15
Flow preservation: CONFIRMED
```

### Lung — CZ CELLxGENE (`fddf43a3-b7c8-49e9-ac0a-5e24b390849f.h5ad`, 65,847 cells → subsampled to 8,000, 2,000 HVGs, 30 cell types)

Verbatim run output:

```
Raw shape: (65847, 60606)
Dropping 4 cell type(s) with < 20 cells
Subsampling 65792 -> 8000 cells
Selecting top 2000 highly variable genes...
Final: 8000 cells x 2000 genes
Cell types (30): B cell 84, CD4-positive alpha-beta T cell 269, CD8-positive
  alpha-beta T cell 217, adventitial cell 73, alveolar adventitial fibroblast 139,
  basal cell 459, basophil 170, bronchial smooth muscle cell 24, capillary
  endothelial cell 907, classical monocyte 191, club cell 195, endothelial cell
  of artery 217, endothelial cell of lymphatic vessel 40, intermediate monocyte 342,
  lung multiciliated epithelial cell 134, macrophage 2009, mature NK T cell 14,
  monocyte 63, myeloid dendritic cell 5, natural killer cell 121, neutrophil 38,
  non-classical monocyte 59, pericyte 84, plasma cell 9, pulmonary alveolar type 1
  cell 379, pulmonary alveolar type 2 cell 1428, pulmonary ionocyte 4, respiratory
  tract goblet cell 129, vascular associated smooth muscle cell 17, vein
  endothelial cell 180
Assay platforms present: ["10x 3' v3", 'Smart-seq2']

lambda in [0.000, 223.141]   s in [0.000, 5.412]
Half-life ratio (large-s / small-s): 3.1x

F_celltype: low-lam=1537.30, mid-lam=122.10, high-lam=113.71
Intra-var:  low-lam=0.0536, high-lam=0.2111
Cell-type F enrichment in low-lam: 13.52x
Intra-noise enrichment in high-lam: 3.94x
Signal structure: CONFIRMED

Marker gene positions (23/30 found):
  SFTPC (AT2): s=4.989, F=1278.2       SFTPB (AT2): s=4.985, F=1920.8
  SFTPA1 (AT2): s=5.258, F=2048.8      NAPSA (AT2): s=5.204, F=2932.5
  AGER (AT1): s=4.740, F=484.0         CAV1 (AT1): s=5.345, F=524.6
  TPPP3 (Ciliated): s=5.050, F=186.4   SCGB1A1 (Club): s=5.099, F=246.9
  SCGB3A1 (Club): s=5.047, F=425.8     PECAM1 (Endothelial): s=4.272, F=706.8
  VWF (Endothelial): s=5.333, F=576.6  CLDN5 (Endothelial): s=4.872, F=930.6
  COL1A2 (Fibroblast): s=5.018, F=874.3
  MARCO (Macrophage): s=5.385, F=1077.1  MRC1 (Macrophage): s=4.293, F=1193.2
  CD3D (T cell): s=5.001, F=789.2      CD3E (T cell): s=4.774, F=665.9
  TRAC (T cell): s=3.824, F=407.5
  NKG7 (NK cell): s=5.347, F=363.1     GNLY (NK cell): s=5.307, F=355.0
  KLRD1 (NK cell): s=5.190, F=219.8
  CD79A (B cell): s=5.347, F=1589.3    MS4A1 (B cell): s=5.060, F=1370.3

Marker gene F before -> after gradient flow (all 23 UP):
  SFTPC 1278.2->1310.9, SFTPB 1920.8->1997.7, SFTPA1 2048.8->2104.8,
  NAPSA 2932.5->3039.2, AGER 484.0->493.8, CAV1 524.6->703.0,
  TPPP3 186.4->189.5, SCGB1A1 246.9->249.8, SCGB3A1 425.8->442.6,
  PECAM1 706.8->775.6, VWF 576.6->848.2, CLDN5 930.6->2186.6,
  COL1A2 874.3->894.0, MARCO 1077.1->1114.5, MRC1 1193.2->1239.2,
  CD3D 789.2->802.4, CD3E 665.9->678.1, TRAC 407.5->414.8,
  NKG7 363.1->367.8, GNLY 355.0->358.2, KLRD1 219.8->222.2,
  CD79A 1589.3->1595.1, MS4A1 1370.3->1376.0

Cell-type separation: 0.8655 -> 0.8927 (retention 103%)
Energy drop: 48%  monotone: True
Marker genes with F retained/increased: 23/23
Flow preservation: CONFIRMED
```

---

## Combined results table (all four runs)

| Dataset | Cells (used) | Cell types | F enrichment (low/high λ) | Intra-noise ratio | Markers found | Markers F↑ | Sep. retention | Energy monotone |
|---|---|---|---|---|---|---|---|---|
| Liver (figshare) | 5,007 | 13 | 4.81× | 1.31× | 4/31 | 4/4 | 107% | True |
| Lung (figshare) | 8,000 (sub) | 33 | 5.93× | 2.15× | 10/30 | 10/10 | 101% | True |
| Liver (CZ CELLxGENE) | 8,000 (sub) | 23 | **6.27×** | **10.49×** | 15/31 | **15/15** | 106% | True |
| Lung (CZ CELLxGENE) | 8,000 (sub) | 30 | **13.52×** | 3.94× | 23/30 | **23/23** | 103% | True |

Every one of the four independent runs — two tissues, two independently
processed data sources, mixed 10x/Smart-seq2 platforms — confirms both
structural predictions with zero exceptions among all marker genes tested
(4+10+15+23 = 52/52 markers show retained or increased F-statistic after the
gradient flow). The CZ CELLxGENE version of each tissue shows a *stronger*
enrichment ratio than the figshare version, plausibly because the larger
original cell pool (22k/66k vs 5k/36k) yields a larger, more informative
subsample even after downsampling to 8,000.

---

## Interpretation

Both tissues independently confirm the two structural predictions from the
scRNA-seq paper:

1. **Signal structure** — cell-type discriminative signal concentrates in
   low-λ (large-s) modes; technical/local noise concentrates in high-λ
   (small-s) modes. Confirmed in both tissues, with Lung showing an even
   stronger enrichment ratio than Liver or the original PBMC68k result.

2. **Flow preservation** — the gradient flow (Proposition 3 / energy
   monotonicity) preserves or improves marker gene discriminability with
   zero exceptions across both tissues (4/4 in Liver, 10/10 in Lung).

Because both datasets already contain a mixture of 10X and Smart-seq2 cells,
this also provides implicit (not yet isolated) evidence that the effect is
not platform-specific.

---

## Remaining work

1. **Platform-split analysis (optional, not yet done)** — rerun Liver and
   Lung filtered to 10x-only and Smart-seq2-only subsets, to give an
   explicit, isolated platform-generalization result rather than relying on
   the mixed sample as implicit evidence. All four runs above already
   contain both platforms mixed together and confirm the structural
   predictions regardless.

2. **Do not contact the journal now.** The submission (BIOINF-2026-2076) is
   still with the editor (desk screen / reviewer assignment stage as of
   this writing). These results are being held in reserve, not sent to the
   editor pre-emptively, because:
   - mid-review manuscript changes are generally not accepted by the
     journal system;
   - proactively flagging "we ran more experiments" before any request adds
     risk (may read as an incomplete original submission) without benefit;
   - the correct point to introduce this material is the revision response,
     if and when the editor's decision requests broader tissue validation.

3. **Decide destination once a decision letter arrives:**
   - If Major/Minor Revision requests generalization evidence → this
     material goes directly into the revision response and/or a new
     supplementary section.
   - If Reject → integrate into a strengthened resubmission, or write up as
     an independent short generalization-focused paper.
