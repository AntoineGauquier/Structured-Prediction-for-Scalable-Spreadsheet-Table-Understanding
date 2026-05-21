# Structured Prediction for Scalable Spreadsheet Table Understanding: From Cell Types to Table Ranges

Official repository for the paper _Structured Prediction for Scalable Spreadsheet Table Understanding: From Cell Types to Table Ranges_.

---

## Repository structure

```
.
├── dataset/                    # StatSheets: annotated spreadsheet dataset
│   ├── manifest.csv            # Index linking each sheet to its annotation file
│   ├── spreadsheets/           # Original spreadsheet files
│   └── annotations/           # Per-sheet annotation files (NPZ format)
├── cell-type-classification/   # Code and baselines for the CTC task
├── table-detection/            # Code and baselines for the TD task
└── extended_version.pdf        # Extended version of the paper
```

---

## StatSheets Dataset

**StatSheets** is a dataset of **737 annotated spreadsheet sheets** collected from national and international organizations publishing statistics. Each sheet is annotated for two tasks:

- **Cell-Type Classification (CTC):** every cell is labeled with one of five types.
- **Table Detection (TD):** the bounding ranges of all tables present on the sheet are identified.

### Sources

The sheets were collected from 14 publicly available sources spanning diverse countries, languages, and formatting conventions:

| Source key | Organization | Country / Scope | Sheets |
|---|---|---|---|
| `abs_spreadsheets` | Australian Bureau of Statistics (ABS) | Australia | 64 |
| `bea_spreadsheets` | Bureau of Economic Analysis (BEA) | United States | 58 |
| `census_spreadsheets` | U.S. Census Bureau | United States | 54 |
| `cnis_spreadsheets` | Conseil National de l'Information Statistique (CNIS) | France | 1 |
| `ilo_spreadsheets` | International Labour Organization (ILO) | International | 48 |
| `insee_spreadsheets` | Institut National de la Statistique et des Études Économiques (INSEE) | France | 57 |
| `interieur_spreadsheets` | French Ministry of the Interior | France | 50 |
| `justice_spreadsheets` | French Ministry of Justice | France | 64 |
| `mic_jp_spreadsheets` | Ministry of Internal Affairs and Communications (MIC) | Japan | 58 |
| `nces_spreadsheets` | National Center for Education Statistics (NCES) | United States | 61 |
| `oecd_spreadsheets` | Organisation for Economic Co-operation and Development (OECD) | International | 63 |
| `stat_sa_spreadsheets` | General Authority for Statistics | Saudi Arabia | 57 |
| `who_spreadsheets` | World Health Organization (WHO) | International | 48 |
| `worldbank_spreadsheets` | World Bank | International | 54 |
| | **Total** | | **737** |

Spreadsheets are provided in their original formats: `.xlsx`, `.xls`, `.ods`, and `.csv` (that can have both `,` and `;` as separators).

### Dataset structure

```
dataset/
├── manifest.csv
├── spreadsheets/
│   ├── <filename>.<ext>
│   └── ...
└── annotations/
    ├── annotations_<uuid>.npz
    └── ...
```

### manifest.csv

The manifest is a CSV file (one row per annotated sheet) with the following columns:

| Column | Description |
|---|---|
| `source` | Source collection key (see table above, e.g. `abs_spreadsheets`) |
| `file_path` | Relative path to the spreadsheet file within the `dataset/` directory |
| `sheet_name` | Name of the specific sheet inside the spreadsheet file which is annotated |
| `mime_type` | MIME type of the spreadsheet file (e.g. `application/vnd.openxmlformats-officedocument.spreadsheetml.sheet`) |
| `labels_path` | Relative path to the corresponding annotation file within `dataset/` |
| `uuid` | Unique identifier for this sheet, matching the UUID in the annotation filename |

### Annotation format

Each annotation file is a NumPy `.npz` archive (e.g. `annotations_<uuid>.npz`) containing four arrays:

| Key | Description |
|---|---|
| `labels` | **CTC annotations**: 2-D integer array of cell-type labels over the **compressed grid** (see below). Values: `0` = EMPTY, `1` = HEADER, `2` = DATA, `3` = TITLE, `4` = OTHER. |
| `anchor_labels` | 2-element array `[row_offset, col_offset]` anchoring `labels` to the compressed grid when the top-most rows or left-most columns of the sheet are empty and were subject to compression. |
| `ranges` | **TD annotations**: Array of Excel-style range strings (e.g. `B10:E83`) identifying table bounding boxes on the **compressed grid**. |
| `ranges_original` | Same table bounding boxes expressed on the **original (uncompressed) grid**. |

#### Grid compression

To keep annotation grids tractable for large, sparse sheets, a **compression step** reduces runs of empty rows and columns: any sequence of **k ≥ 3** consecutive empty rows (respectively columns) is replaced by exactly **2** consecutive empty rows (resp. columns). The `labels` array and `ranges` are defined on this compressed grid, while `ranges_original` gives the coordinates in the original, uncompressed grid. `anchor_labels` handles the edge case where the very first rows or columns of the sheet are empty and are affected by compression, ensuring correct spatial alignment.

---

## Cell-Type Classification

Code and baselines are provided in [`cell-type-classification/`](cell-type-classification/). Each method assigns one of the five following types to each cell: `EMPTY`, `HEADER`, `DATA`, `TITLE`, or `OTHER`.

### Features

Features are extracted at two granularities:

- **Unary features** ([`features/unary_features.py`](cell-type-classification/features/unary_features.py)): per-cell features covering content type (empty, numeric, string, date, formula), text statistics, positional encoding, and spreadsheet formatting (font style, size, color, borders, alignment).
- **Pairwise features** ([`features/pairwise_features.py`](cell-type-classification/features/pairwise_features.py)): 30 features for each pair of 4-connected neighboring cells, encoding content-type compatibility, formatting similarity, and relative position.

### Docker environment

The CRF baselines depend on [pystruct](https://github.com/pystruct/pystruct), which requires Python 3.8 and needs two fixes before it can run:

1. **Build fix**: pystruct's `setup.py` uses the deprecated `use_2to3` flag, which breaks installation under Python 3. The Dockerfile patches this out with `sed` before building pystruct from source.
2. **Runtime patch**: `pystruct.models.utils` is missing from some pystruct builds (the module is referenced at runtime but not always installed). The Dockerfile injects a minimal re-implementation via `PYTHONSTARTUP` that provides the two missing symbols (`loss_augment_unaries`, `crammer_singer_joint_feature`).

Both fixes are applied automatically inside the Docker image. **All baselines should be run inside this container** to ensure the correct Python 3.8 environment and the patched pystruct.

Build the image from the `cell-type-classification/` directory:

```bash
docker build -t ctc-baselines cell-type-classification/
```

Then run any baseline by mounting the dataset and a results directory:

```bash
docker run --rm \
    -v $(pwd)/dataset:/data/dataset \
    -v $(pwd)/results:/data/results \
    ctc-baselines \
    bash -c "cd /opt && python -m cell_type_classification.models.<script> <subcommand> \
        --dataset /data/dataset/manifest.csv \
        --output   /data/results/<output_dir> \
        --feature-cache /data/results/feature_cache"
```

The `--feature-cache` flag is strongly recommended: it caches the extracted features to disk so they are not recomputed for every fold. All scripts support it.

### Baselines

All reproduction scripts use a fixed 5-fold cross-validation with `seed=2112`, matching the fold split used for the initial experiments. If you want to replicate the experiments, **do not change the seed value**, otherwise a different fold split will be computed. 

#### RF and RF-Koci - [`models/rf.py`](cell-type-classification/models/rf.py)

Two variants: **RF** trains on the full unary feature set; **RF-Koci** uses the reduced feature subset presented in the paper. Select the variant with `--variant rf` or `--variant rf-koci`.

```bash
# Training
docker run --rm \
    -v $(pwd)/dataset:/data/dataset -v $(pwd)/results:/data/results \
    ctc-baselines \
    bash -c "cd /opt && python -m cell_type_classification.models.rf train \
        --dataset /data/dataset/manifest.csv \
        --output  /data/results/rf \
        --variant rf --save-fold-models \
        --feature-cache /data/results/feature_cache"

# Inference
docker run --rm \
    -v $(pwd)/dataset:/data/dataset -v $(pwd)/results:/data/results \
    ctc-baselines \
    bash -c "cd /opt && python -m cell_type_classification.models.rf infer \
        /data/results/rf \
        --variant rf \
        --output  /data/results/rf_grids \
        --feature-cache /data/results/feature_cache"
```

#### LightGBM - [`models/lgbm.py`](cell-type-classification/models/lgbm.py)

```bash
# Training
docker run --rm \
    -v $(pwd)/dataset:/data/dataset -v $(pwd)/results:/data/results \
    ctc-baselines \
    bash -c "cd /opt && python -m cell_type_classification.models.lgbm train \
        --dataset /data/dataset/manifest.csv \
        --output  /data/results/lgbm \
        --save-fold-models \
        --feature-cache /data/results/feature_cache"

# Inference
docker run --rm \
    -v $(pwd)/dataset:/data/dataset -v $(pwd)/results:/data/results \
    ctc-baselines \
    bash -c "cd /opt && python -m cell_type_classification.models.lgbm infer \
        /data/results/lgbm \
        --output /data/results/lgbm_grids \
        --feature-cache /data/results/feature_cache"
```

#### CRF-Linear - [`models/crf_linear.py`](cell-type-classification/models/crf_linear.py)

Operates directly on raw unary features (no upstream classifier projection). Two design choices distinguish it from the CRF-RF/GBM variants: sqrt-inverse global class weighting and hard EMPTY constraints. Both are on by default (paper configuration) and can be disabled via CLI flags for ablation studies.

```bash
# Training
docker run --rm \
    -v $(pwd)/dataset:/data/dataset -v $(pwd)/results:/data/results \
    ctc-baselines \
    bash -c "cd /opt && python -m cell_type_classification.models.crf_linear train \
        --dataset /data/dataset/manifest.csv \
        --output  /data/results/crf_linear \
        --save-fold-models \
        --C 0.1 --batch-size 128 \
        --feature-cache /data/results/feature_cache"

# Inference
docker run --rm \
    -v $(pwd)/dataset:/data/dataset -v $(pwd)/results:/data/results \
    ctc-baselines \
    bash -c "cd /opt && python -m cell_type_classification.models.crf_linear infer \
        /data/results/crf_linear \
        --output /data/results/crf_linear_grids \
        --feature-cache /data/results/feature_cache"
```

#### CRF-RF - [`models/rf_crf.py`](cell-type-classification/models/rf_crf.py)

Two-stage pipeline: a fresh RF is trained on each fold's training cells, its `predict_proba` output (one probability per class) replaces the raw unary features, and the CRF is then trained on those projected features. Pairwise features are unchanged.

```bash
# Training
docker run --rm \
    -v $(pwd)/dataset:/data/dataset -v $(pwd)/results:/data/results \
    ctc-baselines \
    bash -c "cd /opt && python -m cell_type_classification.models.rf_crf train \
        --dataset /data/dataset/manifest.csv \
        --output  /data/results/rf_crf \
        --save-fold-models \
        --C 10 --batch-size 32 \
        --feature-cache /data/results/feature_cache"

# Inference
docker run --rm \
    -v $(pwd)/dataset:/data/dataset -v $(pwd)/results:/data/results \
    ctc-baselines \
    bash -c "cd /opt && python -m cell_type_classification.models.rf_crf infer \
        /data/results/rf_crf \
        --output /data/results/rf_crf_grids \
        --feature-cache /data/results/feature_cache"
```

#### CRF-LightGBM - [`models/lgbm_crf.py`](cell-type-classification/models/lgbm_crf.py)

Same two-stage pipeline as CRF-RF, but uses LightGBM (with the best hyperparameters from the standalone LightGBM baseline) as the first-stage classifier.

```bash
# Training
docker run --rm \
    -v $(pwd)/dataset:/data/dataset -v $(pwd)/results:/data/results \
    ctc-baselines \
    bash -c "cd /opt && python -m cell_type_classification.models.lgbm_crf train \
        --dataset /data/dataset/manifest.csv \
        --output  /data/results/lgbm_crf \
        --save-fold-models \
        --C 10 --batch-size 32 \
        --feature-cache /data/results/feature_cache"

# Inference
docker run --rm \
    -v $(pwd)/dataset:/data/dataset -v $(pwd)/results:/data/results \
    ctc-baselines \
    bash -c "cd /opt && python -m cell_type_classification.models.lgbm_crf infer \
        /data/results/lgbm_crf \
        --output /data/results/lgbm_crf_grids \
        --feature-cache /data/results/feature_cache"
```

#### TUTA

We fine-tune [TUTA](https://github.com/microsoft/TUTA_table_understanding/tree/main/tuta) on our five cell-type classes by unfreezing its two last transformer layers and the classification head, using the same 5-fold cross-validation. See the paper for more details.

---

## Table Detection

TBD, provided in [`table-detection/`](table-detection/).

---

## Extended paper

The extended version of the paper (`extended_version.pdf`) includes TBD.
