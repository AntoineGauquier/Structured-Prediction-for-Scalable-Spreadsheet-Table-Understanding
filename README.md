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

TBD, provided in [`cell-type-classification/`](cell-type-classification/).

---

## Table Detection

TBD, provided in [`table-detection/`](table-detection/).

---

## Extended paper

The extended version of the paper (`extended_version.pdf`) includes TBD.
