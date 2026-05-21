"""Unary feature extraction for sheet cells."""

import numpy as np
import pandas as pd
from collections import defaultdict
from scipy.ndimage import convolve
from time import time

from ..loaders import load_spreadsheet
from ..formatting import extract_ods_format_features, extract_xls_format_features, extract_xlsx_format_features
from ..formatting.ods_formatter import create_default_features
from ..utils.text_utils import text_stats, text_starts_with, get_cell_type
from ..utils.compression import compress_empty_rows_with_mapping, compress_empty_columns_with_mapping
from joblib import load as joblib_load


def isna(arr):
    na_mask = pd.isna(arr)
    ws_mask = np.vectorize(
        lambda x: isinstance(x, str) and x.strip() == ""
    )(arr)
    return na_mask | ws_mask


def _apply_annotation_emptiness_rules(cell_values, file_format, sheet,
                                       df_to_logical_row, df_to_logical_col,
                                       merged_map=None):
    """
    Post-process cell_values (compressed-grid space) to fix residual
    annotation/loader divergences for error cells.
    """

    cell_values = cell_values.copy().astype(object)
    h, w = cell_values.shape

    if file_format == 'xls' and sheet is not None:
        import xlrd
        _EMPTY_CTYPES = (xlrd.XL_CELL_EMPTY, xlrd.XL_CELL_ERROR, xlrd.XL_CELL_BLANK)
        for ci in range(h):
            logical_row = df_to_logical_row[ci]
            for cj in range(w):
                logical_col = df_to_logical_col[cj]

                if merged_map:
                    anchor = merged_map.get((logical_row, logical_col))
                    if anchor is not None and anchor != (logical_row, logical_col):
                        continue
                try:
                    if sheet.cell_type(logical_row, logical_col) in _EMPTY_CTYPES:
                        cell_values[ci, cj] = None
                except IndexError:
                    cell_values[ci, cj] = None

    elif file_format == 'xlsx' and sheet is not None:
        for ci in range(h):
            logical_row = df_to_logical_row[ci]
            for cj in range(w):
                logical_col = df_to_logical_col[cj]
                if merged_map:
                    anchor = merged_map.get((logical_row, logical_col))
                    if anchor is not None and anchor != (logical_row, logical_col):
                        continue
                try:
                    cell = sheet.cell(row=logical_row + 1, column=logical_col + 1)
                    if cell.data_type == 'e':
                        cell_values[ci, cj] = None
                except Exception:
                    pass

    # ODS and CSV/TSV: no correction needed here.
    return cell_values


def _get_merge_spans(merged_map, h, w, df_to_logical_row, df_to_logical_col,
                     file_format):
    """
    Return (col_span_grid, row_span_grid), both float32 arrays of shape (h, w).
    All values default to 1.0 (unmerged cell).

    Assumed merged_map structures per format:

    xlsx (openpyxl)
        dict mapping (min_row_1based, min_col_1based)
             -> openpyxl CellRange with .min_row .max_row .min_col .max_col (1-based)

    xls (xlrd)
        list of (row_lo, row_hi, col_lo, col_hi) tuples (0-based, hi exclusive)

    ods (ezodf or variant)
        dict mapping (row_0based, col_0based) -> (row_span_int, col_span_int)

    Any unexpected structure silently returns all-ones grids rather than crashing.
    """
    col_span_grid = np.ones((h, w), dtype=np.float32)
    row_span_grid = np.ones((h, w), dtype=np.float32)

    if not merged_map or file_format in ('csv', 'tsv'):
        return col_span_grid, row_span_grid

    log_to_comp_row = {v: k for k, v in df_to_logical_row.items()}
    log_to_comp_col = {v: k for k, v in df_to_logical_col.items()}

    try:
        # Group all cells by their anchor to recover the bounding box.
        anchor_cells = defaultdict(list)
        for (r, c), (ar, ac) in merged_map.items():
            anchor_cells[(ar, ac)].append((r, c))

        for (ar, ac), cells in anchor_cells.items():
            rows = [r for r, c in cells]
            cols = [c for r, c in cells]
            rs = max(rows) - min(rows) + 1
            cs = max(cols) - min(cols) + 1
            comp_r = log_to_comp_row.get(ar)
            comp_c = log_to_comp_col.get(ac)
            if comp_r is not None and comp_c is not None:
                row_span_grid[comp_r, comp_c] = float(rs)
                col_span_grid[comp_r, comp_c] = float(cs)

    except Exception:
        pass

    return col_span_grid, row_span_grid


def extract_unary_features(path, file_format='auto', sheet_name=None):
    """
    Extract unary features from a spreadsheet file.

    Args:
        path: Path to spreadsheet file
        file_format: 'ods', 'xls', 'xlsx', 'csv', or 'auto' (auto-detect from extension)
        sheet_name: Sheet name or index to load (default: 0)
        
    Returns:
        features: (H, W, 67) numpy array of features, in compressed-grid space.
        row_mapping: dict {compressed_row_idx: original_row_idx}
        col_mapping: dict {compressed_col_idx: original_col_idx}
        original_shape: (H_original, W_original) before any compression
    """

    start_load = time()

    if file_format == 'auto':
        if path.endswith('.ods'):
            file_format = 'ods'
        elif path.endswith('.xls'):
            file_format = 'xls'
        elif path.endswith('.xlsx'):
            file_format = 'xlsx'
        elif path.endswith('.csv') or path.endswith('.tsv'):
            file_format = 'csv'
        else:
            raise ValueError(f"Cannot auto-detect format for {path}")

    # Load spreadsheet.
    df, sheet, workbook, merged_map, cell_map, fmt = load_spreadsheet(
        path, file_format, sheet_name=sheet_name
    )

    h, w = df.shape
    original_shape = (h, w)

    df_to_logical_row = {i: i for i in range(h)}
    df_to_logical_col = {j: j for j in range(w)}

    # Compress empty rows/columns.
    df, df_to_logical_row = compress_empty_rows_with_mapping(df, df_to_logical_row, max_consecutive=2)
    df, df_to_logical_col = compress_empty_columns_with_mapping(df, df_to_logical_col, max_consecutive=2)

    row_mapping = df_to_logical_row
    col_mapping = df_to_logical_col

    h, w = df.shape
    features = np.zeros((h, w, 67), dtype=np.float32)

    start_content = time()
    cell_values = df.values

    cell_values = _apply_annotation_emptiness_rules(cell_values, file_format, sheet,df_to_logical_row, df_to_logical_col,merged_map=merged_map,)

    is_na = isna(cell_values)
    mask_non_empty = ~is_na

    cell_type = np.vectorize(get_cell_type, otypes=[bool]*6)(cell_values)

    is_number_raw = cell_type[0]
    is_int_raw = cell_type[1]
    is_float_raw = cell_type[2]
    is_string_raw = cell_type[3]
    is_date_vec_raw = cell_type[4]
    is_formula_raw = cell_type[5]

    is_number = is_number_raw & mask_non_empty
    is_int = is_int_raw & mask_non_empty
    is_float = is_float_raw & mask_non_empty
    is_string = is_string_raw & mask_non_empty
    is_date_vec = is_date_vec_raw & mask_non_empty
    is_formula = is_formula_raw & mask_non_empty

    features[:, :, 0] = is_na
    features[:, :, 1] = is_number
    features[:, :, 2] = is_string
    features[:, :, 3] = is_date_vec
    features[:, :, 4] = is_formula

    text_stats_vec = np.stack([text_stats(v) for v in cell_values.flat]).reshape(h, w, 5)
    text_stats_vec[is_na] = 0
    features[:, :, 5:10] = text_stats_vec

    features[:, :, 10] = is_int
    features[:, :, 11] = is_float

    starts_with_vec = np.stack([text_starts_with(v) for v in cell_values.flat]).reshape(h, w, 2)
    starts_with_vec[is_na] = 0
    features[:, :, 12:14] = starts_with_vec

    ii, jj = np.indices((h, w))
    features[:, :, 14] = ii / h
    features[:, :, 15] = jj / w
    features[:, :, 16] = (ii == 0) | (jj == 0) | (ii == h - 1) | (jj == w - 1)

    features[:, :, 17] = mask_non_empty.mean(axis=1, keepdims=True)
    features[:, :, 18] = mask_non_empty.mean(axis=0, keepdims=True)

    kernel_4_neighborhood = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])
    neighbor_count = convolve(mask_non_empty.astype(int), kernel_4_neighborhood, mode="constant")

    features[:, :, 19] = neighbor_count
    features[:, :, 20] = neighbor_count == 0

    if file_format == "ods":
        features[:, :, 21] = True
        features[:, :, 22] = False
    elif file_format.startswith("xls"):
        features[:, :, 21] = False
        features[:, :, 22] = True
    else:
        features[:, :, 21] = False
        features[:, :, 22] = False

    cell_type_idx = np.full((h, w), -1, dtype=np.int8)
    cell_type_idx[is_number] = 0
    cell_type_idx[is_string] = 1
    cell_type_idx[is_date_vec] = 2
    cell_type_idx[is_formula] = 3

    type_grid = np.stack([is_number, is_string, is_date_vec, is_formula], axis=2)

    row_nonempty = mask_non_empty.sum(axis=1).astype(np.float32)
    row_denom = row_nonempty.clip(min=1)

    row_frac_num = is_number.sum(axis=1) / row_denom
    row_frac_str = is_string.sum(axis=1) / row_denom

    row_type_counts = type_grid.sum(axis=1)
    row_homogeneity = np.where(row_nonempty > 0, row_type_counts.max(axis=1) / row_denom, 0.0).astype(np.float32)

    row_is_uniform = (row_homogeneity == 1.0) & (row_nonempty > 0)
    is_alone_in_row = (row_nonempty == 1)

    features[:, :, 52] = (row_nonempty / w)[:, np.newaxis]
    features[:, :, 53] = row_frac_num[:, np.newaxis]
    features[:, :, 54] = row_frac_str[:, np.newaxis]
    features[:, :, 55] = row_homogeneity[:, np.newaxis]
    features[:, :, 56] = row_is_uniform[:, np.newaxis]
    features[:, :, 57] = is_alone_in_row[:, np.newaxis]

    col_nonempty = mask_non_empty.sum(axis=0).astype(np.float32)
    col_denom = col_nonempty.clip(min=1)

    col_frac_num = is_number.sum(axis=0) / col_denom
    col_frac_str = is_string.sum(axis=0) / col_denom

    col_type_counts = type_grid.sum(axis=0)
    col_homogeneity = np.where(col_nonempty > 0, col_type_counts.max(axis=1) / col_denom, 0.0).astype(np.float32)

    is_alone_in_col = (col_nonempty == 1)

    features[:, :, 58] = (col_nonempty / h)[np.newaxis, :]
    features[:, :, 59] = col_frac_num[np.newaxis, :]
    features[:, :, 60] = col_frac_str[np.newaxis, :]
    features[:, :, 61] = col_homogeneity[np.newaxis, :]
    features[:, :, 62] = is_alone_in_col[np.newaxis, :]

    col_dominant_type = col_type_counts.argmax(axis=1).astype(np.int8)

    col_type_outlier = (mask_non_empty & (cell_type_idx != col_dominant_type[np.newaxis, :]) & (col_nonempty[np.newaxis, :] > 1))
    features[:, :, 63] = col_type_outlier

    merge_col_span, merge_row_span = _get_merge_spans(merged_map, h, w, df_to_logical_row, df_to_logical_col, file_format)
    features[:, :, 64] = merge_col_span / w
    features[:, :, 65] = merge_row_span / h
    features[:, :, 66] = (merge_col_span >= 3)

    start_fmt = time()

    for i in range(h):
        for j in range(w):
            if file_format == "ods":
                fmt_features = extract_ods_format_features(workbook, i, j, df_to_logical_row, df_to_logical_col, cell_map, fmt, merged_map)
            elif file_format == "xlsx":
                fmt_features = extract_xlsx_format_features(workbook, sheet, i, j, df_to_logical_row, df_to_logical_col, cell_map, merged_map)
            elif file_format == "xls":
                fmt_features = extract_xls_format_features(workbook, sheet, i, j, df_to_logical_row, df_to_logical_col, cell_map, merged_map)
            else:  # CSV or TSV
                fmt_features = create_default_features(10.0, "Arial")

            features[i, j, 23:52] = fmt_features

    return features, row_mapping, col_mapping, original_shape
