"""Utilities for empty rows/columns compression."""

import pandas as pd
import numpy as np
import json
from pathlib import Path

def save_compression_info(path, row_mapping, col_mapping, original_shape):
    """Persist compression mappings to a JSON file.
    
    Args:
        path: Path for the .json file (e.g. cache_dir / "foo.compression.json")
        row_mapping: {compressed_row_idx: original_row_idx}
        col_mapping: {compressed_col_idx: original_col_idx}
        original_shape: (H_original, W_original) before any compression
    """
    payload = {
        "original_shape": list(original_shape),
        "row_mapping": {str(k): int(v) for k, v in row_mapping.items()},
        "col_mapping": {str(k): int(v) for k, v in col_mapping.items()},
    }
    Path(path).write_text(json.dumps(payload, indent=2))

def load_compression_info(path):
    """Load compression mappings saved by save_compression_info.
    
    Returns:
        row_mapping
        col_mapping
        original_shape: (H, W)
    """
    payload = json.loads(Path(path).read_text())
    row_mapping = {int(k): int(v) for k, v in payload["row_mapping"].items()}
    col_mapping = {int(k): int(v) for k, v in payload["col_mapping"].items()}
    original_shape = tuple(payload["original_shape"])
    return row_mapping, col_mapping, original_shape


def expand_to_original(compressed_array, row_mapping, col_mapping,
                        original_shape, fill_value=0):
    """Reinsert compressed predictions into the original grid.

    Args:
        compressed_array: np.ndarray of shape (H_compressed, W_compressed, ...)
        row_mapping: {compressed_row_idx: original_row_idx}
        col_mapping: {compressed_col_idx: original_col_idx}
        original_shape: (H_original, W_original)
        fill_value: Value to place in rows/cols that were fully dropped.

    Returns:
        np.ndarray of shape (H_original, W_original, ...) with predictions
        placed back at their original coordinates; dropped rows/cols get
        fill_value.
    """
    import numpy as np

    H_orig, W_orig = original_shape
    extra_dims = compressed_array.shape[2:]
    out = np.full((H_orig, W_orig) + extra_dims,
                  fill_value, dtype=compressed_array.dtype)

    for ci, oi in row_mapping.items():
        for cj, oj in col_mapping.items():
            out[oi, oj] = compressed_array[ci, cj]

    return out

def is_empty_value(val):
    """Check if a value is considered empty."""
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return True
    
    try:
        if pd.isna(val):
            return True
    except (TypeError, ValueError):
        pass
    
    if isinstance(val, str):
        stripped = val.strip()
        if stripped == "" or stripped.lower() == "none":
            return True
    return False


def is_row_empty(df, row_idx):
    """Check if a row is empty."""
    return all(is_empty_value(val) for val in df.iloc[row_idx])


def is_column_empty(df, col_idx):
    """Check if a column is empty."""
    return all(is_empty_value(val) for val in df.iloc[:, col_idx])


def compress_empty_rows_with_mapping(df, row_mapping, max_consecutive=2):
    """Compress consecutive empty rows while maintaining coordinate mapping.
    
    Args:
        df: DataFrame to compress
        row_mapping: Dict mapping compressed indices to original ones
        max_consecutive: Maximum number of consecutive empty rows to keep
    """
    h, w = df.shape
    rows_to_keep = []
    new_row_mapping = {}
    consecutive_empty = 0
    new_row_idx = 0
    
    for i in range(h):
        row_is_empty = is_row_empty(df, i)
        
        if row_is_empty:
            consecutive_empty += 1
            if consecutive_empty <= max_consecutive:
                rows_to_keep.append(i)
                new_row_mapping[new_row_idx] = row_mapping.get(i, i)
                new_row_idx += 1
        else:
            consecutive_empty = 0
            rows_to_keep.append(i)
            new_row_mapping[new_row_idx] = row_mapping.get(i, i)
            new_row_idx += 1

    while rows_to_keep:
        last_idx = rows_to_keep[-1]
        if is_row_empty(df, last_idx):
            rows_to_keep.pop()
            new_row_mapping.pop(len(rows_to_keep), None)
        else:
            break

    compressed_df = df.iloc[rows_to_keep].reset_index(drop=True)
    return compressed_df, new_row_mapping


def compress_empty_columns_with_mapping(df, col_mapping, max_consecutive=2):
    """Compress consecutive empty columns while maintaining coordinate mapping.
    
    Args:
        df: DataFrame to compress
        col_mapping: Dict mapping compressed indices to original one
        max_consecutive: Maximum number of consecutive empty columns to keep
    """
    h, w = df.shape
    cols_to_keep = []
    new_col_mapping = {}
    consecutive_empty = 0
    new_col_idx = 0
    
    for j in range(w):
        col_is_empty = is_column_empty(df, j)
        
        if col_is_empty:
            consecutive_empty += 1
            if consecutive_empty <= max_consecutive:
                cols_to_keep.append(j)
                new_col_mapping[new_col_idx] = col_mapping.get(j, j)
                new_col_idx += 1
        else:
            consecutive_empty = 0
            cols_to_keep.append(j)
            new_col_mapping[new_col_idx] = col_mapping.get(j, j)
            new_col_idx += 1

    while cols_to_keep:
        last_idx = cols_to_keep[-1]
        if is_column_empty(df, last_idx):
            cols_to_keep.pop()
            new_col_mapping.pop(len(cols_to_keep), None)
        else:
            break
    
    compressed_df = df.iloc[:, cols_to_keep].reset_index(drop=True)
    return compressed_df, new_col_mapping


# Functions for CSV (no mapping)
def empty_mask(arr):
    """Create a mask of empty values in array."""
    is_none = arr == None
    is_empty_str = arr == ""
    is_nan = arr != arr # Only NaN is not equal to itself
    return is_none | is_empty_str | is_nan


def compress_empty_rows(df, max_consecutive=2):
    """Compress consecutive empty rows (simple version for CSV)."""
    arr = df.values
    empty = empty_mask(arr).all(axis=1)

    keep = []
    cpt = 0
    for i, e in enumerate(empty):
        if e:
            cpt += 1
            if cpt <= max_consecutive:
                keep.append(i)
        else:
            cpt = 0
            keep.append(i)

    return df.iloc[keep].reset_index(drop=True)


def compress_empty_columns(df, max_consecutive=2):
    """Compress consecutive empty columns (simple version for CSV)."""
    arr = df.values
    empty = empty_mask(arr).all(axis=0)

    keep = []
    cpt = 0
    for j, e in enumerate(empty):
        if e:
            cpt += 1
            if cpt <= max_consecutive:
                keep.append(j)
        else:
            cpt = 0
            keep.append(j)

    return df.iloc[:, keep]