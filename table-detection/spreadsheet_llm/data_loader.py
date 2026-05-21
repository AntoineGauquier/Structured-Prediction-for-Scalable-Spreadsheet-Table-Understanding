from __future__ import annotations

import os
import re
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from .spreadsheet_io import SpreadsheetData, load_spreadsheet


# ─────────────────────────────────────────────────────────────────────────────
#  K-fold helper  (MUST stay identical to the reference implementation)
# ─────────────────────────────────────────────────────────────────────────────

def make_folds(
    dataset_csv: str,
    k: int,
    random_state: int = 2112,
) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
    """Return k (train_df, val_df) pairs.

    The shuffle + split is identical to the canonical definition so that every
    method uses exactly the same fold boundaries.
    """
    df = pd.read_csv(dataset_csv)
    df_shuffled = df.sample(frac=1, random_state=random_state).reset_index(drop=True)
    kf = KFold(n_splits=k, shuffle=False)

    folds = []
    for train_idx, val_idx in kf.split(df_shuffled):
        folds.append((
            df_shuffled.iloc[train_idx].reset_index(drop=True),
            df_shuffled.iloc[val_idx].reset_index(drop=True),
        ))
    return folds


# ─────────────────────────────────────────────────────────────────────────────
#  Ground-truth loading
# ─────────────────────────────────────────────────────────────────────────────

def load_ground_truth(
    npz_path: str,
    use_original_coords: bool = True,
) -> List[str]:
    """Load table range strings from an annotation .npz file.

    Parameters
    ----------
    npz_path :
        Path to the .npz annotation file.
    use_original_coords :
        When True (default) return the ``ranges_original`` field, which gives
        coordinates in the un-compressed spreadsheet — this is what we need
        when evaluating SpreadsheetLLM predictions against the real file.
        When False, return the ``ranges`` field (compressed coordinates used
        by the label-grid methods).

    Returns
    -------
    List of A1-notation strings like ``['A4:BQ270', 'A280:BQ400']``.
    """
    data = np.load(npz_path, allow_pickle=False)
    field = 'ranges_original' if use_original_coords else 'ranges'
    raw = data[field]
    if raw.dtype.kind == 'S':          # byte strings
        return [r.decode('ascii') for r in raw.ravel()]
    return [str(r) for r in raw.ravel()]


# ─────────────────────────────────────────────────────────────────────────────
#  Single-item loader
# ─────────────────────────────────────────────────────────────────────────────

def load_item(
    manifest_row: pd.Series,
    data_dir: str,
) -> Tuple[SpreadsheetData, List[str]]:
    """Load one manifest row into (SpreadsheetData, gt_ranges).

    Parameters
    ----------
    manifest_row :
        A row from the manifest DataFrame (columns: file_path, sheet_name,
        mime_type, labels_path).
    data_dir :
        Root directory that prefixes both file_path and labels_path.

    Returns
    -------
    (SpreadsheetData, list-of-A1-range-strings)
    """
    fp   = os.path.join(data_dir, manifest_row['file_path'])
    ann  = os.path.join(data_dir, manifest_row['labels_path'])
    name = manifest_row.get('sheet_name') or None
    if isinstance(name, float):   # NaN
        name = None
    mime = manifest_row.get('mime_type') or None

    sheet_data = load_spreadsheet(fp, sheet_name=name, mime_type=mime)
    gt_ranges  = load_ground_truth(ann, use_original_coords=True)
    return sheet_data, gt_ranges


# ─────────────────────────────────────────────────────────────────────────────
#  Address helpers (mirrors utils.py but self-contained here for convenience)
# ─────────────────────────────────────────────────────────────────────────────

def _col_to_idx(letters: str) -> int:
    n = 0
    for ch in letters.upper():
        n = n * 26 + (ord(ch) - 64)
    return n - 1


def _parse_a1_range(rng: str) -> Tuple[int, int, int, int]:
    """'A4:BQ270' → (row_start, col_start, row_end, col_end) 0-indexed."""
    m = re.fullmatch(r'([A-Za-z]+)(\d+):([A-Za-z]+)(\d+)', rng.strip())
    if not m:
        raise ValueError(f"Cannot parse range: {rng!r}")
    r0 = int(m.group(2)) - 1
    c0 = _col_to_idx(m.group(1))
    r1 = int(m.group(4)) - 1
    c1 = _col_to_idx(m.group(3))
    return r0, c0, r1, c1


def gt_ranges_as_boxes(
    gt_ranges: List[str],
) -> List[Tuple[int, int, int, int]]:
    """Convert list of A1-notation strings to (r0,c0,r1,c1) 0-indexed tuples."""
    return [_parse_a1_range(r) for r in gt_ranges]
