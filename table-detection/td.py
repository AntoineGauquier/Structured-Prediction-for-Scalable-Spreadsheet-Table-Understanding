from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.ndimage import label as cc_label

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Hyperparameters 
# ---------------------------------------------------------------------------

MIN_DATA_CELLS             = 4 
MIN_HEADER_ONLY_CELLS      = 8 
HEADER_ROW_THRESHOLD       = 0.4 
HEADER_COL_THRESHOLD       = 0.4 
VERT_GAP_MIN               = 4 
HORIZ_GAP_MIN              = 2 
MIN_DATA_DENSITY           = 0.05 
MIN_HEADER_DENSITY         = 0.5 
DATA_ONLY_MIN_CELLS        = 20 
HEADER_NEIGHBOR_RADIUS     = 3 
DEDUP_IOU_THRESH           = 0.5 
CONTAINMENT_OVERLAP_THRESH = 0.8 

# ---------------------------------------------------------------------------
# Cell-type label constants
# ---------------------------------------------------------------------------

STATE_EMPTY  = 0
STATE_HEADER = 1
STATE_DATA   = 2
STATE_TITLE  = 3
STATE_OTHER  = 4

# ---------------------------------------------------------------------------
# Detected table data structure
# ---------------------------------------------------------------------------

@dataclass
class DetectedTable:
    row_start: int
    row_end: int
    col_start: int
    col_end: int
    top_header_rows: list[int] = field(default_factory=list)
    left_header_cols: list[int] = field(default_factory=list)
    right_header_cols: list[int] = field(default_factory=list)
    sub_header_rows: list[int] = field(default_factory=list)
    data_row_start: int = 0
    data_col_start: int = 0
    data_col_end: int = 0
    data_fill_ratio: float = 1.0
    n_sub_headers: int = 0

    def to_dict(self):
        return {
            "row_start": self.row_start,
            "row_end": self.row_end,
            "col_start": self.col_start,
            "col_end": self.col_end,
            "top_header_rows": self.top_header_rows,
            "left_header_cols": self.left_header_cols,
            "right_header_cols": self.right_header_cols,
            "sub_header_rows": self.sub_header_rows,
            "data_row_start": self.data_row_start,
            "data_col_start": self.data_col_start,
            "data_col_end": self.data_col_end,
            "data_fill_ratio": round(self.data_fill_ratio, 4),
            "n_sub_headers": self.n_sub_headers,
        }

    def as_excel_range(self):
        return (f"{_col_to_letter(self.col_start)}{self.row_start + 1}:{_col_to_letter(self.col_end)}{self.row_end + 1}")

# ---------------------------------------------------------------------------
# Coordinate helpers
# ---------------------------------------------------------------------------

def _col_letters_to_idx(letters):
    idx = 0
    for ch in letters.upper():
        idx = idx * 26 + (ord(ch) - ord('A') + 1)
    return idx - 1


def _col_to_letter(idx):
    result = ""
    n = idx + 1
    while n > 0:
        n, rem = divmod(n - 1, 26)
        result = chr(ord('A') + rem) + result
    return result


def parse_excel_range(range_str):
    m = re.fullmatch(r"([A-Za-z]+)(\d+):([A-Za-z]+)(\d+)", range_str.strip())
    if not m:
        raise ValueError(f"Cannot parse Excel range: {range_str!r}")
    col_s = _col_letters_to_idx(m.group(1))
    row_s = int(m.group(2)) - 1
    col_e = _col_letters_to_idx(m.group(3))
    row_e = int(m.group(4)) - 1
    return row_s, col_s, row_e, col_e


def load_gt_ranges(npz_path, use_original_coords=False):
    data = np.load(npz_path, allow_pickle=False)
    field_name = "ranges_original" if use_original_coords else "ranges"
    ranges_raw = data[field_name]
    if ranges_raw.dtype.kind == 'S':
        ranges_list = [r.decode("ascii") for r in ranges_raw.ravel()]
    else:
        ranges_list = [str(r) for r in ranges_raw.ravel()]
    return [parse_excel_range(r) for r in ranges_list]


# ---------------------------------------------------------------------------
# Compression helpers
# ---------------------------------------------------------------------------

def _feature_cache_path(cache_dir, file_path, sheet_name):
    # Keyed by resolved absolute path + sheet to avoid cross-directory collisions
    key = f"{Path(file_path).resolve()}::{sheet_name}"
    digest = hashlib.sha1(key.encode()).hexdigest()[:16]
    stem = Path(file_path).stem[:50]
    return Path(cache_dir) / f"{stem}_{digest}.npz"


def _compression_cache_path(cache_dir, file_path, sheet_name):
    base = _feature_cache_path(cache_dir, file_path, sheet_name)
    return base.with_suffix(".compression.json")


def _load_compression_info(path):
    payload = json.loads(Path(path).read_text())
    row_mapping = {int(k): int(v) for k, v in payload["row_mapping"].items()}
    col_mapping = {int(k): int(v) for k, v in payload["col_mapping"].items()}
    original_shape = tuple(payload["original_shape"])
    return row_mapping, col_mapping, original_shape


def _expand_to_original(compressed_array, row_mapping, col_mapping, original_shape):
    H_orig, W_orig = original_shape
    extra_dims = compressed_array.shape[2:]
    out = np.full((H_orig, W_orig) + extra_dims, STATE_EMPTY, dtype=compressed_array.dtype)
    for ci, oi in row_mapping.items():
        for cj, oj in col_mapping.items():
            out[oi, oj] = compressed_array[ci, cj]
    return out


def _load_compression_if_available(cache_dir, file_path, sheet_name):
    if cache_dir is None:
        return None, None, None
    comp_path = _compression_cache_path(Path(cache_dir), file_path, sheet_name)
    if not comp_path.exists():
        return None, None, None
    try:
        return _load_compression_info(comp_path)
    except Exception as e:
        print(f"  [warn] could not load compression info from {comp_path}: {e}")
        return None, None, None


def _expand_label_grid(label_grid, row_mapping, col_mapping, original_shape):
    if row_mapping is None or col_mapping is None or original_shape is None:
        return label_grid
    return _expand_to_original(label_grid, row_mapping, col_mapping, original_shape)


# ---------------------------------------------------------------------------
# IoU / matching
# ---------------------------------------------------------------------------

def rect_iou(a, b):
    r0s, c0s, r0e, c0e = a
    r1s, c1s, r1e, c1e = b
    ir_s, ir_e = max(r0s, r1s), min(r0e, r1e)
    ic_s, ic_e = max(c0s, c1s), min(c0e, c1e)
    inter = max(0, ir_e - ir_s + 1) * max(0, ic_e - ic_s + 1)
    if inter == 0:
        return 0.0
    area_a = (r0e - r0s + 1) * (c0e - c0s + 1)
    area_b = (r1e - r1s + 1) * (c1e - c1s + 1)
    return inter / (area_a + area_b - inter)


def match_tables(predicted, gt_ranges):
    if not predicted or not gt_ranges:
        return [], list(range(len(predicted))), list(range(len(gt_ranges)))

    iou_matrix = np.zeros((len(predicted), len(gt_ranges)))
    for pi, pred in enumerate(predicted):
        pred_rect = (pred.row_start, pred.col_start, pred.row_end, pred.col_end)
        for gi, gt in enumerate(gt_ranges):
            iou_matrix[pi, gi] = rect_iou(pred_rect, gt)

    # Greedy matching from highest IoU downward
    matches = []
    used_pred, used_gt = set(), set()
    for idx in np.argsort(iou_matrix, axis=None)[::-1]:
        pi, gi = divmod(int(idx), len(gt_ranges))
        if pi in used_pred or gi in used_gt:
            continue
        iou = iou_matrix[pi, gi]
        if iou == 0.0:
            break
        matches.append((pi, gi, float(iou)))
        used_pred.add(pi)
        used_gt.add(gi)

    unmatched_pred = [i for i in range(len(predicted)) if i not in used_pred]
    unmatched_gt = [i for i in range(len(gt_ranges)) if i not in used_gt]
    return matches, unmatched_pred, unmatched_gt


def compute_iou_metrics(predicted, gt_ranges, thresholds=(0.5, 0.75, 0.95)):
    matches, unmatched_pred, unmatched_gt = match_tables(predicted, gt_ranges)
    matched_ious = [m[2] for m in matches]
    all_ious = matched_ious + [0.0] * (len(unmatched_pred) + len(unmatched_gt))
    n_pred, n_gt = len(predicted), len(gt_ranges)

    result = {
        "mean_iou": float(np.mean(all_ious)) if all_ious else 0.0,
        "matched_ious": matched_ious,
        "n_predicted": n_pred,
        "n_gt": n_gt,
        "n_matched": len(matches),
        "unmatched_pred": len(unmatched_pred),
        "unmatched_gt": len(unmatched_gt),
    }
    for t in thresholds:
        tp = sum(1 for iou in matched_ious if iou >= t)
        precision = tp / n_pred if n_pred > 0 else 0.0
        recall = tp / n_gt if n_gt > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)
              if (precision + recall) > 0 else 0.0)
        result[f"precision@{t}"] = round(precision, 4)
        result[f"recall@{t}"] = round(recall, 4)
        result[f"f1@{t}"] = round(f1, 4)
    return result


# ---------------------------------------------------------------------------
# Table range extractor
# ---------------------------------------------------------------------------

class TableRangeExtractor:
    def __init__(
        self,
        min_data_cells=MIN_DATA_CELLS,
        min_header_only_cells=MIN_HEADER_ONLY_CELLS,
        header_row_threshold=HEADER_ROW_THRESHOLD,
        header_col_threshold=HEADER_COL_THRESHOLD,
        vert_gap_min=VERT_GAP_MIN,
        horiz_gap_min=HORIZ_GAP_MIN,
        min_data_density=MIN_DATA_DENSITY,
        min_header_density=MIN_HEADER_DENSITY,
        data_only_min_cells=DATA_ONLY_MIN_CELLS,
        header_neighbor_radius=HEADER_NEIGHBOR_RADIUS,
        dedup_iou_thresh=DEDUP_IOU_THRESH,
        containment_overlap_thresh=CONTAINMENT_OVERLAP_THRESH,
        sub_header_min_span=0.5,
    ):
        self.min_data_cells = min_data_cells
        self.min_header_only_cells = min_header_only_cells
        self.header_row_threshold = header_row_threshold
        self.header_col_threshold = header_col_threshold
        self.vert_gap_min = vert_gap_min
        self.horiz_gap_min = horiz_gap_min
        self.min_data_density = min_data_density
        self.min_header_density = min_header_density
        self.data_only_min_cells = data_only_min_cells
        self.header_neighbor_radius = header_neighbor_radius
        self.dedup_iou_thresh = dedup_iou_thresh
        self.containment_overlap_thresh = containment_overlap_thresh
        self.sub_header_min_span = sub_header_min_span

    def extract(self, labels):
        H, W = labels.shape
        # Connected components on DATA+HEADER cells are the initial seeds (stage 1)
        seed_mask = np.isin(labels, [STATE_DATA, STATE_HEADER]).astype(np.int32)
        cc_map, n_components = cc_label(seed_mask)

        components = []
        for c in range(1, n_components + 1):
            rows, cols = np.where(cc_map == c)
            data_rows, _ = np.where((cc_map == c) & (labels == STATE_DATA))
            n_data = len(data_rows)

            if n_data >= self.min_data_cells:
                pass
            else:
                header_rows, _ = np.where((cc_map == c) & (labels == STATE_HEADER))
                if len(header_rows) >= self.min_header_only_cells and n_data == 0:
                    pass
                else:
                    continue

            components.append({
                "rows": rows, "cols": cols,
                "r_min": rows.min(), "r_max": rows.max(),
                "c_min": cols.min(), "c_max": cols.max(),
            })

        if not components:
            return []

        components = self._merge_close_components(components, labels)  # stage 2
        components = self._filter_incoherent_components(components, labels)  # stage 3
        tables = [self._expand_component(comp, labels, H, W) for comp in components]  # stage 4
        return self._deduplicate(tables)  # stage 5

    def _merge_close_components(self, components, labels):
        # Iteratively merge until no more pairs qualify
        changed = True
        while changed:
            changed = False
            merged_into = {}

            for i in range(len(components)):
                if i in merged_into:
                    continue
                for j in range(i + 1, len(components)):
                    if j in merged_into:
                        continue
                    if self._should_merge(components[i], components[j], labels):
                        merged_into[j] = i

            if not merged_into:
                break

            absorbed = {}
            for idx, comp in enumerate(components):
                parent = merged_into.get(idx, idx)
                if parent == idx:
                    absorbed[idx] = comp
                else:
                    p = absorbed.setdefault(parent, components[parent])
                    rows = np.concatenate([p["rows"], comp["rows"]])
                    cols = np.concatenate([p["cols"], comp["cols"]])
                    p["rows"], p["cols"] = rows, cols
                    p["r_min"] = rows.min()
                    p["r_max"] = rows.max()
                    p["c_min"] = cols.min()
                    p["c_max"] = cols.max()
                    changed = True

            components = list(absorbed.values())
        return components

    def _filter_incoherent_components(self, components, labels):
        kept = []
        for comp in components:
            r_min, r_max = comp["r_min"], comp["r_max"]
            c_min, c_max = comp["c_min"], comp["c_max"]

            bbox_labels = labels[r_min:r_max + 1, c_min:c_max + 1]
            bbox_size = bbox_labels.size
            n_data = int(np.sum(bbox_labels == STATE_DATA))
            n_header = int(np.sum(bbox_labels == STATE_HEADER))

            if n_data == 0:
                if n_header >= self.min_header_only_cells:
                    kept.append(comp)
                continue

            density = n_data / bbox_size if bbox_size > 0 else 0.0
            h_density = n_header / bbox_size if bbox_size > 0 else 0.0

            # Reject sparse seeds unless header density compensates (ρ_D^min, ρ_H^min)
            if density < self.min_data_density and h_density < self.min_header_density:
                continue

            if n_header == 0:
                if n_data >= self.data_only_min_cells:  # n_D^†: large header-less seed
                    kept.append(comp)
                    continue
                # Smaller header-less seed: accept only if a header cell is nearby (r_N)
                r0 = max(0, r_min - self.header_neighbor_radius)
                r1 = min(labels.shape[0], r_max + self.header_neighbor_radius + 1)
                c0 = max(0, c_min - self.header_neighbor_radius)
                c1 = min(labels.shape[1], c_max + self.header_neighbor_radius + 1)
                if np.any(labels[r0:r1, c0:c1] == STATE_HEADER):
                    kept.append(comp)
                continue

            kept.append(comp)
        return kept

    def _should_merge(self, ci, cj, labels):
        # Vertical gap check: components that overlap column-wise
        col_overlap = not (ci["c_max"] < cj["c_min"] or cj["c_max"] < ci["c_min"])
        if col_overlap:
            r_top = min(ci["r_max"], cj["r_max"])
            r_bottom = max(ci["r_min"], cj["r_min"])
            if r_bottom > r_top + 1:
                c_shared_min = max(ci["c_min"], cj["c_min"])
                c_shared_max = min(ci["c_max"], cj["c_max"])
                if c_shared_min <= c_shared_max:
                    n_cols = c_shared_max - c_shared_min + 1

                    separator_count = 0
                    for r in range(r_top + 1, r_bottom):
                        row_slice = labels[r, c_shared_min:c_shared_max + 1]
                        n_header = int(np.sum(row_slice == STATE_HEADER))
                        if n_header / n_cols >= self.header_row_threshold:
                            # Header row inside the gap -> intra-table sub-header, merge
                            return True
                        # TITLE cells count as separators (not part of either table)
                        if np.all((row_slice == STATE_EMPTY) | (row_slice == STATE_TITLE)):
                            separator_count += 1

                    lower_first = labels[r_bottom, c_shared_min:c_shared_max + 1]
                    n_header_lower = int(np.sum(lower_first == STATE_HEADER))
                    lower_starts_with_header = (
                        n_header_lower / n_cols >= self.header_row_threshold)

                    if lower_starts_with_header:
                        # Self-contained table below: only merge across a minimal gap
                        return separator_count <= 1
                    else:
                        return separator_count < self.vert_gap_min

        # Horizontal gap check: components that overlap row-wise (g_h)
        row_overlap = not (ci["r_max"] < cj["r_min"] or cj["r_max"] < ci["r_min"])
        if row_overlap:
            c_left = min(ci["c_max"], cj["c_max"])
            c_right = max(ci["c_min"], cj["c_min"])
            if c_right > c_left + 1:
                r_shared_min = max(ci["r_min"], cj["r_min"])
                r_shared_max = min(ci["r_max"], cj["r_max"])
                if r_shared_min <= r_shared_max:
                    empty_count = 0
                    for c in range(c_left + 1, c_right):
                        col_slice = labels[r_shared_min:r_shared_max + 1, c]
                        if np.all(col_slice == STATE_EMPTY):
                            empty_count += 1
                    if empty_count < self.horiz_gap_min:
                        return True
        return False

    def _deduplicate(self, tables):
        if self.dedup_iou_thresh >= 1.0 or len(tables) <= 1:
            return tables

        def area(t):
            return (t.row_end - t.row_start + 1) * (t.col_end - t.col_start + 1)

        def overlap_fraction(inner, outer):
            r_ov = max(0, min(inner.row_end, outer.row_end) - max(inner.row_start, outer.row_start) + 1)
            c_ov = max(0, min(inner.col_end, outer.col_end) - max(inner.col_start, outer.col_start) + 1)
            inner_area = area(inner)
            return (r_ov * c_ov) / inner_area if inner_area > 0 else 0.0

        # Process largest tables first; drop smaller ones dominated by an accepted table
        kept = []
        for candidate in sorted(tables, key=area, reverse=True):
            c_rect = (candidate.row_start, candidate.col_start, candidate.row_end, candidate.col_end)
            dominated = any(
                rect_iou(c_rect, (k.row_start, k.col_start, k.row_end, k.col_end)) >= self.dedup_iou_thresh
                or overlap_fraction(candidate, k) >= self.containment_overlap_thresh
                for k in kept
            )
            if not dominated:
                kept.append(candidate)
        return kept

    def _expand_component(self, comp, labels, H, W):
        r_min, r_max = comp["r_min"], comp["r_max"]
        c_min, c_max = comp["c_min"], comp["c_max"]
        col_span = slice(c_min, c_max + 1)

        # Scan outward from the seed bounding box for header rows/cols
        top_header_rows = []
        scan_row = r_min - 1
        while scan_row >= 0:
            if self._header_score_row(scan_row, col_span, labels) >= self.header_row_threshold:
                top_header_rows.insert(0, scan_row)
                scan_row -= 1
            else:
                break

        left_header_cols = []
        scan_col = c_min - 1
        while scan_col >= 0:
            if self._header_score_col(scan_col, r_min, r_max, labels) >= self.header_col_threshold:
                left_header_cols.insert(0, scan_col)
                scan_col -= 1
            else:
                break

        right_header_cols = []
        scan_col = c_max + 1
        while scan_col < W:
            if self._header_score_col(scan_col, r_min, r_max, labels) >= self.header_col_threshold:
                right_header_cols.append(scan_col)
                scan_col += 1
            else:
                break

        bbox_c0 = left_header_cols[0] if left_header_cols else c_min
        bbox_c1 = right_header_cols[-1] if right_header_cols else c_max
        full_n_cols = bbox_c1 - bbox_c0 + 1

        data_row_start = top_header_rows[-1] + 1 if top_header_rows else r_min
        data_col_start = left_header_cols[-1] + 1 if left_header_cols else c_min
        data_col_end = right_header_cols[0] - 1 if right_header_cols else c_max

        sub_header_rows = []
        for r in range(data_row_start, r_max + 1):
            if full_n_cols == 0:
                continue
            n_header = int(np.sum(labels[r, bbox_c0:bbox_c1 + 1] == STATE_HEADER))
            if n_header / full_n_cols < self.sub_header_min_span:
                continue
            if r == data_row_start:
                continue
            if r < r_max and np.any(labels[r + 1:r_max + 1, bbox_c0:bbox_c1 + 1] == STATE_DATA):
                sub_header_rows.append(r)

        data_cells = labels[r_min:r_max + 1, c_min:c_max + 1] == STATE_DATA
        bbox_r0 = top_header_rows[0] if top_header_rows else r_min

        return DetectedTable(
            row_start=bbox_r0, row_end=r_max,
            col_start=bbox_c0, col_end=bbox_c1,
            top_header_rows=top_header_rows,
            left_header_cols=left_header_cols,
            right_header_cols=right_header_cols,
            sub_header_rows=sub_header_rows,
            data_row_start=data_row_start,
            data_col_start=data_col_start,
            data_col_end=data_col_end,
            data_fill_ratio=float(data_cells.mean()),
            n_sub_headers=len(sub_header_rows),
        )

    def _header_score_row(self, row, col_span, labels):
        row_labels = labels[row, col_span]
        non_empty = int(np.sum(row_labels != STATE_EMPTY))
        if non_empty == 0:
            return 0.0
        return float(np.sum(row_labels == STATE_HEADER) / non_empty)

    def _header_score_col(self, col, r_min, r_max, labels):
        col_labels = labels[r_min:r_max + 1, col]
        non_empty = int(np.sum(col_labels != STATE_EMPTY))
        if non_empty == 0:
            return 0.0
        return float(np.sum(col_labels == STATE_HEADER) / non_empty)


# ---------------------------------------------------------------------------
# Per-row processing (runs inside worker processes)
# ---------------------------------------------------------------------------

def _uuid_from_ann_path(ann_path):
    stem = Path(ann_path).stem
    return stem[len("annotations_"):] if stem.startswith("annotations_") else stem


def _file_key(labels_path):
    return _uuid_from_ann_path(labels_path) + ".csv"


def _process_fold_row(row, mode, grids_fold_dir, extractor, cache_dir,
                      iou_thresholds, verbose, grids_subdir=""):
    ann_path = Path(row["labels_path"])
    file_path = row["file_path"]
    sheet_name = row["sheet_name"]

    if not ann_path.exists():
        if verbose:
            print(f"  [skip] annotation not found: {ann_path}")
        return None

    row_mapping, col_mapping, original_shape = _load_compression_if_available(cache_dir, file_path, sheet_name)

    # When compression info is available, use original (uncompressed) coordinates
    use_original_coords = row_mapping is not None
    gt_ranges = load_gt_ranges(ann_path, use_original_coords=use_original_coords)

    try:
        if mode == "oracle":
            ann_data = np.load(str(ann_path), allow_pickle=False)
            if "labels" not in ann_data:
                if verbose:
                    print(f"  [skip] no 'labels' key in {ann_path.name}")
                return None
            label_grid = ann_data["labels"].astype(np.int32)

        else:  # predicted-grids
            uuid = _uuid_from_ann_path(ann_path)
            base_dir = Path(grids_fold_dir) / grids_subdir if grids_subdir else Path(grids_fold_dir)
            grid_path = base_dir / f"{uuid}.npz"
            if not grid_path.exists():
                if verbose:
                    print(f"  [skip] predicted grid not found: {grid_path}")
                return None
            grid_data = np.load(str(grid_path), allow_pickle=False)
            if "labels" not in grid_data:
                if verbose:
                    print(f"  [skip] no 'labels' key in {grid_path.name}")
                return None
            label_grid = grid_data["labels"].astype(np.int32)

        # Reinsert empty rows/cols that were compressed out during CTC feature extraction
        label_grid = _expand_label_grid(label_grid, row_mapping, col_mapping, original_shape)

        t0 = time.time()
        tables = extractor.extract(label_grid)
        detect_s = time.time() - t0

    except Exception as e:
        if verbose:
            print(f"  [error] {ann_path.name}: {e}")
        return None

    metrics = compute_iou_metrics(tables, gt_ranges, thresholds=iou_thresholds)
    metrics["file_path"] = file_path
    metrics["sheet_name"] = sheet_name
    metrics["ann_path"] = str(ann_path)
    metrics["mode"] = mode
    metrics["gt_coords"] = "original" if use_original_coords else "compressed"
    metrics["predicted_ranges"] = [t.as_excel_range() for t in tables]
    metrics["gt_ranges"] = [
        f"{_col_to_letter(c0)}{r0+1}:{_col_to_letter(c1)}{r1+1}"
        for r0, c0, r1, c1 in gt_ranges
    ]

    pred_gt_entry = {
        "file_path": file_path,
        "sheet_name": sheet_name,
        "ground_truth": [
            {"top_lx": [int(r0), int(c0)], "bot_rx": [int(r1), int(c1)]}
            for r0, c0, r1, c1 in gt_ranges
        ],
        "predictions": [
            {"top_lx": [int(t.row_start), int(t.col_start)],
             "bot_rx": [int(t.row_end), int(t.col_end)]}
            for t in tables
        ],
    }

    if verbose:
        mode_tag = "[oracle]" if mode == "oracle" else "[pred]"
        coord_tag = "[orig]" if use_original_coords else "[comp]"
        print(f"  {mode_tag}{coord_tag} {ann_path.stem:55s} "
              f"pred={metrics['n_predicted']} gt={metrics['n_gt']} "
              f"mIoU={metrics['mean_iou']:.3f} F1@.5={metrics.get('f1@0.5', 0.0):.3f} "
              f"[det={detect_s*1e3:.1f}ms]")

    return metrics, pred_gt_entry, {"detect_s": detect_s}


# ---------------------------------------------------------------------------
# Worker pool
# ---------------------------------------------------------------------------

_cv_worker_extractor = None
_cv_worker_mode = None
_cv_worker_grids_fold_dir = None
_cv_worker_grids_subdir = ""

def _cv_worker_init(extractor_kwargs, mode, grids_fold_dir_str, grids_subdir):
    warnings.filterwarnings("ignore")
    global _cv_worker_extractor, _cv_worker_mode, _cv_worker_grids_fold_dir, _cv_worker_grids_subdir
    _cv_worker_extractor = TableRangeExtractor(**extractor_kwargs)
    _cv_worker_mode = mode
    _cv_worker_grids_fold_dir = grids_fold_dir_str
    _cv_worker_grids_subdir = grids_subdir


def _cv_worker_process_row(row, cache_dir, iou_thresholds, verbose):
    return _process_fold_row(
        row, _cv_worker_mode, _cv_worker_grids_fold_dir,
        _cv_worker_extractor, cache_dir, iou_thresholds, verbose,
        grids_subdir=_cv_worker_grids_subdir,
    )


# ---------------------------------------------------------------------------
# Aggregate helpers
# ---------------------------------------------------------------------------

def _aggregate_results(results, thresholds):
    if not results:
        return {}

    all_ious = []
    for r in results:
        all_ious.extend(r["matched_ious"])
        all_ious.extend([0.0] * (r["unmatched_gt"] + r["unmatched_pred"]))

    total_pred = sum(r["n_predicted"] for r in results)
    total_gt = sum(r["n_gt"] for r in results)

    agg = {
        "micro_mean_iou": float(np.mean(all_ious)) if all_ious else 0.0,
        "macro_mean_iou": float(np.mean([r["mean_iou"] for r in results])),
        "total_predicted": total_pred,
        "total_gt": total_gt,
    }
    for t in thresholds:
        tp = sum(int(r[f"precision@{t}"] * r["n_predicted"]) for r in results)
        micro_prec = tp / total_pred if total_pred > 0 else 0.0
        micro_rec = tp / total_gt if total_gt > 0 else 0.0
        micro_f1 = (2 * micro_prec * micro_rec / (micro_prec + micro_rec) if (micro_prec + micro_rec) > 0 else 0.0)
        agg[f"micro_precision@{t}"] = round(micro_prec, 4)
        agg[f"micro_recall@{t}"] = round(micro_rec, 4)
        agg[f"micro_f1@{t}"] = round(micro_f1, 4)
        agg[f"macro_f1@{t}"] = round(float(np.mean([r[f"f1@{t}"] for r in results])), 4)
    return agg


def _print_aggregate(agg, thresholds, prefix=""):
    print(f"{prefix}micro mean IoU : {agg.get('micro_mean_iou', 0.0):.4f}")
    print(f"{prefix}macro mean IoU : {agg.get('macro_mean_iou', 0.0):.4f}")
    for t in thresholds:
        mf1 = agg.get(f"micro_f1@{t}", 0.0)
        mp = agg.get(f"micro_precision@{t}", 0.0)
        mr = agg.get(f"micro_recall@{t}", 0.0)
        maf1 = agg.get(f"macro_f1@{t}", 0.0)
        print(f"{prefix}micro F1 @{t} : {mf1:.4f}  (P={mp:.4f}  R={mr:.4f})")
        print(f"{prefix}macro F1 @{t} : {maf1:.4f}")


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------

def evaluate_cv_folds(
    cv_dir,
    k,
    mode,
    grids_dir=None,
    grids_subdir="",
    cache_dir=None,
    output_dir=None,
    extractor_kwargs=None,
    iou_thresholds=(0.5, 0.75, 0.95),
    verbose=True,
    n_procs=None,
):
    if mode == "predicted-grids" and grids_dir is None:
        raise ValueError("grids_dir is required for 'predicted-grids' mode")

    cv_dir = Path(cv_dir)
    extractor_kwargs = extractor_kwargs or {}
    n_procs = n_procs or max(1, os.cpu_count() - 1)

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    if verbose:
        print(f"\n{'=' * 70}")
        print(f"TABLE DETECTION — {mode.upper()}   k={k}   cv_dir={cv_dir}")
        print(f"{'=' * 70}")

    t_global_start = time.time()
    all_file_results = []
    predictions_vs_gt = {}
    per_fold_summaries = []
    global_detect_s = 0.0

    for fold_num in range(1, k + 1):
        fold_dir = cv_dir / f"fold_{fold_num:02d}"
        val_manifest_path = fold_dir / "val_manifest.csv"

        print(f"\n{'#' * 60}")
        print(f"  FOLD {fold_num}/{k}")
        print(f"{'#' * 60}")

        if not val_manifest_path.exists():
            print(f"  [skip] val_manifest.csv not found: {val_manifest_path}")
            continue

        grids_fold_dir = None
        if mode == "predicted-grids":
            grids_fold_dir = str(Path(grids_dir) / f"fold_{fold_num:02d}")

        with open(val_manifest_path, newline="") as fh:
            val_rows = list(csv.DictReader(fh))
        print(f"  Val set: {len(val_rows)} files  (n_procs={n_procs})")

        fold_results = []
        fold_detect_s = 0.0
        t_fold_start = time.time()

        with ProcessPoolExecutor(
            max_workers=n_procs,
            initializer=_cv_worker_init,
            initargs=(extractor_kwargs, mode, grids_fold_dir, grids_subdir),
        ) as pool:
            futures = {
                pool.submit(_cv_worker_process_row, row, cache_dir, iou_thresholds, verbose): row
                for row in val_rows
            }
            for future in as_completed(futures):
                row = futures[future]
                try:
                    out = future.result()
                except Exception as exc:
                    if verbose:
                        print(f"  [error] worker: {exc}")
                    continue
                if out is None:
                    continue
                metrics, pred_gt_entry, timing = out

                metrics["fold"] = fold_num
                pred_gt_entry["fold"] = fold_num

                fold_results.append(metrics)
                all_file_results.append(metrics)
                predictions_vs_gt[_file_key(row["labels_path"])] = pred_gt_entry
                fold_detect_s += timing["detect_s"]

        t_fold = time.time() - t_fold_start
        global_detect_s += fold_detect_s

        fold_agg = _aggregate_results(fold_results, iou_thresholds)
        fold_agg["fold"] = fold_num
        fold_agg["n_files"] = len(fold_results)
        fold_agg["timing"] = {
            "detect_s": round(fold_detect_s, 3),
            "total_s": round(t_fold, 3),
        }
        per_fold_summaries.append(fold_agg)

        print(f"\n  [FOLD {fold_num}] {len(fold_results)} files evaluated")
        print(f"  [TIMING]  table detection (sum) : {fold_detect_s:.2f}s")
        print(f"  [TIMING]  fold wall-clock        : {t_fold:.2f}s")
        _print_aggregate(fold_agg, iou_thresholds, prefix="  ")

        if output_dir is not None:
            fold_out_dir = output_dir / f"fold_{fold_num:02d}"
            fold_out_dir.mkdir(exist_ok=True)
            metrics_path = fold_out_dir / "metrics.json"
            if not metrics_path.exists():
                with open(metrics_path, "w") as fh:
                    json.dump({"fold_aggregate": fold_agg, "per_file": fold_results},
                              fh, indent=2, default=str)

    t_global = time.time() - t_global_start
    global_agg = _aggregate_results(all_file_results, iou_thresholds)
    global_agg["n_files"] = len(all_file_results)
    global_agg["mode"] = mode
    global_agg["k"] = k
    global_agg["timing"] = {
        "total_detect_s": round(global_detect_s, 3),
        "grand_total_s": round(t_global, 3),
    }

    print(f"\n\n{'=' * 70}")
    print(f"GLOBAL  [{mode.upper()}]  {len(all_file_results)} files  k={k}")
    print(f"{'=' * 70}")
    print(f"  [TIMING] table detection total : {global_detect_s:.2f}s")
    print(f"  [TIMING] grand total (clock)   : {t_global:.2f}s")
    _print_aggregate(global_agg, iou_thresholds)

    result = {
        "aggregate": global_agg,
        "per_fold": per_fold_summaries,
        "per_file": all_file_results,
    }

    if output_dir is not None:
        results_path = output_dir / "results.json"
        pvgt_path = output_dir / "predictions_vs_gt.json"
        if not results_path.exists():
            with open(results_path, "w") as fh:
                json.dump(result, fh, indent=2, default=str)
            print(f"\nResults saved    → {results_path}")
        if not pvgt_path.exists():
            with open(pvgt_path, "w") as fh:
                json.dump(predictions_vs_gt, fh, indent=2, default=str)
            print(f"Pred vs GT saved → {pvgt_path}")

    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _add_extractor_args(p):
    p.add_argument("--min-data-cells", type=int, default=MIN_DATA_CELLS)
    p.add_argument("--min-header-only-cells", type=int, default=MIN_HEADER_ONLY_CELLS)
    p.add_argument("--header-row-threshold", type=float, default=HEADER_ROW_THRESHOLD)
    p.add_argument("--header-col-threshold", type=float, default=HEADER_COL_THRESHOLD)
    p.add_argument("--vert-gap-min", type=int, default=VERT_GAP_MIN)
    p.add_argument("--horiz-gap-min", type=int, default=HORIZ_GAP_MIN)
    p.add_argument("--min-data-density", type=float, default=MIN_DATA_DENSITY)
    p.add_argument("--min-header-density", type=float, default=MIN_HEADER_DENSITY)
    p.add_argument("--data-only-min-cells", type=int, default=DATA_ONLY_MIN_CELLS)
    p.add_argument("--header-neighbor-radius", type=int, default=HEADER_NEIGHBOR_RADIUS)
    p.add_argument("--dedup-iou-thresh", type=float, default=DEDUP_IOU_THRESH)
    p.add_argument("--containment-overlap-thresh", type=float, default=CONTAINMENT_OVERLAP_THRESH)


def _add_cv_args(p):
    p.add_argument("cv_dir", help="Root CV directory (contains fold_01/, fold_02/, …).")
    p.add_argument("--k", type=int, required=True, help="Number of folds.")
    p.add_argument("--cache-dir", required=True, help="Feature cache directory (provides compression mappings).")
    p.add_argument("--output", default=None, help="Output directory. Nothing is saved when omitted.")
    p.add_argument("--iou-thresholds", nargs="+", type=float, default=[0.5, 0.75, 0.95])
    p.add_argument("--n-procs", type=int, default=None, help="Worker processes per fold (default: cpu_count - 1).")
    p.add_argument("--quiet", action="store_true", default=False, help="Suppress per-file output lines.")
    _add_extractor_args(p)


def _extractor_kwargs_from_args(args):
    return {
        "min_data_cells": args.min_data_cells,
        "min_header_only_cells": args.min_header_only_cells,
        "header_row_threshold": args.header_row_threshold,
        "header_col_threshold": args.header_col_threshold,
        "vert_gap_min": args.vert_gap_min,
        "horiz_gap_min": args.horiz_gap_min,
        "min_data_density": args.min_data_density,
        "min_header_density": args.min_header_density,
        "data_only_min_cells": args.data_only_min_cells,
        "header_neighbor_radius": args.header_neighbor_radius,
        "dedup_iou_thresh": args.dedup_iou_thresh,
        "containment_overlap_thresh": args.containment_overlap_thresh,
    }


def _build_parser():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = p.add_subparsers(dest="command", help="Subcommand")

    oracle_p = sub.add_parser(
        "oracle",
        help="Evaluation using GT labels from annotation NPZ.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    _add_cv_args(oracle_p)

    pg_p = sub.add_parser(
        "predicted-grids",
        help="Evaluation with CTC-predicted label grids.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    pg_p.add_argument(
        "--grids-dir", required=True,
        help="Root directory of predicted grids (contains fold_01/, fold_02/, …).",
    )
    pg_p.add_argument(
        "--grids-subdir", default="",
        help=(
            "Subdirectory inside each fold directory holding the NPZ files. "
            "E.g. 'predicted_grids' when files live at "
            "<grids-dir>/fold_01/predicted_grids/<UUID>.npz."
        ),
    )
    _add_cv_args(pg_p)
    return p


def main(argv=None):
    t0 = time.time()
    args = _build_parser().parse_args(argv)

    if args.command is None:
        _build_parser().print_help()
        return 1

    evaluate_cv_folds(
        cv_dir=args.cv_dir,
        k=args.k,
        mode=args.command,
        grids_dir=getattr(args, "grids_dir", None),
        grids_subdir=getattr(args, "grids_subdir", ""),
        cache_dir=args.cache_dir,
        output_dir=args.output,
        extractor_kwargs=_extractor_kwargs_from_args(args),
        iou_thresholds=tuple(args.iou_thresholds),
        verbose=not args.quiet,
        n_procs=args.n_procs,
    )

    print(f"\nTotal wall-clock time: {time.time() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())