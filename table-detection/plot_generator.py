from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker


# ---------------------------------------------------------------------------
# Colour palette  (Wong 2011 — colour-blind safe)
# ---------------------------------------------------------------------------

_PALETTE = [
    "#0072B2",
    "#D55E00",
    "#009E73",
    "#CC79A7",
    "#E69F00",
    "#56B4E9",
    "#F0E442",
    "#000000",
]

_LINESTYLES = [
    "-",
    "--",
    "-.",
    (0, (4, 1)),
    (0, (1, 1)),
    (0, (3, 1, 1, 1)),
]

_Y_PAD_BELOW = 0.05
_Y_PAD_ABOVE = 0.02
_Y_SNAP = 1.0
_ANNOT_GAP_FRAC = 0.042

_UUID_RE = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}")

# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------

FONT_SIZE_LABEL = 10
FONT_SIZE_TICK = 10
FONT_SIZE_ANNOT = 7.5
FONT_SIZE_LEGEND = 9


def _apply_style():
    matplotlib.rcParams.update({
        "font.family": "STIXGeneral",
        "mathtext.fontset": "stix",
        "font.size": FONT_SIZE_LABEL,
        "axes.titlesize": FONT_SIZE_LABEL,
        "axes.labelsize": FONT_SIZE_LABEL,
        "xtick.labelsize": FONT_SIZE_TICK,
        "ytick.labelsize": FONT_SIZE_TICK,
        "legend.fontsize": FONT_SIZE_LEGEND,
        # --- Spines ---
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.75,
        "axes.labelpad": 4,
        # --- Grid ---
        "axes.grid": True,
        "grid.linestyle": ":",
        "grid.linewidth": 0.4,
        "grid.color": "#CCCCCC",
        # --- Ticks ---
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.size": 3,
        "ytick.major.size": 3,
        "xtick.major.width": 0.75,
        "ytick.major.width": 0.75,
        # --- Lines ---
        "lines.linewidth": 1.1,
        "lines.solid_capstyle": "round",
        # --- Legend ---
        "legend.framealpha": 0.93,
        "legend.edgecolor": "#CCCCCC",
        # --- Transparent backgrounds ---
        "figure.facecolor": "none",
        "axes.facecolor": "none",
        "savefig.transparent": True,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
    })


def _save(fig, path, dpi = 200):
    fig.savefig(path, dpi=dpi, facecolor="none", transparent=True, bbox_inches="tight", pad_inches=0.05)
    print(f"Saved: {path}")


# ---------------------------------------------------------------------------
# IoU helpers
# ---------------------------------------------------------------------------

_COL_RE = re.compile(r"^([A-Z]+)(\d+)$")


def _col_letters_to_idx(col):
    idx = 0
    for ch in col.upper():
        idx = idx * 26 + (ord(ch) - ord("A") + 1)
    return idx - 1


def _parse_excel_range(r):
    parts = r.strip().upper().split(":")
    if len(parts) != 2:
        return (-1, -1, -1, -1)
    m0 = _COL_RE.match(parts[0])
    m1 = _COL_RE.match(parts[1])
    if not (m0 and m1):
        return (-1, -1, -1, -1)
    r0 = int(m0.group(2)) - 1
    c0 = _col_letters_to_idx(m0.group(1))
    r1 = int(m1.group(2)) - 1
    c1 = _col_letters_to_idx(m1.group(1))
    return (min(r0, r1), min(c0, c1), max(r0, r1), max(c0, c1))


def _rect_iou(a, b):
    r0 = max(a[0], b[0]);  r1 = min(a[2], b[2])
    c0 = max(a[1], b[1]);  c1 = min(a[3], b[3])
    inter = max(0, r1 - r0 + 1) * max(0, c1 - c0 + 1)
    if inter == 0:
        return 0.0
    area_a = (a[2] - a[0] + 1) * (a[3] - a[1] + 1)
    area_b = (b[2] - b[0] + 1) * (b[3] - b[1] + 1)
    return inter / (area_a + area_b - inter)


def _match_from_ranges(pred_ranges, gt_ranges):
    preds = [_parse_excel_range(r) for r in pred_ranges]
    gts = [_parse_excel_range(r) for r in gt_ranges]
    preds = [p for p in preds if p[0] >= 0]
    gts = [g for g in gts   if g[0] >= 0]
    if not preds or not gts:
        return []
    iou_mat = np.zeros((len(preds), len(gts)), dtype=float)
    for i, p in enumerate(preds):
        for j, g in enumerate(gts):
            iou_mat[i, j] = _rect_iou(p, g)

    matched = []
    used_pred = set()
    used_gt = set()
    flat = np.argsort(-iou_mat, axis=None)
    for idx in flat:
        i, j = divmod(int(idx), len(gts))
        if i in used_pred or j in used_gt:
            continue
        iou = iou_mat[i, j]
        if iou <= 0.0:
            break
        matched.append(float(iou))
        used_pred.add(i)
        used_gt.add(j)
    return matched


# ---------------------------------------------------------------------------
# Curve computation (per fold)
# ---------------------------------------------------------------------------

def _compute_curves(per_file, thresholds, gt_anchored = False):
    """Return (micro_precision, macro_precision) arrays over thresholds."""
    all_pred_ious = []
    per_file_pred = []

    for r in per_file:
        pred_ranges = r.get("predicted_ranges", [])
        gt_ranges = r.get("gt_ranges", [])
        matched = _match_from_ranges(pred_ranges, gt_ranges)
        n_pred = len(pred_ranges)
        n_gt = len(gt_ranges)
        n_match = len(matched)
        cap = min(n_pred, n_gt) if gt_anchored else n_pred
        pred_ious = np.array(matched + [0.0] * (cap - n_match), dtype=float)
        all_pred_ious.extend(pred_ious.tolist())
        per_file_pred.append(pred_ious)

    all_pred = np.array(all_pred_ious, dtype=float)
    micro = np.array([(all_pred >= t).mean() if len(all_pred) > 0 else 0.0 for t in thresholds], dtype=float)
    macro = np.array(
        [np.mean([(p >= t).mean() if len(p) > 0 else 0.0
                  for p in per_file_pred]) if per_file_pred else 0.0
         for t in thresholds], dtype=float,
    )
    return micro, macro


# ---------------------------------------------------------------------------
# Manifest loading and canonical-key extraction
# ---------------------------------------------------------------------------

def _load_manifest(manifest_path):
    """
    Returns ``fp_basename_to_uuid``: maps every known spreadsheet filename
    (with and without extension, lower-cased) to its annotation UUID extracted
    from the ``labels_path`` column.
    """
    mapping = {}
    with open(manifest_path, newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            fp = row.get("file_path", "")
            lp = row.get("labels_path", "")
            if not fp or not lp:
                continue
            m = _UUID_RE.search(lp)
            if not m:
                continue
            uuid = m.group(0)
            basename = Path(fp).name.lower()
            stem = Path(fp).stem.lower()
            mapping[basename] = uuid
            if stem != basename:
                mapping[stem] = uuid
    return mapping


def _canonical_key(record, fp_basename_to_uuid):
    """
    Map a per-file record to its canonical UUID annotation key.

    Strategy (first match wins):
    1. ``ann_path`` / ``labels_path`` field — UUID extracted via regex.
    2. ``file`` field whose stem is a UUID (Mondrian style).
    3. ``file`` / ``file_path`` basename matched against the manifest.
    """

    for key in ("ann_path", "labels_path"):
        val = record.get(key)
        if val:
            m = _UUID_RE.search(str(val))
            if m:
                return m.group(0)

    for key in ("file", "file_path"):
        val = record.get(key)
        if val:
            stem = Path(val).stem
            if _UUID_RE.fullmatch(stem):
                return stem

    for key in ("file", "file_path"):
        val = record.get(key)
        if val:
            basename_lower = Path(val).name.lower()
            stem_lower = Path(val).stem.lower()
            if basename_lower in fp_basename_to_uuid:
                return fp_basename_to_uuid[basename_lower]
            if stem_lower in fp_basename_to_uuid:
                return fp_basename_to_uuid[stem_lower]

    return None


# ---------------------------------------------------------------------------
# Per-file data loading (SpreadsheetLLM-style directory fallback)
# ---------------------------------------------------------------------------

_SKIP_NAMES = {"inference_timing.json", "aggregate.json"}


def _load_per_file(fold_dir):
    """
    Load per-file records from *fold_dir*.

    Returns a list of dicts from ``metrics.json["per_file"]`` if that file
    exists, or from individual ``*.json`` files in the directory, or ``None``.
    """
    metrics_path = fold_dir / "metrics.json"
    if metrics_path.exists():
        with open(metrics_path) as fh:
            data = json.load(fh)
        per_file = data.get("per_file", [])
        if not per_file:
            print(f"  [warn] 'per_file' empty in {metrics_path}", file=sys.stderr)
            return None
        return per_file

    json_files = sorted(
        p for p in fold_dir.glob("*.json")
        if p.name not in _SKIP_NAMES
    )
    if not json_files:
        return None

    records = []
    for jp in json_files:
        try:
            with open(jp) as fh:
                rec = json.load(fh)
            records.append(rec)
        except Exception as exc:
            print(f"  [warn] could not read {jp}: {exc}", file=sys.stderr)
    return records if records else None


# ---------------------------------------------------------------------------
# Y-axis auto-range
# ---------------------------------------------------------------------------

def _auto_ylim(all_vals):
    v_min = float(np.min(all_vals))
    v_max = float(np.max(all_vals))
    span = max(v_max - v_min, 5.0)
    y_lo = max(0.0,   v_min - _Y_PAD_BELOW * span)
    y_hi = min(100.0, v_max + _Y_PAD_ABOVE * span)
    y_lo = float(np.floor(y_lo / _Y_SNAP) * _Y_SNAP)
    y_hi = float(np.ceil (y_hi / _Y_SNAP) * _Y_SNAP)
    return y_lo, y_hi


# ---------------------------------------------------------------------------
# Non-overlapping annotation helpers
# ---------------------------------------------------------------------------

def _spread_positions(raw_values, min_gap, y_lo, y_hi):
    n = len(raw_values)
    if n == 0:
        return []
    if n == 1:
        return [float(np.clip(raw_values[0], y_lo, y_hi))]
    order = sorted(range(n), key=lambda i: -raw_values[i])
    pos = [float(raw_values[i]) for i in order]
    for i in range(1, n):
        if pos[i] > pos[i - 1] - min_gap:
            pos[i] = pos[i - 1] - min_gap
    for i in range(n - 1, -1, -1):
        if pos[i] < y_lo:
            pos[i] = y_lo
        if i < n - 1 and pos[i] <= pos[i + 1] + min_gap:
            pos[i] = pos[i + 1] + min_gap
    for i in range(n):
        pos[i] = min(pos[i], y_hi)
    result = [0.0] * n
    for new_i, orig_i in enumerate(order):
        result[orig_i] = pos[new_i]
    return result


def _annotate_threshold(ax, thresholds, curves_mean, t, y_lo, y_hi):
    """Draw dots + non-overlapping value labels for one threshold (on mean curves)."""
    t_idx = int(np.argmin(np.abs(thresholds - t)))
    min_gap = (y_hi - y_lo) * _ANNOT_GAP_FRAC
    order = sorted(range(len(curves_mean)), key=lambda i: -curves_mean[i][1][t_idx])
    raw = [curves_mean[i][1][t_idx] for i in order]
    spread = _spread_positions(raw, min_gap, y_lo + 0.5, y_hi - 0.5)

    if t >= 0.99:
        x_sign, x_off_base = 1,  0.020
    elif t > 0.80:
        x_sign, x_off_base = -1, 0.016
    else:
        x_sign, x_off_base = 1,  0.016
    ha = "right" if x_sign < 0 else "left"

    for rank, (orig_i, spread_y) in enumerate(zip(order, spread)):
        _, curve, global_idx = curves_mean[orig_i]
        color = _PALETTE[global_idx % len(_PALETTE)]
        actual_y = curve[t_idx]
        x_stagger = rank * 0.0025 * x_sign
        label_x = t + x_off_base + x_stagger

        ax.plot(t, actual_y, "o", color=color, markersize=4.0, markeredgewidth=0.6, markeredgecolor="white", zorder=6, clip_on=False)

        if abs(spread_y - actual_y) > min_gap * 0.25:
            ax.plot([t + 0.003 * x_sign, label_x], [actual_y, spread_y], color=color, lw=0.45, alpha=0.55, zorder=4, clip_on=False)

        ax.text(label_x, spread_y, f"{actual_y:.1f}",
                fontsize=FONT_SIZE_ANNOT, color=color,
                fontweight="semibold", ha=ha, va="center",
                zorder=7, clip_on=False)


# ---------------------------------------------------------------------------
# Panel drawing across-folds: mean curve + shaded min/max band
# ---------------------------------------------------------------------------

CurveEntry = Tuple[str, np.ndarray, np.ndarray, np.ndarray, int]


def _draw_panel(ax, thresholds, curves, iou_markers):
    # Convert to %
    pct = [(name, m * 100, lo * 100, hi * 100, gidx) for name, m, lo, hi, gidx in curves]

    # Auto y-limits spanning both lo and hi bands
    all_vals = np.concatenate([np.concatenate([lo, hi]) for _, _, lo, hi, _ in pct])
    y_lo, y_hi = _auto_ylim(all_vals)
    y_span = y_hi - y_lo

    for name, mean, lo, hi, gidx in pct:
        color = _PALETTE[gidx % len(_PALETTE)]
        ls = _LINESTYLES[gidx % len(_LINESTYLES)]
        ax.fill_between(thresholds, lo, hi, color=color, alpha=0.13, linewidth=0, zorder=2)
        ax.plot(thresholds, mean, label=name, color=color, linestyle=ls, linewidth=1.1, alpha=0.92, zorder=3)

    for t in iou_markers:
        ax.axvline(t, color="#AAAAAA", linewidth=0.75, linestyle="--", zorder=2)

    curves_mean = [(name, mean, gidx) for name, mean, _, _, gidx in pct]
    for t in iou_markers:
        _annotate_threshold(ax, thresholds, curves_mean, t, y_lo, y_hi)

    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(y_lo, y_hi)
    ax.set_xlabel("IoU threshold")
    ax.set_ylabel("Precision (%)")

    major_step = 5.0 if y_span <= 30.0 else 10.0
    ax.yaxis.set_major_locator(mticker.MultipleLocator(major_step))
    ax.yaxis.set_minor_locator(mticker.MultipleLocator(major_step / 2))
    ax.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda v, _: f"{v:.0f}"))
    ax.xaxis.set_major_locator(mticker.MultipleLocator(0.25))
    ax.xaxis.set_minor_locator(mticker.MultipleLocator(0.125))
    ax.legend(loc="lower left", frameon=True)


# ---------------------------------------------------------------------------
# Main logic
# ---------------------------------------------------------------------------

def plot_cv_results(results_dirs, method_names, k, iou_markers, manifest_path, output_dir = None, fmt = "pdf", dpi = 200, matching_modes = None):
    _apply_style()
    thresholds = np.linspace(0, 1, 201)
    n_methods  = len(results_dirs)

    if matching_modes is None:
        matching_modes = ["table"] * n_methods
    else:
        if len(matching_modes) != n_methods:
            raise ValueError(f"--matching-modes must have the same number of entries as --results-dirs ({n_methods}), got {len(matching_modes)}.")
        for mode in matching_modes:
            if mode not in ("table", "region"):
                raise ValueError(f"Unknown matching mode '{mode}'. Valid values: 'table', 'region'.")

    fp_basename_to_uuid = _load_manifest(manifest_path)
    print(f"Manifest loaded: {len(fp_basename_to_uuid)} basename → UUID entries", file=sys.stderr)

    out_path = Path(output_dir) if output_dir is not None else None
    if out_path is not None:
        out_path.mkdir(parents=True, exist_ok=True)

    method_fold_curves = {}
    for global_idx in range(n_methods):
        method_fold_curves[global_idx] = ([], [])

    for fold_num in range(1, k + 1):
        # Step 1: load records + extract canonical keys
        method_data = []

        for global_idx, (results_dir, name, mode) in enumerate(
                zip(results_dirs, method_names, matching_modes)):
            fold_dir = Path(results_dir) / f"fold_{fold_num:02d}"
            if not fold_dir.exists():
                print(f"  [warn] {fold_dir} not found: skipping '{name}' for fold {fold_num}", file=sys.stderr)
                continue

            per_file = _load_per_file(fold_dir)
            if per_file is None:
                print(f"  [warn] no data in {fold_dir}: skipping '{name}' for fold {fold_num}", file=sys.stderr)
                continue

            keys = [_canonical_key(r, fp_basename_to_uuid) for r in per_file]
            n_unmapped = sum(1 for k_ in keys if k_ is None)
            if n_unmapped:
                print(f"  [warn] '{name}' fold {fold_num}: {n_unmapped}/{len(keys)} records could not be mapped to a UUID", file=sys.stderr)

            method_data.append((global_idx, name, mode, per_file, keys))

        if not method_data:
            print(f"  [skip] no data for fold {fold_num}", file=sys.stderr)
            continue

        # Step 2: intersect canonical keys
        key_sets = [
            set(k_ for k_ in keys if k_ is not None)
            for _, _, _, _, keys in method_data
        ]
        common_keys = key_sets[0]
        for s in key_sets[1:]:
            common_keys = common_keys & s

        print(f"Fold {fold_num}: intersection of {[len(s) for s in key_sets]} files = {len(common_keys)} files")

        if not common_keys:
            print(f"  [skip] empty intersection for fold {fold_num}", file=sys.stderr)
            continue

        # Step 3: filter to common subset and compute curves
        for global_idx, name, mode, per_file, keys in method_data:
            filtered = [
                rec for rec, k_ in zip(per_file, keys)
                if k_ in common_keys
            ]
            micro, macro = _compute_curves(
                filtered, thresholds, gt_anchored=(mode == "region")
            )
            method_fold_curves[global_idx][0].append(micro)
            method_fold_curves[global_idx][1].append(macro)

    micro_entries = []
    macro_entries = []

    for global_idx, name in enumerate(method_names):
        fold_micros, fold_macros = method_fold_curves[global_idx]
        if not fold_micros:
            print(f"  [skip] no valid folds found for '{name}'", file=sys.stderr)
            continue

        micro_stack = np.stack(fold_micros)   # (n_folds, n_thresholds)
        macro_stack = np.stack(fold_macros)

        micro_entries.append((
            name,
            micro_stack.mean(axis=0),
            micro_stack.min(axis=0),
            micro_stack.max(axis=0),
            global_idx,
        ))
        macro_entries.append((
            name,
            macro_stack.mean(axis=0),
            macro_stack.min(axis=0),
            macro_stack.max(axis=0),
            global_idx,
        ))

    if not micro_entries:
        print("[error] No data aggregated — check your --results-dirs.", file=sys.stderr)
        return

    for panel_tag, entries in (("micro", micro_entries), ("macro", macro_entries)):
        fig, ax = plt.subplots(figsize=(5.0, 3.6))
        panel_label = "Micro" if panel_tag == "micro" else "Macro"
        ax.set_ylabel(f"{panel_label} precision (%)")
        _draw_panel(ax, thresholds, entries, iou_markers)
        fig.tight_layout()

        if out_path is not None:
            _save(fig, out_path / f"{panel_tag}_precision.{fmt}", dpi=dpi)
            plt.close(fig)
        else:
            plt.show()
            plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser():
    p = argparse.ArgumentParser(
        description=(
            "Plot micro/macro precision curves aggregated across folds "
            "(intersection-aligned, with shaded min/max bands)."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--results-dirs", nargs="+", required=True, metavar="DIR",
        help="One result directory per method (each contains fold_01/, …).",
    )
    p.add_argument(
        "--method-names", nargs="+", required=True, metavar="NAME",
        help="Legend label for each result directory (same order).",
    )
    p.add_argument(
        "--k", type=int, required=True,
        help="Number of folds.",
    )
    p.add_argument(
        "--manifest", required=True, metavar="CSV",
        help=(
            "Manifest CSV with columns file_path,sheet_name,mime_type,labels_path. "
            "Used to map spreadsheet filenames to annotation UUIDs."
        ),
    )
    p.add_argument(
        "--iou-thresholds", nargs="+", type=float,
        default=[0.5, 0.75, 0.95], metavar="T",
        help="IoU thresholds for vertical markers (default: 0.5 0.75 0.95).",
    )
    p.add_argument(
        "--output-dir", default=None, metavar="DIR",
        help="Directory to save figures. Shown interactively when omitted.",
    )
    p.add_argument(
        "--format", default="pdf",
        choices=["pdf", "png", "svg", "eps"],
        help="Output file format (default: pdf).",
    )
    p.add_argument(
        "--dpi", type=int, default=200,
        help="Resolution for raster formats (default: 200).",
    )
    p.add_argument(
        "--matching-modes", nargs="+", metavar="MODE",
        default=None,
        help=(
            "Matching mode per method: 'table' (default, denominator = n_pred) "
            "or 'region' (GT-anchored, denominator = min(n_pred, n_gt)). "
            "Must have the same number of entries as --results-dirs."
        ),
    )
    return p


if __name__ == "__main__":
    args = _build_parser().parse_args()

    if len(args.results_dirs) != len(args.method_names):
        _build_parser().error(
            "--results-dirs and --method-names must have the same "
            "number of entries."
        )

    plot_cv_results(
        results_dirs = args.results_dirs,
        method_names = args.method_names,
        k = args.k,
        iou_markers = sorted(args.iou_thresholds),
        manifest_path = args.manifest,
        output_dir = args.output_dir,
        fmt = args.format,
        dpi = args.dpi,
        matching_modes = args.matching_modes,
    )