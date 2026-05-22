from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import dump as joblib_dump
from joblib import load as joblib_load
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import KFold

from .pipeline import load_and_split_dataset, _load_manifest

# ---------------------------------------------------------------------------
# Fixed experimental constants
# ---------------------------------------------------------------------------

SEED = 2112 # To replicate the paper's results, do not change
K = 5 
STATE_NAMES = ["EMPTY", "HEADER", "DATA", "TITLE", "OTHER"]

# For RF-Koci baseline, we remove indices 17, 18, 20 and the range [52, 67).
_KOCI_EXCLUDED = frozenset([17, 18, 20, *range(52, 67)])
_KOCI_FEATURE_INDICES = [i for i in range(67) if i not in _KOCI_EXCLUDED]

# ---------------------------------------------------------------------------
# Best hyperparameters (found by 5-fold CV tuning with seed=2112, see paper)
# ---------------------------------------------------------------------------

CONFIGS: dict[str, dict] = {
    "rf": {
        "description": "RF: Random Forest on all unary features",
        "n_estimators": 574,
        "max_depth": 20,
        "max_features": "sqrt",
        "min_samples_split": 18,
        "min_samples_leaf": 6,
        "bootstrap": False,
        "max_samples": None,
        "class_weight": "balanced",
        "feature_indices": None, # None = use all features
    },
    "rf-koci": {
        "description": "RF-Koci, Random Forest on filtered unary features",
        "n_estimators": 481,
        "max_depth": 20,
        "max_features": "log2",
        "min_samples_split": 17,
        "min_samples_leaf": 7,
        "bootstrap": True,
        "max_samples": None,
        "class_weight": "balanced",
        "feature_indices": _KOCI_FEATURE_INDICES,
    },
}

# ---------------------------------------------------------------------------
# Fold construction
# ---------------------------------------------------------------------------

def _make_folds(dataset_csv):
    """
    Shuffle the manifest with seed=2112 then split into K folds.
    """
    df = _load_manifest(dataset_csv)
    df_shuffled = df.sample(frac=1, random_state=SEED).reset_index(drop=True)
    kf = KFold(n_splits=K, shuffle=False)   # shuffle already done above
    return [
        (
            df_shuffled.iloc[train_idx].reset_index(drop=True),
            df_shuffled.iloc[val_idx].reset_index(drop=True),
        )
        for train_idx, val_idx in kf.split(df_shuffled)
    ]

# ---------------------------------------------------------------------------
# Feature selection helper
# ---------------------------------------------------------------------------

def _sel(X_nodes, feature_indices):
    return X_nodes if feature_indices is None else X_nodes[:, feature_indices]

def _stack_unary(X_list, y_list, feature_indices):
    """Flatten per-file (X_nodes, edges, pairwise) tuples into flat matrices."""
    X_parts, y_parts = [], []
    for i, (X_nodes, _edges, _pairwise) in enumerate(X_list):
        X_parts.append(_sel(X_nodes, feature_indices))
        y_parts.append(y_list[i])
    return np.concatenate(X_parts), np.concatenate(y_parts)

# ---------------------------------------------------------------------------
# Metrics (file-macro-F1 matching the CRF convention used in the paper)
# ---------------------------------------------------------------------------

def _per_file_metrics(clf, X_list, y_list, feature_indices, y_pred_list = None):
    """
    For each class c: compute F1 on every file where c appears in ground truth
    OR predictions, then average -> per_class_f1[c].
    macro_file_macro_f1 = mean of per_class_f1 values (NaN classes excluded).
    Files absent for a class contribute nothing (not 0).
    """
    preds = []
    all_labels = set()

    for i, (X_nodes, _e, _p) in enumerate(X_list):
        y_pred = (
            y_pred_list[i]
            if y_pred_list is not None
            else clf.predict(_sel(X_nodes, feature_indices))
        )
        preds.append(y_pred)
        all_labels.update(y_list[i].tolist())
        all_labels.update(y_pred.tolist())

    all_labels_sorted = sorted(all_labels)
    n_cls = len(all_labels_sorted)
    n_files = len(X_list)
    f1_mat = np.zeros((n_files, n_cls), dtype=float)
    valid = np.zeros((n_files, n_cls), dtype=bool)
    per_file_f1 = []

    for f, (y_true, y_pred) in enumerate(zip(y_list, preds)):
        for c, lbl in enumerate(all_labels_sorted):
            gt = y_true == lbl
            pr = y_pred == lbl
            TP = np.sum(gt & pr)
            FP = np.sum(~gt & pr)
            FN = np.sum(gt & ~pr)
            d  = 2 * TP + FP + FN
            if d > 0:
                valid[f, c]  = True
                f1_mat[f, c] = (2 * TP) / d
        m = valid[f]
        per_file_f1.append(float(np.mean(f1_mat[f, m])) if m.any() else 0.0)

    per_class_arr = np.full(n_cls, np.nan)
    for c in range(n_cls):
        m = valid[:, c]
        if m.any():
            per_class_arr[c] = np.mean(f1_mat[m, c])
    macro_f1 = float(np.mean(per_class_arr[~np.isnan(per_class_arr)]))

    per_class_f1: dict[str, float | None] = {}
    for c, lbl in enumerate(all_labels_sorted):
        key = STATE_NAMES[lbl] if lbl < len(STATE_NAMES) else str(lbl)
        v   = per_class_arr[c]
        per_class_f1[key] = float(v) if not np.isnan(v) else None

    return macro_f1, per_file_f1, per_class_f1


def _evaluate(clf, X_list, y_list, split_name, feature_indices, y_pred_flat = None):
    print(f"\n{'='*70}\nEVALUATION: {split_name}\n{'='*70}")

    X_flat, y_true_flat = _stack_unary(X_list, y_list, feature_indices)
    if y_pred_flat is None:
        y_pred_flat = clf.predict(X_flat)

    acc = accuracy_score(y_true_flat, y_pred_flat)
    labels = list(range(len(STATE_NAMES)))
    report_str = classification_report(
        y_true_flat, y_pred_flat,
        labels=labels, target_names=STATE_NAMES, zero_division=0,
    )
    report_dict = classification_report(
        y_true_flat, y_pred_flat,
        labels=labels, target_names=STATE_NAMES,
        output_dict=True, zero_division=0,
    )
    cm = confusion_matrix(y_true_flat, y_pred_flat, labels=labels)

    # Split flat predictions back to per-file arrays (for _per_file_metrics)
    sizes = [X[0].shape[0] for X in X_list]
    y_pred_list: list[np.ndarray] = []
    offset = 0
    for s in sizes:
        y_pred_list.append(y_pred_flat[offset: offset + s])
        offset += s

    macro_f1, per_file_f1, per_class_f1 = _per_file_metrics(clf, X_list, y_list, feature_indices, y_pred_list=y_pred_list)

    print(f"\nCell-level accuracy : {acc:.4f}")
    print(
        f"File-macro-F1       : {macro_f1:.4f}  "
        f"(mean of per-class F1s, each averaged over files where class appears)"
    )
    print(f"\nCell-level classification report:\n{report_str}")
    print(f"Confusion matrix (rows=true, cols=pred):\n{cm}")
    print(f"\nPer-class file-macro F1:")
    for cls, f1 in per_class_f1.items():
        print(f"  {cls:<14s}  {f1:.4f}" if f1 is not None else f"  {cls:<14s}  N/A")

    return {
        "accuracy": float(acc),
        "classification_report": report_dict,
        "confusion_matrix": cm.tolist(),
        "macro_file_macro_f1": macro_f1,
        "per_file_f1_list": per_file_f1,
        "per_class_file_macro_f1": per_class_f1,
    }

# ---------------------------------------------------------------------------
# Grid saving
# ---------------------------------------------------------------------------

def _save_predicted_grids(y_pred_flat, grid_shapes_val, val_labels_paths, out_dir):
    """
    Write one <UUID>.npz per val file with a 'labels' array matching the
    ground-truth shape.  Returns a manifest dict for logging.
    """
    manifest: dict = {}
    offset = 0

    for grid_shape, labels_path in zip(grid_shapes_val, val_labels_paths):
        n_cells = grid_shape[0] * grid_shape[1]
        y_pred_file = y_pred_flat[offset: offset + n_cells]
        offset += n_cells

        uuid = os.path.basename(labels_path).split("annotations_")[1].split(".npz")[0]

        try:
            gt_labels = np.load(labels_path, allow_pickle=False)["labels"]
            gt_shape  = gt_labels.shape
        except Exception as exc:
            print(f"  [WARN] Cannot load GT NPZ '{labels_path}': {exc}. Using grid_shape.")
            gt_shape = grid_shape

        expected_cells = gt_shape[0] * gt_shape[1]
        if expected_cells != n_cells:
            print(
                f"  [WARN] UUID={uuid}: grid_shape {grid_shape} gives {n_cells} cells "
                f"but GT shape {gt_shape} needs {expected_cells}.  Skipping."
            )
            manifest[uuid] = {
                "shape": list(gt_shape), "labels_path": str(labels_path),
                "status": "shape_mismatch",
            }
            continue

        np.savez(out_dir / f"{uuid}.npz", labels=y_pred_file.reshape(gt_shape))
        manifest[uuid] = {
            "shape": list(gt_shape), "labels_path": str(labels_path), "status": "ok",
        }

    if offset != len(y_pred_flat):
        print(
            f"  [WARN] Consumed {offset}/{len(y_pred_flat)} predicted cells — "
            f"possible mismatch between grid_shapes_val and y_pred_flat."
        )
    return manifest

# ---------------------------------------------------------------------------
# CV summary aggregation
# ---------------------------------------------------------------------------

def _aggregate(fold_val_metrics):
    def _agg(key):
        vals = [
            m[key] for m in fold_val_metrics
            if m and key in m and m[key] is not None
        ]
        if not vals:
            return {"mean": None, "std": None, "folds": []}
        return {"mean": float(np.mean(vals)), "std": float(np.std(vals)), "folds": vals}

    agg = {"accuracy": _agg("accuracy"), "macro_file_macro_f1": _agg("macro_file_macro_f1")}

    all_cls = []
    seen = set()
    for m in fold_val_metrics:
        for cls in (m.get("per_class_file_macro_f1") or {}):
            if cls not in seen:
                all_cls.append(cls)
                seen.add(cls)

    per_class_agg = {}
    for cls in all_cls:
        scores = [
            m["per_class_file_macro_f1"][cls]
            for m in fold_val_metrics
            if m and (m.get("per_class_file_macro_f1") or {}).get(cls) is not None
        ]
        per_class_agg[cls] = {
            "mean": float(np.mean(scores)) if scores else None,
            "std": float(np.std(scores))  if scores else None,
            "folds": scores,
        }

    return agg, per_class_agg


def _print_cv_summary(variant, agg, per_class_agg):
    print(f"\n{'='*70}")
    print(f"CV SUMMARY  variant={variant}  k={K}  seed={SEED}")
    print(f"{'='*70}")
    for metric, stats in agg.items():
        if not stats["folds"]:
            continue
        fold_str = "  ".join(f"{v:.4f}" for v in stats["folds"])
        print(f"  {metric:30s}  mean={stats['mean']:.4f}  std={stats['std']:.4f}")
        print(f"    per-fold: {fold_str}")
    if per_class_agg:
        print(f"\n  Per-class file-macro F1:")
        for cls, stats in per_class_agg.items():
            if not stats["folds"]:
                continue
            fold_str = "  ".join(f"{v:.4f}" for v in stats["folds"])
            print(f"    {cls:<14s}  mean={stats['mean']:.4f}  std={stats['std']:.4f}")
            print(f"      per-fold: {fold_str}")

# ---------------------------------------------------------------------------
# train subcommand
# ---------------------------------------------------------------------------

def cmd_train(args):
    cfg = CONFIGS[args.variant]
    feature_indices = cfg["feature_indices"]

    # max_samples is only meaningful when bootstrap=True
    effective_max_samples = cfg["max_samples"] if cfg["bootstrap"] else None

    folds = _make_folds(args.dataset)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    n_features_used = len(feature_indices) if feature_indices is not None else 67

    print(f"\n{'='*70}")
    print(f"RF K-FOLD CV (train)  variant={args.variant}  k={K}  seed={SEED}")
    print(f"  {cfg['description']}")
    print(f"  features used    : {n_features_used}")
    print(f"  n_estimators     : {cfg['n_estimators']}")
    print(f"  max_depth        : {cfg['max_depth']}")
    print(f"  max_features     : {cfg['max_features']}")
    print(f"  bootstrap        : {cfg['bootstrap']}")
    print(f"  max_samples      : {effective_max_samples}")
    print(f"  min_samples_split: {cfg['min_samples_split']}")
    print(f"  min_samples_leaf : {cfg['min_samples_leaf']}")
    print(f"  class_weight     : {cfg['class_weight']}")
    print(f"  save_fold_models : {args.save_fold_models}")
    print(f"{'='*70}")

    fold_val_metrics = []
    fold_load_times = []
    fold_train_times = []
    t_grand_start = time.time()

    for fold_idx, (train_df, val_df) in enumerate(folds):
        fold_num = fold_idx + 1
        fold_dir = output_dir / f"fold_{fold_num:02d}"
        fold_dir.mkdir(exist_ok=True)
        train_df.to_csv(fold_dir / "train_manifest.csv", index=False)
        val_df.to_csv(fold_dir / "val_manifest.csv",   index=False)

        print(f"\n\n{'#'*70}")
        print(f"  FOLD {fold_num}/{K}:  train={len(train_df)} rows  val={len(val_df)} rows")
        print(f"{'#'*70}")

        # Load features + labels 
        print(f"\n[Fold {fold_num}]  Loading features …")
        t0 = time.time()
        dataset = load_and_split_dataset(
            dataset_csv=args.dataset,
            validation_split=0.0,
            random_state=SEED,
            max_grid_size=args.max_grid_size,
            early_stopping=True,
            feature_cache_dir=args.feature_cache,
            train_df=train_df,
            val_df=val_df,
            rf_model=None,
            n_states=5,
            n_workers=args.n_jobs,
        )
        t_load = time.time() - t0
        fold_load_times.append(t_load)
        print(f"  [TIMING] loading + feature extraction: {t_load:.2f}s")

        X_val_list = dataset["X_val"]   or []
        y_val_list = dataset["y_val"]   or []

        X_train_flat, y_train_flat = _stack_unary(
            dataset["X_train"], dataset["y_train"], feature_indices,
        )
        print(f"  train files: {len(dataset['X_train'])}  "
              f"train cells: {X_train_flat.shape[0]:,}  "
              f"features: {X_train_flat.shape[1]}")
        print(f"  val files  : {len(X_val_list)}")

        #  Train RF (timed)
        clf = RandomForestClassifier(
            n_estimators = cfg["n_estimators"],
            max_depth = cfg["max_depth"],
            max_features = cfg["max_features"],
            min_samples_split = cfg["min_samples_split"],
            min_samples_leaf = cfg["min_samples_leaf"],
            bootstrap = cfg["bootstrap"],
            max_samples = effective_max_samples,
            class_weight = cfg["class_weight"],
            random_state = SEED,
            n_jobs = args.n_jobs,
        )
        print(f"\n[Fold {fold_num}]  Training Random Forest …")
        t0 = time.time()
        clf.fit(X_train_flat, y_train_flat)
        t_train = time.time() - t0
        fold_train_times.append(t_train)
        print(f"  [TIMING] training: {t_train:.2f}s")

        # Evaluate on val fold
        val_metrics = None
        if X_val_list:
            X_val_flat, _ = _stack_unary(X_val_list, y_val_list, feature_indices)
            y_pred_val = clf.predict(X_val_flat)
            val_metrics = _evaluate(
                clf, X_val_list, y_val_list,
                f"FOLD {fold_num} VALIDATION SET",
                feature_indices, y_pred_flat=y_pred_val,
            )
        fold_val_metrics.append(val_metrics or {})

        # Feature importances
        importances = clf.feature_importances_
        top20 = np.argsort(importances)[::-1][:20]
        print(f"\n  Top-20 feature importances (Gini):")
        for rank, fi in enumerate(top20, 1):
            orig = feature_indices[fi] if feature_indices is not None else fi
            print(f"    #{rank:>2}  feature[{orig:>3}]  {importances[fi]:.5f}")

        # Optionally save model
        if args.save_fold_models:
            joblib_dump(clf, fold_dir / "model.joblib")
            print(f"  Model saved: {fold_dir / 'model.joblib'}")

        # Per-fold log
        fold_log = {
            "fold": fold_num,
            "k": K,
            "seed": SEED,
            "variant": args.variant,
            "n_train_rows": len(train_df),
            "n_val_rows": len(val_df),
            "feature_indices": feature_indices,
            "n_features_used": n_features_used,
            "params": {
                k: v for k, v in cfg.items()
                if k not in ("description", "feature_indices")
            },
            "timing": {
                "load_and_extraction_s": t_load,
                "train_s": t_train,
            },
            "val_metrics": val_metrics,
            "feature_importances": {
                f"feature_{feature_indices[i] if feature_indices else i}": float(v)
                for i, v in enumerate(importances)
            },
        }
        with open(fold_dir / "log.json", "w") as fh:
            json.dump(fold_log, fh, indent=2)

    t_grand = time.time() - t_grand_start

    # Timing summary
    print(f"\n{'='*70}")
    print(f"TIMING SUMMARY  variant={args.variant}  k={K}")
    print(f"{'='*70}")
    print(f"  {'Fold':>6}  {'(a) load+extract':>18}  {'(b) training':>14}")
    for i, (tl, tt) in enumerate(zip(fold_load_times, fold_train_times), 1):
        print(f"  {i:>6}  {tl:>18.2f}s  {tt:>14.2f}s")
    print(f"  {'Total':>6}  {sum(fold_load_times):>18.2f}s  {sum(fold_train_times):>14.2f}s")
    print(f"  Grand total: {t_grand:.2f}s")

    # Aggregate and print CV summary
    agg, per_class_agg = _aggregate(fold_val_metrics)
    _print_cv_summary(args.variant, agg, per_class_agg)

    summary = {
        "variant": args.variant,
        "k": K,
        "seed": SEED,
        "n_features_used": n_features_used,
        "feature_indices": feature_indices,
        "params": {
            k: v for k, v in cfg.items()
            if k not in ("description", "feature_indices")
        },
        "timing": {
            "fold_load_and_extraction_s": fold_load_times,
            "fold_train_s": fold_train_times,
            "total_load_and_extraction_s": sum(fold_load_times),
            "total_train_s": sum(fold_train_times),
            "grand_total_s": t_grand,
        },
        "aggregated_val_metrics": agg,
        "per_class_file_macro_f1": per_class_agg,
    }
    summary_path = output_dir / f"cv_summary_{args.variant}_k{K}_seed{SEED}.json"
    with open(summary_path, "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"\nCV summary → {summary_path}")

# ---------------------------------------------------------------------------
# infer subcommand
# ---------------------------------------------------------------------------

def cmd_infer(args):
    cfg = CONFIGS[args.variant]
    feature_indices = cfg["feature_indices"]
    cv_dir = Path(args.cv_dir)
    output_dir = Path(args.output) if args.output else None
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)

    fold_load_times = []
    fold_infer_times = []
    fold_results = []
    t_grand_start = time.time()

    print(f"\n{'='*70}")
    print(f"RF K-FOLD INFERENCE  variant={args.variant}  k={K}  seed={SEED}")
    print(f"  {cfg['description']}")
    print(f"{'='*70}")

    for fold_num in range(1, K + 1):
        fold_dir = cv_dir / f"fold_{fold_num:02d}"
        model_path = fold_dir / "model.joblib"
        val_manifest_path = fold_dir / "val_manifest.csv"

        print(f"\n{'#'*70}\n  FOLD {fold_num}/{K}\n{'#'*70}")

        if not model_path.exists():
            print(f"  [SKIP] {model_path} not found — re-run train with --save-fold-models")
            continue
        if not val_manifest_path.exists():
            print(f"  [SKIP] {val_manifest_path} not found")
            continue

        val_df = _load_manifest(val_manifest_path)
        empty_train_df  = pd.DataFrame(columns=val_df.columns)

        # Load val features (timed)
        print(f"\n[Fold {fold_num}]  Loading val features ...")
        t0 = time.time()
        dataset = load_and_split_dataset(
            dataset_csv=None, # not needed when both DFs are supplied
            validation_split=0.0,
            random_state=SEED,
            max_grid_size=args.max_grid_size,
            early_stopping=True,
            feature_cache_dir=args.feature_cache,
            train_df=empty_train_df,
            val_df=val_df,
            rf_model=None,
            n_states=5,
            n_workers=args.n_jobs,
        )
        t_load = time.time() - t0
        fold_load_times.append(t_load)
        print(f"  [TIMING] loading + feature extraction: {t_load:.2f}s")

        X_val_list = dataset.get("X_val") or []
        y_val_list = dataset.get("y_val") or []
        grid_shapes_val = dataset.get("grid_shapes_val") or []
        val_labels_paths = dataset.get("val_labels_paths") or []

        if not X_val_list:
            print(f"  [WARN] No val samples loaded for fold {fold_num}, skipping.")
            fold_infer_times.append(0.0)
            continue

        X_val_flat, y_val_flat = _stack_unary(X_val_list, y_val_list, feature_indices)
        print(f"  {len(X_val_list)} val files  |  {X_val_flat.shape[0]:,} cells  "
              f"|  {X_val_flat.shape[1]} features")

        n_skipped = len(val_df) - len(X_val_list)
        if n_skipped > 0:
            print(f"  [INFO] {n_skipped} val file(s) could not be loaded "
                  f"and will be absent from predicted_grids/.")

        clf = joblib_load(model_path)

        # Inference — only clf.predict is timed
        print(f"\n[Fold {fold_num}]  Running inference ...")
        t0 = time.time()
        y_pred_flat = clf.predict(X_val_flat)
        t_infer = time.time() - t0
        fold_infer_times.append(t_infer)
        print(f"  [TIMING] inference (predict only): {t_infer:.2f}s")

        # Metrics (not timed)
        val_metrics = _evaluate(
            clf, X_val_list, y_val_list,
            f"FOLD {fold_num} VALIDATION SET",
            feature_indices, y_pred_flat=y_pred_flat,
        )

        # Save predicted grids
        grid_manifest = {}
        if output_dir is not None:
            fold_out = output_dir / f"fold_{fold_num:02d}"
            grids_dir = fold_out / "predicted_grids"
            grids_dir.mkdir(parents=True, exist_ok=True)
            print(f"\n[Fold {fold_num}]  Saving predicted grids → {grids_dir}")

            grid_manifest = _save_predicted_grids(y_pred_flat, grid_shapes_val, val_labels_paths, grids_dir)
            n_ok = sum(1 for v in grid_manifest.values() if v["status"] == "ok")
            print(f"  {n_ok}/{len(grid_manifest)} grids saved.")
            with open(grids_dir / "manifest.json", "w") as fh:
                json.dump(grid_manifest, fh, indent=2)

        fold_results.append({
            "fold": fold_num,
            "n_val_files": len(X_val_list),
            "n_val_cells": int(X_val_flat.shape[0]),
            "timing": {
                "load_and_extraction_s": t_load,
                "inference_s": t_infer,
            },
            "val_metrics": val_metrics,
            "grid_manifest": grid_manifest,
        })

    t_grand = time.time() - t_grand_start

    # Timing summary
    print(f"\n{'='*70}")
    print(f"INFERENCE TIMING SUMMARY  variant={args.variant}  k={K}")
    print(f"{'='*70}")
    print(f"  {'Fold':>6}  {'(a) load+extract':>18}  {'(b) inference':>14}")
    for i, (tl, ti) in enumerate(zip(fold_load_times, fold_infer_times), 1):
        print(f"  {i:>6}  {tl:>18.2f}s  {ti:>14.2f}s")
    if fold_load_times:
        print(f"  {'Total':>6}  {sum(fold_load_times):>18.2f}s  "
              f"{sum(fold_infer_times):>14.2f}s")
    print(f"  Grand total: {t_grand:.2f}s")

    result = {
        "variant": args.variant,
        "k": K,
        "seed": SEED,
        "timing": {
            "fold_load_and_extraction_s": fold_load_times,
            "fold_inference_s": fold_infer_times,
            "total_load_and_extraction_s": sum(fold_load_times) if fold_load_times else 0.0,
            "total_inference_s": sum(fold_infer_times) if fold_infer_times else 0.0,
            "grand_total_s": t_grand,
        },
        "fold_results": fold_results,
    }
    if output_dir is not None:
        timing_path = output_dir / "inference_timing.json"
        with open(timing_path, "w") as fh:
            json.dump(result, fh, indent=2)
        print(f"\nInference timing: {timing_path}")

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = p.add_subparsers(dest="command", help="Subcommand")

    def _add_shared(sp: argparse.ArgumentParser) -> None:
        sp.add_argument(
            "--variant", required=True, choices=list(CONFIGS),
            help="'rf' (all 67 features) or 'rf-koci' (49 filtered features)",
        )
        sp.add_argument(
            "--feature-cache", default=None, metavar="DIR",
            help="Directory for cached .npz feature files (speeds up re-runs)",
        )
        sp.add_argument(
            "--max-grid-size", type=int, default=2500,
            help="Training grids with more than this many cells are skipped (default: 2500)",
        )

    # train
    tr = sub.add_parser(
        "train",
        help="5-fold CV: train on each fold and evaluate on the held-out val fold.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=(
            "5-fold CV.\n\n"
            "Per-fold timing:\n"
            "  (a) loading + feature extraction  (parallel, all CPU cores)\n"
            "  (b) RandomForest training\n\n"
            "Outputs per fold: fold_NN/log.json, fold_NN/train_manifest.csv,\n"
            "                  fold_NN/val_manifest.csv, [fold_NN/model.joblib]\n"
            "Global output:    cv_summary_<variant>_k5_seed2112.json"
        ),
    )
    tr.add_argument("--dataset",   required=True, metavar="CSV",
                    help="Path to the manifest CSV (e.g. dataset/manifest.csv)")
    tr.add_argument("--output",    required=True, metavar="DIR",
                    help="Output directory")
    tr.add_argument("--save-fold-models", action="store_true",
                    help="Persist model.joblib per fold (required for the infer subcommand)")
    tr.add_argument("--n-jobs", type=int, default=-1,
                    help="Parallel jobs for RandomForestClassifier (default: -1 = all cores)")
    _add_shared(tr)

    # infer
    inf = sub.add_parser(
        "infer",
        help="Fold-by-fold inference on models saved by a prior train run.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=(
            "Fold-by-fold inference.\n\n"
            "Requires a prior train run with --save-fold-models.\n\n"
            "Per-fold timing:\n"
            "  (a) loading + feature extraction  (parallel, all CPU cores)\n"
            "  (b) clf.predict only\n\n"
            "Outputs: output/fold_NN/predicted_grids/<UUID>.npz\n"
            "         output/fold_NN/predicted_grids/manifest.json\n"
            "         output/inference_timing.json"
        ),
    )
    inf.add_argument(
        "cv_dir", metavar="CV_DIR",
        help="Directory produced by the train subcommand (contains fold_01/, fold_02/, …)",
    )
    inf.add_argument(
        "--output", default=None, metavar="DIR",
        help="Directory for predicted grids and timing JSON",
    )
    _add_shared(inf)
    return p


if __name__ == "__main__":
    parser = _build_parser()
    args   = parser.parse_args()
    if args.command is None:
        parser.print_help()
        raise SystemExit(1)
    if args.command == "train":
        cmd_train(args)
    elif args.command == "infer":
        cmd_infer(args)