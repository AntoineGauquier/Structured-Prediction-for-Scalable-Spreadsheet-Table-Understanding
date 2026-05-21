from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from time import time

import numpy as np
import pandas as pd
from joblib import dump as joblib_dump
from joblib import load as joblib_load
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold

from .pipeline import _apply_rf_projection, _build_val_metrics, _validate_rf_model, load_and_split_dataset, train_from_preloaded
from .crf_model import CellTypeClassifierCRF

# ---------------------------------------------------------------------------
# Fixed constants
# ---------------------------------------------------------------------------

SEED = 2112 # To replicate the paper's results, do not change
K = 5
STATE_NAMES = ["EMPTY", "HEADER", "DATA", "TITLE", "OTHER"]

# ---------------------------------------------------------------------------
# RF hyperparameters (hardcoded, best config from standalone RF baseline)
# The RF component inside CRF-RF uses the same hyperparameters as the standalone
# RF baseline: they were selected as globally best across folds before being
# re-used here. See paper for tuning details.
# ---------------------------------------------------------------------------

RF_CONFIG: dict = {
    "description": "RandomForest on all unary features (best params from paper)",
    "n_estimators": 574,
    "max_depth": 20,
    "max_features": "sqrt",
    "min_samples_split": 18,
    "min_samples_leaf":  6,
    "bootstrap": False,
    "max_samples": None,
    "class_weight": "balanced",
}

# ---------------------------------------------------------------------------
# Fold construction (identical across all baselines)
# ---------------------------------------------------------------------------

def _make_folds(dataset_csv):
    df = pd.read_csv(dataset_csv)
    df_shuffled = df.sample(frac=1, random_state=SEED).reset_index(drop=True)
    kf = KFold(n_splits=K, shuffle=False)
    return [
        (
            df_shuffled.iloc[train_idx].reset_index(drop=True),
            df_shuffled.iloc[val_idx].reset_index(drop=True),
        )
        for train_idx, val_idx in kf.split(df_shuffled)
    ]

# ---------------------------------------------------------------------------
# RF helpers
# ---------------------------------------------------------------------------

def _stack_unary(X_list, y_list):
    X_parts, y_parts = [], []
    for (X_nodes, _edges, _pairwise), y in zip(X_list, y_list):
        X_parts.append(X_nodes)
        y_parts.append(y)
    return np.concatenate(X_parts, axis=0), np.concatenate(y_parts, axis=0)


def _train_fold_rf(X_list, y_list, n_jobs):
    """Fit a RandomForestClassifier using the hardcoded RF_CONFIG parameters."""
    X_flat, y_flat = _stack_unary(X_list, y_list)
    clf = RandomForestClassifier(
        n_estimators = RF_CONFIG["n_estimators"],
        max_depth = RF_CONFIG["max_depth"],
        max_features = RF_CONFIG["max_features"],
        min_samples_split = RF_CONFIG["min_samples_split"],
        min_samples_leaf = RF_CONFIG["min_samples_leaf"],
        bootstrap = RF_CONFIG["bootstrap"],
        max_samples = RF_CONFIG["max_samples"],
        class_weight = RF_CONFIG["class_weight"],
        random_state = SEED,
        n_jobs = n_jobs,
    )
    clf.fit(X_flat, y_flat)
    return clf


def _apply_rf_to_dataset(dataset, rf_model):
    """
    Replace raw unary features with RF predict_proba scores (N, n_classes)
    for both X_train and X_val.  Pairwise features and edges are unchanged.
    """
    def _project(X_list: list) -> list:
        return [
            (_apply_rf_projection(rf_model, X_nodes), edges, pairwise)
            for X_nodes, edges, pairwise in X_list
        ]

    projected = dict(dataset)
    projected["X_train"] = _project(dataset["X_train"])
    if dataset.get("X_val"):
        projected["X_val"] = _project(dataset["X_val"])
    projected["n_unary_features"] = len(rf_model.classes_)
    return projected

# ---------------------------------------------------------------------------
# Grid saving
# ---------------------------------------------------------------------------

def _save_predicted_grids(y_pred_list, grid_shapes_val, val_labels_paths, out_dir):
    """
    Write one <UUID>.npz per val file.

    Unlike RF/LightGBM (which work from a flat prediction array), the CRF's
    _predict_parallel already returns per-file 2-D (H, W) arrays — no flat
    slicing needed.
    """
    manifest: dict = {}

    for y_pred, grid_shape, labels_path in zip(
        y_pred_list, grid_shapes_val, val_labels_paths
    ):
        uuid = os.path.basename(labels_path).split("annotations_")[1].split(".npz")[0]

        try:
            gt_labels = np.load(labels_path, allow_pickle=False)["labels"]
            gt_shape = gt_labels.shape
        except Exception as exc:
            print(f"  [WARN] Cannot load GT NPZ '{labels_path}': {exc}. Using grid_shape.")
            gt_shape = grid_shape

        n_cells = grid_shape[0] * grid_shape[1]
        expected = gt_shape[0] * gt_shape[1]
        if expected != n_cells:
            print(
                f"  [WARN] UUID={uuid}: grid_shape {grid_shape} gives {n_cells} cells "
                f"but GT shape {gt_shape} needs {expected}. Skipping."
            )
            manifest[uuid] = {
                "shape": list(gt_shape), "labels_path": str(labels_path),
                "status": "shape_mismatch",
            }
            continue

        np.savez(out_dir / f"{uuid}.npz", labels=y_pred.reshape(gt_shape))
        manifest[uuid] = {
            "shape": list(gt_shape), "labels_path": str(labels_path), "status": "ok",
        }

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

    agg = {
        "accuracy": _agg("accuracy"),
        "macro_file_macro_f1": _agg("macro_file_macro_f1"),
    }

    all_cls = []
    seen = set()
    for m in fold_val_metrics:
        for cls in (m.get("per_class_f1") or {}):
            if cls not in seen:
                all_cls.append(cls)
                seen.add(cls)

    per_class_agg: dict = {}
    for cls in all_cls:
        scores = [
            m["per_class_f1"][cls]
            for m in fold_val_metrics
            if m and (m.get("per_class_f1") or {}).get(cls) is not None
        ]
        per_class_agg[cls] = {
            "mean": float(np.mean(scores)) if scores else None,
            "std": float(np.std(scores))  if scores else None,
            "folds": scores,
        }

    return agg, per_class_agg


def _print_cv_summary(agg, per_class_agg):
    print(f"\n{'='*70}")
    print(f"CV SUMMARY  model=RF-CRF  k={K}  seed={SEED}")
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
# JSON serialization helper
# ---------------------------------------------------------------------------

def _json_default(o):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)

# ---------------------------------------------------------------------------
# train subcommand
# ---------------------------------------------------------------------------

def cmd_train(args):
    folds = _make_folds(args.dataset)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    crf_config = {
        "inference_method": args.inference,
        "C": args.C,
        "max_iter": args.max_iter,
        "patience": args.patience,
        "val_check_every": args.val_check_every,
        "min_delta": args.min_delta,
        "batch_size": args.batch_size,
        "class_balanced": False,
        "weighting_strategy": "none",
        "constrain_empty": False,
    }

    print(f"\n{'='*70}")
    print(f"RF-CRF K-FOLD CV (train)  k={K}  seed={SEED}")
    print(f"  {RF_CONFIG['description']}")
    print(f"  RF  n_estimators={RF_CONFIG['n_estimators']}  max_depth={RF_CONFIG['max_depth']}"
          f"  max_features={RF_CONFIG['max_features']}  bootstrap={RF_CONFIG['bootstrap']}")
    print(f"  CRF C={args.C}  max_iter={args.max_iter}  inference={args.inference}"
          f"  batch_size={args.batch_size}")
    print(f"  Early stopping: patience={args.patience}  val_check_every={args.val_check_every}"
          f"  min_delta={args.min_delta}")
    print(f"  class_balanced=False  constrain_empty=False")
    print(f"  save_fold_models: {args.save_fold_models}")
    print(f"{'='*70}")

    fold_val_metrics = []
    fold_load_times = []
    fold_train_times = []
    t_grand_start = time()

    for fold_idx, (train_df, val_df) in enumerate(folds):
        fold_num = fold_idx + 1
        fold_dir = output_dir / f"fold_{fold_num:02d}"
        fold_dir.mkdir(exist_ok=True)
        train_df.to_csv(fold_dir / "train_manifest.csv", index=False)
        val_df.to_csv(fold_dir   / "val_manifest.csv",   index=False)

        print(f"\n\n{'#'*70}")
        print(f"  FOLD {fold_num}/{K}  —  train={len(train_df)} rows  val={len(val_df)} rows")
        print(f"{'#'*70}")

        # Load raw features + labels
        print(f"\n[Fold {fold_num}]  Loading features ...")
        t0 = time()
        raw_dataset = load_and_split_dataset(
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
        t_load = time() - t0
        fold_load_times.append(t_load)
        print(f"  [TIMING] (a) loading + feature extraction: {t_load:.2f}s")
        print(f"  train files: {len(raw_dataset['X_train'])}  "
              f"val files: {len(raw_dataset['X_val'] or [])}")

        # RF fit -> project -> CRF fit
        t0 = time()

        print(f"\n[Fold {fold_num}/1]  Training fold RF ({RF_CONFIG['n_estimators']} trees) …")
        fold_rf = _train_fold_rf(raw_dataset["X_train"], raw_dataset["y_train"],
                                 n_jobs=args.n_jobs)
        _validate_rf_model(fold_rf, 5)
        X_flat_train, y_flat_train = _stack_unary(raw_dataset["X_train"], raw_dataset["y_train"])
        print(f"  RF fitted on {X_flat_train.shape[0]:,} cells  ({X_flat_train.shape[1]} features)")

        print(f"\n[Fold {fold_num}/2]  Projecting features via RF predict_proba …")
        projected = _apply_rf_to_dataset(raw_dataset, fold_rf)
        print(f"  Unary dim: {raw_dataset['n_unary_features']} → "
              f"{projected['n_unary_features']}  (RF class probabilities)")

        print(f"\n[Fold {fold_num}/3]  Training CRF …")
        fold_crf_path = str(fold_dir / "crf_model.npz")

        _, fold_log = train_from_preloaded(
            dataset=projected,
            output_model_path=fold_crf_path,
            n_states=5,
            inference_method=args.inference,
            random_state=SEED,
            C=args.C,
            max_iter=args.max_iter,
            tol=0.0,
            batch_size=args.batch_size,
            class_balanced=False,
            weighting_strategy="none",
            check_convergence_every=0,
            early_stopping=True,
            patience=args.patience,
            val_check_every=args.val_check_every,
            min_delta=args.min_delta,
            n_jobs=args.n_jobs,
            verbose=1,
            constrain_empty=False,
            state_names=STATE_NAMES,
        )

        t_train = time() - t0
        fold_train_times.append(t_train)
        print(f"  [TIMING] (b) training (RF + projection + CRF): {t_train:.2f}s")

        # Model persistence
        if args.save_fold_models:
            joblib_dump(fold_rf, fold_dir / "rf_model.joblib")
            print(f"  RF saved: {fold_dir / 'rf_model.joblib'}")
            print(f"  CRF saved: {fold_dir / 'crf_model.npz'}")
        else:
            p = Path(fold_crf_path)
            if p.exists():
                p.unlink()

        # Per-fold log
        fold_log["fold"] = fold_num
        fold_log["k"] = K
        fold_log["seed"] = SEED
        fold_log["n_train_rows"] = len(train_df)
        fold_log["n_val_rows"] = len(val_df)
        fold_log["rf_config"] = {k: v for k, v in RF_CONFIG.items() if k != "description"}
        fold_log["crf_config"] = crf_config
        fold_log["timing"] = {
            "load_and_extraction_s": t_load,
            "train_rf_crf_s": t_train,
        }

        with open(fold_dir / "log.json", "w") as fh:
            json.dump(fold_log, fh, indent=2, default=_json_default)

        fold_val_metrics.append(fold_log.get("val_metrics") or {})

    t_grand = time() - t_grand_start

    # Timing summary
    print(f"\n{'='*70}")
    print(f"TIMING SUMMARY  model=RF-CRF  k={K}")
    print(f"{'='*70}")
    print(f"  {'Fold':>6}  {'(a) load+extract':>18}  {'(b) RF+CRF train':>18}")
    for i, (tl, tt) in enumerate(zip(fold_load_times, fold_train_times), 1):
        print(f"  {i:>6}  {tl:>18.2f}s  {tt:>18.2f}s")
    print(f"  {'Total':>6}  {sum(fold_load_times):>18.2f}s  {sum(fold_train_times):>18.2f}s")
    print(f"  Grand total: {t_grand:.2f}s")

    # CV summary
    agg, per_class_agg = _aggregate(fold_val_metrics)
    _print_cv_summary(agg, per_class_agg)

    summary = {
        "model": "RF-CRF",
        "k": K,
        "seed": SEED,
        "rf_config":  {k: v for k, v in RF_CONFIG.items() if k != "description"},
        "crf_config": crf_config,
        "timing": {
            "fold_load_and_extraction_s": fold_load_times,
            "fold_train_rf_crf_s": fold_train_times,
            "total_load_and_extraction_s": sum(fold_load_times),
            "total_train_rf_crf_s": sum(fold_train_times),
            "grand_total_s": t_grand,
        },
        "aggregated_val_metrics": agg,
        "per_class_file_macro_f1": per_class_agg,
    }
    summary_path = output_dir / f"cv_summary_rf_crf_k{K}_seed{SEED}.json"
    with open(summary_path, "w") as fh:
        json.dump(summary, fh, indent=2, default=_json_default)
    print(f"\nCV summary → {summary_path}")

# ---------------------------------------------------------------------------
# infer subcommand
# ---------------------------------------------------------------------------

def cmd_infer(args):
    cv_dir = Path(args.cv_dir)
    output_dir = Path(args.output) if args.output else None
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)

    fold_load_times = []
    fold_infer_times = []
    fold_results = []
    t_grand_start = time()

    print(f"\n{'='*70}")
    print(f"RF-CRF K-FOLD INFERENCE  k={K}  seed={SEED}")
    print(f"{'='*70}")

    for fold_num in range(1, K + 1):
        fold_dir = cv_dir / f"fold_{fold_num:02d}"
        rf_path = fold_dir / "rf_model.joblib"
        crf_path = fold_dir / "crf_model.npz"
        val_manifest_path = fold_dir / "val_manifest.csv"

        print(f"\n{'#'*70}\n  FOLD {fold_num}/{K}\n{'#'*70}")

        if not rf_path.exists():
            print(f"  [SKIP] {rf_path} not found: re-run train with --save-fold-models")
            continue
        if not crf_path.exists():
            print(f"  [SKIP] {crf_path} not found: re-run train with --save-fold-models")
            continue
        if not val_manifest_path.exists():
            print(f"  [SKIP] {val_manifest_path} not found")
            continue

        val_df = pd.read_csv(val_manifest_path)
        empty_train_df = pd.DataFrame(columns=val_df.columns)

        # Load val features 
        print(f"\n[Fold {fold_num}]  Loading val features ...")
        t0 = time()
        dataset = load_and_split_dataset(
            dataset_csv=None,
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
        t_load = time() - t0
        fold_load_times.append(t_load)
        print(f"  [TIMING] (a) loading + feature extraction: {t_load:.2f}s")

        X_val_list = dataset.get("X_val") or []
        y_val_list = dataset.get("y_val") or []
        grid_shapes_val = dataset.get("grid_shapes_val") or []
        val_labels_paths = dataset.get("val_labels_paths") or []

        if not X_val_list:
            print(f"  [WARN] No val samples loaded for fold {fold_num}, skipping.")
            fold_infer_times.append(0.0)
            continue

        n_skipped = len(val_df) - len(X_val_list)
        if n_skipped > 0:
            print(f"  [INFO] {n_skipped} val file(s) could not be loaded.")

        # RF projection
        print(f"\n[Fold {fold_num}]  Loading fold RF model + projecting features ...")
        fold_rf = joblib_load(rf_path)
        X_val_proj = [
            (_apply_rf_projection(fold_rf, X_nodes), edges, pairwise)
            for X_nodes, edges, pairwise in X_val_list
        ]
        print(f"  RF projection applied: unary dim → {len(fold_rf.classes_)}")

        # Load CRF model
        print(f"  Loading CRF model from {crf_path}")
        crf_model = CellTypeClassifierCRF.load_model(str(crf_path))

        # CRF inference (timed)
        print(f"\n[Fold {fold_num}]  Running CRF inference on {len(X_val_proj)} val files …")
        t0 = time()
        y_pred_list = crf_model._predict_parallel(X_val_proj, grid_shapes_val,
                                                   n_jobs=args.n_jobs)
        t_infer = time() - t0
        fold_infer_times.append(t_infer)
        print(f"  [TIMING] (b) CRF inference: {t_infer:.2f}s")

        # Metrics
        metrics = _build_val_metrics(
            crf_model, X_val_proj, y_val_list, grid_shapes_val,
            y_pred_list=y_pred_list,
        )
        acc = metrics.get("accuracy")
        f1 = metrics.get("macro_file_macro_f1")
        print(f"  Accuracy : {acc:.4f}" if acc is not None else "  Accuracy : N/A")
        print(f"  Macro F1 : {f1:.4f}"  if f1  is not None else "  Macro F1 : N/A")

        # Save predicted grids
        grid_manifest: dict = {}
        if output_dir is not None:
            fold_out  = output_dir / f"fold_{fold_num:02d}"
            grids_dir = fold_out / "predicted_grids"
            grids_dir.mkdir(parents=True, exist_ok=True)
            print(f"\n[Fold {fold_num}]  Saving predicted grids → {grids_dir}")
            grid_manifest = _save_predicted_grids(
                y_pred_list, grid_shapes_val, val_labels_paths, grids_dir,
            )
            n_ok = sum(1 for v in grid_manifest.values() if v["status"] == "ok")
            print(f"  {n_ok}/{len(grid_manifest)} grids saved.")
            with open(grids_dir / "manifest.json", "w") as fh:
                json.dump(grid_manifest, fh, indent=2)

        fold_results.append({
            "fold": fold_num,
            "n_val_files": len(X_val_list),
            "timing": {
                "load_and_extraction_s": t_load,
                "inference_s": t_infer,
            },
            "val_metrics": metrics,
            "grid_manifest": grid_manifest,
        })

    t_grand = time() - t_grand_start

    # Timing summary
    print(f"\n{'='*70}")
    print(f"INFERENCE TIMING SUMMARY  model=RF-CRF  k={K}")
    print(f"{'='*70}")
    print(f"  {'Fold':>6}  {'(a) load+extract':>18}  {'(b) inference':>14}")
    for i, (tl, ti) in enumerate(zip(fold_load_times, fold_infer_times), 1):
        print(f"  {i:>6}  {tl:>18.2f}s  {ti:>14.2f}s")
    if fold_load_times:
        print(f"  {'Total':>6}  {sum(fold_load_times):>18.2f}s  "
              f"{sum(fold_infer_times):>14.2f}s")
    print(f"  Grand total: {t_grand:.2f}s")

    result = {
        "model": "RF-CRF",
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
            json.dump(result, fh, indent=2, default=_json_default)
        print(f"\nInference timing → {timing_path}")

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = p.add_subparsers(dest="command", help="Subcommand")

    def _add_shared(sp):
        sp.add_argument(
            "--feature-cache", default=None, metavar="DIR",
            help="Directory for cached .npz feature files (speeds up re-runs)",
        )
        sp.add_argument(
            "--max-grid-size", type=int, default=2500,
            help="Training grids with more than this many cells are skipped (default: 2500)",
        )
        sp.add_argument(
            "--n-jobs", type=int, default=-1,
            help="Parallel jobs for RF training and CRF inference (default: -1 = all cores)",
        )

    # train
    tr = sub.add_parser(
        "train",
        help="5-fold CV: train RF+CRF and evaluate on each held-out val fold.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=(
            "5-fold CV.\n\n"
            "Per-fold timing:\n"
            "  (a) loading + feature extraction  (parallel, all CPU cores)\n"
            "  (b) RF training + projection + CRF training\n\n"
            "RF hyperparameters are hardcoded (best from standalone RF baseline).\n"
            "CRF hyperparameters are set via the flags below; defaults are the\n"
            "paper values.\n\n"
            "Outputs per fold: fold_NN/log.json, fold_NN/train_manifest.csv,\n"
            "                  fold_NN/val_manifest.csv,\n"
            "                  [fold_NN/rf_model.joblib, fold_NN/crf_model.npz]\n"
            "Global output:    cv_summary_rf_crf_k5_seed2112.json"
        ),
    )
    tr.add_argument("--dataset", required=True, metavar="CSV",
                    help="Path to the manifest CSV (e.g. dataset/manifest.csv)")
    tr.add_argument("--output", required=True, metavar="DIR",
                    help="Output directory")
    tr.add_argument("--save-fold-models", action="store_true",
                    help="Persist rf_model.joblib + crf_model.npz per fold "
                         "(required for the infer subcommand)")

    crf_grp = tr.add_argument_group(
        "CRF hyperparameters",
        "Defaults are the values used in the paper (see paper for tuning details).",
    )
    crf_grp.add_argument("--inference", default="qpbo", choices=["ad3", "qpbo", "lp"],
                         help="Inference algorithm (default: qpbo)")
    crf_grp.add_argument("--C", type=float, default=10.0,
                         help="SSVM regularisation constant (paper: C=10, default: 10.0)")
    crf_grp.add_argument("--max-iter", type=int, default=10000,
                         help="Max cutting-plane iterations (default: 10000; rarely reached "
                              "— early stopping triggers first)")
    crf_grp.add_argument("--patience", type=int, default=10,
                         help="Early-stopping patience: # consecutive val checks with no "
                              "improvement before stopping (paper: 10, default: 10)")
    crf_grp.add_argument("--val-check-every", type=int, default=100,
                         help="CRF iterations between val evaluations (paper: 100, default: 100)")
    crf_grp.add_argument("--min-delta", type=float, default=0.0,
                         help="Minimum val macro-F1 gain to count as an improvement "
                              "(default: 0.0 — any gain counts)")
    crf_grp.add_argument("--batch-size", type=int, default=None,
                         help="Mini-batch size for SSVM (None = full-batch; paper used batching "
                              "— set to e.g. 32 to replicate)")

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
            "  (b) CRF predict only (RF projection applied outside timed region)\n\n"
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