from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from time import time

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from .pipeline import _build_val_metrics, load_and_split_dataset, train_from_preloaded
from .crf_model import CellTypeClassifierCRF

# ---------------------------------------------------------------------------
# Fixed constants
# ---------------------------------------------------------------------------

SEED = 2112 # To replicate the paper's results, do not change
K = 5
STATE_NAMES = ["EMPTY", "HEADER", "DATA", "TITLE", "OTHER"]

# Feature 0 (is_na) is the is-empty flag in the current feature extraction code.
IS_EMPTY_FEATURE_INDEX = 0
EMPTY_CLASS_INDEX = 0

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
# Grid saving
# ---------------------------------------------------------------------------

def _save_predicted_grids(y_pred_list, grid_shapes_val, val_labels_paths, out_dir):
    """
    Write one <UUID>.npz per val file.
    CRF _predict_parallel returns per-file 2-D (H, W) arrays directly.
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
            print(f"  [WARN] UUID={uuid}: grid_shape {grid_shape} gives {n_cells} cells but GT shape {gt_shape} needs {expected}. Skipping.")
            manifest[uuid] = {
                "shape": list(gt_shape), "labels_path": str(labels_path),
                "status": "shape_mismatch",
            }
            continue

        np.savez(out_dir / f"{uuid}.npz", labels=y_pred.reshape(gt_shape))
        manifest[uuid] = {"shape": list(gt_shape), "labels_path": str(labels_path), "status": "ok"}
    return manifest

# ---------------------------------------------------------------------------
# CV summary aggregation
# ---------------------------------------------------------------------------

def _aggregate(fold_val_metrics):
    def _agg(key):
        vals = [m[key] for m in fold_val_metrics if m and key in m and m[key] is not None]
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

    per_class_agg = {}
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
    print(f"CV SUMMARY  model=CRF-Linear  k={K}  seed={SEED}")
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

    class_balanced = args.weighting_strategy != "none"

    crf_config = {
        "inference_method":    args.inference,
        "C": args.C,
        "max_iter": args.max_iter,
        "patience": args.patience,
        "val_check_every": args.val_check_every,
        "min_delta": args.min_delta,
        "batch_size": args.batch_size,
        "class_balanced": class_balanced,
        "weighting_strategy": args.weighting_strategy,
        "class_weight_strategy": args.class_weight_strategy,
        "constrain_empty": args.constrain_empty,
        "is_empty_feature_index": args.is_empty_feature_index,
        "empty_class_index": args.empty_class_index,
        "empty_penalty": args.empty_penalty,
    }

    print(f"\n{'='*70}")
    print(f"CRF-Linear K-FOLD CV (train)  k={K}  seed={SEED}")
    print(f"  CRF C={args.C}  max_iter={args.max_iter}  inference={args.inference}"
          f"  batch_size={args.batch_size}")
    print(f"  Early stopping: patience={args.patience}  val_check_every={args.val_check_every}"
          f"  min_delta={args.min_delta}")
    print(f"  Loss weighting: strategy={args.weighting_strategy}"
          f"  class_weight_strategy={args.class_weight_strategy}")
    print(f"  EMPTY constraint: {args.constrain_empty}"
          + (f"  (feature={args.is_empty_feature_index}, class={args.empty_class_index},"
             f" penalty={args.empty_penalty:.2e})" if args.constrain_empty else ""))
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
        val_df.to_csv(fold_dir / "val_manifest.csv",   index=False)

        print(f"\n\n{'#'*70}")
        print(f"  FOLD {fold_num}/{K}  —  train={len(train_df)} rows  val={len(val_df)} rows")
        print(f"{'#'*70}")

        # Load raw features + labels 
        print(f"\n[Fold {fold_num}]  Loading features ...")
        t0 = time()
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
        t_load = time() - t0
        fold_load_times.append(t_load)
        print(f"  [TIMING] (a) loading + feature extraction: {t_load:.2f}s")
        print(f"  train files: {len(dataset['X_train'])}  "
              f"val files: {len(dataset['X_val'] or [])}")

        # CRF training 
        print(f"\n[Fold {fold_num}]  Training CRF ...")
        fold_crf_path = str(fold_dir / "crf_model.npz")
        t0 = time()

        _, fold_log = train_from_preloaded(
            dataset=dataset,
            output_model_path=fold_crf_path,
            n_states=5,
            inference_method=args.inference,
            random_state=SEED,
            C=args.C,
            max_iter=args.max_iter,
            tol=0.0,
            batch_size=args.batch_size,
            class_balanced=class_balanced,
            weighting_strategy=args.weighting_strategy,
            class_weight_strategy=args.class_weight_strategy,
            check_convergence_every=0,
            early_stopping=True,
            patience=args.patience,
            val_check_every=args.val_check_every,
            min_delta=args.min_delta,
            n_jobs=args.n_jobs,
            verbose=1,
            constrain_empty=args.constrain_empty,
            is_empty_feature_index=args.is_empty_feature_index,
            empty_class_index=args.empty_class_index,
            empty_penalty=args.empty_penalty,
            state_names=STATE_NAMES,
        )

        t_train = time() - t0
        fold_train_times.append(t_train)
        print(f"  [TIMING] (b) CRF training: {t_train:.2f}s")

        #  Model persistence
        if not args.save_fold_models:
            p = Path(fold_crf_path)
            if p.exists():
                p.unlink()
        else:
            print(f"  CRF saved → {fold_dir / 'crf_model.npz'}")

        #  Per-fold log 
        fold_log["fold"] = fold_num
        fold_log["k"] = K
        fold_log["seed"] = SEED
        fold_log["n_train_rows"] = len(train_df)
        fold_log["n_val_rows"] = len(val_df)
        fold_log["crf_config"] = crf_config
        fold_log["timing"] = {
            "load_and_extraction_s": t_load,
            "train_crf_s": t_train,
        }

        with open(fold_dir / "log.json", "w") as fh:
            json.dump(fold_log, fh, indent=2, default=_json_default)

        fold_val_metrics.append(fold_log.get("val_metrics") or {})

    t_grand = time() - t_grand_start

    # Timing summary
    print(f"\n{'='*70}")
    print(f"TIMING SUMMARY  model=CRF-Linear  k={K}")
    print(f"{'='*70}")
    print(f"  {'Fold':>6}  {'Load+extract':>18}  {'CRF train':>15}")
    for i, (tl, tt) in enumerate(zip(fold_load_times, fold_train_times), 1):
        print(f"  {i:>6}  {tl:>18.2f}s  {tt:>15.2f}s")
    print(f"  {'Total':>6}  {sum(fold_load_times):>18.2f}s  {sum(fold_train_times):>15.2f}s")
    print(f"  Grand total: {t_grand:.2f}s")

    # CV summary
    agg, per_class_agg = _aggregate(fold_val_metrics)
    _print_cv_summary(agg, per_class_agg)

    summary = {
        "model": "CRF-Linear",
        "k": K,
        "seed": SEED,
        "crf_config": crf_config,
        "timing": {
            "fold_load_and_extraction_s": fold_load_times,
            "fold_train_crf_s": fold_train_times,
            "total_load_and_extraction_s": sum(fold_load_times),
            "total_train_crf_s": sum(fold_train_times),
            "grand_total_s": t_grand,
        },
        "aggregated_val_metrics": agg,
        "per_class_file_macro_f1": per_class_agg,
    }
    summary_path = output_dir / f"cv_summary_crf_linear_k{K}_seed{SEED}.json"
    with open(summary_path, "w") as fh:
        json.dump(summary, fh, indent=2, default=_json_default)
    print(f"\nCV summary: {summary_path}")

# ---------------------------------------------------------------------------
# infer subcommand
# ---------------------------------------------------------------------------

def cmd_infer(args):
    cv_dir = Path(args.cv_dir)
    output_dir = Path(args.output) if args.output else None
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)

    fold_load_times: list[float] = []
    fold_infer_times: list[float] = []
    fold_results: list[dict]  = []
    t_grand_start = time()

    print(f"\n{'='*70}")
    print(f"CRF-Linear K-FOLD INFERENCE  k={K}  seed={SEED}")
    print(f"{'='*70}")

    for fold_num in range(1, K + 1):
        fold_dir = cv_dir / f"fold_{fold_num:02d}"
        crf_path = fold_dir / "crf_model.npz"
        val_manifest_path = fold_dir / "val_manifest.csv"

        print(f"\n{'#'*70}\n  FOLD {fold_num}/{K}\n{'#'*70}")

        if not crf_path.exists():
            print(f"  [SKIP] {crf_path} not found — re-run train with --save-fold-models")
            continue
        if not val_manifest_path.exists():
            print(f"  [SKIP] {val_manifest_path} not found")
            continue

        val_df         = pd.read_csv(val_manifest_path)
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

        # Load CRF model
        print(f"\n[Fold {fold_num}]  Loading CRF model from {crf_path}")
        crf_model = CellTypeClassifierCRF.load_model(str(crf_path))

        # CRF inference (timed)
        print(f"\n[Fold {fold_num}]  Running CRF inference on {len(X_val_list)} val files …")
        t0 = time()
        y_pred_list = crf_model._predict_parallel(X_val_list, grid_shapes_val,
                                                   n_jobs=args.n_jobs)
        t_infer = time() - t0
        fold_infer_times.append(t_infer)
        print(f"  [TIMING] (b) CRF inference: {t_infer:.2f}s")

        # Metrics (not timed)
        metrics = _build_val_metrics(
            crf_model, X_val_list, y_val_list, grid_shapes_val,
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

    #  Timing summary
    print(f"\n{'='*70}")
    print(f"INFERENCE TIMING SUMMARY  model=CRF-Linear  k={K}")
    print(f"{'='*70}")
    print(f"  {'Fold':>6}  {'(a) load+extract':>18}  {'(b) inference':>14}")
    for i, (tl, ti) in enumerate(zip(fold_load_times, fold_infer_times), 1):
        print(f"  {i:>6}  {tl:>18.2f}s  {ti:>14.2f}s")
    if fold_load_times:
        print(f"  {'Total':>6}  {sum(fold_load_times):>18.2f}s  "
              f"{sum(fold_infer_times):>14.2f}s")
    print(f"  Grand total: {t_grand:.2f}s")

    result = {
        "model": "CRF-Linear",
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
            help="Parallel jobs for CRF inference (default: -1 = all cores)",
        )

    # train
    tr = sub.add_parser(
        "train",
        help="5-fold CV: train CRF and evaluate on each held-out val fold.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=(
            "5-fold CV.\n\n"
            "Per-fold timing:\n"
            "  (a) loading + feature extraction  (parallel, all CPU cores)\n"
            "  (b) CRF training\n\n"
            "CRF hyperparameters are set via the flags below; defaults are the\n"
            "paper values.  Loss weighting and EMPTY constraints are ON by\n"
            "default (paper configuration) and can be disabled for ablations.\n\n"
            "Outputs per fold: fold_NN/log.json, fold_NN/train_manifest.csv,\n"
            "                  fold_NN/val_manifest.csv,\n"
            "                  [fold_NN/crf_model.npz]\n"
            "Global output:    cv_summary_crf_linear_k5_seed2112.json"
        ),
    )
    tr.add_argument("--dataset", required=True, metavar="CSV",
                    help="Path to the manifest CSV (e.g. dataset/manifest.csv)")
    tr.add_argument("--output", required=True, metavar="DIR",
                    help="Output directory")
    tr.add_argument("--save-fold-models", action="store_true",
                    help="Persist crf_model.npz per fold (required for the infer subcommand)")

    crf_grp = tr.add_argument_group(
        "CRF hyperparameters",
        "Defaults are the values used in the paper (see paper for tuning details).",
    )
    crf_grp.add_argument("--inference", default="qpbo", choices=["ad3", "qpbo", "lp"],
                         help="Inference algorithm (default: qpbo)")
    crf_grp.add_argument("--C", type=float, default=0.1,
                         help="SSVM regularisation constant (paper: C=0.1, default: 0.1)")
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
    crf_grp.add_argument("--batch-size", type=lambda x: None if int(x) == 0 else int(x),
                         default=128, metavar="INT",
                         help="Mini-batch size for SSVM (paper: 128, default: 128; "
                              "set to 0 for full-batch NeverStopOneSlackSSVM)")

    wt_grp = tr.add_argument_group(
        "Loss weighting",
        "Class-rebalancing strategy for the structured loss. "
        "Defaults are the paper configuration (sqrt-inverse global weighting). "
        "Set --weighting-strategy none to disable.",
    )
    wt_grp.add_argument(
        "--weighting-strategy",
        choices=["none", "global", "per_file"],
        default="global",
        help="'global': sqrt-inverse weights computed once from training labels (paper default). "
             "'per_file': inverse-freq weights recomputed per spreadsheet. "
             "'none': standard unweighted Hamming loss.",
    )
    wt_grp.add_argument(
        "--class-weight-strategy",
        choices=["inverse", "sqrt_inverse", "uniform"],
        default="sqrt_inverse",
        help="Formula for global weights: 'sqrt_inverse' = sqrt(1/freq_c) (paper default), "
             "'inverse' = 1/freq_c (stronger rebalancing), 'uniform' = equal weights. "
             "Only applies when --weighting-strategy is 'global'.",
    )

    empty_grp = tr.add_argument_group(
        "EMPTY constraints",
        "Hard label constraints for cells with a known-empty indicator feature. "
        "Enabled by default (paper configuration). "
        "Use --no-constrain-empty to disable.",
    )
    empty_grp.add_argument(
        "--no-constrain-empty", dest="constrain_empty", action="store_false",
        help="Disable hard EMPTY-class constraints (enabled by default).",
    )
    empty_grp.add_argument(
        "--is-empty-feature-index", type=int, default=IS_EMPTY_FEATURE_INDEX,
        metavar="INT",
        help=f"Column index of the is_empty flag in the node-feature matrix "
             f"(default: {IS_EMPTY_FEATURE_INDEX} = is_na).",
    )
    empty_grp.add_argument(
        "--empty-class-index", type=int, default=EMPTY_CLASS_INDEX, metavar="INT",
        help=f"Label index for the EMPTY class (default: {EMPTY_CLASS_INDEX}).",
    )
    empty_grp.add_argument(
        "--empty-penalty", type=float, default=1e6, metavar="FLOAT",
        help="Score penalty for forbidden (node, label) pairs "
             "(default: 1e6; must be < 2e6 for QPBO).",
    )
    tr.set_defaults(constrain_empty=True)
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
            "  (b) CRF predict\n\n"
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
    args = parser.parse_args()
    if args.command is None:
        parser.print_help()
        raise SystemExit(1)
    if args.command == "train":
        cmd_train(args)
    elif args.command == "infer":
        cmd_infer(args)