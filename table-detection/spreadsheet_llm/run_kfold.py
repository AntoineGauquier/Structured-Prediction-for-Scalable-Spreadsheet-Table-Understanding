from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from .data_loader import make_folds, load_item
from .encoding import encode_sheet_compressor
from .metrics import evaluate_detection, evaluate_detection_iou
from .tasks import detect_tables


# ─────────────────────────────────────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _file_id(row: pd.Series) -> str:
    key = f"{row['file_path']}|{row.get('sheet_name', '')}"
    return hashlib.md5(key.encode()).hexdigest()


def _result_path(output_dir: Path, fold_idx: int, file_id: str) -> Path:
    return output_dir / f"fold_{fold_idx}" / f"{file_id}.json"


def _token_count_heuristic(text: str) -> int:
    return max(1, len(text) // 4)


# ─────────────────────────────────────────────────────────────────────────────
#  Process a single validation row
# ─────────────────────────────────────────────────────────────────────────────

def process_row(
    row: pd.Series,
    data_dir: str,
    llm,                  # spreadsheet_llm.llm.LLMClient or None in dry-run
    *,
    k_anchor: int,
    dry_run: bool,
) -> Dict[str, Any]:
    """Load, encode, (optionally) infer, evaluate one spreadsheet."""
    t0 = time.time()

    sheet_data, gt_ranges = load_item(row, data_dir)

    encoded, _row_map, _col_map = encode_sheet_compressor(
        sheet_data,
        k=k_anchor,
        use_extraction=True,
        use_translation=True,
        use_aggregation=False,
    )
    compressed_tokens = _token_count_heuristic(encoded)

    base = {
        'file_path':        row['file_path'],
        'sheet_name':       row.get('sheet_name'),
        'labels_path':      row.get('labels_path'),
        'n_rows':           sheet_data.n_rows,
        'n_cols':           sheet_data.n_cols,
        'k_anchor':         k_anchor,
        'compressed_tokens': compressed_tokens,
        'gt_ranges':        gt_ranges,
    }

    if dry_run or llm is None:
        base['dry_run'] = True
        base['elapsed_sec'] = round(time.time() - t0, 3)
        return base

    det = detect_tables(sheet_data, llm, k=k_anchor)

    metrics     = evaluate_detection(det['predicted_ranges'], gt_ranges)
    metrics_iou = evaluate_detection_iou(det['predicted_ranges'], gt_ranges)

    return {
        **base,
        'predicted_ranges':            det['predicted_ranges'],
        'predicted_ranges_compressed': det['predicted_ranges_compressed'],
        'range_corners_merged':        det['range_corners_merged'],
        'no_valid_prediction':         det['no_valid_prediction'],
        'llm_response':  det['llm_record']['response'],
        'llm_tokens':    det['llm_record']['total_tokens'],
        'llm_elapsed':   round(det['llm_record']['elapsed_sec'], 3),
        'metrics':       metrics,
        'metrics_iou':   metrics_iou,
        'elapsed_sec':   round(time.time() - t0, 3),
    }


# ─────────────────────────────────────────────────────────────────────────────
#  Fold runner
# ─────────────────────────────────────────────────────────────────────────────

def run_fold(
    fold_idx: int,
    val_df: pd.DataFrame,
    output_dir: Path,
    data_dir: str,
    finetune_dir: Path,
    base_model_name: str,
    *,
    k_anchor: int,
    dry_run: bool,
    verbose: bool,
) -> List[Dict[str, Any]]:
    fold_dir = output_dir / f"fold_{fold_idx}"
    fold_dir.mkdir(parents=True, exist_ok=True)

    t_fold_start = time.time()

    # ── Load model for this fold ──────────────────────────────────────────
    llm = None
    if not dry_run:
        from .llm import LLMClient
        adapter_path = finetune_dir / f"fold_{fold_idx}" / "adapter"
        if not adapter_path.exists():
            print(f"  [fold {fold_idx}] ERROR: adapter not found at {adapter_path}",
                  file=sys.stderr)
            print(f"  Run: python -m spreadsheet_llm.finetune train "
                  f"--output {finetune_dir} --fold {fold_idx}", file=sys.stderr)
            return []

        if verbose:
            print(f"  [fold {fold_idx}] loading model "
                  f"{base_model_name} + adapter {adapter_path} …")
        t_load_start = time.time()
        llm = LLMClient(
            base_model_name=base_model_name,
            adapter_path=str(adapter_path),
        )
        t_load_sec = round(time.time() - t_load_start, 1)
        if verbose:
            print(f"  [fold {fold_idx}] model loaded in {t_load_sec:.1f}s")
    else:
        t_load_sec = 0.0

    # ── Process validation split ──────────────────────────────────────────
    results: List[Dict[str, Any]] = []
    n = len(val_df)
    t_infer_start = time.time()

    for i, (_, row) in enumerate(val_df.iterrows()):
        fid   = _file_id(row)
        rpath = _result_path(output_dir, fold_idx, fid)

        if rpath.exists():
            if verbose:
                print(f"  [fold {fold_idx}] ({i+1}/{n}) SKIP (cached): "
                      f"{row['file_path']}")
            with open(rpath) as f:
                results.append(json.load(f))
            continue

        if verbose:
            print(f"  [fold {fold_idx}] ({i+1}/{n}) {row['file_path']}")

        try:
            res = process_row(
                row, data_dir, llm,
                k_anchor=k_anchor,
                dry_run=dry_run,
            )
            res['error'] = None
        except Exception as e:
            tb = traceback.format_exc()
            print(f"  [fold {fold_idx}] ERROR {row['file_path']}: {e}",
                  file=sys.stderr)
            res = {
                'file_path':   row['file_path'],
                'sheet_name':  row.get('sheet_name'),
                'labels_path': row.get('labels_path'),
                'error':       str(e),
                'traceback':   tb,
            }

        with open(rpath, 'w') as f:
            json.dump(res, f, indent=2, default=str)
        results.append(res)

        if verbose and not dry_run and res.get('error') is None:
            m = res.get('metrics', {})
            print(f"    gt={res.get('gt_ranges')}  "
                  f"pred={res.get('predicted_ranges')}  "
                  f"EoB-0 F1={m.get('f1', 0):.3f}  "
                  f"IoU@.5 F1={res.get('metrics_iou', {}).get('f1@0.5', 0):.3f}")

    t_infer_sec  = round(time.time() - t_infer_start, 1)
    t_total_sec  = round(time.time() - t_fold_start, 1)

    # ── Unload model to free GPU memory before next fold ──────────────────
    if llm is not None:
        import torch
        del llm.model
        del llm
        torch.cuda.empty_cache()
        if verbose:
            print(f"  [fold {fold_idx}] model unloaded")

    # ── Save fold timing ──────────────────────────────────────────────────
    n_ok = sum(1 for r in results if not r.get('error'))
    timing = {
        "fold":          fold_idx,
        "model_load_sec": t_load_sec,
        "inference_sec":  t_infer_sec,
        "total_sec":      t_total_sec,
        "n_files":        n,
        "n_ok":           n_ok,
        "avg_sec_per_file": round(t_infer_sec / max(1, n_ok), 2),
    }
    timing_path = output_dir / f"fold_{fold_idx}" / "inference_timing.json"
    with open(timing_path, "w") as tf:
        json.dump(timing, tf, indent=2)

    if verbose and not dry_run:
        print(f"  [fold {fold_idx}] inference done: "
              f"load={t_load_sec:.0f}s  infer={t_infer_sec:.0f}s  "
              f"avg={timing['avg_sec_per_file']:.1f}s/file  "
              f"total={t_total_sec:.0f}s")

    return results


# ─────────────────────────────────────────────────────────────────────────────
#  Aggregate across folds
# ─────────────────────────────────────────────────────────────────────────────

def aggregate_fold_results(
    fold_results: List[List[Dict[str, Any]]],
) -> Dict[str, Any]:
    """Micro-average EoB-0 and IoU P/R/F1 over all folds."""
    from .metrics import aggregate_metrics, aggregate_iou_metrics

    all_eob:  List[Dict] = []
    all_iou:  List[Dict] = []
    per_fold_agg: List[Dict] = []

    for fold_idx, results in enumerate(fold_results):
        fold_eob = [r['metrics']     for r in results if r.get('metrics')]
        fold_iou = [r['metrics_iou'] for r in results if r.get('metrics_iou')]
        all_eob.extend(fold_eob)
        all_iou.extend(fold_iou)
        if fold_eob:
            per_fold_agg.append({
                'fold': fold_idx,
                **aggregate_metrics(fold_eob),
                'iou': aggregate_iou_metrics(fold_iou) if fold_iou else {},
            })

    overall_eob = aggregate_metrics(all_eob) if all_eob else {}
    overall_iou = aggregate_iou_metrics(all_iou) if all_iou else {}
    return {
        'overall':           overall_eob,
        'overall_iou':       overall_iou,
        'per_fold':          per_fold_agg,
        'n_files_evaluated': len(all_eob),
    }


# ─────────────────────────────────────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog='spreadsheet_llm.run_kfold',
        description='K-fold inference with fine-tuned Mistral-7B-Instruct-v0.2.',
    )
    p.add_argument('--manifest', required=True)
    p.add_argument('--data-dir', default='.')
    p.add_argument('--finetune-dir', required=True,
                   help='Root directory written by finetune.py '
                        '(contains fold_*/adapter/ subdirectories).')
    p.add_argument('--output', required=True,
                   help='Output directory for per-file JSON results.')
    p.add_argument('--k', type=int, default=5)
    p.add_argument('--random-state', type=int, default=2112)
    p.add_argument('--fold', type=int, default=None,
                   help='Run only this fold index (0-based). Default: all folds.')
    p.add_argument('--base-model', default='mistralai/Mistral-7B-Instruct-v0.2')
    p.add_argument('--k-anchor', type=int, default=4)
    p.add_argument('--dry-run', action='store_true',
                   help='Encode only; do not load the model or call inference.')
    p.add_argument('--quiet', action='store_true')
    return p


def main(argv: Optional[list] = None) -> int:
    args = build_parser().parse_args(argv)
    verbose = not args.quiet

    output_dir    = Path(args.output)
    finetune_dir  = Path(args.finetune_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    folds = make_folds(args.manifest, k=args.k, random_state=args.random_state)
    fold_indices = list(range(args.k)) if args.fold is None else [args.fold]

    if verbose:
        print(f"[run_kfold] {args.k} folds  manifest={args.manifest}")
        print(f"  base_model={args.base_model}")
        print(f"  finetune_dir={finetune_dir}")
        for fi in fold_indices:
            _, val_df = folds[fi]
            print(f"  fold {fi}: val={len(val_df)}")

    if args.dry_run and verbose:
        print("[run_kfold] dry-run mode — encoding only, no model loading")

    all_fold_results: List[List[Dict]] = []

    for fold_idx in fold_indices:
        _, val_df = folds[fold_idx]
        if verbose:
            print(f"\n[fold {fold_idx}] val size = {len(val_df)}")

        fold_res = run_fold(
            fold_idx, val_df, output_dir, args.data_dir, finetune_dir,
            args.base_model,
            k_anchor=args.k_anchor,
            dry_run=args.dry_run,
            verbose=verbose,
        )
        all_fold_results.append(fold_res)

    # ── Dry-run token summary ─────────────────────────────────────────────
    if args.dry_run and verbose:
        total_tokens = sum(
            r.get('compressed_tokens', 0)
            for fold_res in all_fold_results
            for r in fold_res
            if not r.get('error')
        )
        n_files = sum(
            1 for fold_res in all_fold_results
            for r in fold_res
            if not r.get('error')
        )
        print(f"\n[dry-run] files={n_files}  "
              f"total approx tokens={total_tokens:,}  "
              f"avg={total_tokens // max(1, n_files):,}/file")

    # ── Aggregate ─────────────────────────────────────────────────────────
    if not args.dry_run:
        agg = aggregate_fold_results(all_fold_results)
        agg_path = output_dir / 'aggregate.json'
        with open(agg_path, 'w') as f:
            json.dump(agg, f, indent=2, default=str)

        if verbose:
            ov  = agg.get('overall', {})
            iov = agg.get('overall_iou', {})
            print(f"\n[aggregate] EoB-0  F1={ov.get('f1', 0):.4f}  "
                  f"P={ov.get('precision', 0):.4f}  "
                  f"R={ov.get('recall', 0):.4f}  "
                  f"files={agg.get('n_files_evaluated', 0)}")
            print(f"[aggregate] IoU@.5  F1={iov.get('f1@0.5', 0):.4f}  "
                  f"P={iov.get('precision@0.5', 0):.4f}  "
                  f"R={iov.get('recall@0.5', 0):.4f}")
            print(f"[aggregate] IoU@.75 F1={iov.get('f1@0.75', 0):.4f}  "
                  f"IoU@.95 F1={iov.get('f1@0.95', 0):.4f}")
            for pf in agg.get('per_fold', []):
                print(f"  fold {pf['fold']}: EoB-0 F1={pf.get('f1', 0):.4f}  "
                      f"IoU@.5 F1={pf.get('iou', {}).get('f1@0.5', 0):.4f}  "
                      f"n={pf.get('n_sheets', 0)}")
            print(f"[aggregate] saved → {agg_path}")

    return 0


if __name__ == '__main__':
    sys.exit(main())
