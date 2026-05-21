from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _deduplicate(ranges):
    """Return ranges with duplicates removed, preserving first-occurrence order."""
    seen = set()
    result = []
    for r in ranges:
        if r not in seen:
            seen.add(r)
            result.append(r)
    return result


def _convert_fold(fold_dir):
    """
    Read all per-file JSONs in a fold directory.

    Returns (per_file_records, n_skipped_errors, n_deduplicated_files).
    """
    per_file = []
    n_errors = 0
    n_dedup  = 0

    for json_path in sorted(fold_dir.glob("*.json")):
        if json_path.name == "inference_timing.json":
            continue

        with open(json_path) as fh:
            record = json.load(fh)

        # Skip error records
        if record.get("error") is not None:
            n_errors += 1
            continue

        pred_ranges = record.get("predicted_ranges") or []
        gt_ranges = record.get("gt_ranges") or []

        # Deduplicate predicted ranges (LLM can repeat identical ranges)
        deduped = _deduplicate(pred_ranges)
        if len(deduped) < len(pred_ranges):
            n_dedup += 1

        per_file.append({
            "file": record.get("file_path", json_path.stem),
            "predicted_ranges": deduped,
            "gt_ranges": gt_ranges,
            "n_predicted": len(deduped),
            "n_gt": len(gt_ranges),
        })

    return per_file, n_errors, n_dedup


def _build_parser():
    p = argparse.ArgumentParser(
        description=(
            "Convert SpreadsheetLLM per-file JSON results to the metrics.json "
            "format expected by plot_generator.py."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--input-dir", required=True, metavar="DIR",
        help="Root directory containing fold_0/, fold_1/, … sub-directories.",
    )
    p.add_argument(
        "--output-dir", required=True, metavar="DIR",
        help="Root directory where fold_01/metrics.json … files will be written.",
    )
    return p


def main():
    args = _build_parser().parse_args()

    input_root = Path(args.input_dir)
    output_root = Path(args.output_dir)

    fold_dirs = sorted(input_root.glob("fold_*/"))
    if not fold_dirs:
        print(f"[error] No fold_* directories found under {input_root}", file=sys.stderr)
        sys.exit(1)

    for fold_dir in fold_dirs:
        per_file, n_errors, n_dedup = _convert_fold(fold_dir)

        fold_idx = int(fold_dir.name.split("_")[1])
        out_fold = output_root / f"fold_{fold_idx + 1:02d}"
        out_fold.mkdir(parents=True, exist_ok=True)
        out_path = out_fold / "metrics.json"

        with open(out_path, "w") as fh:
            json.dump({"per_file": per_file}, fh, indent=2)

        print(
            f"Written {out_path}  "
            f"({len(per_file)} files, "
            f"{n_errors} skipped [error], "
            f"{n_dedup} deduplicated)"
        )


if __name__ == "__main__":
    main()