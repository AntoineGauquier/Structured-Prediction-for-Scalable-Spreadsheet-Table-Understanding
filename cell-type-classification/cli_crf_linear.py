import argparse, json, sys
import numpy as np
from time import time
from pathlib import Path

from .models.pipeline import train_crf_crossval, make_folds, train_from_dataset
from .models.crf_model import CellTypeClassifierCRF
from .features import extract_features


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest='command', help='Command to run')

    # TRAIN command
    train_parser = subparsers.add_parser('train', help='Train model from dataset CSV')
    train_parser.add_argument('dataset_csv', help='Path to dataset CSV file')
    train_parser.add_argument('--output', required=True,
                              help='Path to save trained model (.npz) or output directory '
                                   'when --crossval is set')
    train_parser.add_argument('--n-states', type=int, default=5)
    train_parser.add_argument('--inference', default='qpbo',
                              choices=['ad3', 'qpbo', 'lp'],
                              help='Inference method used during SSVM training (default: qpbo)')
    train_parser.add_argument('--C', type=float, default=0.1)
    train_parser.add_argument('--max-iter', type=int, default=10000)
    train_parser.add_argument('--tol', type=float, default=0.0)
    train_parser.add_argument('--batch-size', type=int, default=None)
    train_parser.add_argument('--check-convergence-every', type=int, default=0)
    train_parser.add_argument('--max-grid-size', type=int, default=2500,
                              help='Skip spreadsheets with more than this many cells '
                                   '(default: 2500)')
    train_parser.add_argument(
        '--class-weight-strategy',
        choices=['none', 'global_sqrt', 'global_inverse', 'per_file'],
        default='global_sqrt',
        help=(
            "Class weighting strategy for the structured loss.\n"
            "  none           — standard unweighted Hamming loss\n"
            "  global_sqrt    — sqrt(inverse-frequency) global weights (paper default)\n"
            "  global_inverse — inverse-frequency global weights\n"
            "  per_file       — inverse-frequency weights recomputed per spreadsheet"
        ),
    )
    train_parser.add_argument('--no-early-stopping', action='store_true')
    train_parser.add_argument('--patience', type=int, default=10)
    train_parser.add_argument('--val-check-every', type=int, default=100,
                              help='Evaluate val macro-F1 every N cutting-plane iterations '
                                   '(default: 100)')
    train_parser.add_argument('--min-delta', type=float, default=0.0,
                              help='Minimum F1 improvement to reset patience counter '
                                   '(default: 0.0)')
    train_parser.add_argument('--val-split', type=float, default=0.25)
    train_parser.add_argument('--checkpoint-dir', default=None)
    train_parser.add_argument('--checkpoint-every', type=int, default=50)
    train_parser.add_argument('--n-jobs', type=int, default=-2,
                              help='Number of parallel jobs (-2 = all CPUs minus one). '
                                   'Default: -2.')
    train_parser.add_argument('--verbose', type=int, default=1, choices=[0, 1, 2])
    train_parser.add_argument('--random-seed', type=int, default=2112)
    train_parser.add_argument('--cache-dir', default=None)
    train_parser.add_argument('--state-names', nargs='+', default=None, metavar='NAME',
                              help='Ordered list of class names (default: EMPTY HEADER DATA '
                                   'TITLE OTHER). Length must equal --n-states if both given.')

    cv_grp = train_parser.add_argument_group(
        'Cross-validation',
        'Run k-fold cross-validation instead of a single train/val split. '
        'When --crossval is set, --output is treated as a directory; each fold '
        'writes its outputs under fold_01/, fold_02/, … and an aggregated '
        'cv_summary_k{k}_seed{random-seed}.json is placed at the root. '
        'Use --random-seed 2112 to match the fold splits used in the paper.'
    )
    cv_grp.add_argument('--crossval', action='store_true', default=False,
                        help='Enable k-fold cross-validation.')
    cv_grp.add_argument('--k', type=int, default=5,
                        help='Number of folds (default: 5).')
    cv_grp.add_argument('--save-fold-models', action='store_true', default=False,
                        help='Keep the .npz weight file for every fold '
                             '(default: discard to save disk space).')

    empty_grp = train_parser.add_argument_group(
        'EMPTY constraints',
        'Hard label constraints for cells with a known-empty indicator feature. '
        'When enabled, cells flagged as empty can only be predicted as EMPTY, '
        'and non-flagged cells are never predicted as EMPTY.'
    )
    empty_grp.add_argument('--constrain-empty', action='store_true', default=False, help='Enable hard EMPTY-class constraints.')
    empty_grp.add_argument('--is-empty-feature-index', type=int, default=None, metavar='INT',
                           help='Column index of the is_empty flag in the node-feature matrix. '
                                'Required when --constrain-empty is set. '
                                'Feature 0 (is_na) is correct for the current feature code.')
    empty_grp.add_argument('--empty-class-index', type=int, default=None, metavar='INT', help='Label index for the EMPTY class (default: 0).')
    empty_grp.add_argument('--empty-penalty', type=float, default=1e6, metavar='FLOAT',
                           help='Score penalty for forbidden (node, label) pairs. '
                                'Must be < 2e6 for QPBO. Default: 1e6.')

    # EXTRACT command
    extract_parser = subparsers.add_parser('extract', help='Extract features from a spreadsheet file')
    extract_parser.add_argument('file', help='Path to spreadsheet file')
    extract_parser.add_argument('--format',
                                choices=['auto', 'ods', 'xls', 'xlsx', 'csv'],
                                default='auto')
    extract_parser.add_argument('--sheet', default=None)
    extract_parser.add_argument('--output', required=True, help='Path to save features (.npz)')

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return 1

    dispatch = {
        'train': run_train,
        'extract': run_extract,
    }
    return dispatch[args.command](args)


def run_train(args):
    """Train model from dataset CSV, optionally with k-fold cross-validation."""

    if args.constrain_empty and args.is_empty_feature_index is None:
        print("error: --is-empty-feature-index is required when --constrain-empty is set.", file=sys.stderr)
        return 1

    weighting_strategy = (
        args.class_weight_strategy
        .replace('global_sqrt', 'global')
        .replace('global_inverse', 'global')
    )
    class_weight_strategy = (
        'sqrt_inverse' if args.class_weight_strategy == 'global_sqrt'
        else 'inverse'  if args.class_weight_strategy in ('global_inverse', 'per_file')
        else 'uniform'
    )
    class_balanced = args.class_weight_strategy != 'none'

    state_names = args.state_names
    if state_names is not None and args.n_states != len(state_names):
        if args.n_states == 5:
            args.n_states = len(state_names)
        else:
            print(f"error: --n-states {args.n_states} conflicts with "
                  f"{len(state_names)} names in --state-names", file=sys.stderr)
            return 1

    shared_train_kwargs = dict(
        n_states=args.n_states,
        inference_method=args.inference,
        C=args.C,
        max_iter=args.max_iter,
        tol=args.tol,
        batch_size=args.batch_size,
        class_balanced=class_balanced,
        weighting_strategy=weighting_strategy,
        class_weight_strategy=class_weight_strategy,
        check_convergence_every=args.check_convergence_every,
        early_stopping=not args.no_early_stopping,
        patience=args.patience,
        val_check_every=args.val_check_every,
        min_delta=args.min_delta,
        n_jobs=args.n_jobs,
        verbose=args.verbose,
        random_state=args.random_seed,
        constrain_empty=args.constrain_empty,
        is_empty_feature_index=args.is_empty_feature_index,
        empty_class_index=args.empty_class_index,
        empty_penalty=args.empty_penalty,
        state_names=state_names,
        feature_cache_dir=args.cache_dir,
        max_grid_size=args.max_grid_size,
    )

    print("\n" + "="*80)
    print("TRAINING CONFIGURATION")
    print("="*80)
    print(f"Dataset: {args.dataset_csv}")
    print(f"Output:  {args.output}")
    if args.crossval:
        print(f"\nMode: k-fold cross-validation  (k={args.k}, seed={args.random_seed})")
        print(f"  Save fold models: {args.save_fold_models}")
    else:
        print(f"\nMode: single train/val split  (val_split={args.val_split})")
    print(f"\nCRF: C={args.C}  max_iter={args.max_iter}  inference={args.inference}")
    print(f"  class_weight_strategy={args.class_weight_strategy}"
          f"  constrain_empty={args.constrain_empty}")
    print("="*80)

    if args.crossval:
        train_crf_crossval(
            dataset_csv=args.dataset_csv,
            output_dir=args.output,
            k=args.k,
            save_fold_models=args.save_fold_models,
            **shared_train_kwargs,
        )
        print("\n" + "="*80)
        print("CROSS-VALIDATION COMPLETED")
        print("="*80)
        print(f"\nFold outputs: {args.output}/")
        print(f"Summary JSON: {args.output}/cv_summary_k{args.k}_seed{args.random_seed}.json")
    else:
        train_from_dataset(
            dataset_csv=args.dataset_csv,
            output_model_path=args.output,
            validation_split=args.val_split,
            checkpoint_dir=args.checkpoint_dir,
            checkpoint_every=args.checkpoint_every,
            **shared_train_kwargs,
        )
        print("\n" + "="*80)
        print("TRAINING COMPLETED")
        print("="*80)
        print(f"\nModel saved to: {args.output}")

    return 0


def run_extract(args):
    """Extract features from a file."""
    print(f"Extracting features from: {args.file}")
    start = time()
    unary, edges, pairwise = extract_features(
        args.file, file_format=args.format, sheet_name=args.sheet
    )
    H, W, F_u = unary.shape
    print(f"\nExtracted features:")
    print(f"  Grid size: {H}x{W}")
    print(f"  Unary features: {F_u}")
    print(f"  Edges: {edges.shape[0]}")
    print(f"  Pairwise features: {pairwise.shape[1]}")
    print(f"  Extraction time: {time() - start:.2f}s")
    np.savez(args.output, unary=unary, edges=edges, pairwise=pairwise)
    print(f"\nFeatures saved to: {args.output}")
    return 0


if __name__ == '__main__':
    sys.exit(main())