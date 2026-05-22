import contextlib, json, hashlib, zipfile
from concurrent.futures import ProcessPoolExecutor, as_completed
import pandas as pd
import numpy as np
from pathlib import Path
from time import time
from tqdm import tqdm
from joblib import Parallel, delayed

from .crf_model import CellTypeClassifierCRF
from ..features import extract_features
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from ..utils.compression import save_compression_info

# Must stay in sync with EXCLUDED_FEATURE_INDICES in rf_baseline.py.
_RF_EXCLUDED_FEATURE_INDICES = []  # frozenset([17, 18, 20, *range(52, 67)])
_RF_KEEP_INDICES = [i for i in range(67) if i not in _RF_EXCLUDED_FEATURE_INDICES]


def _apply_rf_projection(rf_model, X_nodes):
    """
    Project raw (N, 67) unary features to (N, n_classes) RF class probabilities.
    Applies the same feature selection used during RF training.

    Args:
        rf_model: fitted sklearn RandomForestClassifier
        X_nodes:  float32 array of shape (N, 67)

    Returns:
        float32 array of shape (N, n_classes), writeable, C-contiguous
    """
    if X_nodes.ndim != 2 or X_nodes.shape[1] != 67:
        raise ValueError(
            f"_apply_rf_projection expects X_nodes of shape (N, 67), "
            f"got {X_nodes.shape}"
        )
    X_selected = X_nodes[:, _RF_KEEP_INDICES] # (N, len(_RF_KEEP_INDICES))
    proba = rf_model.predict_proba(X_selected) # (N, n_classes)
    if proba.ndim != 2:
        raise ValueError(
            f"rf_model.predict_proba returned unexpected shape {proba.shape}"
        )
    out = np.ascontiguousarray(proba.astype(np.float32))
    out.flags.writeable = True
    return out # (N, n_classes)


def _validate_rf_model(rf_model, n_states):
    """
    Warn (or raise) if the RF's class count doesn't match the CRF's n_states.
    """
    n_rf_classes = len(rf_model.classes_)
    if n_rf_classes != n_states:
        raise ValueError(
            f"RF model has {n_rf_classes} classes but CRF n_states={n_states}. "
            f"They must match because RF predict_proba output becomes the CRF "
            f"unary features (shape N x n_classes)."
        )


def _compression_cache_path(feature_cache_dir, file_path, sheet_name):
    base = _feature_cache_path(feature_cache_dir, file_path, sheet_name)
    return base.with_suffix(".compression.json")


def _feature_cache_path(cache_dir, file_path, sheet_name):
    key = f"{Path(file_path).resolve()}::{sheet_name}"
    digest = hashlib.sha1(key.encode()).hexdigest()[:16]
    stem = Path(file_path).stem[:50]
    return Path(cache_dir) / f"{stem}_{digest}.npz"


def _load_cached_features(cache_path):
    d = np.load(cache_path, allow_pickle=False)
    return d["unary"], d["edges"], d["pairwise"]


def _save_cached_features(cache_path, unary, edges, pairwise):
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache_path, unary=unary, edges=edges, pairwise=pairwise)

MIME_TO_FMT = {
    'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet': 'xlsx',
    'application/vnd.ms-excel': 'xls',
    'application/vnd.oasis.opendocument.spreadsheet': 'ods',
    'text/csv': 'csv',
    'text/tab-separated-values': 'tsv',
}

FORMAT_TO_EXT = {'xlsx': '.xlsx', 'xls': '.xls', 'ods': '.ods', 'csv': '.csv', 'tsv': '.tsv'}
EXT_TO_FMT = {v: k for k, v in FORMAT_TO_EXT.items()}
_KNOWN_EXTS = set(EXT_TO_FMT.keys())


def _sniff_format(filepath):
    try:
        with open(filepath, 'rb') as f:
            magic = f.read(4)
    except OSError:
        return None

    if magic == b'\xD0\xCF\x11\xE0':
        return 'xls'

    if magic[:2] == b'PK':
        try:
            with zipfile.ZipFile(filepath) as zf:
                names = zf.namelist()
                if 'mimetype' in names:
                    mt = zf.read('mimetype').decode('ascii', errors='ignore').strip()
                    if 'opendocument.spreadsheet' in mt:
                        return 'ods'
                if '[Content_Types].xml' in names:
                    return 'xlsx'
        except Exception:
            pass
        return 'xlsx'

    return None


def detect_format(filepath, mime_type):
    mime_clean = (mime_type or '').split(';')[0].strip().lower()
    if mime_clean in MIME_TO_FMT:
        return MIME_TO_FMT[mime_clean]

    sniffed = _sniff_format(filepath)
    if sniffed:
        return sniffed

    ext = Path(filepath).suffix.lower()
    if ext in _KNOWN_EXTS:
        return EXT_TO_FMT[ext]

    raise ValueError(
        f'Unsupported format: extension={ext!r}, mime_type={mime_type!r}. '
        f'Known extensions: {list(EXT_TO_FMT)}, known MIME types: {list(MIME_TO_FMT)}'
    )


def _build_val_metrics(model, X_val, y_val, grid_shapes_val, y_pred_list=None):
    cell_metrics = model.score(X_val, y_val, grid_shapes=grid_shapes_val, detailed=True, y_pred_list=y_pred_list)

    cm = cell_metrics.get("confusion_matrix")
    cm_list = cm.tolist() if isinstance(cm, np.ndarray) else cm

    clf_report_raw = cell_metrics.get("classification_report", {})
    clf_report = {}
    for key, val in clf_report_raw.items():
        if isinstance(val, dict):
            clf_report[key] = {k: float(v) for k, v in val.items()}
        else:
            clf_report[key] = float(val) if val is not None else None

    file_metrics = model.score_file_macro_f1(X_val, y_val, grid_shapes=grid_shapes_val, detailed=True, y_pred_list=y_pred_list)

    def _safe_float(v):
        if v is None:
            return None
        try:
            f = float(v)
            return None if f != f else f
        except (TypeError, ValueError):
            return None

    per_class_f1 = {cls: _safe_float(v) for cls, v in file_metrics.get("per_class_f1", {}).items()}
    support_files = {
        cls: (int(v) if v is not None else None)
        for cls, v in file_metrics.get("support_files_per_class", {}).items()
    }

    return {
        "accuracy": _safe_float(cell_metrics.get("accuracy")),
        "classification_report": clf_report,
        "confusion_matrix": cm_list,
        "macro_file_macro_f1": _safe_float(file_metrics.get("macro_file_macro_f1")),
        "per_class_f1": per_class_f1,
        "support_files_per_class": support_files,
    }

def make_folds(dataset_csv, k, random_state=2112):
    from sklearn.model_selection import KFold

    df = _load_manifest(dataset_csv)
    df_shuffled = df.sample(frac=1, random_state=random_state).reset_index(drop=True)
    kf = KFold(n_splits=k, shuffle=False)

    folds = []
    for train_idx, val_idx in kf.split(df_shuffled):
        folds.append((
            df_shuffled.iloc[train_idx].reset_index(drop=True),
            df_shuffled.iloc[val_idx].reset_index(drop=True),
        ))
    return folds


def _load_manifest(path):
    return pd.read_csv(path)


def _load_single_sample(args):
    """
    Load features + labels for one row.  Runs inside a worker process.

    Returns (index, X, y, grid_shape, file_name)  on success,
            (index, None, None, None, file_name)   on failure.
    """
    (
        idx, file_path, sheet_name, labels_path, mime_type,
        feature_cache_dir, rf_model_path, n_states, verbosity,
    ) = args

    from pathlib import Path
    import numpy as np

    file_name = Path(file_path).name

    for n in _SUBSET:
        if file_name in n: 
            return idx, None, None, None, Path(file_path).name

    rf_model = None
    if rf_model_path is not None:
        import joblib
        rf_model = joblib.load(rf_model_path)
        _validate_rf_model(rf_model, n_states)

    try:
        mime_type = mime_type or ""
        file_format = detect_format(file_path, mime_type)

        if feature_cache_dir is not None:
            import fcntl

            cache_path = _feature_cache_path(feature_cache_dir, file_path, sheet_name)
            comp_cache_path = _compression_cache_path(feature_cache_dir, file_path, sheet_name)
            lock_path = cache_path.with_suffix(".lock")

            _y_data_pre = np.load(labels_path, allow_pickle=True)
            _y_pre = _y_data_pre["labels"]
            _anchor_pre = _y_data_pre["anchor_labels"]
            _expected_H = int(_y_pre.shape[0]) + int(_anchor_pre[0])
            _expected_W = int(_y_pre.shape[1]) + int(_anchor_pre[1])

            with open(lock_path, "w") as _lf:
                fcntl.flock(_lf, fcntl.LOCK_EX)
                try:
                    _cache_valid = False
                    if cache_path.exists():
                        unary_features, edges, pairwise_features = _load_cached_features(cache_path)
                        _cached_H, _cached_W = unary_features.shape[0], unary_features.shape[1]
                        if _cached_H != _expected_H or _cached_W != _expected_W:
                            if verbosity:
                                print(
                                    f"  [cache stale] {file_name}: "
                                    f"cached ({_cached_H},{_cached_W}) != "
                                    f"label ({_expected_H},{_expected_W}) — re-extracting."
                                )
                            cache_path.unlink(missing_ok=True)
                        else:
                            _cache_valid = True
                            if verbosity:
                                print(f"  [cache hit]  {file_name}")

                    if not _cache_valid:
                        (unary_features, edges, pairwise_features,
                         row_mapping, col_mapping, original_shape) = extract_features(
                            file_path, file_format=file_format, sheet_name=sheet_name
                        )
                        _save_cached_features(cache_path, unary_features, edges, pairwise_features)
                        save_compression_info(comp_cache_path, row_mapping, col_mapping, original_shape)
                        if verbosity:
                            print(f"  [extracted]  {file_name} -> cached")
                finally:
                    fcntl.flock(_lf, fcntl.LOCK_UN)

        else:
            (unary_features, edges, pairwise_features,
             row_mapping, col_mapping, original_shape) = extract_features(
                file_path, file_format=file_format, sheet_name=sheet_name
            )

        # labels
        y_data = np.load(labels_path, allow_pickle=True)
        y_true_original = y_data["labels"]
        anchor_row, anchor_col = y_data["anchor_labels"]

        y_true = np.array(y_true_original, copy=True)
        h, w = y_true.shape
        y_true_padded = np.zeros((h + anchor_row, w + anchor_col), dtype=y_true.dtype)
        y_true_padded[anchor_row:anchor_row + h, anchor_col:anchor_col + w] = y_true
        y_true = y_true_padded

        H, W, _ = unary_features.shape
        if y_true.shape != (H, W):
            raise ValueError(f"Label shape {y_true.shape} doesn't match grid {(H, W)}")

        # nodes
        X_nodes_raw = unary_features.reshape(-1, unary_features.shape[2]).copy()
        X_nodes = _apply_rf_projection(rf_model, X_nodes_raw) if rf_model is not None else X_nodes_raw

        # edges
        edges_w = np.array(edges, copy=True) if isinstance(edges, np.ndarray) else edges
        pf_w = np.array(pairwise_features, copy=True) if isinstance(pairwise_features, np.ndarray) else pairwise_features

        X = (X_nodes, edges_w, pf_w)
        y = y_true.ravel().copy()

        return idx, X, y, (H, W), file_name

    except Exception as exc:
        print(f"\nError processing {file_path}: {exc}")
        return idx, None, None, None, Path(file_path).name


def load_and_split_dataset(
    dataset_csv,
    validation_split=0.2,
    random_state=0,
    max_grid_size=2500,
    early_stopping=True,
    feature_cache_dir=None,
    verbosity=1,
    train_df=None,
    val_df=None,
    rf_model=None,
    n_states=5,
    n_workers=4,
    rf_model_path=None,
):
    """
    Parallel version of load_and_split_dataset.

    Parameters
    ----------
    dataset_csv : str or Path
        Path to the CSV listing file_path / sheet_name / labels_path columns.
    validation_split : float
        Fraction of samples held out for validation when no explicit split is given.
    random_state : int
        Seed for the train/val shuffle.
    max_grid_size : int
        Training samples whose grid exceeds this cell count are filtered out.
    early_stopping : bool
        When True (and no explicit split), a validation set is carved out.
    feature_cache_dir : str or Path or None
        Directory for caching extracted features.  Disabled when None.
    verbosity : int
        0 = silent per-file messages, 1 = print cache hit/miss per file.
    train_df, val_df : pd.DataFrame or None
        Pre-split DataFrames.  When both are supplied, dataset_csv is ignored.
    rf_model : fitted RandomForestClassifier or None
        Used only for validation / n_classes reporting in the main process.
        The actual projection runs inside workers via rf_model_path.
    n_states : int
        Expected CRF state count; used to validate the RF class count.
    n_workers : int
        Worker processes.  Use 1 for sequential / debug mode.
    rf_model_path : str or Path or None
        Path to a joblib-serialised RandomForestClassifier.  Each worker loads
        it independently — faster than pickling the object when n_workers > 1.
        When supplied, rf_model is ignored for the projection itself.

    Returns
    -------
    dict with keys:
        X_train, y_train, grid_shapes_train,
        X_val, y_val, grid_shapes_val,
        n_unary_features, n_pairwise_features,
        val_labels_paths,   # [GRID-SAVING] labels_path for each loaded val file
        val_file_names,     # [GRID-SAVING] file_path for each loaded val file
    """
    print("=" * 80)
    print("LOADING DATASET (parallel)")
    print("=" * 80)

    import os
    start = time()
    if n_workers <= 0:
        n_workers = os.cpu_count()

    if rf_model is not None:
        _validate_rf_model(rf_model, n_states)
        n_rf_classes = len(rf_model.classes_)
        print(f"\nRF-scoring mode: raw (N,67) unary features -> RF predict_proba (N,{n_rf_classes})")
        print(f"  RF classes ({n_rf_classes}): {list(rf_model.classes_)}")

    if feature_cache_dir is not None:
        feature_cache_dir = Path(feature_cache_dir)
        feature_cache_dir.mkdir(parents=True, exist_ok=True)
        print(f"\nFeature cache: {feature_cache_dir}")
    else:
        print("\nFeature cache: disabled")

    if train_df is not None and val_df is not None:
        n_train_rows = len(train_df)
        df = pd.concat([train_df, val_df], ignore_index=True)
        explicit_split = True
        print(f"\nUsing pre-computed split: {n_train_rows} train / {len(val_df)} val")
    else:
        df = _load_manifest(dataset_csv)
        explicit_split = False
        print(f"\nLoading dataset from: {dataset_csv}")

    for col in ("file_path", "sheet_name", "labels_path"):
        if col not in df.columns:
            raise ValueError(f"Dataset CSV must have column: {col}")

    print(f"Found {len(df)} samples — dispatching to {n_workers} worker(s)...\n")

    # Build per-worker argument tuples 
    worker_args = []
    for idx, row in df.iterrows():
        fp = row["file_path"]

        worker_args.append((
            idx,
            fp,
            row["sheet_name"] if pd.notna(row.get("sheet_name")) else None,
            row["labels_path"],
            row.get("mime_type", ""),
            feature_cache_dir,
            rf_model_path,
            n_states,
            verbosity,
        ))

    # dispatch
    results_by_idx = {}

    if n_workers == 1:
        for args in tqdm(worker_args, desc="Loading (sequential)"):
            idx, X, y, gs, fname = _load_single_sample(args)
            if X is not None:
                results_by_idx[idx] = (X, y, gs, fname)
    else:
        nb_cells = 0
        with ProcessPoolExecutor(max_workers=n_workers) as pool:
            futures = {pool.submit(_load_single_sample, a): a[0] for a in worker_args}
            for future in tqdm(as_completed(futures), total=len(futures), desc="Loading"):
                idx, X, y, gs, fname = future.result()
                if X is not None:
                    gs_product = gs[0] * gs[1]
                    nb_cells += gs_product
                    results_by_idx[idx] = (X, y, gs, fname)

        print("NB_CELLS:", nb_cells)

    ordered = sorted(results_by_idx.items())
    ordered_keys = [k for k, _ in ordered]
    X_all        = [v[0] for _, v in ordered]
    y_all        = [v[1] for _, v in ordered]
    grid_shapes  = [v[2] for _, v in ordered]
    file_names   = [v[3] for _, v in ordered]

    print(f"\nSuccessfully loaded {len(X_all)} samples")
    if not X_all:
        raise ValueError("No samples could be loaded from dataset")

    n_unary_features = X_all[0][0].shape[1]
    n_pairwise_features = X_all[0][2].shape[1]
    print(f"\nFeature dimensions:")
    print(f"  Unary:    {n_unary_features}"
          + (" (RF predict_proba)" if rf_model is not None or rf_model_path else " (raw)"))
    print(f"  Pairwise: {n_pairwise_features}")

    # Sanity-check feature dimensionality across all samples.
    for i, (X, y, gs) in enumerate(zip(X_all, y_all, grid_shapes)):
        if X[0].shape[1] != n_unary_features:
            raise ValueError(f"Sample {i}: unary dim {X[0].shape[1]} != {n_unary_features}")
        if X[0].shape[0] != gs[0] * gs[1]:
            raise ValueError(f"Sample {i}: X_nodes rows {X[0].shape[0]} != {gs[0] * gs[1]}")
        if X[2].shape[1] != n_pairwise_features:
            raise ValueError(f"Sample {i}: pairwise dim {X[2].shape[1]} != {n_pairwise_features}")

    # train / val split
    if explicit_split or (early_stopping and validation_split > 0):
        if explicit_split:
            train_idx = list(range(min(n_train_rows, len(X_all))))
            val_idx = list(range(n_train_rows, len(X_all)))
        else:
            n_val = int(len(X_all) * validation_split)
            indices = np.random.RandomState(random_state).permutation(len(X_all))
            train_idx = indices[:-n_val]
            val_idx = indices[-n_val:]

            val_file_names_out = [file_names[i] for i in val_idx]
            with open("validation_files.json", "w", encoding="utf-8") as f:
                json.dump(val_file_names_out, f, indent=2)
            print(f"\nSaved validation file names to validation_files.json")

        nb_filtered = 0
        train_indices = []
        for i in train_idx:
            if grid_shapes[i][0] * grid_shapes[i][1] <= max_grid_size:
                train_indices.append(i)
            else:
                nb_filtered += 1

        X_train = [X_all[i] for i in train_indices]
        y_train = [y_all[i] for i in train_indices]
        grid_shapes_train = [grid_shapes[i] for i in train_indices]
        X_val = [X_all[i] for i in val_idx]
        y_val = [y_all[i] for i in val_idx]
        grid_shapes_val = [grid_shapes[i] for i in val_idx]

        val_labels_paths = [df.iloc[ordered_keys[i]]["labels_path"] for i in val_idx]
        val_file_names = [file_names[i] for i in val_idx]

        print(f"\nDataset split (random_state={random_state}, val_split={validation_split}):")
        print(f"  Training:   {len(X_train)} samples ({nb_filtered} too large filtered out)")
        print(f"  Validation: {len(X_val)} samples")
    else:
        X_train = X_all
        y_train = y_all
        grid_shapes_train = grid_shapes
        X_val = y_val = grid_shapes_val = None
        val_labels_paths = []
        val_file_names   = []
        print(f"\nUsing all {len(X_train)} samples for training (no split)")

    print("ELAPSED:", time()-start)
    return {
        "X_train": X_train,
        "y_train": y_train,
        "grid_shapes_train": grid_shapes_train,
        "X_val": X_val,
        "y_val": y_val,
        "grid_shapes_val": grid_shapes_val,
        "n_unary_features": n_unary_features,
        "n_pairwise_features": n_pairwise_features,
        "val_labels_paths": val_labels_paths,
        "val_file_names": val_file_names,
    }


def train_from_preloaded(
    dataset,
    output_model_path,
    n_states=5,
    inference_method='qpbo',
    random_state=0,
    C=1.0,
    max_iter=1000,
    tol=1e-3,
    batch_size=None,
    class_balanced=False,
    weighting_strategy='none',
    class_weight_strategy='inverse',
    use_batch_weights=False,
    check_convergence_every=100,
    early_stopping=True,
    patience=5,
    val_check_every=100,
    min_delta=1e-4,
    n_jobs=-1,
    verbose=1,
    checkpoint_dir=None,
    checkpoint_every=50,
    constrain_empty=False,
    is_empty_feature_index=None,
    empty_class_index=None,
    empty_penalty=1e6,
    state_names=None,
):
    X_train = dataset['X_train']
    y_train = dataset['y_train']
    grid_shapes_train = dataset['grid_shapes_train']
    X_val = dataset['X_val']
    y_val = dataset['y_val']
    grid_shapes_val = dataset['grid_shapes_val']
    n_unary_features = dataset['n_unary_features']
    n_pairwise_features = dataset['n_pairwise_features']

    if weighting_strategy != 'none' and not class_balanced:
        print(
            f"  [INFO] weighting_strategy='{weighting_strategy}' requested but "
            f"class_balanced=False: changing class_balanced to True for consistency."
        )
        class_balanced = True

    print("=" * 80)
    print("TRAINING CRF MODEL")
    print("=" * 80)
    print(f"\nFeature dimensions:")
    print(f"  Unary features:    {n_unary_features}")
    print(f"  Pairwise features: {n_pairwise_features}")
    print(f"  States:            {n_states}")

    if n_unary_features == n_states:
        print(f"\n  [INFO] n_unary_features == n_states == {n_states}: CRF is operating on RF predict_proba scores as unary input.")

    training_log = {
        'iterations': [],
        'objective': [],
        'train_accuracy': [],
        'val_accuracy': [],
        'best_val_accuracy': 0.0,
        'best_iteration': 0,
        'config': {
            'C': C,
            'max_iter': max_iter,
            'tol': tol,
            'batch_size': batch_size,
            'class_balanced': class_balanced,
            'weighting_strategy': weighting_strategy,
            'n_jobs': n_jobs,
            'early_stopping': early_stopping,
            'patience': patience,
            'val_check_every': val_check_every,
            'min_delta': min_delta,
            'constrain_empty': constrain_empty,
            'is_empty_feature_index': is_empty_feature_index,
            'empty_class_index': empty_class_index,
            'empty_penalty': empty_penalty,
            'n_unary_features': n_unary_features,
            'n_pairwise_features': n_pairwise_features,
        }
    }

    if checkpoint_dir:
        checkpoint_dir = Path(checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        print(f"\nCheckpoints will be saved to: {checkpoint_dir}")

    print("\nInitializing model...")
    print(f"  Class-balanced loss: {class_balanced}")
    if class_balanced:
        print(f"  Weighting strategy:  {weighting_strategy}")
    if batch_size is not None:
        print(f"  Mini-batch size: {batch_size}")
    else:
        print(f"  Training mode: Full-batch")

    if constrain_empty:
        print(f"\n  constrain_empty: {constrain_empty}")
        print(f"  empty_penalty:   {empty_penalty}")

    model = CellTypeClassifierCRF(
        n_unary_features=n_unary_features,
        n_pairwise_features=n_pairwise_features,
        n_states=n_states,
        inference_method=inference_method,
        random_state=random_state,
        class_balanced=class_balanced,
        weighting_strategy=weighting_strategy,
        constrain_empty=constrain_empty,
        is_empty_feature_index=is_empty_feature_index,
        empty_class_index=empty_class_index,
        empty_penalty=empty_penalty,
        state_names=state_names,
    )

    val_es_kwargs = {}
    if early_stopping and X_val is not None:
        val_es_kwargs = dict(
            X_val=X_val,
            y_val=y_val,
            grid_shapes_val=grid_shapes_val,
            val_check_every=val_check_every,
            patience=patience,
            min_delta=min_delta,
        )

    print(f"\nStarting training...")
    start_time = time()
    model.fit(
        X_train, y_train,
        C=C, max_iter=max_iter, tol=tol,
        verbose=verbose, n_jobs=n_jobs,
        batch_size=batch_size,
        class_weight_strategy=class_weight_strategy,
        check_convergence_every=check_convergence_every,
        grid_shapes_train=grid_shapes_train,
        **val_es_kwargs,
    )
    training_time = time() - start_time
    training_log['training_time'] = training_time

    print(f"\nTraining completed in {training_time:.2f}s")

    if hasattr(model.ssvm, 'objective_curve_'):
        training_log['objective'] = [float(x) for x in model.ssvm.objective_curve_]
        training_log['iterations'] = list(range(len(model.ssvm.objective_curve_)))
        training_log['objective_curve'] = training_log['objective']

    if model.val_scores_:
        training_log['val_curve'] = model.val_scores_
        training_log['best_val_iter'] = int(model.best_val_iter_)
        training_log['best_val_macro_f1'] = float(model.best_val_score_)
        print(
            f"  Best val macro-F1 during training: "
            f"{model.best_val_score_:.4f} at iter {model.best_val_iter_}"
        )

    y_pred_train = model._predict_parallel(X_train, grid_shapes_train, n_jobs=n_jobs)
    train_acc = model.score(
        X_train, y_train, grid_shapes=grid_shapes_train, y_pred_list=y_pred_train
    )
    print(f"Training accuracy: {train_acc:.4f}")
    training_log['train_accuracy'].append(train_acc)

    y_pred_val = None
    if X_val is not None:
        y_pred_val = model._predict_parallel(X_val, grid_shapes_val, n_jobs=n_jobs)
        val_acc = model.score(
            X_val, y_val, grid_shapes=grid_shapes_val, y_pred_list=y_pred_val
        )
        print(f"Validation accuracy (best weights): {val_acc:.4f}")
        training_log['val_accuracy'].append(val_acc)
        training_log['best_val_accuracy'] = val_acc
        training_log['val_metrics'] = _build_val_metrics(
            model, X_val, y_val, grid_shapes_val, y_pred_list=y_pred_val
        )

    print(f"\nSaving final model to: {output_model_path}")
    model.save_model(output_model_path)

    log_path = Path(output_model_path).with_suffix('.json')
    with open(log_path, 'w') as f:
        json.dump(training_log, f, indent=2)
    print(f"Training log saved to: {log_path}")

    if X_val is not None:
        print("\n" + "=" * 80)
        print("FINAL EVALUATION ON VALIDATION SET")
        print("=" * 80)
        model.evaluate(
            X_val, y_val,
            grid_shapes=grid_shapes_val,
            y_pred_list=y_pred_val,
            n_jobs=n_jobs,
            verbose=True,
        )

    return model, training_log

_SUBSET = frozenset({"EQ08__13"})

def train_from_dataset(
    dataset_csv,
    output_model_path,
    n_states=5,
    inference_method='qpbo',
    random_state=0,
    C=1.0,
    max_iter=1000,
    tol=1e-3,
    batch_size=None,
    class_balanced=False,
    weighting_strategy='none',
    class_weight_strategy='inverse',
    use_batch_weights=False,
    check_convergence_every=100,
    early_stopping=True,
    patience=5,
    val_check_every=50,
    min_delta=1e-4,
    validation_split=0.2,
    n_jobs=-1,
    verbose=1,
    checkpoint_dir=None,
    checkpoint_every=50,
    max_grid_size=2500,
    feature_cache_dir=None,
    constrain_empty=False,
    is_empty_feature_index=None,
    empty_class_index=None,
    empty_penalty=1e6,
    state_names=None,
    rf_model=None,
):
    dataset = load_and_split_dataset(
        dataset_csv=dataset_csv,
        validation_split=validation_split,
        random_state=random_state,
        max_grid_size=max_grid_size,
        early_stopping=early_stopping,
        feature_cache_dir=feature_cache_dir,
        rf_model=rf_model,
        n_states=n_states,
    )
    return train_from_preloaded(
        dataset=dataset,
        output_model_path=output_model_path,
        n_states=n_states,
        inference_method=inference_method,
        random_state=random_state,
        C=C,
        max_iter=max_iter,
        tol=tol,
        batch_size=batch_size,
        class_balanced=class_balanced,
        weighting_strategy=weighting_strategy,
        class_weight_strategy=class_weight_strategy,
        use_batch_weights=use_batch_weights,
        check_convergence_every=check_convergence_every,
        early_stopping=early_stopping,
        patience=patience,
        val_check_every=val_check_every,
        min_delta=min_delta,
        n_jobs=n_jobs,
        verbose=verbose,
        checkpoint_dir=checkpoint_dir,
        checkpoint_every=checkpoint_every,
        constrain_empty=constrain_empty,
        is_empty_feature_index=is_empty_feature_index,
        empty_class_index=empty_class_index,
        empty_penalty=empty_penalty,
        state_names=state_names,
    )


def train_crf_crossval(
    dataset_csv,
    output_dir,
    k=5,
    random_state=2112,
    n_states=5,
    inference_method='qpbo',
    C=1.0,
    max_iter=1000,
    tol=1e-3,
    batch_size=None,
    class_balanced=False,
    weighting_strategy='none',
    class_weight_strategy='inverse',
    check_convergence_every=100,
    early_stopping=True,
    patience=5,
    val_check_every=100,
    min_delta=1e-4,
    n_jobs=-1,
    verbose=1,
    max_grid_size=2500,
    feature_cache_dir=None,
    constrain_empty=False,
    is_empty_feature_index=None,
    empty_class_index=None,
    empty_penalty=1e6,
    state_names=None,
    save_fold_models=False,
    rf_model=None,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    folds = make_folds(dataset_csv, k=k, random_state=random_state)

    print(f"\n{'=' * 70}")
    print(f"CRF K-FOLD CV  k={k}  random_state={random_state}")
    print(f"  Each row appears in val exactly once. No overlap between val sets.")
    print(f"  Total rows: {sum(len(v) for _, v in folds)}  "
          f"val per fold: {[len(v) for _, v in folds]}")
    if rf_model is not None:
        print(f"  RF-projection mode: unary features = RF predict_proba "
              f"(n_classes={len(rf_model.classes_)})")
    print(f"{'=' * 70}")

    fold_val_metrics = []

    for fold_idx, (train_df, val_df) in enumerate(folds):
        print(f"\n\n{'#' * 70}")
        print(f"  FOLD {fold_idx + 1}/{k}  ->  train={len(train_df)} rows  "
              f"val={len(val_df)} rows")
        print(f"{'#' * 70}")

        fold_dir = output_dir / f"fold_{fold_idx + 1:02d}"
        fold_dir.mkdir(exist_ok=True)

        train_df.to_csv(fold_dir / "train_manifest.csv", index=False)
        val_df.to_csv(fold_dir / "val_manifest.csv", index=False)

        fold_model_path = str(fold_dir / "model.npz")

        dataset = load_and_split_dataset(
            dataset_csv=dataset_csv,
            validation_split=0.0,
            random_state=random_state,
            max_grid_size=max_grid_size,
            early_stopping=early_stopping,
            feature_cache_dir=feature_cache_dir,
            train_df=train_df,
            val_df=val_df,
            rf_model=rf_model,
            n_states=n_states,
        )

        _, fold_log = train_from_preloaded(
            dataset=dataset,
            output_model_path=fold_model_path,
            n_states=n_states,
            inference_method=inference_method,
            random_state=random_state,
            C=C,
            max_iter=max_iter,
            tol=tol,
            batch_size=batch_size,
            class_balanced=class_balanced,
            weighting_strategy=weighting_strategy,
            class_weight_strategy=class_weight_strategy,
            check_convergence_every=check_convergence_every,
            early_stopping=early_stopping,
            patience=patience,
            val_check_every=val_check_every,
            min_delta=min_delta,
            n_jobs=n_jobs,
            verbose=verbose,
            constrain_empty=constrain_empty,
            is_empty_feature_index=is_empty_feature_index,
            empty_class_index=empty_class_index,
            empty_penalty=empty_penalty,
            state_names=state_names,
        )

        fold_log["fold"] = fold_idx + 1
        fold_log["k"] = k
        fold_log["n_train_rows"] = len(train_df)
        fold_log["n_val_rows"] = len(val_df)

        with open(fold_dir / "log.json", "w") as f:
            json.dump(fold_log, f, indent=2)

        if not save_fold_models:
            fold_model_path_obj = Path(fold_model_path)
            if fold_model_path_obj.exists():
                fold_model_path_obj.unlink()

        fold_val_metrics.append(fold_log.get("val_metrics", {}))

    def _agg(key):
        vals = [m[key] for m in fold_val_metrics if m and key in m and m[key] is not None]
        if not vals:
            return {"mean": None, "std": None, "folds": []}
        return {"mean": float(np.mean(vals)), "std": float(np.std(vals)), "folds": vals}

    summary = {
        "k": k,
        "random_state": random_state,
        "rf_projection": rf_model is not None,
        "aggregated_val_metrics": {
            "accuracy": _agg("accuracy"),
            "macro_file_macro_f1": _agg("macro_file_macro_f1"),
        },
    }

    if fold_val_metrics and fold_val_metrics[0].get("per_class_f1"):
        class_names = list(fold_val_metrics[0]["per_class_f1"].keys())
        per_class_agg = {}
        for cls in class_names:
            vals = [
                m["per_class_f1"][cls]
                for m in fold_val_metrics
                if m and "per_class_f1" in m and m["per_class_f1"].get(cls) is not None
            ]
            per_class_agg[cls] = {
                "mean": float(np.mean(vals)) if vals else None,
                "std":  float(np.std(vals))  if vals else None,
                "folds": vals,
            }
        summary["aggregated_val_metrics"]["per_class_f1"] = per_class_agg

    print(f"\n\n{'=' * 70}")
    print(f"CRF CV SUMMARY  k={k}  random_state={random_state}"
          + ("  [RF-projection]" if rf_model is not None else ""))
    print(f"{'=' * 70}")
    for metric, stats in summary["aggregated_val_metrics"].items():
        if metric == "per_class_f1":
            print(f"\n  Per-class file-macro F1:")
            for cls, cstats in stats.items():
                fold_str = "  ".join(f"{v:.4f}" for v in (cstats["folds"] or []))
                mean_s = f"{cstats['mean']:.4f}" if cstats["mean"] is not None else "N/A"
                std_s  = f"{cstats['std']:.4f}"  if cstats["std"]  is not None else "N/A"
                print(f"    {cls:12s}  mean={mean_s}  std={std_s}")
                if fold_str:
                    print(f"      per-fold : {fold_str}")
        else:
            fold_str = "  ".join(f"{v:.4f}" for v in (stats["folds"] or []))
            mean_s = f"{stats['mean']:.4f}" if stats["mean"] is not None else "N/A"
            std_s  = f"{stats['std']:.4f}"  if stats["std"]  is not None else "N/A"
            print(f"  {metric:30s}  mean={mean_s}  std={std_s}")
            if fold_str:
                print(f"    per-fold : {fold_str}")

    summary_path = output_dir / f"cv_summary_k{k}_seed{random_state}.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nCV summary saved to: {summary_path}")

    return summary


# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------

@contextlib.contextmanager
def _maybe_clamp_crf(model, force_clamp, clamp_feature_index, clamp_empty_class, clamp_penalty):
    if not force_clamp:
        yield
        return

    if getattr(model, 'constrain_empty', False):
        model_feat  = model.is_empty_feature_index
        model_label = model.empty_class_index
        model_pen = model.empty_penalty
        mismatches = []
        if clamp_feature_index is not None and clamp_feature_index != model_feat:
            mismatches.append(
                f"clamp_feature_index={clamp_feature_index} ignored (model uses {model_feat})"
            )
        if clamp_empty_class is not None and clamp_empty_class != model_label:
            mismatches.append(
                f"clamp_empty_class={clamp_empty_class} ignored (model uses {model_label})"
            )
        if mismatches:
            print(
                f"[Clamping] Model already has built-in clamping. "
                f"Caller overrides ignored: {'; '.join(mismatches)}"
            )
        else:
            print(
                f"[Clamping] Using model's built-in clamping "
                f"(feature={model_feat}, label={model_label}, penalty={model_pen:.2e})."
            )
        yield
        return

    if clamp_feature_index is None:
        raise ValueError(
            "force_clamp=True but the model was not trained with clamping "
            "and clamp_feature_index was not provided."
        )

    _MAX_SAFE = 2_000_000
    if clamp_penalty > _MAX_SAFE:
        raise ValueError(f"clamp_penalty={clamp_penalty} >= {_MAX_SAFE}. QPBO scales by 1000 before int32 conversion, this would overflow.")

    from .crf_model import ConstrainedEdgeFeatureGraphCRF

    print(f"[Clamping] Swapping model.crf for a ConstrainedEdgeFeatureGraphCRF (feature={clamp_feature_index}, label={clamp_empty_class}, penalty={clamp_penalty:.2e}).")

    original_crf = model.crf
    constrained_crf = ConstrainedEdgeFeatureGraphCRF(
        is_empty_feature_index=clamp_feature_index,
        empty_class_index=clamp_empty_class,
        penalty=clamp_penalty,
        n_states=model.n_states,
        n_features=model.n_unary_features,
        n_edge_features=model.n_pairwise_features,
        inference_method=model.inference_method,
    )
    model.crf = constrained_crf
    try:
        yield
    finally:
        model.crf = original_crf


def _process_one_row(idx, row, model, compute_metrics, has_labels, output_dir, save_predictions):
    file_path  = row['file_path']
    sheet_name = row['sheet_name'] if pd.notna(row['sheet_name']) else None
    labels_path = row.get('labels_path', None)
    mime_type = row.get('mime_type', "")

    try:
        file_format = detect_format(file_path, mime_type)
    except Exception as e:
        return {'skip': 'extraction', 'file': file_path, 'reason': str(e), 'raise': True}

    try:
        unary_features, edges, pairwise_features, _, _, _ = extract_features(
            file_path, file_format=file_format, sheet_name=sheet_name
        )
    except Exception as e:
        return {'skip': 'extraction', 'file': file_path, 'reason': str(e)}

    H, W, _ = unary_features.shape
    X_nodes = unary_features.reshape(-1, unary_features.shape[2])
    X = (X_nodes, edges, pairwise_features)

    try:
        y_pred = model.predict(X, grid_shape=(H, W))
    except Exception as e:
        return {'skip': 'extraction', 'file': file_path, 'reason': str(e)}

    y_true = None
    if compute_metrics and has_labels and pd.notna(labels_path):
        try:
            y_data = np.load(labels_path, allow_pickle=True)
            y_true = np.array(y_data['labels'], copy=True)
            y_true.flags.writeable = True
        except Exception as e:
            return {'skip': 'label_load', 'file': file_path, 'reason': str(e)}

        if y_true.shape != (H, W):
            return {
                'skip': 'shape_mismatch',
                'file': file_path,
                'label_shape': list(y_true.shape),
                'grid_shape': [H, W],
            }

    if save_predictions:
        pred_filename = f"pred_{idx}_{Path(file_path).stem}.npy"
        np.save(Path(output_dir) / pred_filename, y_pred)

    return {
        'result': {
            'file_path': file_path,
            'sheet_name': sheet_name,
            'y_pred': y_pred,
            'grid_shape': (H, W),
            'state_counts': model.get_state_counts(y_pred),
            'X': X,
            'y_true': y_true,
        }
    }


def infer_from_dataset(
    dataset_csv,
    model_path,
    output_dir,
    compute_metrics=True,
    save_predictions=True,
    n_jobs=-1,
    verbose=1,
    force_clamp=False,
    clamp_feature_index=0,
    clamp_empty_class=0,
    clamp_penalty=1_000_000,
):
    print("=" * 80)
    print("RUNNING INFERENCE ON DATASET")
    print("=" * 80)

    print(f"\nLoading model from: {model_path}")
    model = CellTypeClassifierCRF.load_model(model_path)

    print(f"\nLoading dataset from: {dataset_csv}")
    df = _load_manifest(dataset_csv)

    required_cols = ['file_path', 'sheet_name']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Dataset CSV must have column: {col}")

    has_labels = 'labels_path' in df.columns
    if compute_metrics and not has_labels:
        print("Warning: No 'labels_path' column found. Metrics will not be computed.")
        compute_metrics = False

    print(f"Found {len(df)} samples in dataset")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\nRunning inference...")

    with _maybe_clamp_crf(
        model,
        force_clamp=force_clamp,
        clamp_feature_index=clamp_feature_index,
        clamp_empty_class=clamp_empty_class,
        clamp_penalty=clamp_penalty,
    ):
        raw_outputs = Parallel(n_jobs=n_jobs, verbose=verbose)(
            delayed(_process_one_row)(
                idx, row, model, compute_metrics, has_labels,
                output_dir, save_predictions
            )
            for idx, row in df.iterrows()
        )

    results = {
        'file_paths': [],
        'sheet_names': [],
        'predictions': [],
        'grid_shapes': [],
        'state_counts': [],
    }
    X_list = []
    y_true_list = []
    grid_shapes_eval = []

    skipped_extraction = []
    skipped_shape_mismatch = []
    skipped_label_load = []

    for out in raw_outputs:
        if 'skip' in out:
            if out.get('raise'):
                raise RuntimeError(
                    f"Format detection failed for {out['file']}: {out['reason']}"
                )
            if out['skip'] == 'extraction':
                print(f"\n[SKIP] {out['file']}: {out['reason']}")
                skipped_extraction.append(out['file'])
            elif out['skip'] == 'label_load':
                print(f"\n[SKIP] Could not load labels for {out['file']}: {out['reason']}")
                skipped_label_load.append(out['file'])
            elif out['skip'] == 'shape_mismatch':
                print(
                    f"\n[SKIP] Shape mismatch for {out['file']}: "
                    f"label {out['label_shape']} != grid {out['grid_shape']}"
                )
                skipped_shape_mismatch.append(out)
            continue

        r = out['result']
        results['file_paths'].append(r['file_path'])
        results['sheet_names'].append(r['sheet_name'])
        results['predictions'].append(r['y_pred'])
        results['grid_shapes'].append(r['grid_shape'])
        results['state_counts'].append(r['state_counts'])

        if r['y_true'] is not None:
            X_list.append(r['X'])
            y_true_list.append(r['y_true'])
            grid_shapes_eval.append(r['grid_shape'])

    n_processed = len(results['predictions'])
    n_total = len(df)
    n_skipped = (len(skipped_extraction) + len(skipped_shape_mismatch) + len(skipped_label_load))

    print(f"\nSuccessfully processed {n_processed}/{n_total} samples ({n_skipped} skipped)")

    if skipped_shape_mismatch:
        print(f"\n  {len(skipped_shape_mismatch)} file(s) skipped due to label/grid shape mismatch:")
        for entry in skipped_shape_mismatch:
            print(f"    {entry['file']}: label {entry['label_shape']} vs grid {entry['grid_shape']}")

    metrics_cells = None
    if compute_metrics and X_list:
        print(f"\nEvaluating on {len(X_list)} labelled files...")
        metrics_cells = model.evaluate(
            X_list, y_true_list, grid_shapes=grid_shapes_eval, verbose=True
        )
        results['metrics'] = metrics_cells

    summary_path = output_dir / "inference_results.json"
    summary = {
        'n_samples_total': n_total,
        'n_samples_processed': n_processed,
        'n_labelled_evaluated': len(X_list),
        'n_skipped_extraction': len(skipped_extraction),
        'n_skipped_shape_mismatch': len(skipped_shape_mismatch),
        'n_skipped_label_load': len(skipped_label_load),
        'skipped_shape_mismatch': skipped_shape_mismatch,
        'file_paths': results['file_paths'],
        'sheet_names': [str(s) for s in results['sheet_names']],
        'grid_shapes': results['grid_shapes'],
        'state_counts': results['state_counts'],
    }
    if metrics_cells:
        summary['accuracy'] = float(metrics_cells['accuracy'])
        summary['per_class_report'] = {
            k: v for k, v in metrics_cells['classification_report'].items()
            if isinstance(v, dict)
        }

    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\nResults summary saved to: {summary_path}")
    return results