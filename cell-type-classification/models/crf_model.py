"""
CRF model for table structure detection in spreadsheets.
Wraps pystruct's EdgeFeatureGraphCRF with additional functions.
Modifies also OneSlackSSVM behavior for clamping.
"""

import os
import numpy as np
from pystruct.models import EdgeFeatureGraphCRF
from pystruct.learners import OneSlackSSVM
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


class ClassBalancedEdgeFeatureGraphCRF(EdgeFeatureGraphCRF):
    """
    EdgeFeatureGraphCRF with class-balanced loss.

    weighting_strategy
    ------------------
    'global': fixed weight vector computed once from the full training-set
                 distribution and stored in self.class_weights.
    'per_file': inverse-frequency weights recomputed from each sample's
                 label vector at loss-evaluation time. class_weights is
                 ignored.
    """

    def __init__(self, class_weights=None, weighting_strategy='global', **kwargs):
        super().__init__(**kwargs)
        self.class_weights = class_weights
        self.weighting_strategy = weighting_strategy

    def _get_class_weights(self, y=None):
        if self.weighting_strategy == 'per_file' and y is not None:
            counts = np.bincount(y, minlength=self.n_states).astype(np.float64)
            n = float(len(y))
            n_present = float(np.sum(counts > 0))
            if n_present == 0:
                return np.ones(self.n_states, dtype=np.float64)
            weights = np.ones(self.n_states, dtype=np.float64)
            for c in range(self.n_states):
                if counts[c] > 0:
                    weights[c] = n / (n_present * counts[c])
            return weights

        if self.class_weights is not None:
            return np.asarray(self.class_weights, dtype=np.float64)
        return np.ones(self.n_states, dtype=np.float64)

    def loss(self, y, y_hat):
        w = self._get_class_weights(y)
        return float(np.sum(w[y] * (y != y_hat)))

    def max_loss(self, y):
        return float(np.sum(self._get_class_weights(y)[y]))

    def continuous_loss(self, y, y_hat):
        w = self._get_class_weights(y)
        return float(np.sum(w[y] * (1.0 - y_hat[np.arange(len(y)), y])))


class _EmptyConstraintMixin:
    """
    Mixin enforcing hard EMPTY constraints via unary-potential clamping.

    Two cases are enforced:
      is_empty==1: all non-EMPTY labels penalized
      is_empty==0: EMPTY label penalized
    """

    _MAX_SAFE_PENALTY = 2_000_000

    def _get_unary_potentials(self, x, w):
        unary = super()._get_unary_potentials(x, w)
        feat_col = x[0][:, self._constrain_is_empty_feature_index]
        feat_max = feat_col.max()

        if feat_max > 1.0: # Safety net
            raise ValueError(f"Feature at index {self._constrain_is_empty_feature_index} does not look binary (max={feat_max:.4f}).")

        penalty = self._constrain_penalty
        if penalty > self._MAX_SAFE_PENALTY:
            raise ValueError(f"empty_penalty={penalty} >= {self._MAX_SAFE_PENALTY}. Would overflow int32 after x1000 QPBO scaling.")

        empty_idx = self._constrain_empty_class_index
        non_empty_idx  = [i for i in range(self.n_states) if i != empty_idx]
        is_empty = feat_col != 0
        not_empty = ~is_empty

        if is_empty.any():
            unary[np.ix_(np.where(is_empty)[0], non_empty_idx)] -= penalty
        if not_empty.any():
            unary[not_empty, empty_idx] -= penalty
        return unary


class ConstrainedEdgeFeatureGraphCRF(_EmptyConstraintMixin, EdgeFeatureGraphCRF):
    def __init__(self, is_empty_feature_index, empty_class_index=0, penalty=1e6, **kwargs):
        self._constrain_is_empty_feature_index = is_empty_feature_index
        self._constrain_empty_class_index = empty_class_index
        self._constrain_penalty = penalty
        super().__init__(**kwargs)


class ConstrainedClassBalancedEdgeFeatureGraphCRF(_EmptyConstraintMixin, ClassBalancedEdgeFeatureGraphCRF):
    def __init__(self, is_empty_feature_index, empty_class_index=0, penalty=1e6, **kwargs):
        self._constrain_is_empty_feature_index = is_empty_feature_index
        self._constrain_empty_class_index = empty_class_index
        self._constrain_penalty = penalty
        super().__init__(**kwargs)

class NeverStopOneSlackSSVM(OneSlackSSVM):
    """
    Disables the 'constraint too weak' stopping criterion of OneSlackSSVM (test).
    """
    def _check_bad_constraint(self, violation, djoint_feature, loss, old_constraints, break_on_bad, tol=None):
        # Keep duplicate-constraint detection (to avoid infinite loops)
        equals = [True for djf_, loss_ in old_constraints if np.all(djf_ == djoint_feature) and loss == loss_]

        if np.any(equals):
            return True  # exact duplicate: useless

        # Original check_constraints logic if enabled
        if self.check_constraints:
            for con in old_constraints:
                violation_tmp = max(con[1] - np.dot(self.w, con[0]), 0)
                if violation - violation_tmp < -1e-5:
                    if break_on_bad:
                        raise ValueError("Bad inference")
                    return True

        return False  # Never stops on weak-but-not-duplicate constraints

class MiniBatchOneSlackSSVM(OneSlackSSVM):
    """Mini-batch variant of OneSlackSSVM."""

    def __init__(self, model, batch_size=32, sampling_strategy='uniform', check_convergence_every=100, **kwargs):
        super().__init__(model, **kwargs)
        self.batch_size = batch_size
        self.sampling_strategy = sampling_strategy
        self.violation_weights = None
        self.check_convergence_every = check_convergence_every
        self.iteration_count = 0

    def fit(self, X, Y, constraints=None, warm_start=False, initialize=True):
        X_writable = []
        for x in X:
            if isinstance(x, tuple):
                X_writable.append(tuple(arr.copy() if isinstance(arr, np.ndarray) and not arr.flags.writeable else arr for arr in x))
            else:
                X_writable.append(x)

        Y_writable = []
        for y in Y:
            if isinstance(y, np.ndarray):
                y = y.copy() if not y.flags.writeable else y
            else:
                y = np.array(y)
            Y_writable.append(y)

        self.X = X_writable
        self.Y = Y_writable

        if self.sampling_strategy == 'weighted':
            self.violation_weights = np.ones(len(X))

        if not warm_start:
            self.w = np.zeros(self.model.size_joint_feature)
            self.constraints_ = []
            self.objective_curve_ = []
            self.primal_objective_curve_ = []
            self.cached_constraint_ = []
            self.iteration_count = 0

        self._original_find_constraint = self._find_new_constraint
        self._find_new_constraint = self._find_new_constraint_minibatch
        try:
            super().fit(X_writable, Y_writable, constraints=constraints, warm_start=warm_start, initialize=False)
        finally:
            if hasattr(self, '_original_find_constraint'):
                self._find_new_constraint = self._original_find_constraint
        return self

    def _sample_batch(self, n_samples):
        if self.batch_size >= n_samples:
            return np.arange(n_samples)
        if self.sampling_strategy == 'uniform':
            return np.random.choice(n_samples, self.batch_size, replace=False)
        elif self.sampling_strategy == 'weighted':
            if self.violation_weights is None:
                self.violation_weights = np.ones(n_samples)
            p = self.violation_weights / self.violation_weights.sum()
            return np.random.choice(n_samples, self.batch_size, replace=False, p=p)
        else:
            raise ValueError(f"Unknown sampling strategy: {self.sampling_strategy}")

    def _check_full_convergence(self):
        from pystruct.utils import find_constraint
        max_v = 0.0
        for i in range(len(self.X)):
            _, _, slack, _ = find_constraint(self.model, self.X[i], self.Y[i], self.w)
            max_v = max(max_v, slack)
        converged = max_v < self.tol
        if self.verbose > 0:
            print(f"  [Full-dataset check: max_violation={max_v:.6f}, converged={converged}]")
        return converged, max_v

    def _find_new_constraint_minibatch(self, X, Y, psi_gt, constraints, check=True):
        self.iteration_count += 1

        if (self.check_convergence_every > 0 and self.iteration_count % self.check_convergence_every == 0):
            converged, _ = self._check_full_convergence()
            if converged:
                if self.verbose > 0:
                    print(f"  Converged after {self.iteration_count} iterations!")
                from pystruct.utils import find_constraint
                y_hat, djf, _, _ = find_constraint(self.model, X[0], Y[0], self.w)
                return [y_hat], djf * 0.0, 0.0

        batch_indices = self._sample_batch(len(X))

        from pystruct.utils import find_constraint
        from joblib import Parallel, delayed

        if self.n_jobs != 1:
            batch_X, batch_Y = [], []
            for i in batch_indices:
                x_i = X[i]
                batch_X.append(tuple(
                    arr.copy() if isinstance(arr, np.ndarray) and not arr.flags.writeable
                    else arr for arr in x_i
                ) if isinstance(x_i, tuple) else x_i)
                y_i = Y[i]
                batch_Y.append(y_i.copy() if isinstance(y_i, np.ndarray) and not y_i.flags.writeable else y_i)
            w = self.w.copy() if not self.w.flags.writeable else self.w
            results = Parallel(n_jobs=self.n_jobs,
                               verbose=max(0, self.verbose - 3),
                               max_nbytes=None)(
                delayed(find_constraint)(self.model, batch_X[k], batch_Y[k], w)
                for k in range(len(batch_indices))
            )
        else:
            results = [find_constraint(self.model, X[i], Y[i], self.w) for i in batch_indices]

        djoint = np.zeros(self.model.size_joint_feature)
        loss_mean = 0.0
        Y_hat = []
        for k, i in enumerate(batch_indices):
            y_hat, djf, _, loss = results[k]
            Y_hat.append(y_hat)
            djoint += djf
            loss_mean += loss
            if self.sampling_strategy == 'weighted':
                self.violation_weights[i] = max(0, loss - np.dot(self.w, djf))

        djoint /= len(batch_indices)
        loss_mean /= len(batch_indices)
        return Y_hat, djoint, loss_mean


class CellTypeClassifierCRF:
    """
    Cell-type classification using CRF.

    States: 0=EMPTY  1=HEADER  2=DATA  3=TITLE  4=OTHER

    weighting_strategy
    ------------------
    'none': standard Hamming loss (class_balanced=False, EdgeFeatureGraphCRF)
    'global': inverse-frequency global weights (set class_balanced=True)
    'per_file': per-file inverse-frequency weights (set class_balanced=True)

    The exact formula for 'global' weights (inverse vs sqrt_inverse) is
    controlled by class_weight_strategy in fit().

    constrain_empty
    ---------------
    See the _EmptyConstraintMixin class.
    """

    STATE_EMPTY = 0
    STATE_HEADER = 1
    STATE_DATA = 2
    STATE_TITLE = 3
    STATE_OTHER = 4

    DEFAULT_STATE_NAMES = ['EMPTY', 'HEADER', 'DATA', 'TITLE', 'OTHER']

    def __init__(self, n_unary_features, n_pairwise_features, n_states=None,
                 state_names=None, inference_method='ad3', random_state=0,
                 class_balanced=False,
                 weighting_strategy='none',
                 use_batch_weights=False,
                 constrain_empty=False,
                 is_empty_feature_index=None,
                 empty_class_index=None,
                 empty_penalty=1e6):

        if state_names is None:
            state_names = list(self.DEFAULT_STATE_NAMES)
        if n_states is None:
            n_states = len(state_names)
        elif n_states != len(state_names):
            raise ValueError(f"n_states={n_states} conflicts with len(state_names)={len(state_names)}")

        self.STATE_NAMES = state_names
        self.n_states = n_states
        self.n_unary_features = n_unary_features
        self.n_pairwise_features = n_pairwise_features
        self.inference_method = inference_method
        self.random_state = random_state
        self.class_balanced = class_balanced
        self.weighting_strategy = weighting_strategy
        self.use_batch_weights = use_batch_weights
        self.class_weights = None

        self.constrain_empty = constrain_empty
        self.is_empty_feature_index = is_empty_feature_index
        self.empty_class_index = empty_class_index if empty_class_index is not None else self.STATE_EMPTY
        self.empty_penalty = empty_penalty

        if constrain_empty and is_empty_feature_index is None:
            raise ValueError("is_empty_feature_index required when constrain_empty=True.")
        if constrain_empty and empty_penalty >= _EmptyConstraintMixin._MAX_SAFE_PENALTY:
            raise ValueError(f"empty_penalty={empty_penalty} would overflow int32 in QPBO.")

        if weighting_strategy != 'none' and not class_balanced:
            self.class_balanced = True

        crf_kw = dict(n_states=n_states, n_features=n_unary_features, n_edge_features=n_pairwise_features, inference_method=inference_method)
        con_kw = dict(is_empty_feature_index=is_empty_feature_index, empty_class_index=self.empty_class_index, penalty=empty_penalty)

        if self.class_balanced and constrain_empty:
            self.crf = ConstrainedClassBalancedEdgeFeatureGraphCRF(
                **con_kw, class_weights=None,
                weighting_strategy=weighting_strategy, **crf_kw)
        elif self.class_balanced:
            self.crf = ClassBalancedEdgeFeatureGraphCRF(
                class_weights=None,
                weighting_strategy=weighting_strategy, **crf_kw)
        elif constrain_empty:
            self.crf = ConstrainedEdgeFeatureGraphCRF(**con_kw, **crf_kw)
        else:
            self.crf = EdgeFeatureGraphCRF(**crf_kw)

        self.ssvm = None
        self._weights = None
        self.training_history = {'objective': [], 'train_accuracy': [], 'val_accuracy': []}

        self.val_scores_ = []
        self.best_val_score_= -np.inf
        self.best_val_iter_ = 0

    @property
    def weights(self):
        if self.ssvm is not None and hasattr(self.ssvm, 'w'):
            return self.ssvm.w
        return self._weights

    @weights.setter
    def weights(self, w):
        if w is not None:
            expected = (self.n_states * self.n_unary_features + self.n_states ** 2 * self.n_pairwise_features)
            if len(w) != expected:
                raise ValueError(f"Weight size mismatch: expected {expected}, got {len(w)}")
        self._weights = w
        if self.ssvm is not None:
            self.ssvm.w = w

    def compute_class_weights(self, y_train, strategy='inverse'):
        all_labels = np.concatenate([y.ravel() for y in y_train])
        counts = np.bincount(all_labels, minlength=self.n_states).astype(float)
        counts[counts == 0] = 1
        if strategy == 'uniform':
            return np.ones(self.n_states)
        elif strategy == 'inverse':
            return len(all_labels) / (self.n_states * counts)
        elif strategy == 'sqrt_inverse':
            return np.sqrt(len(all_labels) / (self.n_states * counts))
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

    def fit(self, X_train, y_train, C=1.0, max_iter=1000, verbose=1, n_jobs=-1,
            tol=1e-3, inference_cache=0, warm_start=False,
            batch_size=None, class_weight_strategy='inverse',
            check_convergence_every=100,
            X_val=None,
            y_val=None,
            grid_shapes_val=None,
            grid_shapes_train=None,
            val_check_every=100, # evaluate val every N cutting-plane iters
            patience=5, # stop after k checks with no improvement (after k*N iterations)
            min_delta=1e-4, # minimum improvement to count as "better"
            **kwargs):

        verbose = 1
        y_train = [
            (y.copy() if isinstance(y, np.ndarray) and not y.flags.writeable
             else np.array(y) if not isinstance(y, np.ndarray) else y)
            for y in y_train
        ]

        if self.class_balanced:
            if verbose:
                print("\nComputing class weights for balanced training...")
            if self.weighting_strategy == 'global':
                self.class_weights = self.compute_class_weights(y_train, strategy=class_weight_strategy)
                self.crf.class_weights = self.class_weights
                if verbose:
                    print("\nClass distribution:")
                    all_labels = np.concatenate([y.ravel() for y in y_train])
                    for c in range(self.n_states):
                        cnt = np.sum(all_labels == c)
                        pct = 100 * cnt / len(all_labels)
                        print(f"  {self.STATE_NAMES[c]:10s}: {cnt:6d} ({pct:5.1f}%) -> weight={self.class_weights[c]:.3f}")
            else:
                if verbose:
                    print(f"  Weighting strategy: {self.weighting_strategy}  (weights computed per-file at loss-evaluation time)")

        if self.constrain_empty and verbose:
            print(f"\n  EMPTY constraint: ENABLED (feature_col={self.is_empty_feature_index},  empty_label={self.empty_class_index},  penalty={self.empty_penalty:.2e})")

        if batch_size is None:
            if verbose:
                print(f"\nTraining with full-batch NeverStopOneSlackSSVM...")
                print(f"  C={C}, max_iter={max_iter}, tol={tol}, n_jobs={n_jobs}")
            self.ssvm = NeverStopOneSlackSSVM(
                model=self.crf, C=C, max_iter=max_iter, verbose=verbose,
                n_jobs=n_jobs, tol=tol, inference_cache=inference_cache, **kwargs)
        else:
            if verbose:
                print(f"\nTraining with mini-batch OneSlackSSVM...")
                print(f"  Batch size: {batch_size}, C={C}, max_iter={max_iter}, tol={tol}, check_every={check_convergence_every}, n_jobs={n_jobs}")
            self.ssvm = MiniBatchOneSlackSSVM(
                model=self.crf, batch_size=batch_size, sampling_strategy='uniform',
                check_convergence_every=check_convergence_every,
                C=C, max_iter=max_iter, verbose=verbose, n_jobs=n_jobs,
                tol=tol, inference_cache=inference_cache, **kwargs)

        if warm_start and self._weights is not None:
            self.ssvm.w = self._weights

        use_early_stopping = X_val is not None and y_val is not None and grid_shapes_val is not None and val_check_every > 0 and patience > 0

        if use_early_stopping:
            if verbose:
                print(f"\n  Early stopping: ENABLED  (val_check_every={val_check_every}, patience={patience}, min_delta={min_delta})")
            self._fit_early_stopping(
                X_train=X_train,
                y_train=y_train,
                X_val=X_val,
                y_val=y_val,
                grid_shapes_val=grid_shapes_val,
                grid_shapes_train=grid_shapes_train,
                max_iter=max_iter,
                val_check_every=val_check_every,
                patience=patience,
                min_delta=min_delta,
                verbose=verbose,
                n_jobs=n_jobs,
            )
        else:
            self.ssvm.fit(X_train, y_train)

        if hasattr(self.ssvm, 'objective_curve_'):
            self.training_history['objective'] = self.ssvm.objective_curve_

        if verbose:
            print(f"\nTraining completed.")
            if hasattr(self.ssvm, 'objective_curve_') and self.ssvm.objective_curve_:
                print(f"Final objective: {self.ssvm.objective_curve_[-1]:.4f}")
                print(f"Total cutting-plane iterations: {len(self.ssvm.objective_curve_)}")
        return self

    def _fit_early_stopping(
        self,
        X_train, y_train,
        X_val, y_val, grid_shapes_val,
        grid_shapes_train,
        max_iter,
        val_check_every,
        patience,
        min_delta,
        verbose,
        n_jobs,
    ):
        """
        Run SSVM training in chunks of `val_check_every` cutting-plane
        iterations, evaluating val macro-F1 after every chunk.

        The SSVM is continued via warm_start=True so that the accumulated
        constraint set and QP solution carry over between chunks: this is
        equivalent to running one long training session, just with
        periodic val evaluation and early termination.

        After training, self.ssvm.w gets the weights that achieved the best
        val macro-F1.
        """

        verbose = 1
        best_score = -np.inf
        best_weights = None
        no_improve = 0
        self.val_scores_ = []
        self.best_val_score_ = -np.inf
        self.best_val_iter_ = 0

        total_iters = 0
        first_chunk = True

        while total_iters < max_iter:
            chunk = min(val_check_every, max_iter - total_iters)

            self.ssvm.max_iter = chunk
            self.ssvm.fit(X_train, y_train, warm_start=not first_chunk)
            first_chunk = False

            new_total = len(getattr(self.ssvm, 'objective_curve_', []))
            new_iters = new_total - total_iters
            total_iters = new_total

            y_pred_val = self._predict_parallel(X_val, grid_shapes_val, n_jobs=n_jobs)
            score = self.score_file_macro_f1(X_val, y_val,grid_shapes=grid_shapes_val,y_pred_list=y_pred_val)

            entry = {'iter': total_iters, 'val_macro_f1': float(score)}
            self.val_scores_.append(entry)

            if verbose >= 1:
                print(
                    f"  [ES  iter={total_iters:4d}/{max_iter}]  val_macro_f1={score:.4f}   best={best_score:.4f}   no_improve={no_improve}/{patience}")

            if score > best_score + min_delta:
                best_score = score
                best_weights = self.ssvm.w.copy()
                no_improve = 0
                self.best_val_score_ = best_score
                self.best_val_iter_ = total_iters
            else:
                no_improve += 1

            natural_convergence = (new_iters < chunk)
            patience_exceeded = (no_improve >= patience)

            if natural_convergence or patience_exceeded:
                if verbose >= 1:
                    reason = "SSVM tol reached (natural convergence)" if natural_convergence else f"no val improvement for {patience} consecutive checks"
                    print(f"  [ES] Stopping: {reason}.\n       Best val macro-F1={best_score:.4f} at iter {self.best_val_iter_}")
                break

        if best_weights is not None:
            self.ssvm.w = best_weights
            if verbose >= 1:
                print(f"  [ES] Restored best weights  (val macro-F1={best_score:.4f} at iter {self.best_val_iter_})")

    def predict(self, X, grid_shape=None):
        if self.weights is None:
            raise ValueError("Model not trained.")
        if not isinstance(X, tuple) or len(X) != 3:
            raise ValueError("X must be (unary_features, edges, pairwise_features)")
        unary = X[0]
        if grid_shape is None:
            n = unary.shape[0]
            H = int(np.sqrt(n)); W = n // H
            if H * W != n:
                raise ValueError(f"Cannot infer grid shape from {n} nodes.")
            grid_shape = (H, W)
        H, W = grid_shape
        if H * W != unary.shape[0]:
            raise ValueError(f"Grid {grid_shape} doesn't match {unary.shape[0]} nodes.")
        return self.crf.inference(X, self.weights).reshape(H, W)

    def _predict_parallel(self, X_list, grid_shapes, n_jobs=1):
        """
        Run inference on a list of samples in parallel.

        Uses prefer="threads" because QPBO releases the GIL, giving 
        parallelism without the serialisation cost of forked processes.
        Switch to prefer="processes" if you use AD3 inference.
        """
        if self.weights is None:
            raise ValueError("Model not trained.")
        from joblib import Parallel, delayed
        return Parallel(n_jobs=n_jobs, prefer="threads")(
            delayed(self.predict)(X, grid_shape=gs)
            for X, gs in zip(X_list, grid_shapes)
        )

    def predict_batch(self, X_list, grid_shapes=None):
        if grid_shapes is None:
            grid_shapes = [None] * len(X_list)
        return [self.predict(X, grid_shape=gs) for X, gs in zip(X_list, grid_shapes)]

    def _resolve_grid_shapes(self, y_test):
        return [
            y.shape if y.ndim == 2
            else (int(np.sqrt(len(y))), len(y) // int(np.sqrt(len(y))))
            for y in y_test
        ]

    def score(self, X_test, y_test, grid_shapes=None, detailed=False, y_pred_list=None, n_jobs=1):
        """
        Cell-micro accuracy (and optionally a full sklearn report).
        """

        if not isinstance(X_test, list):
            X_test = [X_test]; y_test = [y_test]
        if grid_shapes is None:
            grid_shapes = self._resolve_grid_shapes(y_test)
        if y_pred_list is None:
            y_pred_list = self._predict_parallel(X_test, grid_shapes, n_jobs=n_jobs)

        y_true_all, y_pred_all = [], []
        for y_true, y_pred in zip(y_test, y_pred_list):
            y_true_all.extend(
                y_true.ravel() if hasattr(y_true, 'ravel') else np.array(y_true).ravel())
            y_pred_all.extend(y_pred.ravel())

        y_true_all = np.array(y_true_all)
        y_pred_all = np.array(y_pred_all)
        accuracy   = accuracy_score(y_true_all, y_pred_all)

        if not detailed:
            return accuracy

        report = classification_report(y_true_all, y_pred_all, target_names=self.STATE_NAMES, output_dict=True, zero_division=0)
        cm = confusion_matrix(y_true_all, y_pred_all)
        return {
            'accuracy': accuracy,
            'classification_report': report,
            'confusion_matrix': cm,
            'report_string': classification_report(y_true_all, y_pred_all, target_names=self.STATE_NAMES, zero_division=0),
        }

    def score_file_macro_f1(self, X_test, y_test, grid_shapes=None, detailed=False, y_pred_list=None, n_jobs=1):
        """
        File-macro F1: per-file F1 averaged over classes.
        """
        if not isinstance(X_test, list):
            X_test = [X_test]; y_test = [y_test]
        if grid_shapes is None:
            grid_shapes = self._resolve_grid_shapes(y_test)
        if y_pred_list is None:
            y_pred_list = self._predict_parallel(X_test, grid_shapes, n_jobs=n_jobs)

        n_classes  = len(self.STATE_NAMES)
        n_files = len(X_test)
        f1_matrix  = np.zeros((n_files, n_classes), dtype=float)
        valid_mask = np.zeros((n_files, n_classes), dtype=bool)

        for f_idx, (y_true, y_pred) in enumerate(zip(y_test, y_pred_list)):
            ytf = y_true.ravel() if hasattr(y_true, "ravel") else np.array(y_true).ravel()
            ypf = y_pred.ravel()
            for c in range(n_classes):
                gt = ytf == c; pr = ypf == c
                TP = np.sum(gt & pr)
                FP = np.sum(~gt & pr)
                FN = np.sum(gt & ~pr)
                d  = 2 * TP + FP + FN
                if d > 0:
                    valid_mask[f_idx, c] = True
                    f1_matrix[f_idx, c]  = (2 * TP) / d

        per_class_f1 = np.zeros(n_classes, dtype=float)
        support_files = np.zeros(n_classes, dtype=int)
        for c in range(n_classes):
            m = valid_mask[:, c]
            support_files[c] = np.sum(m)
            per_class_f1[c] = np.mean(f1_matrix[m, c]) if support_files[c] > 0 else np.nan

        macro_f1 = np.mean(per_class_f1[~np.isnan(per_class_f1)])

        if not detailed:
            return macro_f1
        return {
            "macro_file_macro_f1": macro_f1,
            "per_class_f1": dict(zip(self.STATE_NAMES, per_class_f1)),
            "support_files_per_class": dict(zip(self.STATE_NAMES, support_files)),
            "f1_matrix": f1_matrix,
            "valid_mask": valid_mask,
        }

    def evaluate(self, X_test, y_test, grid_shapes=None, verbose=True, y_pred_list=None, n_jobs=1):
        """
        Full evaluation: cell-micro + file-macro metrics.
        """

        if not isinstance(X_test, list):
            X_test = [X_test]; y_test = [y_test]
        if grid_shapes is None:
            grid_shapes = self._resolve_grid_shapes(y_test)

        if y_pred_list is None:
            y_pred_list = self._predict_parallel(X_test, grid_shapes, n_jobs=n_jobs)

        metrics_cells = self.score( X_test, y_test, grid_shapes=grid_shapes, detailed=True, y_pred_list=y_pred_list)
        metrics_files = self.score_file_macro_f1(X_test, y_test, grid_shapes=grid_shapes, detailed=True, y_pred_list=y_pred_list)

        if verbose:
            print("\n" + "="*60)
            print("EVALUATION RESULTS")
            print("="*60)
            print(f"\nOverall Accuracy (cell-micro): {metrics_cells['accuracy']:.4f}")
            print(f"Macro F1 (file-macro, class-macro): {metrics_files['macro_file_macro_f1']:.4f}")
            print("\nPer-Class Metrics (FILE-MACRO FAIR F1):")
            for sn in self.STATE_NAMES:
                f1 = metrics_files["per_class_f1"][sn]
                nf = metrics_files["support_files_per_class"][sn]
                print(f"  {sn:10s}: F1={'N/A' if np.isnan(f1) else f'{f1:.3f}'}, files_used={nf}")
            print("\nPer-Class Metrics (CELL-MICRO sklearn report):")
            for sn in self.STATE_NAMES:
                r = metrics_cells['classification_report'].get(sn, {})
                if r:
                    print(f"  {sn:10s}: P={r['precision']:.3f}, R={r['recall']:.3f}, F1={r['f1-score']:.3f}, n={int(r['support'])}")
            print("\nConfusion Matrix:")
            print("Rows=True, Cols=Predicted")
            print("States:", self.STATE_NAMES)
            print(metrics_cells['confusion_matrix'])
            print("="*60)

        return metrics_cells

    def get_state_name(self, state_id):
        return self.STATE_NAMES[state_id] if 0 <= state_id < len(self.STATE_NAMES) else f"UNKNOWN_{state_id}"

    def get_state_counts(self, predictions):
        flat = predictions.ravel() if hasattr(predictions, 'ravel') else np.array(predictions).ravel()
        return {self.get_state_name(c): int(np.sum(flat == c)) for c in range(self.n_states)}

    def save_model(self, filepath):
        if self.weights is None:
            raise ValueError("No weights to save.")
        d = {
            'weights': self.weights,
            'n_states': self.n_states,
            'state_names': np.array(self.STATE_NAMES),
            'n_unary_features': self.n_unary_features,
            'n_pairwise_features': self.n_pairwise_features,
            'inference_method': self.inference_method,
            'random_state': self.random_state,
            'class_balanced': self.class_balanced,
            'weighting_strategy': self.weighting_strategy,
            'use_batch_weights': self.use_batch_weights,
            'constrain_empty': self.constrain_empty,
            'is_empty_feature_index':self.is_empty_feature_index if self.is_empty_feature_index is not None else -1,
            'empty_class_index': self.empty_class_index,
            'empty_penalty': self.empty_penalty,
        }
        if self.class_weights is not None:
            d['class_weights'] = self.class_weights
        np.savez(filepath, **d)
        print(f"Model saved to {filepath}")

    @classmethod
    def load_model(cls, filepath):
        data = np.load(filepath, allow_pickle=True)

        def _b(k, default=False): return bool(data[k]) if k in data else default
        def _i(k, default=0): return int(data[k]) if k in data else default
        def _f(k, default=0.0): return float(data[k]) if k in data else default
        def _s(k, default=''): return str(data[k]) if k in data else default

        class_balanced = _b('class_balanced')
        use_batch_weights = _b('use_batch_weights')
        weighting_strategy = _s('weighting_strategy', 'none')
        constrain_empty = _b('constrain_empty')
        _fi = _i('is_empty_feature_index', -1)
        is_empty_feature_index = _fi if _fi >= 0 else None
        empty_class_index = _i('empty_class_index', 0)
        empty_penalty = _f('empty_penalty', 1e6)
        state_names = list(data['state_names'].astype(str)) if 'state_names' in data else list(cls.DEFAULT_STATE_NAMES)

        model = cls(
            n_unary_features = _i('n_unary_features'),
            n_pairwise_features = _i('n_pairwise_features'),
            n_states = _i('n_states'),
            inference_method = _s('inference_method', 'ad3'),
            random_state = _i('random_state'),
            class_balanced = class_balanced,
            weighting_strategy = weighting_strategy,
            use_batch_weights = use_batch_weights,
            constrain_empty = constrain_empty,
            is_empty_feature_index = is_empty_feature_index,
            empty_class_index = empty_class_index,
            empty_penalty = empty_penalty,
            state_names = state_names,
        )
        model.weights = data['weights']

        if 'class_weights' in data:
            model.class_weights = data['class_weights']
            if model.class_balanced and hasattr(model.crf, 'class_weights'):
                model.crf.class_weights = model.class_weights

        print(f"Model loaded from {filepath}")
        print(f"  Class-balanced:     {model.class_balanced}")
        print(f"  Weighting strategy: {model.weighting_strategy}")
        print(f"  EMPTY constraint:   {model.constrain_empty}", end="")
        if model.constrain_empty:
            print(f"  (feature_col={model.is_empty_feature_index}, label={model.empty_class_index}, penalty={model.empty_penalty:.2e})")
        else:
            print()
        return model