# experiment_utils.py
"""
Shared logic for running a single (dataset, outlier_method, norm_method,
metric) configuration n_runs times and aggregating the results.

Used by both run_one_dataset.py (single hardcoded dataset) and
run_all_dataset.py (loops over every dataset in configs.DATASETS) so the
evaluation logic only exists in one place.

Fixes applied vs. the earlier version:
  1. Outlier removal and normalisation are deterministic (no random_state)
     but were previously recomputed once per run inside the n_runs loop.
     They're now computed once per configuration, outside the loop.
  2. If outlier removal eliminates an entire true class, n_clusters (chosen
     from the ORIGINAL label set) would then be evaluated against fewer
     remaining classes than it was set up for -- a silent methodological
     mismatch. This is now detected and the run is skipped (NaN), consistent
     with how the existing "too few samples" case is already handled.
"""
import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    silhouette_score,
    f1_score,
)

from datasets import load_dataset
from outlier_handling import zscore_filter, iqr_filter
from normalisation import minmax_scale, standard_scale, robust_scale
from kmeans_core import KMeans

OUTLIER_METHODS = ['none', 'zscore', 'iqr']
NORM_METHODS = ['none', 'minmax', 'standard', 'robust']
DISTANCE_METRICS = ['euclidean', 'manhattan', 'cosine', 'minkowski',
                     'chebyshev', 'correlation', 'hamming']

METRICS_LIST = [
    'inertia', 'silhouette', 'ari', 'nmi', 'acc', 'macro_f1',
    'removed_count', 'removed_pct',
]


def optimal_mapping(y_true, y_pred):
    """Map predicted clusters to true classes using a one-to-one assignment."""
    true_labels, true_ids = np.unique(y_true, return_inverse=True)
    pred_labels, pred_ids = np.unique(y_pred, return_inverse=True)
    contingency = np.zeros((len(pred_labels), len(true_labels)), dtype=int)
    np.add.at(contingency, (pred_ids, true_ids), 1)

    pred_idx, true_idx = linear_sum_assignment(-contingency)
    return {pred_labels[p]: true_labels[t] for p, t in zip(pred_idx, true_idx)}


def _mapped_labels(y_true, y_pred):
    """Apply the optimal cluster-to-class mapping; unmatched clusters are errors."""
    mapping = optimal_mapping(y_true, y_pred)
    return np.array([mapping.get(label, -1) for label in y_pred])


def compute_accuracy(y_true, y_pred):
    y_mapped = _mapped_labels(y_true, y_pred)
    return np.mean(y_mapped == y_true)


def compute_macro_f1(y_true, y_pred):
    y_mapped = _mapped_labels(y_true, y_pred)
    return f1_score(
        y_true, y_mapped, labels=np.unique(y_true), average='macro', zero_division=0
    )


def _nan_row(metrics_list=METRICS_LIST, removed_count=np.nan, removed_pct=np.nan):
    row = {m: (np.nan, np.nan, np.nan, np.nan) for m in metrics_list}
    row['removed_count'] = (removed_count, 0.0, removed_count, removed_count)
    row['removed_pct'] = (removed_pct, 0.0, removed_pct, removed_pct)
    return row


def run_single_config(dataset_name, outlier_method, norm_method, metric,
                       n_runs=100, random_state_base=42, min_per_class=None):
    """
    Runs one (outlier_method, norm_method, metric) configuration n_runs
    times on dataset_name and returns aggregated (mean, std, min, max)
    per evaluation metric.

    min_per_class : int or None
        Passed through to zscore_filter/iqr_filter. If given, guarantees
        outlier removal never eliminates a whole class entirely (e.g.
        ecoli's imL/imS, 2 samples each) -- see outlier_handling.py for
        details. None reproduces the original (unprotected) behaviour.
    """
    X, y_true = load_dataset(dataset_name)
    n_clusters = len(np.unique(y_true))

    # ---- Preprocessing: deterministic, so computed ONCE for this config ----
    X_proc, y_proc = X.copy(), y_true.copy()

    if outlier_method == 'zscore':
        X_proc, y_proc, _ = zscore_filter(X_proc, y_proc, threshold=3.0,
                                           min_per_class=min_per_class)
    elif outlier_method == 'iqr':
        X_proc, y_proc, _ = iqr_filter(X_proc, y_proc, multiplier=1.5,
                                        min_per_class=min_per_class)
    # 'none': X_proc, y_proc stay as the original copies

    removed_count = X.shape[0] - X_proc.shape[0]
    removed_pct = 100.0 * removed_count / X.shape[0]

    if X_proc.shape[0] < n_clusters:
        # Not enough points left to even attempt n_clusters clusters.
        return _nan_row(removed_count=removed_count, removed_pct=removed_pct)

    n_classes_remaining = len(np.unique(y_proc))
    if n_classes_remaining < n_clusters:
        # Outlier removal wiped out at least one whole true class.
        # Evaluating with the ORIGINAL n_clusters against fewer remaining
        # classes silently mismatches cluster count to class count, so
        # this configuration is skipped rather than silently biased.
        return _nan_row(removed_count=removed_count, removed_pct=removed_pct)

    if norm_method == 'minmax':
        X_proc = minmax_scale(X_proc)
    elif norm_method == 'standard':
        X_proc = standard_scale(X_proc)
    elif norm_method == 'robust':
        X_proc = robust_scale(X_proc)
    # 'none': leave X_proc unscaled

    results = {m: [] for m in METRICS_LIST}

    for run_idx in range(n_runs):
        seed = random_state_base + run_idx

        model = KMeans(
            n_clusters=n_clusters,
            max_iter=300,
            tol=1e-4,
            random_state=seed,
            metric=metric,
            n_init=10,
        ).fit(X_proc)

        labels = model.labels_

        # Preprocessing is deterministic, but store it with every run so its
        # aggregate columns use the same output schema as the other metrics.
        results['removed_count'].append(removed_count)
        results['removed_pct'].append(removed_pct)

        if len(np.unique(labels)) < 2 or X_proc.shape[0] < 2:
            results['inertia'].append(model.inertia_)
            for m in ['silhouette', 'ari', 'nmi', 'acc', 'macro_f1']:
                results[m].append(np.nan)
            continue

        results['inertia'].append(model.inertia_)
        results['silhouette'].append(silhouette_score(X_proc, labels))
        results['ari'].append(adjusted_rand_score(y_proc, labels))
        results['nmi'].append(normalized_mutual_info_score(y_proc, labels))
        results['acc'].append(compute_accuracy(y_proc, labels))
        results['macro_f1'].append(compute_macro_f1(y_proc, labels))

    agg = {}
    for m in METRICS_LIST:
        vals = np.array(results[m])
        vals = vals[~np.isnan(vals)]
        if len(vals) > 0:
            agg[m] = (np.mean(vals), np.std(vals), np.min(vals), np.max(vals))
        else:
            agg[m] = (np.nan, np.nan, np.nan, np.nan)
    return agg


def run_all_configs(dataset_name, n_runs=100, random_state_base=42, verbose=True,
                     min_per_class=None):
    """
    Runs every (outlier_method, norm_method, metric) combination for one
    dataset and returns a list of result rows (dicts), ready to build a
    DataFrame from.

    min_per_class : int or None
        Forwarded to run_single_config -> zscore_filter/iqr_filter.
    """
    rows = []
    total_combos = len(OUTLIER_METHODS) * len(NORM_METHODS) * len(DISTANCE_METRICS)
    count = 0
    for out_method in OUTLIER_METHODS:
        for norm in NORM_METHODS:
            for metric in DISTANCE_METRICS:
                count += 1
                if verbose:
                    print(f"[{count}/{total_combos}] {dataset_name} | "
                          f"outlier={out_method} | norm={norm} | metric={metric}")
                agg = run_single_config(
                    dataset_name, out_method, norm, metric,
                    n_runs=n_runs, random_state_base=random_state_base,
                    min_per_class=min_per_class,
                )
                row = {
                    'dataset': dataset_name,
                    'outlier_method': out_method,
                    'norm': norm,
                    'metric': metric,
                }
                for m, (mean, std, vmin, vmax) in agg.items():
                    row[f'{m}_mean'] = mean
                    row[f'{m}_std'] = std
                    row[f'{m}_min'] = vmin
                    row[f'{m}_max'] = vmax
                rows.append(row)
    return rows
