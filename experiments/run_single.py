import numpy as np
from sklearn.datasets import load_iris, load_wine   # add more later
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    silhouette_score,
    accuracy_score,
    confusion_matrix
)
from scipy.optimize import linear_sum_assignment
from scipy.stats import iqr

# Import your own modules
from normalisation import minmax_scale, standard_scale, robust_scale
from distance_metrics import run_kmeans


def compute_accuracy(y_true, y_pred):
    """
    Map cluster labels to true labels using the Hungarian algorithm,
    then compute classification accuracy.
    """
    cm = confusion_matrix(y_true, y_pred)
    row_ind, col_ind = linear_sum_assignment(-cm)
    mapping = dict(zip(col_ind, row_ind))
    y_pred_mapped = np.array([mapping[label] for label in y_pred])
    return accuracy_score(y_true, y_pred_mapped)


def run_single_config(dataset_name, outlier, norm, metric, n_runs=30):
    """
    Runs a single experimental configuration.
    Returns a dictionary with mean, std, min, max for each metric.
    """
    # Load dataset
    if dataset_name == 'iris':
        data = load_iris()
        X, y_true = data.data, data.target
    elif dataset_name == 'wine':
        data = load_wine()
        X, y_true = data.data, data.target
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    # Storage for metrics across runs
    metrics_list = ['inertia', 'silhouette', 'ari', 'nmi', 'acc']
    results = {m: [] for m in metrics_list}

    for seed in range(n_runs):
        X_temp = X.copy()
        y_temp = y_true.copy()

        # --- Outlier handling ---
        if outlier == 'zscore':
            z_scores = np.abs((X_temp - X_temp.mean(axis=0)) / X_temp.std(axis=0))
            mask = (z_scores < 3.0).all(axis=1)
            X_temp = X_temp[mask]
            y_temp = y_temp[mask]
        elif outlier == 'iqr':
            Q1 = np.percentile(X_temp, 25, axis=0)
            Q3 = np.percentile(X_temp, 75, axis=0)
            IQR = iqr(X_temp, axis=0)
            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR
            mask = ((X_temp >= lower) & (X_temp <= upper)).all(axis=1)
            X_temp = X_temp[mask]
            y_temp = y_temp[mask]
        # else 'none': do nothing

        # --- Normalisation ---
        if norm == 'minmax':
            X_temp = minmax_scale(X_temp)
        elif norm == 'standard':
            X_temp = standard_scale(X_temp)
        elif norm == 'robust':
            X_temp = robust_scale(X_temp)
        # else 'none': do nothing

        # --- K‑means clustering ---
        n_clusters = len(np.unique(y_true))   # true number of classes
        labels, inertia = run_kmeans(X_temp, n_clusters=n_clusters,
                                     metric=metric, random_state=seed)

        # --- Compute evaluation metrics ---
        # Only compute if more than one cluster exists
        if len(np.unique(labels)) > 1:
            results['inertia'].append(inertia)
            results['silhouette'].append(silhouette_score(X_temp, labels))
            results['ari'].append(adjusted_rand_score(y_temp, labels))
            results['nmi'].append(normalized_mutual_info_score(y_temp, labels))
            results['acc'].append(compute_accuracy(y_temp, labels))
        # If only one cluster, metrics are undefined; skip that run

    # Aggregate statistics: mean, std, min, max
    agg = {}
    for m in metrics_list:
        vals = results[m]
        if vals:
            agg[m] = (np.mean(vals), np.std(vals), np.min(vals), np.max(vals))
        else:
            agg[m] = (np.nan, np.nan, np.nan, np.nan)

    return agg