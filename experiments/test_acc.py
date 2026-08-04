import numpy as np
from datasets import load_dataset
from outlier_handling import inject_outliers, zscore_filter, iqr_filter
from normalisation import minmax_scale, standard_scale, robust_scale
from kmeans_core import KMeans
from sklearn.metrics import confusion_matrix
import numpy as np

# ------------------------------------------------------------
# Robust accuracy function (same as in run_experiments.py)
# ------------------------------------------------------------
def compute_clustering_accuracy(y_true, y_pred):
    # Majority‑vote mapping per cluster
    mapping = {}
    for cluster in np.unique(y_pred):
        mask = (y_pred == cluster)
        true_in_cluster = y_true[mask]
        if len(true_in_cluster) > 0:
            mapping[cluster] = np.bincount(true_in_cluster).argmax()
        else:
            mapping[cluster] = 0   # fallback
    y_mapped = np.array([mapping[label] for label in y_pred])
    return np.mean(y_mapped == y_true)

# ------------------------------------------------------------
# Configuration (edit these values as needed)
# ------------------------------------------------------------
dataset_name = 'iris'
contamination = 0.05          # 0.0 or 0.05
outlier_method = 'zscore'     # 'none', 'zscore', 'iqr'
norm_method = 'standard'      # 'none', 'minmax', 'standard', 'robust'
metric = 'euclidean'
n_runs = 100                  # number of repetitions
random_state_base = 42

# ------------------------------------------------------------
# Experiment loop
# ------------------------------------------------------------
X, y_true = load_dataset(dataset_name)
n_clusters = len(np.unique(y_true))

accs = []
for run_idx in range(n_runs):
    seed = random_state_base + run_idx
    X_temp, y_temp = X.copy(), y_true.copy()

    # 1. Contamination
    if contamination > 0:
        X_temp, y_temp, _ = inject_outliers(
            X_temp, y_temp, contamination=contamination,
            scale_factor=5.0, random_state=seed
        )

    # 2. Outlier removal
    if outlier_method == 'zscore':
        X_temp, y_temp, _ = zscore_filter(X_temp, y_temp, threshold=3.0)
    elif outlier_method == 'iqr':
        X_temp, y_temp, _ = iqr_filter(X_temp, y_temp, multiplier=1.5)

    # Skip if too few samples
    if X_temp.shape[0] < n_clusters:
        continue

    # 3. Normalisation
    if norm_method == 'minmax':
        X_temp = minmax_scale(X_temp)
    elif norm_method == 'standard':
        X_temp = standard_scale(X_temp)
    elif norm_method == 'robust':
        X_temp = robust_scale(X_temp)

    # 4. K-means
    model = KMeans(
        n_clusters=n_clusters, max_iter=300, tol=1e-4,
        random_state=seed, metric=metric
    ).fit(X_temp)

    # 5. Accuracy (only if valid clustering)
    if len(np.unique(model.labels_)) > 1:
        acc = compute_clustering_accuracy(y_temp, model.labels_)
        accs.append(acc)

# ------------------------------------------------------------
# Display results
# ------------------------------------------------------------
if accs:
    print(f"Configuration: {dataset_name} | cont={contamination} | "
          f"outlier={outlier_method} | norm={norm_method} | metric={metric}")
    print(f"Runs completed: {len(accs)}/{n_runs}")
    print(f"Accuracy: mean={np.mean(accs):.4f}, std={np.std(accs):.4f}, "
          f"min={np.min(accs):.4f}, max={np.max(accs):.4f}")
else:
    print("No valid runs – check data or parameters.")