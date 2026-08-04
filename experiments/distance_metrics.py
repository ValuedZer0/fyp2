# kmeans/distances.py
"""
Distance metrics
Each function takes (X, centroids) -> (n_samples, n_clusters) distance array.

"""
import numpy as np


def euclidean(X, centroids):
    diffs = X[:, np.newaxis, :] - centroids[np.newaxis, :, :]
    return np.sqrt(np.sum(diffs ** 2, axis=2))


def manhattan(X, centroids):
    diffs = X[:, np.newaxis, :] - centroids[np.newaxis, :, :]
    return np.sum(np.abs(diffs), axis=2)


def cosine(X, centroids):
    norm_X = np.linalg.norm(X, axis=1, keepdims=True)
    norm_c = np.linalg.norm(centroids, axis=1, keepdims=True)
    sim = np.dot(X / (norm_X + 1e-10), (centroids / (norm_c + 1e-10)).T)
    return 1.0 - np.clip(sim, -1.0, 1.0)


def chebyshev(X, centroids):
    diffs = X[:, np.newaxis, :] - centroids[np.newaxis, :, :]
    return np.max(np.abs(diffs), axis=2)


def minkowski(X, centroids, p=3):
    diffs = X[:, np.newaxis, :] - centroids[np.newaxis, :, :]
    return np.sum(np.abs(diffs) ** p, axis=2) ** (1.0 / p)


def correlation(X, centroids):
    # Center the data
    X_c = X - X.mean(axis=1, keepdims=True)
    C_c = centroids - centroids.mean(axis=1, keepdims=True)

    m = X.shape[1]   # number of features

    # Covariance (properly normalised by m)
    cov = np.dot(X_c, C_c.T) / m

    # Standard deviations (population, ddof=0)
    std_X = np.std(X, axis=1, ddof=0).reshape(-1, 1)
    std_C = np.std(centroids, axis=1, ddof=0).reshape(1, -1)

    rho = cov / (std_X * std_C + 1e-10)
    rho = np.clip(rho, -1.0, 1.0)
    return 1.0 - rho


def hamming(X, centroids):
    diffs = X[:, np.newaxis, :] != centroids[np.newaxis, :, :]
    return np.sum(diffs, axis=2)

METRICS = {
    'euclidean': euclidean,
    'manhattan': manhattan,
    'cosine': cosine,
    'chebyshev': chebyshev,
    'minkowski': minkowski,
    'correlation': correlation,
    'hamming': hamming,
}

def get_metric(metric):
    """
    Returns the distance function for selected metrics
    """
    if callable(metric):
        return metric
    try:
        return METRICS[metric]
    except KeyError:
        raise ValueError(
            f"Unsupported metric: {metric!r}. "
            f"Available: {list(METRICS)} or pass a callable."
        )