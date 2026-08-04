# kmeans_core.py
import numpy as np
from distance_metrics import get_metric

class KMeans:
    def __init__(self, n_clusters, max_iter=300, tol=1e-4,
                 random_state=None, metric='euclidean', n_init=10):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.metric = metric
        self.n_init = n_init
        self._distance_fn = get_metric(metric)

        self.centroids = None
        self.labels_ = None
        self.inertia_ = None
        self.n_iter_ = 0

    def fit(self, X):
        X = np.asarray(X, dtype=float)
        n_samples = X.shape[0]

        if self.n_clusters > n_samples:
            raise ValueError(
                f"n_clusters={self.n_clusters} cannot exceed "
                f"n_samples={n_samples}"
            )

        rng = np.random.default_rng(self.random_state)

        best_inertia = np.inf
        best_labels = None
        best_centroids = None
        best_iters = None          # winner’s iterations
        total_iters = 0            # total across restarts

        for _ in range(self.n_init):
            seed = rng.integers(0, 2**31)
            labels, centroids, inertia, iters = self._fit_one_init(X, seed)
            total_iters += iters
            if inertia < best_inertia:
                best_inertia = inertia
                best_labels = labels
                best_centroids = centroids
                best_iters = iters

        self.centroids = best_centroids
        self.labels_ = best_labels
        self.inertia_ = best_inertia
        self.n_iter_ = best_iters              # sklearn convention
        self.total_iter_ = total_iters          # optional, total compute
        return self

    def _fit_one_init(self, X, seed):
        n = X.shape[0]
        rng = np.random.default_rng(seed)
        centroids = self._init_centroids(X, rng)

        for it in range(self.max_iter):
            labels = self._assign(X, centroids)
            # pass centroids explicitly to _update
            new_centroids = self._update(X, labels, rng, centroids)

            shift = np.linalg.norm(new_centroids - centroids)
            centroids = new_centroids
            if shift < self.tol:
                break

        labels = self._assign(X, centroids)
        inertia = self._compute_inertia_centroids(X, labels, centroids)
        return labels, centroids, inertia, it + 1

    def _init_centroids(self, X, rng):
        indices = rng.choice(len(X), size=self.n_clusters, replace=False)
        return X[indices].copy()

    def _distances(self, X, centroids):
        return self._distance_fn(X, centroids)

    def _assign(self, X, centroids):
        return np.argmin(self._distances(X, centroids), axis=1)

    def _update(self, X, labels, rng, centroids=None):
        """Update centroids using means, or component-wise modes for Hamming."""
        if centroids is None:
            centroids = self.centroids
        new_centroids = np.zeros_like(centroids)
        for j in range(self.n_clusters):
            members = X[labels == j]
            if len(members) > 0:
                if self.metric == 'hamming':
                    new_centroids[j] = self._componentwise_mode(members)
                else:
                    new_centroids[j] = members.mean(axis=0)
            else:
                new_centroids[j] = X[rng.integers(len(X))]
        return new_centroids

    @staticmethod
    def _componentwise_mode(X):
        """Return a deterministic categorical prototype for Hamming distance."""
        modes = np.empty(X.shape[1], dtype=X.dtype)
        for feature_idx in range(X.shape[1]):
            values, counts = np.unique(X[:, feature_idx], return_counts=True)
            modes[feature_idx] = values[np.argmax(counts)]
        return modes

    @staticmethod
    def _compute_inertia_centroids(X, labels, centroids):
        return float(np.sum((X - centroids[labels]) ** 2))

    def predict(self, X):
        if self.centroids is None:
            raise RuntimeError("Call fit() before predict().")
        X = np.asarray(X, dtype=float)
        return self._assign(X, self.centroids)

    def fit_predict(self, X):
        self.fit(X)
        return self.labels_


def run_kmeans(X, n_clusters, metric='euclidean', random_state=42, n_init=10):
    model = KMeans(
        n_clusters=n_clusters,
        metric=metric,
        random_state=random_state,
        n_init=n_init
    ).fit(X)
    return model.labels_, model.inertia_
