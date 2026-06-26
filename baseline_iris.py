import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score

# Load data
iris = load_iris()
X = iris.data
y_true = iris.target

# Run K-means once
kmeans = KMeans(n_clusters=3, random_state=42, n_init=1)
y_pred = kmeans.fit_predict(X)

# Compute metrics
ari = adjusted_rand_score(y_true, y_pred)
nmi = normalized_mutual_info_score(y_true, y_pred)
sil = silhouette_score(X, y_pred)
inertia = kmeans.inertia_

print(f"Inertia: {inertia:.4f}")
print(f"Silhouette: {sil:.4f}")
print(f"ARI: {ari:.4f}")
print(f"NMI: {nmi:.4f}")