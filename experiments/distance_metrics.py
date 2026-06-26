from sklearn.preprocessing import Normalizer
from sklearn.cluster import KMeans

def run_kmeans(X, n_clusters, metric='euclidean', random_state=42, n_init=1):
    if metric == 'cosine':
        # Normalize to unit length, then use Euclidean
        X = Normalizer(norm='l2').fit_transform(X)
        metric = 'euclidean'
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=n_init)
    kmeans.fit(X)
    return kmeans.labels_, kmeans.inertia_