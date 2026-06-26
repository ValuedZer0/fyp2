import numpy as np
from scipy.stats import iqr

def zscore_filter(X, threshold=3.0):
    """Remove instances where any feature has |z| > threshold."""
    z_scores = np.abs((X - X.mean(axis=0)) / X.std(axis=0))
    mask = (z_scores < threshold).all(axis=1)
    removed = (~mask).sum()
    return X[mask], removed

def iqr_filter(X, multiplier=1.5):
    """Remove instances outside [Q1 - multiplier*IQR, Q3 + multiplier*IQR] on any feature."""
    Q1 = np.percentile(X, 25, axis=0)
    Q3 = np.percentile(X, 75, axis=0)
    IQR = iqr(X, axis=0)
    lower = Q1 - multiplier * IQR
    upper = Q3 + multiplier * IQR
    mask = ((X >= lower) & (X <= upper)).all(axis=1)
    removed = (~mask).sum()
    return X[mask], removed