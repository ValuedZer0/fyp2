
import numpy as np

def minmax_scale(X):
    """Min‑Max scaling: x' = (x - min) / (max - min)  (Eq. 2.13)"""
    X = np.asarray(X, dtype=float)
    min_vals = np.min(X, axis=0)
    max_vals = np.max(X, axis=0)
    range_vals = max_vals - min_vals
    range_vals[range_vals == 0] = 1.0   
    return (X - min_vals) / range_vals

def standard_scale(X):
    """Z‑score standardisation: x' = (x - μ) / σ  (Eq. 2.14)"""
    X = np.asarray(X, dtype=float)
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0, ddof=0)   
    std[std == 0] = 1.0
    return (X - mean) / std

def robust_scale(X):
    """Robust scaling: x' = (x - median) / IQR  (Eq. 2.15)"""
    X = np.asarray(X, dtype=float)
    median = np.median(X, axis=0)
    q1 = np.percentile(X, 25, axis=0)
    q3 = np.percentile(X, 75, axis=0)
    iqr = q3 - q1
    iqr[iqr == 0] = 1.0
    return (X - median) / iqr