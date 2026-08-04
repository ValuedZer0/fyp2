# outlier_handling.py
import numpy as np


# def inject_outliers(X, y=None, contamination=0.05, scale_factor=5.0, random_state=None):
#     """
#     Add synthetic outliers to a copy of X (and y if given).

#     Parameters
#     ----------
#     X : ndarray (n_samples, n_features)
#     y : ndarray (n_samples,) or None
#     contamination : float, fraction of points to perturb (e.g. 0.05 = 5%)
#     scale_factor : float, noise magnitude = std(X) * scale_factor
#     random_state : int or None for reproducibility

#     Returns
#     -------
#     X_cont : ndarray, contaminated data
#     y_cont : ndarray (if y provided), otherwise None
#     outlier_idx : indices of injected outliers
#     """
#     rng = np.random.default_rng(random_state)
#     n_samples = X.shape[0]
#     n_outliers = int(n_samples * contamination)
#     if n_outliers == 0:
#         return (X.copy(), y.copy() if y is not None else None, np.array([], dtype=int))

#     idx = rng.choice(n_samples, size=n_outliers, replace=False)
#     stds = np.std(X, axis=0) + 1e-10
#     noise = rng.normal(loc=0, scale=stds * scale_factor, size=(n_outliers, X.shape[1]))
#     X_cont = X.copy()
#     X_cont[idx] = X[idx] + noise

#     if y is not None:
#         y_cont = y.copy()
#         return X_cont, y_cont, idx
#     else:
#         return X_cont, idx


def _protect_min_class_size(X, y, mask, min_per_class, extremeness_score):
    """
    Given a boolean keep-mask, ensure every class in y retains at least
    min_per_class points, by adding back the LEAST extreme removed points
    for any class that would otherwise fall below that threshold.

    This exists because small/imbalanced classes (e.g. ecoli's imL/imS,
    2 samples each) can sit at genuinely extreme values relative to the
    GLOBAL feature distribution, causing global z-score/IQR filtering to
    eliminate them entirely -- which then makes downstream evaluation
    invalid, since n_clusters (chosen from the original class count) no
    longer matches the number of classes actually remaining.
    """
    mask = mask.copy()
    for cls in np.unique(y):
        cls_idx = np.where(y == cls)[0]
        n_kept = mask[cls_idx].sum()
        if n_kept < min_per_class:
            removed_idx = cls_idx[~mask[cls_idx]]
            # add back the least extreme of the removed points for this class
            order = removed_idx[np.argsort(extremeness_score[removed_idx])]
            need = min_per_class - n_kept
            mask[order[:need]] = True
    return mask


def zscore_filter(X, y=None, threshold=3.0, min_per_class=None):
    """
    Remove points where any feature has |z| > threshold.

    Parameters
    ----------
    X : ndarray (n_samples, n_features)
    y : ndarray (n_samples,) or None
    threshold : float
    min_per_class : int or None
        If given AND y is provided, guarantees every class retains at
        least this many points, protecting small/extreme classes from
        being entirely eliminated. Ignored if y is None.

    Returns
    -------
    X_clean : ndarray
    y_clean : ndarray (if y provided), else None
    mask : boolean array indicating which rows were kept
    """
    z_scores = np.abs((X - X.mean(axis=0)) / (X.std(axis=0) + 1e-10))
    mask = (z_scores < threshold).all(axis=1)
    if y is not None:
        if min_per_class is not None:
            extremeness = z_scores.max(axis=1)  # higher = more extreme
            mask = _protect_min_class_size(X, y, mask, min_per_class, extremeness)
        return X[mask], y[mask], mask
    return X[mask], mask


def iqr_filter(X, y=None, multiplier=1.5, min_per_class=None):
    """
    Remove points outside Tukey's fences: [Q1 - multiplier*IQR, Q3 + multiplier*IQR].

    Parameters
    ----------
    X : ndarray (n_samples, n_features)
    y : ndarray (n_samples,) or None
    multiplier : float
    min_per_class : int or None
        If given AND y is provided, guarantees every class retains at
        least this many points, protecting small/extreme classes from
        being entirely eliminated. Ignored if y is None.

    Returns
    -------
    X_clean : ndarray
    y_clean : ndarray (if y provided), else None
    mask : boolean array indicating which rows were kept
    """
    Q1 = np.percentile(X, 25, axis=0)
    Q3 = np.percentile(X, 75, axis=0)
    IQR = Q3 - Q1 + 1e-10
    lower = Q1 - multiplier * IQR
    upper = Q3 + multiplier * IQR
    mask = ((X >= lower) & (X <= upper)).all(axis=1)
    if y is not None:
        if min_per_class is not None:
            # distance beyond the nearest fence, per feature, as an
            # extremeness score (0 if within fences on that feature)
            beyond_lower = np.maximum(lower - X, 0)
            beyond_upper = np.maximum(X - upper, 0)
            extremeness = (beyond_lower + beyond_upper).max(axis=1)
            mask = _protect_min_class_size(X, y, mask, min_per_class, extremeness)
        return X[mask], y[mask], mask
    return X[mask], mask