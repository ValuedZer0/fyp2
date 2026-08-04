# configs.py

# All nine UCI benchmark datasets
DATASETS = [
    'iris',
    'glass',
    'balance',
    'cancer',
    'wine',
    'vertebral',
    'ecoli',
    'blood',
    'seeds'
]

# Outlier handling methods
OUTLIER_METHODS = [
    'none',    # no outlier removal
    'zscore',  # Z-score filtering (|z| > 3)
    'iqr'      # IQR filtering (1.5 * IQR)
]

# Data normalisation techniques
NORM_METHODS = [
    'none',    # no scaling
    'minmax',  # Min-Max scaling to [0,1]
    'standard',# Z-score standardisation
    'robust'   # Robust scaling (median/IQR)
]

# Distance metrics for K-means
DISTANCE_METRICS = [
    'euclidean',
    'manhattan',
    'cosine',
    'minkowski',   # p=3
    'chebyshev',
    'correlation',
    'hamming'
]

# Contamination levels (fraction of injected outliers)
CONTAMINATION_LEVELS = [
    0.0,   # clean (original data)

]

# Number of K-means repetitions per configuration
N_RUNS = 100