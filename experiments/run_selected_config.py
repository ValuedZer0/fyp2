"""
Run three manually selected configurations for one dataset
Usage: python run_selected_config.py
Edit DATASET_NAME and SELECTED_CONFIGS below before running
"""
import os

import pandas as pd

from experiment_utils import run_selected_configs


DATASET_NAME = 'iris'
N_RUNS = 100
RANDOM_STATE_BASE = 42
RESULTS_DIR = 'results'

# (outlier method, normalisation method, distance metric).
SELECTED_CONFIGS = [
    ('none', 'minmax', 'euclidean'),
    ('zscore', 'standard', 'manhattan'),
    ('zscore_robust_2.5', 'robust', 'cosine'),
]


if __name__ == '__main__':
    if len(SELECTED_CONFIGS) != 3:
        raise ValueError("SELECTED_CONFIGS must contain exactly three configurations")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    rows = run_selected_configs(
        DATASET_NAME,
        SELECTED_CONFIGS,
        n_runs=N_RUNS,
        random_state_base=RANDOM_STATE_BASE,
        min_per_class=2,
    )

    output_path = os.path.join(RESULTS_DIR, f'{DATASET_NAME}_selected_results.csv')
    pd.DataFrame(rows).to_csv(output_path, index=False)
    print(f"Saved {len(rows)} rows to {output_path}")
