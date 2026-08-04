# run_one_dataset.py
"""
Run all experiment combinations for a SINGLE dataset.
Usage: python run_one_dataset.py
Change DATASET_NAME below to select the dataset.
"""
import os
import pandas as pd

from experiment_utils import run_all_configs

# ---------------- Configuration ----------------
DATASET_NAME = 'glass'
N_RUNS = 100
RANDOM_STATE_BASE = 42
RESULTS_DIR = 'results'
# ------------------------------------------------

if __name__ == '__main__':
    os.makedirs(RESULTS_DIR, exist_ok=True)

    rows = run_all_configs(
        DATASET_NAME, n_runs=N_RUNS, random_state_base=RANDOM_STATE_BASE,
        min_per_class=2,
    )

    df = pd.DataFrame(rows)
    out_path = os.path.join(RESULTS_DIR, f'{DATASET_NAME}_results.csv')
    df.to_csv(out_path, index=False)
    print(f"\nSaved {len(rows)} rows to {out_path}")
