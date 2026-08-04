# run_all_dataset.py
"""
Run all experiment combinations for EVERY dataset listed in configs.DATASETS.
Usage: python run_all_dataset.py
"""
import os
import pandas as pd

from configs import DATASETS
from experiment_utils import run_all_configs

# ---------------- Configuration ----------------
N_RUNS = 100
RANDOM_STATE_BASE = 42
RESULTS_DIR = 'results'
# ------------------------------------------------

if __name__ == '__main__':
    os.makedirs(RESULTS_DIR, exist_ok=True)

    for ds in DATASETS:
        print(f"\n========== Starting {ds} ==========")
        rows = run_all_configs(
            ds, n_runs=N_RUNS, random_state_base=RANDOM_STATE_BASE,
            min_per_class=2,
        )
        df = pd.DataFrame(rows)
        out_path = os.path.join(RESULTS_DIR, f'{ds}_results.csv')
        df.to_csv(out_path, index=False)
        print(f"Saved {len(rows)} rows to {out_path}")
