# analyse_results.py  (best per factor tables + best per dataset by accuracy)
import pandas as pd
import numpy as np
import os

RESULT_DIR = 'results'

# ------------------------------------------------------------------
# 1. Load and combine all datasets
# ------------------------------------------------------------------
all_files = [f for f in os.listdir(RESULT_DIR) if f.endswith('_results.csv')]
dfs = []
for f in all_files:
    df = pd.read_csv(os.path.join(RESULT_DIR, f))
    dataset_name = f.replace('_results.csv', '')
    df['dataset'] = dataset_name
    dfs.append(df)
full_df = pd.concat(dfs, ignore_index=True)
print(f"Loaded {len(full_df)} rows from {len(all_files)} datasets.")

# ------------------------------------------------------------------
# Helper: best per factor per dataset (ranked by acc_mean)
# ------------------------------------------------------------------
def best_per_factor(data, factor_col, other_cols, metric='acc_mean'):
    idx = data.groupby(['dataset', factor_col])[metric].idxmax()
    best = data.loc[idx]
    cols = (['dataset', factor_col] + other_cols +
            ['ari_mean', 'ari_min', 'ari_max', 'ari_std',
             'acc_mean', 'acc_min', 'acc_max', 'acc_std',
             'silhouette_mean', 'silhouette_min', 'silhouette_max', 'silhouette_std'])
    return best[cols].sort_values(['dataset', factor_col])

# ------------------------------------------------------------------
# 2. Table A: best (norm, metric) per outlier method per dataset
# ------------------------------------------------------------------
print("\n" + "="*70)
print("TABLE A: Best (norm, metric) per dataset and outlier method")
table_outlier = best_per_factor(full_df, 'outlier_method', ['norm', 'metric'])
print(table_outlier.to_string(index=False))
table_outlier.to_csv('best_per_outlier_by_dataset.csv', index=False)

# ------------------------------------------------------------------
# 3. Table B: best (outlier, metric) per normalisation per dataset
# ------------------------------------------------------------------
print("\n" + "="*70)
print("TABLE B: Best (outlier, metric) per dataset and normalisation method")
table_norm = best_per_factor(full_df, 'norm', ['outlier_method', 'metric'])
print(table_norm.to_string(index=False))
table_norm.to_csv('best_per_norm_by_dataset.csv', index=False)

# ------------------------------------------------------------------
# 4. Table C: best (outlier, norm) per distance metric per dataset
# ------------------------------------------------------------------
print("\n" + "="*70)
print("TABLE C: Best (outlier, norm) per dataset and distance metric")
table_metric = best_per_factor(full_df, 'metric', ['outlier_method', 'norm'])
print(table_metric.to_string(index=False))
table_metric.to_csv('best_per_metric_by_dataset.csv', index=False)

# ------------------------------------------------------------------
# 5. Best configuration per dataset (ranked by acc_mean)
# ------------------------------------------------------------------
print("\n" + "="*70)
print("BEST CONFIGURATION PER DATASET (by ACCURACY mean)")
best_per_ds = full_df.loc[full_df.groupby('dataset')['acc_mean'].idxmax()]
best_per_ds_out = best_per_ds[['dataset', 'outlier_method', 'norm', 'metric',
                               'ari_mean', 'ari_min', 'ari_max', 'ari_std',
             'acc_mean', 'acc_min', 'acc_max', 'acc_std',
             'silhouette_mean', 'silhouette_min', 'silhouette_max', 'silhouette_std']]
print(best_per_ds_out.to_string(index=False))
best_per_ds_out.to_csv('best_per_dataset_by_accuracy.csv', index=False)

print("\nAnalysis complete. Tables saved as CSV.")
