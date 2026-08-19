# analyse_results.py  (best per factor tables + best per dataset by accuracy)
import pandas as pd
import numpy as np
import os

RESULT_DIR = 'results'

# 1. Load and combine all datasets
all_files = [f for f in os.listdir(RESULT_DIR) if f.endswith('_results.csv')]
dfs = []
for f in all_files:
    df = pd.read_csv(os.path.join(RESULT_DIR, f))
    dataset_name = f.replace('_results.csv', '')
    df['dataset'] = dataset_name
    dfs.append(df)
full_df = pd.concat(dfs, ignore_index=True)
print(f"Loaded {len(full_df)} rows from {len(all_files)} datasets.")


# Helper: best per factor per dataset (ranked by ari_mean)
def best_per_factor(data, factor_col, other_cols, metric='ari_mean'):
    """
    For each (dataset, factor_col) group, keep ALL rows that achieve the
    maximum value of `metric`. This correctly displays ties instead of
    silently picking the first one.
    """
    max_vals = data.groupby(['dataset', factor_col])[metric].transform('max')
    best = data[data[metric] == max_vals].copy()

    cols = (['dataset', factor_col] + other_cols +
            ['ari_mean', 'ari_min', 'ari_max', 'ari_std',
             'acc_mean', 'acc_min', 'acc_max', 'acc_std',
             'hungarian_acc_mean', 'hungarian_acc_std', 'hungarian_acc_min', 'hungarian_acc_max',
             'silhouette_mean', 'silhouette_min', 'silhouette_max', 'silhouette_std'])
    return best[cols].sort_values(['dataset', factor_col])


# 2. Table A: best (norm, metric) per outlier method per dataset
print("\n" + "="*70)
print("TABLE A: Best (norm, metric) per dataset and outlier method")
table_outlier = best_per_factor(full_df, 'outlier_method', ['norm', 'metric'])
print(table_outlier.to_string(index=False))
table_outlier.to_csv('best_per_outlier_by_dataset.csv', index=False)

# 3. Table B: best (outlier, metric) per normalisation per dataset
print("\n" + "="*70)
print("TABLE B: Best (outlier, metric) per dataset and normalisation method")
table_norm = best_per_factor(full_df, 'norm', ['outlier_method', 'metric'])
print(table_norm.to_string(index=False))
table_norm.to_csv('best_per_norm_by_dataset.csv', index=False)

# 4. Table C: best (outlier, norm) per distance metric per dataset
print("\n" + "="*70)
print("TABLE C: Best (outlier, norm) per dataset and distance metric")
table_metric = best_per_factor(full_df, 'metric', ['outlier_method', 'norm'])
print(table_metric.to_string(index=False))
table_metric.to_csv('best_per_metric_by_dataset.csv', index=False)

# 5. Best configuration per dataset (ranked by acc_mean)
print("\n" + "="*70)
print("BEST CONFIGURATION PER DATASET (by ACCURACY mean)")
max_acc = full_df.groupby('dataset')['acc_mean'].transform('max')
best_per_ds = full_df[full_df['acc_mean'] == max_acc]

best_per_ds_out = best_per_ds[['dataset', 'outlier_method', 'norm', 'metric',
                               'ari_mean', 'ari_min', 'ari_max', 'ari_std',
                               'acc_mean', 'acc_min', 'acc_max', 'acc_std',
                               'hungarian_acc_mean', 'hungarian_acc_std', 'hungarian_acc_min', 'hungarian_acc_max',
                               'silhouette_mean', 'silhouette_min', 'silhouette_max', 'silhouette_std']]
print(best_per_ds_out.to_string(index=False))
best_per_ds_out.to_csv('best_per_dataset_by_accuracy.csv', index=False)

# best overall configurations (full 9/9 coverage)
combo_group = full_df.groupby(['outlier_method', 'norm', 'metric'])
acc_mean = combo_group['acc_mean'].mean()
n_datasets = combo_group['acc_mean'].count()

fully_covered = acc_mean[n_datasets == 9].reset_index()
fully_covered = fully_covered.sort_values('acc_mean', ascending=False)

print("\nTOP 20 OVERALL CONFIGURATIONS:")
for rank, (_, row) in enumerate(fully_covered.head(60).iterrows(), start=1):
    print(f"  {rank}. {row['outlier_method']:15s} {row['norm']:10s} {row['metric']:15s}  "
          f"acc = {row['acc_mean']:.4f}")

# Per‑dataset accuracy for the proposed configuration
best = fully_covered.iloc[0]
best_outlier, best_norm, best_metric = best['outlier_method'], best['norm'], best['metric']
print(f"\nPROPOSED configuration (chosen): {best_outlier} + {best_norm} + {best_metric}")

proposed = full_df[(full_df['outlier_method'] == best_outlier) &
                   (full_df['norm'] == best_norm) &
                   (full_df['metric'] == best_metric)]

per_ds = proposed.groupby('dataset')[['acc_mean', 'acc_std']].first()
print("\nPer‑dataset accuracy (PROPOSED):")
for ds in per_ds.index:
    m = per_ds.loc[ds, 'acc_mean']
    s = per_ds.loc[ds, 'acc_std']
    print(f"  {ds}: {m:.4f} ± {s:.4f}")

overall_acc = per_ds['acc_mean'].mean()
overall_acc_std = per_ds['acc_mean'].std()
print(f"\nOverall accuracy (PROPOSED): {overall_acc:.2f} ± {overall_acc_std:.2f}")

print("\nAnalysis complete. Tables saved as CSV.")