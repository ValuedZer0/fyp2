# analyse_results.py  (best per factor tables + best per dataset + proposed configuration)
"""
Load all result CSVs, compute best/worst per factor tables ranked by ARI,
identify the proposed configuration (best overall by ARI, full dataset coverage),
and report its per-dataset purity and Hungarian-matched accuracy.

All rankings are based on Adjusted Rand Index (ARI), which is independent of
the cluster-to-class mapping method and corrects for chance agreement.
"""
import pandas as pd
import numpy as np
import os

RESULT_DIR = 'results'
pd.set_option('display.max_colwidth', None)
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
# Helper functions: best / worst per factor per dataset (ranked by ARI)
# ------------------------------------------------------------------
def best_per_factor(data, factor_col, other_cols, metric='ari_mean'):
    """
    For each (dataset, factor_col) group, keep ALL rows that achieve the
    maximum value of `metric`. This correctly displays ties.
    """
    max_vals = data.groupby(['dataset', factor_col])[metric].transform('max')
    best = data[data[metric] == max_vals].copy()

    cols = (['dataset', factor_col] + other_cols +
            ['ari_mean', 'ari_min', 'ari_max', 'ari_std',
             'acc_mean', 'acc_min', 'acc_max', 'acc_std',
             'hungarian_acc_mean', 'hungarian_acc_std', 'hungarian_acc_min', 'hungarian_acc_max',
             'silhouette_mean', 'silhouette_min', 'silhouette_max', 'silhouette_std'])
    return best[cols].sort_values(['dataset', factor_col])


def worst_per_factor(data, factor_col, other_cols, metric='ari_mean'):
    """
    For each (dataset, factor_col) group, keep ALL rows that achieve the
    minimum value of `metric`. Displays ties as well.
    """
    min_vals = data.groupby(['dataset', factor_col])[metric].transform('min')
    worst = data[data[metric] == min_vals].copy()

    cols = (['dataset', factor_col] + other_cols +
            ['ari_mean', 'ari_min', 'ari_max', 'ari_std',
             'acc_mean', 'acc_min', 'acc_max', 'acc_std',
             'hungarian_acc_mean', 'hungarian_acc_std', 'hungarian_acc_min', 'hungarian_acc_max',
             'silhouette_mean', 'silhouette_min', 'silhouette_max', 'silhouette_std'])
    return worst[cols].sort_values(['dataset', factor_col])


# ------------------------------------------------------------------
# 2. Table A: best (norm, metric) per outlier method per dataset
# ------------------------------------------------------------------
print("\n" + "="*70)
print("TABLE A: Best (norm, metric) per dataset and outlier method [ranked by ARI]")
table_outlier = best_per_factor(full_df, 'outlier_method', ['norm', 'metric'])
print(table_outlier.to_string(index=False))
table_outlier.to_csv('best_per_outlier_by_dataset.csv', index=False)

# ------------------------------------------------------------------
# 3. Table B: best (outlier, metric) per normalisation per dataset
# ------------------------------------------------------------------
print("\n" + "="*70)
print("TABLE B: Best (outlier, metric) per dataset and normalisation method [ranked by ARI]")
table_norm = best_per_factor(full_df, 'norm', ['outlier_method', 'metric'])
print(table_norm.to_string(index=False))
table_norm.to_csv('best_per_norm_by_dataset.csv', index=False)

# ------------------------------------------------------------------
# 4. Table C: best (outlier, norm) per distance metric per dataset
# ------------------------------------------------------------------
print("\n" + "="*70)
print("TABLE C: Best (outlier, norm) per dataset and distance metric [ranked by ARI]")
table_metric = best_per_factor(full_df, 'metric', ['outlier_method', 'norm'])
print(table_metric.to_string(index=False))
table_metric.to_csv('best_per_metric_by_dataset.csv', index=False)

# ------------------------------------------------------------------
# 4b. Worst counterparts (needed for "worst" column in ablation tables)
# ------------------------------------------------------------------
print("\n" + "="*70)
print("TABLE A-WORST: Worst (norm, metric) per dataset and outlier method [ranked by ARI]")
table_outlier_worst = worst_per_factor(full_df, 'outlier_method', ['norm', 'metric'])
print(table_outlier_worst.to_string(index=False))
table_outlier_worst.to_csv('worst_per_outlier_by_dataset.csv', index=False)

print("TABLE B-WORST: Worst (outlier, metric) per dataset and normalisation method [ranked by ARI]")
table_norm_worst = worst_per_factor(full_df, 'norm', ['outlier_method', 'metric'])
print(table_norm_worst.to_string(index=False))
table_norm_worst.to_csv('worst_per_norm_by_dataset.csv', index=False)

print("TABLE C-WORST: Worst (outlier, norm) per dataset and distance metric [ranked by ARI]")
table_metric_worst = worst_per_factor(full_df, 'metric', ['outlier_method', 'norm'])
print(table_metric_worst.to_string(index=False))
table_metric_worst.to_csv('worst_per_metric_by_dataset.csv', index=False)

# ------------------------------------------------------------------
# 5. Best configuration per dataset (ranked by ARI)
# ------------------------------------------------------------------
print("\n" + "="*70)
print("BEST CONFIGURATION PER DATASET [ranked by ARI]")
max_ari = full_df.groupby('dataset')['ari_mean'].transform('max')
best_per_ds = full_df[full_df['ari_mean'] == max_ari]

best_per_ds_out = best_per_ds[['dataset', 'outlier_method', 'norm', 'metric',
                               'ari_mean', 'ari_min', 'ari_max', 'ari_std',
                               'acc_mean', 'acc_min', 'acc_max', 'acc_std',
                               'hungarian_acc_mean', 'hungarian_acc_std', 'hungarian_acc_min', 'hungarian_acc_max',
                               'silhouette_mean', 'silhouette_min', 'silhouette_max', 'silhouette_std']]
print(best_per_ds_out.to_string(index=False))
best_per_ds_out.to_csv('best_per_dataset_by_ari.csv', index=False)

# ------------------------------------------------------------------
# 6. Overall best configuration (full 9/9 coverage, ranked by mean ARI)
# ------------------------------------------------------------------
combo_group = full_df.groupby(['outlier_method', 'norm', 'metric'])
mean_ari_by_config = combo_group['ari_mean'].mean()
n_datasets = combo_group['ari_mean'].count()

fully_covered = mean_ari_by_config[n_datasets == 9].reset_index()
fully_covered = fully_covered.sort_values('ari_mean', ascending=False)

print("\nTOP 20 OVERALL CONFIGURATIONS (by mean ARI):")
for rank, (_, row) in enumerate(fully_covered.head(20).iterrows(), start=1):
    print(f"  {rank}. {row['outlier_method']:15s} {row['norm']:10s} {row['metric']:15s}  "
          f"ARI = {row['ari_mean']:.4f}")

# Proposed configuration = top-ranked, full coverage
best = fully_covered.iloc[0]
best_outlier, best_norm, best_metric = best['outlier_method'], best['norm'], best['metric']
print(f"\nPROPOSED configuration (chosen by ARI): {best_outlier} + {best_norm} + {best_metric}")

# ------------------------------------------------------------------
# 7. Extract purity and Hungarian accuracy for the proposed configuration
# ------------------------------------------------------------------
proposed = full_df[(full_df['outlier_method'] == best_outlier) &
                   (full_df['norm'] == best_norm) &
                   (full_df['metric'] == best_metric)]

per_ds = proposed.groupby('dataset')[['acc_mean', 'acc_std', 'hungarian_acc_mean', 'hungarian_acc_std']].first()

overall_acc = per_ds['acc_mean'].mean()

# Between-dataset std-dev spread 
overall_acc_spread = per_ds['acc_mean'].mean()

# Mean run-to-run SD, comparable to the literature benchmark's convention
overall_acc_runtorun_sd = per_ds['acc_std'].mean()

print(f"Overall accuracy: {overall_acc:.2f}")
print(f"  Between-dataset SD (spread across datasets): {overall_acc_spread:.2f}")
print(f"  Mean run-to-run SD (benchmark-comparable stability): {overall_acc_runtorun_sd:.2f}")
print("\nPer-dataset accuracy (PROPOSED):")
for ds in per_ds.index:
    purity_mean = per_ds.loc[ds, 'acc_mean']
    purity_std = per_ds.loc[ds, 'acc_std']
    hung_mean = per_ds.loc[ds, 'hungarian_acc_mean']
    hung_std = per_ds.loc[ds, 'hungarian_acc_std']
    print(f"  {ds}: Purity = {purity_mean:.4f} ± {purity_std:.4f} | Hungarian = {hung_mean:.4f} ± {hung_std:.4f}")

overall_purity_mean = per_ds['acc_mean'].mean()
overall_purity_std = per_ds['acc_mean'].std()
overall_hungarian_mean = per_ds['hungarian_acc_mean'].mean()
overall_hungarian_std = per_ds['hungarian_acc_mean'].std()

print(f"\nOverall purity (PROPOSED): {overall_purity_mean:.2f} ± {overall_purity_std:.2f}")
print(f"Overall Hungarian (PROPOSED): {overall_hungarian_mean:.2f} ± {overall_hungarian_std:.2f}")

print("\nAnalysis complete. Tables saved as CSV.")

# ------------------------------------------------------------------
# Outlier ablation: fix norm = minmax, metric = euclidean
# Modified to display ties instead of silently picking one via idxmax()
# ------------------------------------------------------------------
subset = full_df[(full_df['norm'] == 'minmax') & (full_df['metric'] == 'euclidean')]

# No removal
no_rem = subset[subset['outlier_method'] == 'none'].set_index('dataset')[
    ['acc_mean', 'hungarian_acc_mean']
].rename(columns={'acc_mean': 'no_rem_purity',
                  'hungarian_acc_mean': 'no_rem_hung'})

# Best removal by ARI among outlier_method != none — TIE-SAFE VERSION
rem = subset[subset['outlier_method'] != 'none'].copy()
max_ari_per_ds = rem.groupby('dataset')['ari_mean'].transform('max')
tied = rem[rem['ari_mean'] == max_ari_per_ds]

# Collapse ties into one row per dataset: join method names, keep the
# (identical, since ARI/accuracy are the same for an exact tie) accuracy values
best = (
    tied.groupby('dataset')
    .agg(
        best_method=('outlier_method', lambda s: ', '.join(sorted(s))),
        n_tied=('outlier_method', 'count'),
        acc_mean=('acc_mean', 'first'),
        hungarian_acc_mean=('hungarian_acc_mean', 'first'),
    )
)
best = best.rename(columns={'acc_mean': 'best_rem_purity',
                             'hungarian_acc_mean': 'best_rem_hung'})

# Merge
out_tbl = pd.concat([no_rem, best], axis=1)
out_tbl['delta_purity'] = out_tbl['best_rem_purity'] - out_tbl['no_rem_purity']
out_tbl['delta_hung'] = out_tbl['best_rem_hung'] - out_tbl['no_rem_hung']

# Purity table
purity_table = out_tbl[['no_rem_purity', 'best_method', 'n_tied', 'best_rem_purity', 'delta_purity']]
purity_table.columns = ['No removal', 'Best method', '# tied', 'Best removal', 'Δ']
print("Purity table (outlier):\n", purity_table.round(4))

# Hungarian table
hung_table = out_tbl[['no_rem_hung', 'best_method', 'n_tied', 'best_rem_hung', 'delta_hung']]
hung_table.columns = ['No removal', 'Best method', '# tied', 'Best removal', 'Δ']
print("\nHungarian table (outlier):\n", hung_table.round(4))


# ------------------------------------------------------------------
# Normalisation ablation — TIE-SAFE VERSION
# ------------------------------------------------------------------
subset = full_df[
    (full_df['outlier_method'] == 'zscore_robust_2.5') &
    (full_df['metric'] == 'euclidean')
].copy()

none_df = subset[subset['norm'] == 'none'].set_index('dataset')[
    ['acc_mean', 'hungarian_acc_mean']
].rename(columns={'acc_mean': 'none_purity', 'hungarian_acc_mean': 'none_hung'})

non_none = subset[subset['norm'] != 'none'].copy()
max_ari_per_ds = non_none.groupby('dataset')['ari_mean'].transform('max')
tied = non_none[non_none['ari_mean'] == max_ari_per_ds]

best_df = (
    tied.groupby('dataset')
    .agg(
        best_norm=('norm', lambda s: ', '.join(sorted(s))),
        n_tied=('norm', 'count'),
        acc_mean=('acc_mean', 'first'),
        hungarian_acc_mean=('hungarian_acc_mean', 'first'),
    )
)
best_df = best_df.rename(columns={'acc_mean': 'best_purity', 'hungarian_acc_mean': 'best_hung'})

norm_tbl = pd.concat([none_df, best_df], axis=1)
norm_tbl['delta_purity'] = norm_tbl['best_purity'] - norm_tbl['none_purity']
norm_tbl['delta_hung'] = norm_tbl['best_hung'] - norm_tbl['none_hung']

purity_out = norm_tbl[['none_purity', 'best_norm', 'n_tied', 'best_purity', 'delta_purity']]
purity_out.columns = ['No normalisation', 'Best norm', '# tied', 'Best purity', 'Δ Purity']
print("\nPurity table (normalisation):\n", purity_out.round(4))

hung_out = norm_tbl[['none_hung', 'best_norm', 'n_tied', 'best_hung', 'delta_hung']]
hung_out.columns = ['No normalisation', 'Best norm', '# tied', 'Best Hungarian', 'Δ Hungarian']
print("\nHungarian table (normalisation):\n", hung_out.round(4))


# ------------------------------------------------------------------
# Distance metric ablation — TIE-SAFE VERSION
# ------------------------------------------------------------------
subset = full_df[
    (full_df['outlier_method'] == 'zscore_robust_2.5') &
    (full_df['norm'] == 'minmax')
].copy()

euclidean_df = subset[subset['metric'] == 'euclidean'].set_index('dataset')[
    ['acc_mean', 'hungarian_acc_mean']
].rename(columns={'acc_mean': 'euclidean_purity', 'hungarian_acc_mean': 'euclidean_hung'})

non_euclidean = subset[subset['metric'] != 'euclidean'].copy()
max_ari_per_ds = non_euclidean.groupby('dataset')['ari_mean'].transform('max')
tied = non_euclidean[non_euclidean['ari_mean'] == max_ari_per_ds]

best_df = (
    tied.groupby('dataset')
    .agg(
        best_metric=('metric', lambda s: ', '.join(sorted(s))),
        n_tied=('metric', 'count'),
        acc_mean=('acc_mean', 'first'),
        hungarian_acc_mean=('hungarian_acc_mean', 'first'),
    )
)
best_df = best_df.rename(columns={'acc_mean': 'best_purity', 'hungarian_acc_mean': 'best_hung'})

metric_tbl = pd.concat([euclidean_df, best_df], axis=1)
metric_tbl['delta_purity'] = metric_tbl['best_purity'] - metric_tbl['euclidean_purity']
metric_tbl['delta_hung'] = metric_tbl['best_hung'] - metric_tbl['euclidean_hung']

purity_out = metric_tbl[['euclidean_purity', 'best_metric', 'n_tied', 'best_purity', 'delta_purity']]
purity_out.columns = ['Euclidean', 'Best metric', '# tied', 'Best purity', 'Δ Purity']
print("\nPurity table (metric):\n", purity_out.round(4))

hung_out = metric_tbl[['euclidean_hung', 'best_metric', 'n_tied', 'best_hung', 'delta_hung']]
hung_out.columns = ['Euclidean', 'Best metric', '# tied', 'Best Hungarian', 'Δ Hungarian']
print("\nHungarian table (metric):\n", hung_out.round(4))