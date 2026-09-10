"""
min_per_class ablation check.

Tests whether the min_per_class=2 class guard (used in run_one_dataset.py /
run_all_dataset.py) has a measurable effect on the proposed algorithm's
pipeline (outlier_method='zscore_robust_2.5', norm='minmax', metric='euclidean')
on the three datasets where KMN-RobustZ scored best in Table 4.x of 4.2.2:
Glass, Balance, and Ecoli.

For each dataset, runs the proposed configuration twice:
    (a) min_per_class=2   (as actually used in the reported experiments)
    (b) min_per_class=None (guard disabled)
and reports removed_pct, ARI, purity (acc), Hungarian accuracy, and macro-F1
for both, plus the difference.

This directly tests the hypothesis raised in 4.2.2: that the class guard
is responsible for part of KMN-RobustZ's advantage on severely imbalanced
datasets, by showing what happens to a class (in particular, whether it
gets stripped out) when the guard is removed.

Run from the experiments/ directory:
    python min_per_class_ablation.py
"""
import numpy as np
from experiment_utils import run_single_config
from datasets import load_dataset
from outlier_handling import zscore_robust_filter

DATASETS = ['iris', 'glass', 'balance', 'cancer', 'wine',
            'vertebral', 'ecoli', 'blood', 'seeds']
OUTLIER_METHOD = 'zscore_robust_2.5'
NORM = 'minmax'
METRIC = 'euclidean'
N_RUNS = 100
RANDOM_STATE_BASE = 42


def class_survival_report(dataset_name):
    """Show, per class, how many points survive zscore_robust_2.5 filtering
    with vs without the min_per_class=2 guard."""
    X, y = load_dataset(dataset_name)
    threshold = 2.5

    _, y_guarded, _ = zscore_robust_filter(X, y, threshold=threshold, min_per_class=2)
    _, y_unguarded, _ = zscore_robust_filter(X, y, threshold=threshold, min_per_class=None)

    print(f"  Per-class survival ({dataset_name}):")
    classes = np.unique(y)
    for cls in classes:
        n_orig = (y == cls).sum()
        n_guarded = (y_guarded == cls).sum()
        n_unguarded = (y_unguarded == cls).sum()
        flag = "  <-- guard intervened" if n_unguarded < 2 <= n_guarded else ""
        print(f"    class {cls}: original={n_orig}, "
              f"guarded(min=2)={n_guarded}, unguarded={n_unguarded}{flag}")


def main():
    print(f"{'Dataset':<10}{'Guard':<10}{'Removed%':>10}{'ARI':>8}{'Purity':>8}{'Hungarian':>11}{'MacroF1':>9}")
    print("-" * 66)

    for name in DATASETS:
        for guard_label, min_per_class in [('ON(=2)', 2), ('OFF(None)', None)]:
            agg = run_single_config(
                name, OUTLIER_METHOD, NORM, METRIC,
                n_runs=N_RUNS, random_state_base=RANDOM_STATE_BASE,
                min_per_class=min_per_class,
            )
            removed_pct = agg['removed_pct'][0]
            ari = agg['ari'][0]
            purity = agg['acc'][0]
            hung = agg['hungarian_acc'][0]
            f1 = agg['macro_f1'][0]
            print(f"{name:<10}{guard_label:<10}{removed_pct:>9.2f}%{ari:>8.4f}"
                  f"{purity:>8.4f}{hung:>11.4f}{f1:>9.4f}")
        class_survival_report(name)
        print()


if __name__ == "__main__":
    main()