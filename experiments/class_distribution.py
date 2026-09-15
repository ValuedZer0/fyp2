"""
Compute class distribution (imbalance ratio) for all 9 benchmark datasets
using the project's own datasets.load_dataset(), so counts match exactly
what K-means / evaluation actually sees

Run from the experiments/ directory:
    python class_distribution.py
"""
import numpy as np
from datasets import load_dataset

DATASET_NAMES = [
    'iris', 'wine', 'cancer', 'glass', 'balance',
    'vertebral', 'ecoli', 'blood', 'seeds'
]

# Human-readable class names, matching the encoding each dataset uses in datasets.py
LABEL_NAMES = {
    'iris':      {0: 'setosa', 1: 'versicolor', 2: 'virginica'},
    'wine':      {0: 'class_0', 1: 'class_1', 2: 'class_2'},
    'cancer':    {0: 'malignant', 1: 'benign'},
    'glass':     {0: 'type1(building_float)', 1: 'type2(building_nonfloat)', 2: 'type3(vehicle_float)',
                  4: 'type5(containers)', 5: 'type6(tableware)', 6: 'type7(headlamps)'},
    'balance':   {0: 'B(balanced)', 1: 'L(left-tip)', 2: 'R(right-tip)'},
    'vertebral': {0: 'DH(Disk Hernia)', 1: 'SL(Spondylolisthesis)', 2: 'NO(Normal)'},
    'ecoli':     {0: 'cp', 1: 'im', 2: 'imS', 3: 'imL', 4: 'imU', 5: 'om', 6: 'omL', 7: 'pp'},
    'blood':     {0: 'non-donor', 1: 'repeat-donor'},
    'seeds':     {0: 'Kama', 1: 'Rosa', 2: 'Canadian'},
}

def main():
    print(f"{'Dataset':<12}{'N':>5}{'#Cls':>6}   Class counts (largest -> smallest, with real labels)                          Ratio(max/min)")
    print("-" * 115)
    for name in DATASET_NAMES:
        X, y = load_dataset(name)
        classes, counts = np.unique(y, return_counts=True)
        order = np.argsort(-counts)
        counts_sorted = counts[order]
        classes_sorted = classes[order]
        ratio = counts_sorted[0] / counts_sorted[-1]
        names = LABEL_NAMES.get(name, {})
        counts_str = ", ".join(
            f"{names.get(c, c)}:{n}" for c, n in zip(classes_sorted, counts_sorted)
        )
        print(f"{name:<12}{len(y):>5}{len(classes):>6}   {counts_str:<70} {ratio:>6.2f}:1")

if __name__ == "__main__":
    main()
