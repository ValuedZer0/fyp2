import numpy as np
import pandas as pd
from sklearn.datasets import load_iris, load_wine, load_breast_cancer

def load_dataset(name):
    """
    Load a benchmark dataset by name.
    Returns X (features) and y_true (ground truth labels).
    """
    name = name.lower()
    if name == 'iris':
        data = load_iris()
        X, y = data.data, data.target
    elif name == 'wine':
        data = load_wine()
        X, y = data.data, data.target
    elif name == 'cancer':
        data = load_breast_cancer()
        X, y = data.data, data.target
    elif name == 'glass':
        # 10 columns: ID, RI, Na, Mg, Al, Si, K, Ca, Ba, Fe (type)
        df = pd.read_csv('data/glass.data', header=None)
        # Drop ID column (index 0)
        X = df.iloc[:, 1:-1].values.astype(float)
        y = df.iloc[:, -1].values.astype(int) - 1  # types 1..7 -> 0..6
    elif name == 'balance':
        # Columns: class (B,L,R), left-weight, left-distance, right-weight, right-distance
        df = pd.read_csv('data/balance-scale.data', header=None)
        # Convert class to integer: B=0, L=1, R=2
        class_map = {'B':0, 'L':1, 'R':2}
        y = df.iloc[:, 0].map(class_map).values
        X = df.iloc[:, 1:].values.astype(float)
    elif name == 'vertebral':
        # 7 columns: 6 features + class (3 classes: DH, SL, NO)
        df = pd.read_csv('data/column_3C.dat', sep=r'\s+', header=None, engine='python')
        # Features are columns 0-5, class is column 6
        X = df.iloc[:, :6].values.astype(float)
        class_map = {'DH':0, 'SL':1, 'NO':2}
        y = df.iloc[:, 6].map(class_map).values
    elif name == 'ecoli':
        # 9 columns: sequence name, 7 features, class
        df = pd.read_csv('data/ecoli.data', sep=r'\s+', header=None, engine='python')
        # Drop first column (sequence name)
        X = df.iloc[:, 1:-1].values.astype(float)
        # Encode class labels (8 classes) to 0..7
        y = pd.factorize(df.iloc[:, -1])[0]
    elif name == 'blood':
        # 5 columns: R, F, M, T, whether donated (binary)
        df = pd.read_csv('data/transfusion.data')
        X = df.iloc[:, :4].values.astype(float)
        y = df.iloc[:, -1].values.astype(int)
    elif name == 'seeds':
        # 8 columns: 7 features + class (1,2,3)
        df = pd.read_csv('data/seeds_dataset.txt', sep=r'\s+', header=None, engine='python')
        X = df.iloc[:, :7].values.astype(float)
        y = df.iloc[:, -1].values.astype(int) - 1  # 1..3 -> 0..2
    else:
        raise ValueError(f"Unknown dataset: {name}")

    # Verify no missing values
    assert not np.isnan(X).any(), f"NaN in features of {name}"
    assert not np.isnan(y).any(), f"NaN in labels of {name}"
    return X.astype(float), y.astype(int)