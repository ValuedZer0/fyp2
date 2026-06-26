from sklearn.datasets import load_iris
import pandas as pd

iris = load_iris()
data = pd.DataFrame(iris.data, columns=iris.feature_names)
data['target'] = iris.target

print("Shape:", data.shape)
print(data.head())
print("\nMissing values?")
print(data.isnull().sum().sum())