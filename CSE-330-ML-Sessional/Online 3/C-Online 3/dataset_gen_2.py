import numpy as np
import pandas as pd
from sklearn.datasets import make_classification

np.random.seed(42)

X, y = make_classification(
    n_samples=2500,
    n_features=12,
    n_informative=8,
    n_redundant=2,
    n_classes=3,
    n_clusters_per_class=1,
    random_state=42
)

df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(12)])

# One-hot encode target
df['class_0'] = (y == 0).astype(int)
df['class_1'] = (y == 1).astype(int)
df['class_2'] = (y == 2).astype(int)

df.to_csv("stacking_dataset.csv", index=False)