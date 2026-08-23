from sklearn.datasets import load_iris
from sklearn.feature_selection import SelectKBest

# Load dataset
X, y = load_iris(return_X_y=True)

# Select top 2 features
selector = SelectKBest(k=2)

X_new = selector.fit_transform(X, y)

print(X_new.shape)