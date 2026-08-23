from sklearn.datasets import load_iris
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

X, y = load_iris(return_X_y=True)

params = {
    'gamma': [1, 0.1, 0.01, 0.001],
    'C': [0.1, 1, 10, 100],
    'kernel': ['rbf']
}

grid = GridSearchCV(SVC(), params, cv=5)

grid.fit(X, y)

print("Best Parameters:", grid.best_params_)
print("Best Score:", grid.best_score_)