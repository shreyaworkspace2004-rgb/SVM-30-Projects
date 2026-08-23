from sklearn.datasets import load_iris
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

iris = load_iris()

parameters = {
    'C': [0.1, 1, 10],
    'gamma': [1, 0.1, 0.01],
    'kernel': ['rbf']
}

grid = GridSearchCV(SVC(), parameters, cv=5)

grid.fit(iris.data, iris.target)

print("Best Parameters:", grid.best_params_)