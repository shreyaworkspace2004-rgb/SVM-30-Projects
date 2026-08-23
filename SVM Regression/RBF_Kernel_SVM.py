from sklearn.datasets import load_iris
from sklearn.svm import SVC

iris = load_iris()
X = iris.data
y = iris.target

model = SVC(kernel='rbf')
model.fit(X, y)

print("Training Completed")