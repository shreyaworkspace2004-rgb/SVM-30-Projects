from sklearn.datasets import load_iris
from sklearn.svm import SVC

iris = load_iris()

model = SVC(kernel='sigmoid')
model.fit(iris.data, iris.target)

print("Model Trained")