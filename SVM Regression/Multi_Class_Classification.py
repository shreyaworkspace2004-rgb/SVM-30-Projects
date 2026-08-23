from sklearn.datasets import load_iris
from sklearn.svm import SVC

iris = load_iris()

model = SVC(decision_function_shape='ovo')
model.fit(iris.data, iris.target)

print("Classes:", model.classes_)