from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

iris = load_iris()

scaler = StandardScaler()
X_scaled = scaler.fit_transform(iris.data)

model = SVC()
model.fit(X_scaled, iris.target)

print("SVM trained on scaled data")