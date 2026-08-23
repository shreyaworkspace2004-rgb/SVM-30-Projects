from sklearn.datasets import load_wine
from sklearn.svm import SVC

X,y = load_wine(return_X_y=True)

model = SVC()
model.fit(X,y)