from sklearn.datasets import load_digits
from sklearn.svm import SVC

X,y = load_digits(return_X_y=True)

model = SVC()
model.fit(X,y)