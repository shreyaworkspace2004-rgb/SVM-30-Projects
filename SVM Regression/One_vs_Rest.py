from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC

model = OneVsRestClassifier(SVC())