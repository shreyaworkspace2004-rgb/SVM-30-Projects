from sklearn.svm import SVC

model = SVC(
    decision_function_shape='ovo'
)