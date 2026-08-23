from sklearn.datasets import load_iris
from sklearn.svm import SVC

# Load dataset
X, y = load_iris(return_X_y=True)

# Create model
model = SVC()

# Train model
model.fit(X, y)

# Decision scores
scores = model.decision_function(X)

print(scores[:5])