from sklearn.datasets import load_iris
from sklearn.svm import SVC

# Load dataset
X, y = load_iris(return_X_y=True)

# Create model
model = SVC(probability=True)

# Train model
model.fit(X, y)

# Predict probabilities
print(model.predict_proba(X[:5]))