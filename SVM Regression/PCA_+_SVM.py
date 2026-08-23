from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from sklearn.svm import SVC

# Load dataset
X, y = load_iris(return_X_y=True)

# Apply PCA
pca = PCA(n_components=2)

X_new = pca.fit_transform(X)

# Train SVM
model = SVC()

model.fit(X_new, y)

print("Model Trained Successfully")