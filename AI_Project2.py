# Import required libraries
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    precision_score,
    recall_score,
    f1_score
)

# Load Iris dataset
iris = load_iris()

# Features and target labels
X = iris.data
y = iris.target

# Display dataset information
print("=" * 50)
print("IRIS DATASET INFORMATION")
print("=" * 50)
print(f"Total Samples : {X.shape[0]}")
print(f"Total Features: {X.shape[1]}")
print(f"Total Classes : {len(set(y))}")

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print(f"\nTraining Samples: {len(X_train)}")
print(f"Testing Samples : {len(X_test)}")

# Scale feature values
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Create KNN classifier
knn = KNeighborsClassifier(n_neighbors=5)

# Train model
knn.fit(X_train, y_train)

# Make predictions
y_pred = knn.predict(X_test)

# Calculate evaluation metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average="weighted")
recall = recall_score(y_test, y_pred, average="weighted")
f1 = f1_score(y_test, y_pred, average="weighted")

# Display results
print("\n" + "=" * 50)
print("MODEL EVALUATION")
print("=" * 50)

print(f"Accuracy : {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall   : {recall:.2f}")
print(f"F1 Score : {f1:.2f}")

print("\nConfusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print(cm)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Visualize Confusion Matrix
plt.figure(figsize=(6, 5))
plt.imshow(cm)

plt.title("Confusion Matrix")
plt.colorbar()

plt.xticks(np.arange(3), iris.target_names)
plt.yticks(np.arange(3), iris.target_names)

plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")

for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j,i,cm[i, j],ha="center",va="center"
        )

plt.tight_layout()
plt.show()

# Visualize Evaluation Metrics
metrics = ["Accuracy", "Precision", "Recall", "F1 Score"]
scores = [accuracy, precision, recall, f1]

plt.figure(figsize=(6, 5))
plt.bar(metrics, scores)

plt.title("Model Performance Metrics")
plt.ylabel("Score")
plt.ylim(0, 1.1)

plt.tight_layout()
plt.show()