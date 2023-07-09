
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np
import matplotlib.pyplot as plt

# Load data
train_pixels = np.load("D:/Github/ProjectMidterm_ML/FacialExpressionRicognition/data/dataset3/train_pixels.npy")
train_labels = np.load("D:/Github/ProjectMidterm_ML/FacialExpressionRicognition/data/dataset3/train_labels.npy")
test_pixels = np.load("D:/Github/ProjectMidterm_ML/FacialExpressionRicognition/data/dataset3/test_pixels.npy")
test_labels = np.load("D:/Github/ProjectMidterm_ML/FacialExpressionRicognition/data/dataset3/test_labels.npy")

# Reshape data
train_pixels = train_pixels.reshape(train_pixels.shape[0], -1)
test_pixels = test_pixels.reshape(test_pixels.shape[0], -1)

# Define best params
best_params = {'n_neighbors': 5, 'weights': 'distance', 'p': 1}

# Create KNN classifier with best params
knn_classifier = KNeighborsClassifier(**best_params)

# Train the model
knn_classifier.fit(train_pixels, train_labels)

# Make predictions on train and test sets
train_predictions = knn_classifier.predict(train_pixels)
test_predictions = knn_classifier.predict(test_pixels)

# Calculate evaluation metrics for train set
train_accuracy = accuracy_score(train_labels, train_predictions)
train_precision = precision_score(train_labels, train_predictions, average='weighted')
train_recall = recall_score(train_labels, train_predictions, average='weighted')
train_f1_score = f1_score(train_labels, train_predictions, average='weighted')

# Calculate evaluation metrics for test set
test_accuracy = accuracy_score(test_labels, test_predictions)
test_precision = precision_score(test_labels, test_predictions, average='weighted')
test_recall = recall_score(test_labels, test_predictions, average='weighted')
test_f1_score = f1_score(test_labels, test_predictions, average='weighted')

# Print evaluation metrics

print("Test Accuracy:", test_accuracy)
print("Test Precision:", test_precision)
print("Test Recall:", test_recall)
print("Test F1 Score:", test_f1_score)

# Plot learning curve
plt.plot(range(len(knn_classifier.loss_curve_)), knn_classifier.loss_curve_)
plt.xlabel('Number of iterations')
plt.ylabel('Loss')
plt.title('Learning Curve')
plt.grid(True)
plt.savefig('knn_learning_curve.png')
plt.show()

