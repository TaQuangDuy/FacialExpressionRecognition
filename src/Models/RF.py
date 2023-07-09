import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, f1_score, recall_score
import matplotlib.pyplot as plt
import joblib

# Load data
train_pixels = np.load("D:/Github/ProjectMidterm_ML/FacialExpressionRicognition/data/dataset/train_pixels.npy")
train_labels = np.load("D:/Github/ProjectMidterm_ML/FacialExpressionRicognition/data/dataset/train_labels.npy")
val_pixels = np.load("D:/Github/ProjectMidterm_ML/FacialExpressionRicognition/data/dataset/val_pixels.npy")
val_labels = np.load("D:/Github/ProjectMidterm_ML/FacialExpressionRicognition/data/dataset/val_labels.npy")
train_pixels = train_pixels.reshape(train_pixels.shape[0], -1)
val_pixels = val_pixels.reshape(val_pixels.shape[0], -1)

# Best parameters from Grid Search
best_rf_params = {'n_estimators': 70, 'max_depth': 20, 'min_samples_split': 5, 'min_samples_leaf': 2}

# Train Random Forest
random_forest_classifier = RandomForestClassifier(**best_rf_params)

# Lists to store accuracy values
train_accuracy_values = []
val_accuracy_values = []

# Training loop
for epoch in range(10):
    random_forest_classifier.fit(train_pixels, train_labels)

    # Calculate accuracy on training set
    train_predictions = random_forest_classifier.predict(train_pixels)
    train_accuracy = accuracy_score(train_labels, train_predictions)
    train_accuracy_values.append(train_accuracy)

    # Calculate accuracy on validation set
    val_predictions = random_forest_classifier.predict(val_pixels)
    val_accuracy = accuracy_score(val_labels, val_predictions)
    val_accuracy_values.append(val_accuracy)

# Plot accuracy values
epochs = range(1, 11)
plt.plot(epochs, train_accuracy_values, label='Train Accuracy')
plt.plot(epochs, val_accuracy_values, label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()
plt.grid(True)
plt.savefig('accuracy_plot.png')
plt.show()
