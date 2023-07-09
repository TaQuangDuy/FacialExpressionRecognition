import numpy as np
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
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
random_forest_classifier.fit(train_pixels, train_labels)

# Train RF with AdaBoost
adaboost_classifier = AdaBoostClassifier(base_estimator=random_forest_classifier, n_estimators=50, learning_rate=1)
adaboost_classifier.fit(train_pixels, train_labels)

# Evaluation
val_predictions = adaboost_classifier.predict(val_pixels)
accuracy = accuracy_score(val_labels, val_predictions)
precision = precision_score(val_labels, val_predictions, average='weighted')
f1 = f1_score(val_labels, val_predictions, average='weighted')
recall = recall_score(val_labels, val_predictions, average='weighted')
print("Train Accuracy:", random_forest_classifier.score(train_pixels, train_labels))
print("Test Accuracy:", accuracy)
print("Precision:", precision)
print("F1 Score:", f1)
print("Recall:", recall)

joblib.dump(adaboost_classifier, 'random_forest_model.pkl')
