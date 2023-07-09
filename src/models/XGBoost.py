import numpy as np
import xgboost as xgb
from sklearn.metrics import accuracy_score
import joblib

# Load data
train_pixels = np.load("D:/Github/ProjectMidterm_ML/FacialExpressionRicognition/data/dataset2/train_pixels.npy")
train_labels = np.load("D:/Github/ProjectMidterm_ML/FacialExpressionRicognition/data/dataset2/train_labels.npy")
val_pixels = np.load("D:/Github/ProjectMidterm_ML/FacialExpressionRicognition/data/dataset2/val_pixels.npy")
val_labels = np.load("D:/Github/ProjectMidterm_ML/FacialExpressionRicognition/data/dataset2/val_labels.npy")
test_pixels = np.load("D:/Github/ProjectMidterm_ML/FacialExpressionRicognition/data/dataset2/test_pixels.npy")
test_labels = np.load("D:/Github/ProjectMidterm_ML/FacialExpressionRicognition/data/dataset2/test_labels.npy")

# Reshape data
train_pixels = train_pixels.reshape(train_pixels.shape[0], -1)
val_pixels = val_pixels.reshape(val_pixels.shape[0], -1)
test_pixels = test_pixels.reshape(test_pixels.shape[0], -1)

# Determine number of classes
num_classes = 7

# XGBoost with best parameters
best_xgb_params = {'max_depth': 12, 'learning_rate': 0.01, 'n_estimators': 400, 'num_class': num_classes}
xgb_classifier = xgb.XGBClassifier(**best_xgb_params)
xgb_classifier.fit(train_pixels, train_labels)

# Save trained model
model_path = "xgboost_model.joblib"
joblib.dump(xgb_classifier, model_path)

# Predict on test data
test_predictions = xgb_classifier.predict(test_pixels)

# Calculate accuracy on test data
test_accuracy = accuracy_score(test_labels, test_predictions)

# Print test accuracy
print("Test Accuracy:", test_accuracy)
