import numpy as np
import xgboost as xgb
from sklearn.metrics import accuracy_score

def load_data():
    # Load data
    train_pixels = np.load(r"D:\Facial Expression Reconization\data\dataset\train_pixels.npy")
    train_labels = np.load(r"D:\Facial Expression Reconization\data\dataset\train_labels.npy")
    val_pixels = np.load(r"D:\Facial Expression Reconization\data\dataset\val_pixels.npy")
    val_labels = np.load(r"D:\Facial Expression Reconization\data\dataset\val_labels.npy")
    test_pixels = np.load(r"D:\Facial Expression Reconization\data\dataset\test_pixels.npy")
    test_labels = np.load(r"D:\Facial Expression Reconization\data\dataset\test_labels.npy")
    return train_pixels, train_labels, val_pixels, val_labels, test_pixels, test_labels

def reshape_data(data):
    # Reshape data
    return data.reshape(data.shape[0], -1)

# Load data
train_pixels, train_labels, val_pixels, val_labels, test_pixels, test_labels = load_data()

# Reshape data
train_pixels = reshape_data(train_pixels)
val_pixels = reshape_data(val_pixels)
test_pixels = reshape_data(test_pixels)

# Convert data to DMatrix format
dtrain = xgb.DMatrix(train_pixels, label=train_labels)
dval = xgb.DMatrix(val_pixels, label=val_labels)
dtest = xgb.DMatrix(test_pixels, label=test_labels)

# Set parameters for XGBoost model
params = {
    'max_depth': 3,
    'eta': 0.1,
    'objective': 'multi:softmax',
    'num_class': 7
}

# Train XGBoost model
model = xgb.train(params, dtrain, num_boost_round=100)

# Make predictions on validation set
val_preds = model.predict(dval)

# Evaluate accuracy on validation set
val_accuracy = accuracy_score(val_labels, val_preds)
print("Validation Accuracy:", val_accuracy)

# Make predictions on test set
test_preds = model.predict(dtest)

# Evaluate accuracy on test set
test_accuracy = accuracy_score(test_labels, test_preds)
print("Test Accuracy:", test_accuracy)
