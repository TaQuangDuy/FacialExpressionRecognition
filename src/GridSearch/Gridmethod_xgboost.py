import numpy as np
import xgboost as xgb
from sklearn.metrics import accuracy_score, precision_score, f1_score, recall_score
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import joblib


# Load data
train_pixels = np.load("D:/Github/ProjectMidterm_ML/FacialExpressionRicognition/data/dataset/train_pixels.npy")
train_labels = np.load("D:/Github/ProjectMidterm_ML/FacialExpressionRicognition/data/dataset/train_labels.npy")
val_pixels = np.load("D:/Github/ProjectMidterm_ML/FacialExpressionRicognition/data/dataset/val_pixels.npy")
val_labels = np.load("D:/Github/ProjectMidterm_ML/FacialExpressionRicognition/data/dataset/val_labels.npy")

# Reshape data
train_pixels = train_pixels.reshape(train_pixels.shape[0], -1)
val_pixels = val_pixels.reshape(val_pixels.shape[0], -1)

# XGBoost
param_grid = {
    'max_depth': [5,7,10],
    'learning_rate': [0.01, 0.001,0.0001],
    'n_estimators': [200,300,400]
}

xgb_classifier = xgb.XGBClassifier()
grid_search_xgb = GridSearchCV(xgb_classifier, param_grid, cv=2)
grid_search_xgb.fit(train_pixels, train_labels)

best_xgb_params = grid_search_xgb.best_params_

# Print best parameters
print("Best Parameters - XGBoost:")
print(best_xgb_params)

# Plot Grid Search results
mean_scores = grid_search_xgb.cv_results_['mean_test_score']
params = grid_search_xgb.cv_results_['params']
plt.figure(figsize=(12, 6))
plt.plot(range(len(mean_scores)), mean_scores)
plt.xticks(range(len(params)), [str(param) for param in params], rotation='vertical')
plt.xlabel('Parameters')
plt.ylabel('Mean Test Score')
plt.title('Grid Search - XGBoost')
plt.tight_layout()
plt.savefig('xgb_grid_search.png')
plt.show()
