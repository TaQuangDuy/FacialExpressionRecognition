import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

# Load data
train_pixels = np.load("D:/Github/ProjectMidterm_ML/FacialExpressionRicognition/data/dataset/train_pixels.npy")
train_labels = np.load("D:/Github/ProjectMidterm_ML/FacialExpressionRicognition/data/dataset/train_labels.npy")
train_pixels = train_pixels.reshape(train_pixels.shape[0], -1)

# Grid search
param_grid = {
    'n_estimators': [60, 70, 80],
    'max_depth': [19, 20, 21],
    'min_samples_split': [3,4,5,6],
    'min_samples_leaf': [1,2,3]
}

random_forest_classifier = RandomForestClassifier()
grid_search_rf = GridSearchCV(random_forest_classifier, param_grid, cv=5)
grid_search_rf.fit(train_pixels, train_labels)

best_rf_params = grid_search_rf.best_params_

print("Best Parameters - Random Forest:")
print(best_rf_params)
