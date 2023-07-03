import os
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
from sklearn.ensemble import VotingClassifier
import matplotlib.pyplot as plt
import joblib

# Load data
data_path = os.path.join(os.getcwd(), '..', 'data', 'dataset2')
train_pixels = np.load(os.path.join(data_path, 'train_pixels.npy'))
train_labels = np.load(os.path.join(data_path, 'train_labels.npy'))
val_pixels = np.load(os.path.join(data_path, 'val_pixels.npy'))
val_labels = np.load(os.path.join(data_path, 'val_labels.npy'))

# Reshape data
train_pixels = train_pixels.reshape(train_pixels.shape[0], -1)
val_pixels = val_pixels.reshape(val_pixels.shape[0], -1)

# Set parameter grid for Grid Search
param_grid = {
    'max_depth': [3, 5, 7],
    'learning_rate': [0.1, 0.01, 0.001],
    'n_estimators': [100, 200, 300],
    'gamma': [0, 0.1, 0.2],
    'subsample': [0.8, 0.9, 1],
    'colsample_bytree': [0.8, 0.9, 1]
}

# Perform Grid Search
xgb_model = XGBClassifier(random_state=42)
grid_search = GridSearchCV(xgb_model, param_grid, cv=5)
grid_search.fit(train_pixels, train_labels)

# Get the best parameters
best_params = grid_search.best_params_

# Plot the Grid Search process
scores = grid_search.cv_results_['mean_test_score']
scores = np.array(scores).reshape(len(param_grid['max_depth']), len(param_grid['learning_rate']))
plt.figure(figsize=(12, 6))
for i, max_depth in enumerate(param_grid['max_depth']):
    plt.plot(param_grid['learning_rate'], scores[i], label=f'max_depth={max_depth}')
plt.xlabel('Learning Rate')
plt.ylabel('Mean Test Score')
plt.legend()
plt.title('Grid Search Results - XGBoost')
plt.grid(True)
plt.savefig('grid_search_results_xgb.png')

# Train the XGBoost model with best parameters
xgb_model = XGBClassifier(random_state=42, **best_params)
xgb_model.fit(train_pixels, train_labels)

# Evaluate the model
val_predictions = xgb_model.predict(val_pixels)
accuracy = accuracy_score(val_labels, val_predictions)
f1 = f1_score(val_labels, val_predictions, average='weighted')
recall = recall_score(val_labels, val_predictions, average='weighted')
precision = precision_score(val_labels, val_predictions, average='weighted')

# Print evaluation metrics
print('Accuracy:', accuracy)
print('F1 Score:', f1)
print('Recall:', recall)
print('Precision:', precision)

# Save the model
joblib.dump(xgb_model, 'xgb_model.pkl')

# Save the Grid Search results plot
plt.savefig('grid_search_results_xgb.png')
