import os
import numpy as np
from sklearn.ensemble import RandomForestClassifier
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
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 5, 10],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['auto', 'sqrt', 'log2']
}

# Perform Grid Search
rf_model = RandomForestClassifier(random_state=42)
grid_search = GridSearchCV(rf_model, param_grid, cv=5)
grid_search.fit(train_pixels, train_labels)

# Get the best parameters
best_params = grid_search.best_params_

# Plot the Grid Search process
scores = grid_search.cv_results_['mean_test_score']
scores = np.array(scores).reshape(len(param_grid['n_estimators']), len(param_grid['max_depth']))
plt.figure(figsize=(12, 6))
for i, n_estimators in enumerate(param_grid['n_estimators']):
    plt.plot(param_grid['max_depth'], scores[i], label=f'n_estimators={n_estimators}')
plt.xlabel('Max Depth')
plt.ylabel('Mean Test Score')
plt.legend()
plt.title('Grid Search Results - Random Forest')
plt.grid(True)
plt.savefig('grid_search_results_rf.png')

# Train the RF model with best parameters
rf_model = RandomForestClassifier(random_state=42, **best_params)
rf_model.fit(train_pixels, train_labels)

# Evaluate the model
val_predictions = rf_model.predict(val_pixels)
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
joblib.dump(rf_model, 'rf_model.pkl')

# Save the Grid Search results plot
plt.savefig('grid_search_results_rf.png')
