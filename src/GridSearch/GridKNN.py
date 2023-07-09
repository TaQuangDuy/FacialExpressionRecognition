from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
import numpy as np
import matplotlib.pyplot as plt

# Load data
train_pixels = np.load("D:/Github/ProjectMidterm_ML/FacialExpressionRicognition/data/dataset/train_pixels.npy")
train_labels = np.load("D:/Github/ProjectMidterm_ML/FacialExpressionRicognition/data/dataset/train_labels.npy")
val_pixels = np.load("D:/Github/ProjectMidterm_ML/FacialExpressionRicognition/data/dataset/val_pixels.npy")
val_labels = np.load("D:/Github/ProjectMidterm_ML/FacialExpressionRicognition/data/dataset/val_labels.npy")

# Reshape data
train_pixels = train_pixels.reshape(train_pixels.shape[0], -1)
val_pixels = val_pixels.reshape(val_pixels.shape[0], -1)

# Define the parameter grid for Grid Search
param_grid = {
    'n_neighbors': [3, 5, 7, 9],
    'weights': ['uniform', 'distance'],
    'p': [1, 2]
}

# Create KNN model
knn_model = KNeighborsClassifier()

# Perform Grid Search
grid_search = GridSearchCV(knn_model, param_grid, scoring='f1_weighted', cv=5)
grid_search.fit(train_pixels, train_labels)

# Get the best parameters and best score
best_params = grid_search.best_params_
best_score = grid_search.best_score_

# Print the best parameters and best score
print("Best Parameters:", best_params)
print("Best Score:", best_score)

# Extract the results from Grid Search
results = grid_search.cv_results_
mean_scores = results['mean_test_score']
n_neighbors = param_grid['n_neighbors']
weights = param_grid['weights']
p_values = param_grid['p']

# Plot the results
plt.figure(figsize=(10, 6))
for weight in weights:
    for p_value in p_values:
        scores = mean_scores[(results['param_weights'] == weight) & (results['param_p'] == p_value)]
        label = f'Weight={weight}, p={p_value}'
        plt.plot(n_neighbors, scores, marker='o', label=label)

plt.xlabel('Number of Neighbors (K)')
plt.ylabel('Mean F1 Score')
plt.title('Grid Search Results')
plt.legend()
plt.grid(True)
plt.savefig('grid_search_results.png')
plt.show()
