import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV


def load_data():
    train_pixels = np.load(r"D:\Facial Expression Reconization\data\dataset\train_pixels.npy")
    train_labels = np.load(r"D:\Facial Expression Reconization\data\dataset\train_labels.npy")
    val_pixels = np.load(r"D:\Facial Expression Reconization\data\dataset\val_pixels.npy")
    val_labels = np.load(r"D:\Facial Expression Reconization\data\dataset\val_labels.npy")
    test_pixels = np.load(r"D:\Facial Expression Reconization\data\dataset\test_pixels.npy")
    test_labels = np.load(r"D:\Facial Expression Reconization\data\dataset\test_labels.npy")
    return train_pixels, train_labels, val_pixels, val_labels, test_pixels, test_labels


def reshape_data(data):
    return data.reshape(data.shape[0], -1)


def plot_grid_search_results(grid_search, model_name):
    params = grid_search.cv_results_['params']
    mean_test_scores = grid_search.cv_results_['mean_test_score']
    std_test_scores = grid_search.cv_results_['std_test_score']

    for param, mean_score, std_score in zip(params, mean_test_scores, std_test_scores):
        print(f"Parameters: {param}, Mean Score: {mean_score:.4f}, Std Score: {std_score:.4f}")

    plt.errorbar(range(len(params)), mean_test_scores, yerr=std_test_scores, fmt='o')
    plt.xticks(range(len(params)), params, rotation=45)
    plt.xlabel('Parameters')
    plt.ylabel('Mean Test Score')
    plt.title(f'Grid Search Results ({model_name})')
    plt.savefig(f'{model_name}_grid_search_results.png')
    plt.close()


# Load data
train_pixels, train_labels, val_pixels, val_labels, test_pixels, test_labels = load_data()

# Reshape data
train_pixels = reshape_data(train_pixels)
val_pixels = reshape_data(val_pixels)
test_pixels = reshape_data(test_pixels)

# Grid Search for SVM
svm_param_grid = {'C': [0.1, 1, 10], 'gamma': ['scale', 'auto']}
svm_grid_search = GridSearchCV(SVC(), svm_param_grid, cv=3)
svm_grid_search.fit(train_pixels, train_labels)

# Plot SVM Grid Search Results
plot_grid_search_results(svm_grid_search, 'SVM')

# Get best parameters for SVM
svm_best_params = svm_grid_search.best_params_
print("Best Parameters (SVM):", svm_best_params)

# Grid Search for KNN
knn_param_grid = {'n_neighbors': [3, 5, 7]}
knn_grid_search = GridSearchCV(KNeighborsClassifier(), knn_param_grid, cv=3)
knn_grid_search.fit(train_pixels, train_labels)

# Plot KNN Grid Search Results
plot_grid_search_results(knn_grid_search, 'KNN')

# Get best parameters for KNN
knn_best_params = knn_grid_search.best_params_
print("Best Parameters (KNN):", knn_best_params)

# Grid Search for Decision Tree
dt_param_grid = {'max_depth': [3, 6, 10]}
dt_grid_search = GridSearchCV(DecisionTreeClassifier(), dt_param_grid, cv=3)
dt_grid_search.fit(train_pixels, train_labels)

# Plot Decision Tree Grid Search Results
plot_grid_search_results(dt_grid_search, 'Decision Tree')

# Get best parameters for Decision Tree
dt_best_params = dt_grid_search.best_params_
print("Best Parameters (Decision Tree):", dt_best_params)

# Grid Search for Logistic Regression
lr_param_grid = {'C': [0.1, 1, 10]}
lr_grid_search = GridSearchCV(LogisticRegression(), lr_param_grid, cv=3)
lr_grid_search.fit(train_pixels, train_labels)

# Plot Logistic Regression Grid Search Results
plot_grid_search_results(lr_grid_search, 'Logistic Regression')

# Get best parameters for Logistic Regression
lr_best_params = lr_grid_search.best_params_
print("Best Parameters (Logistic Regression):", lr_best_params)
