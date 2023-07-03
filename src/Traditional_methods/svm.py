import os
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
from sklearn.ensemble import VotingClassifier
import matplotlib.pyplot as plt
import joblib


data_path = os.path.join(os.getcwd(), '..', 'data', 'dataset2')
train_pixels = np.load(os.path.join(data_path, 'train_pixels.npy'))
train_labels = np.load(os.path.join(data_path, 'train_labels.npy'))
val_pixels = np.load(os.path.join(data_path, 'val_pixels.npy'))
val_labels = np.load(os.path.join(data_path, 'val_labels.npy'))
train_pixels = train_pixels.reshape(train_pixels.shape[0], -1)
val_pixels = val_pixels.reshape(val_pixels.shape[0], -1)

param_grid = {
    'C': [0.1, 1, 10],
    'kernel': ['linear', 'rbf'],
    'gamma': ['scale', 'auto']
}

svm_model = SVC(random_state=42, probability=True)  # Set probability=True
grid_search = GridSearchCV(svm_model, param_grid, cv=5)
grid_search.fit(train_pixels, train_labels)
best_params = grid_search.best_params_
scores = grid_search.cv_results_['mean_test_score']
scores = np.array(scores).reshape(len(param_grid['C']), len(param_grid['kernel']))
plt.figure(figsize=(12, 6))
for i, C in enumerate(param_grid['C']):
    plt.plot(param_grid['kernel'], scores[i], label=f'C={C}')
plt.xlabel('Kernel')
plt.ylabel('Mean Test Score')
plt.legend()
plt.title('Grid Search Results')
plt.grid(True)
plt.savefig('grid_search_results_svm.png')
svm_model = SVC(random_state=42, probability=True, **best_params)  # Set probability=True
svm_model.fit(train_pixels, train_labels)
val_predictions = svm_model.predict(val_pixels)
accuracy = accuracy_score(val_labels, val_predictions)
f1 = f1_score(val_labels, val_predictions, average='weighted')
recall = recall_score(val_labels, val_predictions, average='weighted')
precision = precision_score(val_labels, val_predictions, average='weighted')
print('Accuracy:', accuracy)
print('F1 Score:', f1)
print('Recall:', recall)
print('Precision:', precision)
joblib.dump(svm_model, 'svm_model.pkl')
plt.savefig('grid_search_results_svm.png')
