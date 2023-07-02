from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
from joblib import load
import numpy as np

# Load saved models
knn_model = load('knn_model.joblib')
decision_tree_model = load('decision_tree_model.joblib')
logistic_regression_model = load('logistic_regression_model.joblib')
svm_model = load('svm_model.joblib')


def load_data():
    # Load data
    test_pixels = np.load(r"D:\Facial Expression Reconization\data\dataset\test_pixels.npy")
    test_labels = np.load(r"D:\Facial Expression Reconization\data\dataset\test_labels.npy")
    return test_pixels, test_labels


def reshape_data(data):
    # Reshape data
    return data.reshape(data.shape[0], -1)


# Load data
test_pixels, test_labels = load_data()

# Reshape data
test_pixels = reshape_data(test_pixels)

# Hard Voting
voting_hard = VotingClassifier(estimators=[
    ('knn', knn_model),
    ('decision_tree', decision_tree_model),
    ('logistic_regression', logistic_regression_model),
    ('svm', svm_model)
], voting='hard')

# Soft Voting
voting_soft = VotingClassifier(estimators=[
    ('knn', knn_model),
    ('decision_tree', decision_tree_model),
    ('logistic_regression', logistic_regression_model),
    ('svm', svm_model)
], voting='soft')

# Weighted Voting
voting_weighted = VotingClassifier(estimators=[
    ('knn', knn_model),
    ('decision_tree', decision_tree_model),
    ('logistic_regression', logistic_regression_model),
    ('svm', svm_model)
], voting='soft', weights=[319, 302, 365, 453])  # Adjust the weights as desired

# Fit the voting classifiers
voting_hard.fit(test_pixels, test_labels)
voting_soft.fit(test_pixels, test_labels)
voting_weighted.fit(test_pixels, test_labels)

# Make predictions
hard_voting_predictions = voting_hard.predict(test_pixels)
weighted_voting_predictions = voting_weighted.predict(test_pixels)

# Calculate evaluation metrics for each voting method
hard_voting_accuracy = accuracy_score(test_labels, hard_voting_predictions)
hard_voting_precision = precision_score(test_labels, hard_voting_predictions, average='weighted')
hard_voting_recall = recall_score(test_labels, hard_voting_predictions, average='weighted')
hard_voting_f1 = f1_score(test_labels, hard_voting_predictions, average='weighted')


weighted_voting_accuracy = accuracy_score(test_labels, weighted_voting_predictions)
weighted_voting_precision = precision_score(test_labels, weighted_voting_predictions, average='weighted')
weighted_voting_recall = recall_score(test_labels, weighted_voting_predictions, average='weighted')
weighted_voting_f1 = f1_score(test_labels, weighted_voting_predictions, average='weighted')

print("Hard Voting - Accuracy:", hard_voting_accuracy)
print("Hard Voting - Precision:", hard_voting_precision)
print("Hard Voting - Recall:", hard_voting_recall)
print("Hard Voting - F1-score:", hard_voting_f1)


print("Weighted Voting - Accuracy:", weighted_voting_accuracy)
print("Weighted Voting - Precision:", weighted_voting_precision)
print("Weighted Voting - Recall:", weighted_voting_recall)
print("Weighted Voting - F1-score:", weighted_voting_f1)
