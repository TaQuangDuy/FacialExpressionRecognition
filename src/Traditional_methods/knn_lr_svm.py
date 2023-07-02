import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
from joblib import dump


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


def train_knn(X_train, y_train, n_neighbors=5):
    # Create KNN classifier
    knn = KNeighborsClassifier(n_neighbors=n_neighbors)

    # Train the model
    knn.fit(X_train, y_train)

    return knn


def train_decision_tree(X_train, y_train, max_depth=6):
    # Create Decision Tree classifier
    decision_tree = DecisionTreeClassifier(max_depth=max_depth)

    # Train the model
    decision_tree.fit(X_train, y_train)

    return decision_tree


def train_logistic_regression(X_train, y_train, C=1):
    # Create Logistic Regression classifier
    logistic_regression = LogisticRegression(C=C)

    # Train the model
    logistic_regression.fit(X_train, y_train)

    return logistic_regression


def train_svm(X_train, y_train, C=10, gamma='scale'):
    # Create SVM classifier
    svm = SVC(C=C, gamma=gamma)

    # Train the model
    svm.fit(X_train, y_train)

    return svm


def train_decision_tree_with_adaboost(X_train, y_train, max_depth=6, n_estimators=50):
    # Create Decision Tree classifier
    base_classifier = DecisionTreeClassifier(max_depth=max_depth)

    # Create AdaBoost classifier
    adaboost = AdaBoostClassifier(base_estimator=base_classifier, n_estimators=n_estimators)

    # Train the model
    adaboost.fit(X_train, y_train)

    return adaboost


def train_logistic_regression_with_adaboost(X_train, y_train, n_estimators=50):
    # Create Logistic Regression classifier
    base_classifier = LogisticRegression()

    # Create AdaBoost classifier
    adaboost = AdaBoostClassifier(base_estimator=base_classifier, n_estimators=n_estimators)

    # Train the model
    adaboost.fit(X_train, y_train)

    return adaboost


def train_svm_with_adaboost(X_train, y_train, C=10, gamma='scale', n_estimators=50):
    # Create SVM classifier
    base_classifier = SVC(C=C, gamma=gamma)

    # Create AdaBoost classifier
    adaboost = AdaBoostClassifier(base_estimator=base_classifier, n_estimators=n_estimators)

    # Train the model
    adaboost.fit(X_train, y_train)

    return adaboost


def evaluate_model(model, X_test, y_test):
    # Make predictions
    y_pred = model.predict(X_test)

    # Calculate evaluation metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')

    return accuracy, precision, recall, f1


# Load data
train_pixels, train_labels, val_pixels, val_labels, test_pixels, test_labels = load_data()

# Reshape data
train_pixels = reshape_data(train_pixels)
val_pixels = reshape_data(val_pixels)
test_pixels = reshape_data(test_pixels)

# Train KNN
knn_model = train_knn(train_pixels, train_labels, n_neighbors=5)

# Save the model
dump(knn_model, 'knn_model.joblib')

# Evaluate KNN
knn_accuracy, knn_precision, knn_recall, knn_f1 = evaluate_model(knn_model, test_pixels, test_labels)

print("KNN - Accuracy:", knn_accuracy)
print("KNN - Precision:", knn_precision)
print("KNN - Recall:", knn_recall)
print("KNN - F1-score:", knn_f1)

# Train Decision Tree
decision_tree_model = train_decision_tree(train_pixels, train_labels, max_depth=6)

# Save the model
dump(decision_tree_model, 'decision_tree_model.joblib')

# Evaluate Decision Tree
decision_tree_accuracy, decision_tree_precision, decision_tree_recall, decision_tree_f1 = evaluate_model(decision_tree_model, test_pixels, test_labels)

print("Decision Tree - Accuracy:", decision_tree_accuracy)
print("Decision Tree - Precision:", decision_tree_precision)
print("Decision Tree - Recall:", decision_tree_recall)
print("Decision Tree - F1-score:", decision_tree_f1)

# Train Logistic Regression
logistic_regression_model = train_logistic_regression(train_pixels, train_labels, C=1)

# Save the model
dump(logistic_regression_model, 'logistic_regression_model.joblib')

# Evaluate Logistic Regression
lr_accuracy, lr_precision, lr_recall, lr_f1 = evaluate_model(logistic_regression_model, test_pixels, test_labels)

print("Logistic Regression - Accuracy:", lr_accuracy)
print("Logistic Regression - Precision:", lr_precision)
print("Logistic Regression - Recall:", lr_recall)
print("Logistic Regression - F1-score:", lr_f1)

# Train SVM
svm_model = train_svm(train_pixels, train_labels, C=10, gamma='scale')

# Save the model
dump(svm_model, 'svm_model.joblib')

# Evaluate SVM
svm_accuracy, svm_precision, svm_recall, svm_f1 = evaluate_model(svm_model, test_pixels, test_labels)

print("SVM - Accuracy:", svm_accuracy)
print("SVM - Precision:", svm_precision)
print("SVM - Recall:", svm_recall)
print("SVM - F1-score:", svm_f1)
