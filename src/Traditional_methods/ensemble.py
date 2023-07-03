import os
import numpy as np
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from joblib import dump, load

data_path = os.path.join(os.getcwd(), '..', 'data')
dataset_path = os.path.join(data_path, 'dataset2')

test_pixels = np.load(os.path.join(dataset_path, 'test_pixels.npy'))
test_labels = np.load(os.path.join(dataset_path, 'test_labels.npy'))

test_pixels = test_pixels.reshape((-1, 48 * 48))

rf_model = load('rf_model.joblib')
svm_model = load('svm_model.joblib')
xgb_model = load('xgb_model.joblib')

ensemble_voting_hard = VotingClassifier(estimators=[('rf', rf_model), ('svm', svm_model), ('xgb', xgb_model)], voting='hard')
ensemble_voting_soft = VotingClassifier(estimators=[('rf', rf_model), ('svm', svm_model), ('xgb', xgb_model)], voting='soft')
ensemble_voting_weighted = VotingClassifier(estimators=[('rf', rf_model), ('svm', svm_model), ('xgb', xgb_model)], voting='soft', weights=[1, 2, 1])

ensemble_voting_hard.fit(test_pixels, test_labels)
ensemble_voting_soft.fit(test_pixels, test_labels)
ensemble_voting_weighted.fit(test_pixels, test_labels)

ensemble_hard_predictions = ensemble_voting_hard.predict(test_pixels)
ensemble_soft_predictions = ensemble_voting_soft.predict(test_pixels)
ensemble_weighted_predictions = ensemble_voting_weighted.predict(test_pixels)

accuracy_hard = accuracy_score(test_labels, ensemble_hard_predictions)
accuracy_soft = accuracy_score(test_labels, ensemble_soft_predictions)
accuracy_weighted = accuracy_score(test_labels, ensemble_weighted_predictions)

f1_hard = f1_score(test_labels, ensemble_hard_predictions, average='macro')
f1_soft = f1_score(test_labels, ensemble_soft_predictions, average='macro')
f1_weighted = f1_score(test_labels, ensemble_weighted_predictions, average='macro')

precision_hard = precision_score(test_labels, ensemble_hard_predictions, average='macro')
precision_soft = precision_score(test_labels, ensemble_soft_predictions, average='macro')
precision_weighted = precision_score(test_labels, ensemble_weighted_predictions, average='macro')

recall_hard = recall_score(test_labels, ensemble_hard_predictions, average='macro')
recall_soft = recall_score(test_labels, ensemble_soft_predictions, average='macro')
recall_weighted = recall_score(test_labels, ensemble_weighted_predictions, average='macro')

dump(ensemble_voting_hard, 'ensemble_hard_model.joblib')
dump(ensemble_voting_soft, 'ensemble_soft_model.joblib')
dump(ensemble_voting_weighted, 'ensemble_weighted_model.joblib')

print('Hard Voting - Accuracy:', accuracy_hard, 'F1-Score:', f1_hard, 'Precision:', precision_hard, 'Recall:', recall_hard)
print('Soft Voting - Accuracy:', accuracy_soft, 'F1-Score:', f1_soft, 'Precision:', precision_soft, 'Recall:', recall_soft)
print('Weighted Voting - Accuracy:', accuracy_weighted, 'F1-Score:', f1_weighted, 'Precision:', precision_weighted, 'Recall:', recall_weighted)
