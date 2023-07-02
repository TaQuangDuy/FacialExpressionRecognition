import os
import numpy as np
import matplotlib.pyplot as plt
from keras.applications import ResNet50
from keras.layers import Dense, GlobalAveragePooling2D
from keras.models import Model
from keras.optimizers import Adam, RMSprop
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc

# Load preprocessed data
data_dir = os.path.join(os.getcwd(), '..', 'data', 'dataset')
train_pixels = np.load(os.path.join(data_dir, 'train_pixels.npy'))
train_labels = np.load(os.path.join(data_dir, 'train_labels.npy'))
val_pixels = np.load(os.path.join(data_dir, 'val_pixels.npy'))
val_labels = np.load(os.path.join(data_dir, 'val_labels.npy'))
test_pixels = np.load(os.path.join(data_dir, 'test_pixels.npy'))
test_labels = np.load(os.path.join(data_dir, 'test_labels.npy'))

# Convert grayscale images to RGB
train_pixels = np.repeat(train_pixels[..., np.newaxis], 3, -1)
val_pixels = np.repeat(val_pixels[..., np.newaxis], 3, -1)
test_pixels = np.repeat(test_pixels[..., np.newaxis], 3, -1)

# Define ResNet50 model
base_model = ResNet50(include_top=False, weights='imagenet', input_shape=(48, 48, 3))
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(256, activation='relu')(x)
predictions = Dense(7, activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=predictions)

# Define optimizer and learning rate
optimizer = Adam(lr=0.001)  # or RMSprop(lr=0.001)

# Compile the model
model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Define callbacks
early_stop = EarlyStopping(patience=5, verbose=1)
reduce_lr = ReduceLROnPlateau(factor=0.1, patience=3, verbose=1)

# Train the model
history = model.fit(train_pixels, train_labels, batch_size=32, epochs=50, validation_data=(val_pixels, val_labels),
                    callbacks=[early_stop, reduce_lr])

# Evaluate on test set
test_loss, test_accuracy = model.evaluate(test_pixels, test_labels)

# Generate predictions
test_predictions = model.predict(test_pixels)
test_predictions = np.argmax(test_predictions, axis=1)

# Compute evaluation metrics
classification_rep = classification_report(test_labels, test_predictions)
confusion_mtx = confusion_matrix(test_labels, test_predictions)

# Calculate F1 score, recall, and precision
f1_score = 2 * (classification_rep[1] * classification_rep[2]) / (classification_rep[1] + classification_rep[2])
recall = classification_rep[1]
precision = classification_rep[2]

# Compute ROC curve and AUC
n_classes = 7
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(test_labels, test_predictions, pos_label=i)
    roc_auc[i] = auc(fpr[i], tpr[i])

# Plot accuracy and loss curves
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.show()
