import os
import pandas as pd
import numpy as np
import albumentations as A
import cv2
import random
from imblearn.combine import SMOTEENN

data_path = os.path.join(os.getcwd(), '..', 'data')
df = pd.read_csv(os.path.join(data_path, 'fer2013.csv'))


def load_data(data_dir):
    train_pixels = np.load(os.path.join(data_dir, 'train_pixels.npy'))
    train_labels = np.load(os.path.join(data_dir, 'train_labels.npy'))
    test_pixels = np.load(os.path.join(data_dir, 'test_pixels.npy'))
    test_labels = np.load(os.path.join(data_dir, 'test_labels.npy'))
    return train_pixels, train_labels, test_pixels, test_labels


# Extract the pixel values and labels for training and validation sets
train_pixels = []
train_labels = []
val_pixels = []
val_labels = []
test_pixels = []
test_labels = []
class_counts = {}

for i, row in df.iterrows():
    pixels = np.asarray([int(p) for p in row['pixels'].split()])
    pixels = pixels.reshape((48, 48))
    if row['Usage'] == 'Training':
        train_pixels.append(pixels)
        train_labels.append(row['emotion'])
    elif row['Usage'] == 'PublicTest':
        val_pixels.append(pixels)
        val_labels.append(row['emotion'])
    else:
        test_pixels.append(pixels)
        test_labels.append(row['emotion'])

    # Count the number of images in each class
    if row['emotion'] in class_counts:
        class_counts[row['emotion']] += 1
    else:
        class_counts[row['emotion']] = 1

# Print the number of images in each class before oversampling
print("Class Counts (Before Oversampling):")
for emotion, count in class_counts.items():
    print(f"Emotion: {emotion} - Count: {count}")

# Convert the pixel values to numpy arrays
train_pixels = np.array(train_pixels, dtype=np.float32)
train_labels = np.array(train_labels)
val_pixels = np.array(val_pixels, dtype=np.float32)
val_labels = np.array(val_labels)
test_pixels = np.array(test_pixels, dtype=np.float32)
test_labels = np.array(test_labels)

# Normalize pixel values to range [0, 1]
train_pixels /= 255.0
val_pixels /= 255.0
test_pixels /= 255.0

# Reshape the training set to 2D
train_pixels_2d = train_pixels.reshape(train_pixels.shape[0], -1)

# Apply SMOTEENN oversampling and undersampling
smoteenn = SMOTEENN()
train_pixels_combined, train_labels_combined = smoteenn.fit_resample(train_pixels_2d, train_labels)

# Reshape the combined training set back to 3D
train_pixels_combined = train_pixels_combined.reshape(-1, 48, 48)

# Count the number of images in each class after combining oversampling and undersampling
class_counts_combined = {}
for emotion in train_labels_combined:
    if emotion in class_counts_combined:
        class_counts_combined[emotion] += 1
    else:
        class_counts_combined[emotion] = 1

# Print the number of images in each class after combining oversampling and undersampling
print("\nClass Counts (After Combining Oversampling and Undersampling):")
for emotion, count in class_counts_combined.items():
    print(f"Emotion: {emotion} - Count: {count}")

# Define the augmentation sequence
augmentations = [
    A.HorizontalFlip(p=0.5),
    A.RandomCrop(height=48, width=48, p=0.1),
    A.Rotate(limit=8, border_mode=cv2.BORDER_CONSTANT, value=0, p=0.5),
    A.GaussianBlur(blur_limit=(3, 7), sigma_limit=(0, 0.5), p=0.5),
]

# Define the maximum number of augmentations to apply
max_num_augmentations = 3

# Apply the augmentations to the training set
aug_train_pixels = np.empty((0, 48, 48), dtype=np.float32)
aug_train_labels = []
for i in range(len(train_pixels_combined)):
    img = train_pixels_combined[i]
    label = train_labels_combined[i]

    # Choose a random subset of augmentations to apply
    num_augmentations = random.randint(1, max_num_augmentations)
    chosen_augmentations = random.sample(augmentations, num_augmentations)

    # Apply the chosen augmentations to the image
    img_aug = A.Compose(chosen_augmentations, p=1.0)(image=img)

    # Add the augmented image and label to the training set
    aug_train_pixels = np.concatenate((aug_train_pixels, np.expand_dims(img_aug['image'], axis=0)))
    aug_train_labels.append(label)

# Convert the augmented training set to numpy arrays
aug_train_labels = np.array(aug_train_labels)

# Apply Histogram Equalization to the training set
for i in range(len(aug_train_pixels)):
    img = aug_train_pixels[i]
    img_eq = cv2.equalizeHist(img.astype(np.uint8))
    aug_train_pixels[i] = img_eq

# Apply Local Binary Patterns (LBP) to the training set
radius = 3
n_points = 8 * radius
for i in range(len(aug_train_pixels)):
    img = aug_train_pixels[i]
    img_lbp = np.zeros_like(img)
    for row in range(1, img.shape[0] - 1):
        for col in range(1, img.shape[1] - 1):
            center_pixel = img[row, col]
            binary_code = 0
            for i, (dx, dy) in enumerate([(1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1)]):
                neighbor_pixel = img[row + dy, col + dx]
                if neighbor_pixel >= center_pixel:
                    binary_code |= 1 << i
            img_lbp[row, col] = binary_code
    aug_train_pixels[i] = img_lbp

# Count the number of images in each class after augmentation
class_counts_augmented = {}
for emotion in aug_train_labels:
    if emotion in class_counts_augmented:
        class_counts_augmented[emotion] += 1
    else:
        class_counts_augmented[emotion] = 1

# Print the number of images in each class after augmentation
print("\nClass Counts (After Augmentation):")
for emotion, count in class_counts_augmented.items():
    print(f"Emotion: {emotion} - Count: {count}")

# Save the augmented and combined training set
dataset_path = os.path.join(data_path, 'dataset3')
os.makedirs(dataset_path, exist_ok=True)
np.save(os.path.join(dataset_path, 'train_pixels.npy'), aug_train_pixels.astype(np.float32))
np.save(os.path.join(dataset_path, 'train_labels.npy'), aug_train_labels)
np.save(os.path.join(dataset_path, 'val_pixels.npy'), val_pixels.astype(np.float32))
np.save(os.path.join(dataset_path, 'val_labels.npy'), val_labels)
np.save(os.path.join(dataset_path, 'test_pixels.npy'), test_pixels.astype(np.float32))
np.save(os.path.join(dataset_path, 'test_labels.npy'), test_labels)
