import os
import pandas as pd
import numpy as np
import albumentations as A
import cv2
import random

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
for i in range(len(train_pixels)):
    img = train_pixels[i]
    label = train_labels[i]

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

# Save the preprocessed data to disk
dataset_path = os.path.join(data_path, 'dataset')
os.makedirs(dataset_path, exist_ok=True)
np.save(os.path.join(dataset_path, 'train_pixels.npy'), aug_train_pixels.astype(np.float32))
np.save(os.path.join(dataset_path, 'train_labels.npy'), aug_train_labels)
np.save(os.path.join(dataset_path, 'val_pixels.npy'), val_pixels.astype(np.float32))
np.save(os.path.join(dataset_path, 'val_labels.npy'), val_labels)
np.save(os.path.join(dataset_path, 'test_pixels.npy'), test_pixels.astype(np.float32))
np.save(os.path.join(dataset_path, 'test_labels.npy'), test_labels)
