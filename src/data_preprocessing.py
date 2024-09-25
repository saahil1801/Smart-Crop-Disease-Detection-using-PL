import os
import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# Constants
IMAGE_SIZE = (224, 224)
DATA_DIR = 'data/PlantVillage'
SELECTED_CLASSES = ['Potato___Late_blight', 'Potato___Early_blight', 'Potato___healthy']
PROCESSED_DIR = 'data/processed'

class PlantDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label

def load_data(data_dir, selected_classes):
    images = []
    labels = []
    class_names = []

    for label, class_name in enumerate(selected_classes):
        class_dir = os.path.join(data_dir, class_name)
        if os.path.exists(class_dir):
            class_names.append(class_name)
            for img_name in os.listdir(class_dir):
                img_path = os.path.join(class_dir, img_name)
                img = cv2.imread(img_path)
                if img is not None:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img = cv2.resize(img, IMAGE_SIZE)
                    images.append(img)
                    labels.append(label)
        else:
            print(f"Warning: Directory for class {class_name} does not exist. Skipping.")

    return np.array(images), np.array(labels), class_names

def preprocess_data():
    images, labels, class_names = load_data(DATA_DIR, SELECTED_CLASSES)
    images = images / 255.0  # Normalize
    X_train, X_val, y_train, y_val = train_test_split(images, labels, test_size=0.2, random_state=42, stratify=labels)

    os.makedirs(PROCESSED_DIR, exist_ok=True)
    np.save(os.path.join(PROCESSED_DIR, 'X_train.npy'), X_train)
    np.save(os.path.join(PROCESSED_DIR, 'X_val.npy'), X_val)
    np.save(os.path.join(PROCESSED_DIR, 'y_train.npy'), y_train)
    np.save(os.path.join(PROCESSED_DIR, 'y_val.npy'), y_val)
    pd.DataFrame(class_names).to_csv(os.path.join(PROCESSED_DIR, 'class_names.csv'), index=False)

if __name__ == "__main__":
    preprocess_data()
