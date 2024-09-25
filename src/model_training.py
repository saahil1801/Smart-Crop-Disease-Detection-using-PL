import os
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from torchvision import transforms
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger  
from data_preprocessing import PlantDataset, IMAGE_SIZE

PROCESSED_DIR = 'data/processed'
LOG_DIR = 'data/logs' 

class PlantModel(pl.LightningModule):
    def __init__(self, input_shape, num_classes, learning_rate=1e-4):
        super(PlantModel, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.BatchNorm2d(32),
            nn.Dropout(0.1),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.BatchNorm2d(64),
            nn.Dropout(0.1),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.BatchNorm2d(128),
            nn.Dropout(0.1),

            nn.Flatten(),
            nn.Linear(128 * (input_shape[1] // 8) * (input_shape[2] // 8), 512),
            nn.ReLU(),
            nn.Linear(512, num_classes),
        )
        self.learning_rate = learning_rate
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        # Ensure the input is in float32 before passing to the model
        x = x.float()
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        val_loss = self.criterion(y_hat, y)
        self.log('val_loss', val_loss)
        return val_loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

def train_model():
    # Load the processed data
    X_train = np.load(os.path.join(PROCESSED_DIR, 'X_train.npy')).astype(np.float32)  # Ensure the data is float32
    X_val = np.load(os.path.join(PROCESSED_DIR, 'X_val.npy')).astype(np.float32)      # Ensure the data is float32
    y_train = np.load(os.path.join(PROCESSED_DIR, 'y_train.npy')).astype(np.float32)  # Ensure labels are float32
    y_val = np.load(os.path.join(PROCESSED_DIR, 'y_val.npy')).astype(np.float32)      # Ensure labels are float32

    class_names = pd.read_csv(os.path.join(PROCESSED_DIR, 'class_names.csv')).values.flatten().tolist()
    num_classes = len(class_names)

    # Data transformations
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(20),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dataset = PlantDataset(X_train, y_train, transform=transform)
    val_dataset = PlantDataset(X_val, y_val, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)

    # Model and training setup
    input_shape = (3, IMAGE_SIZE[0], IMAGE_SIZE[1])
    model = PlantModel(input_shape=input_shape, num_classes=num_classes)

    checkpoint_callback = ModelCheckpoint(monitor='val_loss', save_top_k=1, mode='min', dirpath='data/models', filename='best_model')
    early_stopping_callback = EarlyStopping(monitor='val_loss', patience=10, mode='min')

    logger = CSVLogger(save_dir=LOG_DIR, name='my_model')

    # Use MPS if available, otherwise fall back to CPU
    trainer = pl.Trainer(
        max_epochs=50, 
        callbacks=[early_stopping_callback, checkpoint_callback], 
        accelerator="mps" if torch.backends.mps.is_available() else "cpu", 
        devices=1,
        logger=logger
    )

    trainer.fit(model, train_loader, val_loader)

if __name__ == "__main__":
    train_model()
