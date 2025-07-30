# scripts/train_segmenter.py
#
# Description:
# Trains the U-Net segmentation model for identifying soft actuators.
# This script is integrated with MLFlow for experiment tracking and
# model versioning.

import os
import sys
import logging
import pathlib
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import cv2
import mlflow
import mlflow.pytorch

# --- Path Setup ---
# Ensures that the script can find the 'pkg' directory when run from the root
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from pkg.segmentation_model import UNet

# --- Configuration & Hyperparameters ---
LEARNING_RATE = 1e-4
BATCH_SIZE = 4
EPOCHS = 50
VALIDATION_SPLIT = 0.2
IMG_SIZE = 256
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Corrected Paths ---
# These paths reflect the final, streamlined MLOps folder structure.
DATASET_DIR = os.path.join("ml", "datasets", "processed")
IMAGE_DIR = os.path.join(DATASET_DIR, "images")
MASK_DIR = os.path.join(DATASET_DIR, "masks")
MLRUNS_DIR = os.path.join(project_root, "ml", "mlruns")
MODEL_OUTPUT_DIR = os.path.join(project_root, "ml", "models", "segmentation")

# MLFlow settings
MLFLOW_TRACKING_URI = pathlib.Path(MLRUNS_DIR).as_uri()
MLFLOW_EXPERIMENT_NAME = "Actuator Segmentation"
MLFLOW_MODEL_NAME = "Actuator-Segmenter"

# --- Dataset Definition ---
class SegmentationDataset(Dataset):
    """Custom PyTorch Dataset for loading actuator images and masks."""
    def __init__(self, image_dir, mask_dir, size=(IMG_SIZE, IMG_SIZE)):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.size = size
        # Ensure filenames match between images and masks
        self.ids = [fname for fname in os.listdir(image_dir) if os.path.isfile(os.path.join(mask_dir, fname))]

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        img_name = self.ids[idx]
        img_path = os.path.join(self.image_dir, img_name)
        mask_path = os.path.join(self.mask_dir, img_name)

        # Load and resize image
        image = cv2.imread(img_path, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, self.size, interpolation=cv2.INTER_AREA)

        # Load and resize mask
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, self.size, interpolation=cv2.INTER_NEAREST)

        # Normalize and transpose
        image = (image.astype(np.float32) / 255.0)
        mask = (mask.astype(np.float32) / 255.0)
        
        # Convert to torch tensor
        image_tensor = torch.from_numpy(image).permute(2, 0, 1)
        mask_tensor = torch.from_numpy(mask).unsqueeze(0)

        return image_tensor, mask_tensor

# --- Main Training Function ---
def train_model():
    """
    Executes the full training and model registration pipeline using MLFlow.
    """
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    os.makedirs(MODEL_OUTPUT_DIR, exist_ok=True)

    # --- MLFlow Setup ---
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    with mlflow.start_run() as run:
        logging.info(f"🚀 Starting MLFlow Run: {run.info.run_name}")
        
        # --- 1. Log Hyperparameters ---
        params = {
            "learning_rate": LEARNING_RATE,
            "batch_size": BATCH_SIZE,
            "epochs": EPOCHS,
            "validation_split": VALIDATION_SPLIT,
            "image_size": IMG_SIZE,
            "device": str(DEVICE),
            "architecture": "UNet"
        }
        mlflow.log_params(params)
        logging.info("Parameters logged to MLFlow.")

        # --- 2. Load Data ---
        dataset = SegmentationDataset(image_dir=IMAGE_DIR, mask_dir=MASK_DIR)
        
        val_size = int(len(dataset) * VALIDATION_SPLIT)
        train_size = len(dataset) - val_size
        train_set, val_set = random_split(dataset, [train_size, val_size])

        train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
        logging.info(f"Dataset loaded: {train_size} training samples, {val_size} validation samples.")

        # --- 3. Initialize Model, Optimizer, and Loss ---
        model = UNet(n_channels=3, n_classes=1).to(DEVICE)
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
        criterion = nn.BCEWithLogitsLoss()

        # --- 4. Training and Validation Loop ---
        best_val_loss = float('inf')
        best_model_path = os.path.join(MODEL_OUTPUT_DIR, "best_segmentation_model.pth") 

        for epoch in range(EPOCHS):
            # Training
            model.train()
            train_loss = 0.0
            pbar_train = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]")
            for images, masks in pbar_train:
                images, masks = images.to(DEVICE), masks.to(DEVICE)
                
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, masks)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
                pbar_train.set_postfix({"Loss": loss.item()})
            
            avg_train_loss = train_loss / len(train_loader)

            # Validation
            model.eval()
            val_loss = 0.0
            pbar_val = tqdm(val_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Val]")
            with torch.no_grad():
                for images, masks in pbar_val:
                    images, masks = images.to(DEVICE), masks.to(DEVICE)
                    outputs = model(images)
                    loss = criterion(outputs, masks)
                    val_loss += loss.item()
                    pbar_val.set_postfix({"Loss": loss.item()})
            
            avg_val_loss = val_loss / len(val_loader)
            
            mlflow.log_metric("train_loss", avg_train_loss, step=epoch)
            mlflow.log_metric("val_loss", avg_val_loss, step=epoch)

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save(model.state_dict(), best_model_path)
                logging.info(f"✅ New best model found at epoch {epoch+1} with validation loss: {best_val_loss:.4f}")
        
        mlflow.log_metric("best_val_loss", best_val_loss)

        # --- 5. Log and Register Model ---
        logging.info("Logging best model to MLFlow...")
        model.load_state_dict(torch.load(best_model_path))

        # An input example to auto-infer the model signature.
        input_example = next(iter(train_loader))[0].to(DEVICE)
        
        model_info = mlflow.pytorch.log_model(
            pytorch_model=model,
            name="model",
            registered_model_name=MLFLOW_MODEL_NAME
        )
        
        registered_version = model_info.registered_model_version
        logging.info(f"✅ Model '{MLFLOW_MODEL_NAME}' registered with version {registered_version}.")

        # Clean up the temporary local model file
        os.remove(best_model_path)


if __name__ == '__main__':
    train_model()
