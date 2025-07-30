# Copyright 2025 Xuanjiang Liu
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# services/vision_analysis/vision_utils.py
#
# Description:
# This module contains utility functions specifically for the VisionAnalysis
# service, such as loading the model and calibration data.

import os
import torch
import logging
import numpy as np

# Assuming the new project structure where pkg is a top-level package
from pkg.segmentation_model import UNet

def load_vision_service_dependencies(model_path: str, calib_path: str) -> tuple:
    """
    Loads and prepares all dependencies required by the VisionAnalysisService.

    Args:
        model_path (str): The full path to the trained .pth model file.
        calib_path (str): The full path to the .npz calibration file.

    Returns:
        tuple: A tuple containing (model, intrinsics_matrix, distortion_coeffs).
               Returns (None, None, None) if any dependency fails to load.
    """
    logging.info("🧠 Loading resources for the VisionAnalysisService...")

    model = _load_segmentation_model(model_path)
    intrinsics, dist_coeffs = _load_calibration_data(calib_path)

    if model is None or intrinsics is None:
        logging.error("❌ FATAL: Could not load model or calibration data. Shutting down.")
        return None, None, None

    logging.info("✅ VisionAnalysisService initialized successfully.")
    return model, intrinsics, dist_coeffs


def _load_segmentation_model(model_path: str):
    """Loads the trained U-Net segmentation model."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if not os.path.exists(model_path):
        logging.error(f"❌ FATAL: Trained model file not found at '{model_path}'.")
        logging.error("   -> Please run the training script first.")
        return None
    try:
        model = UNet(n_channels=3, n_classes=1).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        logging.info("✅ Segmentation model loaded successfully.")
        return model
    except Exception as e:
        logging.error(f"❌ FAILED to load model: {e}")
        return None

def _load_calibration_data(calib_path: str):
    """Loads camera calibration data (intrinsics and distortion)."""
    if not os.path.exists(calib_path):
        logging.error(f"❌ FATAL: Camera calibration file not found at '{calib_path}'.")
        logging.error("   -> Please run the calibration script first.")
        return None, None

    with np.load(calib_path) as data:
        intrinsics = data['mtx']
        dist_coeffs = data['dist']
        logging.info("✅ Camera intrinsics loaded successfully.")
        return intrinsics, dist_coeffs
