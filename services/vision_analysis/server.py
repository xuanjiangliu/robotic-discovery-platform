# services/vision_analysis/server.py
#
# Description:
# This script runs the gRPC server for the VisionAnalysisService. It handles
# incoming requests, performs real-time actuator analysis, and returns the
# curvature profile.

import os
import sys
import grpc
import logging
import cv2
import numpy as np
import torch
import mlflow
import pathlib
import time # <-- ADD THIS
from concurrent import futures
from dataclasses import asdict
from torchvision import transforms

# --- Path Setup ---
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
protos_path = os.path.join(project_root, 'pkg', 'protos')
if project_root not in sys.path:
    sys.path.insert(0, project_root)
if protos_path not in sys.path:
    sys.path.append(protos_path)
# --- End Path Setup ---

from pkg.protos import vision_pb2
from pkg.protos import vision_pb2_grpc
from pkg.geometry_utils import compute_curvature_profile
from pkg.segmentation_model import UNet

# --- Configuration ---
# Setup for structured logging
LOGS_DIR = os.path.join(project_root, "logs")
os.makedirs(LOGS_DIR, exist_ok=True)
METRICS_LOG_FILE = os.path.join(LOGS_DIR, "vision_service_metrics.csv")

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_IMG_SIZE = 256
DEFAULT_DEPTH_SCALE = 0.001

# --- MLflow and File Paths ---
MLRUNS_DIR = os.path.join(project_root, "ml", "mlruns")
MLFLOW_TRACKING_URI = pathlib.Path(MLRUNS_DIR).as_uri()
MLFLOW_MODEL_NAME = "Actuator-Segmenter"
CALIB_FILE = os.path.join(project_root, "ml", "configs", "calibration_data.npz")


def _setup_metrics_log():
    """Creates the metrics log file and writes the header if it doesn't exist."""
    if not os.path.exists(METRICS_LOG_FILE):
        with open(METRICS_LOG_FILE, "w") as f:
            f.write("timestamp,mean_curvature,max_curvature,mask_coverage_percent\n")

def _load_resources():
    # ... (no changes in this function)
    model = None
    intrinsics = None
    depth_scale = None
    try:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        model_uri = f"models:/{MLFLOW_MODEL_NAME}/latest"
        model = mlflow.pytorch.load_model(model_uri, map_location=device)
        model.eval()
        logging.info(f"✅ Segmentation model '{MLFLOW_MODEL_NAME}' (latest) loaded from MLflow.")
    except Exception as e:
        logging.error(f"❌ FATAL: Failed to load model from MLflow: {e}")
        return None, None, None
    if not os.path.exists(CALIB_FILE):
        logging.error(f"❌ FATAL: Calibration data not found at '{CALIB_FILE}'.")
        return model, None, None
    try:
        with np.load(CALIB_FILE) as data:
            intrinsics = data['mtx']
            depth_scale = data.get('depth_scale', DEFAULT_DEPTH_SCALE)
        logging.info("✅ Camera intrinsics and depth scale loaded.")
    except Exception as e:
        logging.error(f"❌ FATAL: Failed to load intrinsics: {e}")
        return model, None, None
    return model, intrinsics, depth_scale

class VisionAnalysisService(vision_pb2_grpc.VisionAnalysisServiceServicer):
    # ... (no changes in __init__)
    def __init__(self, model, intrinsics, depth_scale):
        self.model = model
        self.intrinsics = intrinsics
        self.depth_scale = depth_scale
        self.preprocess = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((MODEL_IMG_SIZE, MODEL_IMG_SIZE), antialias=True),
        ])
        logging.info("✅ VisionAnalysisService initialized successfully.")

    def AnalyzeActuatorPerformance(self, request_iterator, context):
        logging.info("🤝 Received new request for actuator analysis.")
        try:
            for request in request_iterator:
                color_image = cv2.imdecode(np.frombuffer(request.color_image.data, np.uint8), cv2.IMREAD_COLOR)
                depth_image = cv2.imdecode(np.frombuffer(request.depth_image.data, np.uint8), cv2.IMREAD_UNCHANGED)

                image_rgb = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
                input_tensor = self.preprocess(image_rgb).unsqueeze(0).to(device)
                with torch.no_grad():
                    output = self.model(input_tensor)
                    mask_resized = (torch.sigmoid(output) > 0.5).squeeze(0).cpu().numpy().transpose(1, 2, 0).astype(np.uint8)
                final_mask = cv2.resize(mask_resized, (color_image.shape[1], color_image.shape[0]), interpolation=cv2.INTER_NEAREST)

                curvature_results = compute_curvature_profile(
                    depth_image=depth_image, mask=final_mask,
                    intrinsics=self.intrinsics, depth_scale=self.depth_scale
                )

                # Calculate mask coverage for logging
                mask_coverage = (np.count_nonzero(final_mask) / final_mask.size) * 100

                response = vision_pb2.AnalysisResponse()
                if curvature_results:
                    response.mean_curvature = curvature_results.mean_curvature
                    response.max_curvature = curvature_results.max_curvature
                    if hasattr(curvature_results, 'spline_points') and curvature_results.spline_points:
                        response.spline_points.extend([vision_pb2.Point3D(**asdict(p)) for p in curvature_results.spline_points])

                _, mask_bytes = cv2.imencode('.png', final_mask * 255)
                response.mask = mask_bytes.tobytes()

                # Log key metrics to the CSV file
                with open(METRICS_LOG_FILE, "a") as f:
                    timestamp = time.time()
                    mean_k = response.mean_curvature
                    max_k = response.max_curvature
                    f.write(f"{timestamp},{mean_k},{max_k},{mask_coverage}\n")

                yield response

        except Exception as e:
            logging.error(f"An unhandled exception occurred during analysis: {e}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f"Internal error during analysis: {e}")
            yield vision_pb2.AnalysisResponse()


def serve():
    logging.info("🧠 Loading resources for the VisionAnalysisService...")
    # The log file is set up on start
    _setup_metrics_log()
    
    model, intrinsics, depth_scale = _load_resources()

    if model is None or intrinsics is None or depth_scale is None:
        logging.error("❌ FATAL: Could not load all required resources. Shutting down.")
        return
    
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    vision_pb2_grpc.add_VisionAnalysisServiceServicer_to_server(
        VisionAnalysisService(model, intrinsics, depth_scale), server
    )
    server.add_insecure_port('[::]:50051')
    server.start()
    logging.info("🚀 VisionAnalysisService started. Listening on [::]:50051...")
    server.wait_for_termination()


if __name__ == '__main__':
    serve()
