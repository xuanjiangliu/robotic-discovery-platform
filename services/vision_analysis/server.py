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
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_IMG_SIZE = 256
DEFAULT_DEPTH_SCALE = 0.001 # Default for Intel RealSense cameras

# --- MLflow and File Paths ---
MLRUNS_DIR = os.path.join(project_root, "ml", "mlruns")
MLFLOW_TRACKING_URI = pathlib.Path(MLRUNS_DIR).as_uri()
MLFLOW_MODEL_NAME = "Actuator-Segmenter"
CALIB_FILE = os.path.join(project_root, "ml", "configs", "calibration_data.npz")


def _load_resources():
    """
    Loads all necessary resources for the service: the ML model, camera
    intrinsics, and depth scale.
    """
    model = None
    intrinsics = None
    depth_scale = None

    # --- Load Model from MLflow ---
    try:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        model_uri = f"models:/{MLFLOW_MODEL_NAME}/latest"
        model = mlflow.pytorch.load_model(model_uri, map_location=device)
        model.eval()
        logging.info(f"✅ Segmentation model '{MLFLOW_MODEL_NAME}' (latest) loaded from MLflow.")
    except Exception as e:
        logging.error(f"❌ FATAL: Failed to load model from MLflow: {e}")
        return None, None, None

    # --- Load Calibration Data ---
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
    """
    The gRPC service implementation for analyzing actuator performance.
    """
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
                # Decode the compressed image data from the client.
                color_image = cv2.imdecode(np.frombuffer(request.color_image.data, np.uint8), cv2.IMREAD_COLOR)
                depth_image = cv2.imdecode(np.frombuffer(request.depth_image.data, np.uint8), cv2.IMREAD_UNCHANGED)

                # --- 1. Get Segmentation Mask from U-Net ---
                image_rgb = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
                input_tensor = self.preprocess(image_rgb).unsqueeze(0).to(device)
                with torch.no_grad():
                    output = self.model(input_tensor)
                    mask_resized = (torch.sigmoid(output) > 0.5).squeeze(0).cpu().numpy().transpose(1, 2, 0).astype(np.uint8)
                final_mask = cv2.resize(mask_resized, (color_image.shape[1], color_image.shape[0]), interpolation=cv2.INTER_NEAREST)

                # --- 2. Analyze Geometry ---
                curvature_results = compute_curvature_profile(
                    depth_image=depth_image,
                    mask=final_mask,
                    intrinsics=self.intrinsics,
                    depth_scale=self.depth_scale
                )

                # --- 3. Construct and Send Response ---
                response = vision_pb2.AnalysisResponse()
                if curvature_results:
                    response.mean_curvature = curvature_results.mean_curvature
                    response.max_curvature = curvature_results.max_curvature
                    if hasattr(curvature_results, 'spline_points') and curvature_results.spline_points is not None:
                        response.spline_points.extend([vision_pb2.Point3D(**asdict(p)) for p in curvature_results.spline_points])

                _, mask_bytes = cv2.imencode('.png', final_mask * 255)
                response.mask = mask_bytes.tobytes()
                
                yield response

        except Exception as e:
            logging.error(f"An unhandled exception occurred during analysis: {e}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f"Internal error during analysis: {e}")
            yield vision_pb2.AnalysisResponse()


def serve():
    """
    Starts the gRPC server and waits for connections.
    """
    logging.info("🧠 Loading resources for the VisionAnalysisService...")
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
