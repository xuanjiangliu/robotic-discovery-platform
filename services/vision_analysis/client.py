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

# services/vision_analysis/client.py
#
# Description:
# A gRPC client for the VisionAnalysisService. It captures live data from a
# RealSense camera, streams it to the server, and visualizes all the
# analysis results in real-time.

import os
import sys
import grpc
import cv2
import numpy as np
import logging
from collections import deque

# --- Path Setup ---
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
protos_path = os.path.join(project_root, 'pkg', 'protos')
if project_root not in sys.path:
    sys.path.insert(0, project_root)
if protos_path not in sys.path:
    sys.path.append(protos_path)
# --- End Path Setup ---

from pkg.protos import vision_pb2, vision_pb2_grpc
from pkg.camera import Camera

# --- Configuration ---
SERVER_ADDRESS = 'localhost:50051'
CALIB_FILE = os.path.join(project_root, "ml", "configs", "calibration_data.npz")
SMOOTHING_WINDOW = 10 # Number of frames to average metrics over for display

def generate_requests(cam: Camera, frame_queue: deque):
    """
    A generator function that yields camera frame requests and queues the
    corresponding color image for synchronized visualization.
    """
    while True:
        depth_frame_obj, color_image = cam.get_frames()
        if color_image is None or depth_frame_obj is None:
            continue

        frame_queue.append(color_image)
        
        h, w, _ = color_image.shape
        depth_image = np.asanyarray(depth_frame_obj.get_data())
        depth_h, depth_w = depth_image.shape

        # Encode images to compressed formats before sending.
        # Use JPEG for the color image (lossy but high compression).
        _, color_bytes = cv2.imencode('.jpg', color_image)
        # Use PNG for the depth image (lossless to preserve data).
        _, depth_bytes = cv2.imencode('.png', depth_image)

        # Construct the request message with the compressed bytes.
        yield vision_pb2.AnalysisRequest(
            color_image=vision_pb2.Image(data=color_bytes.tobytes(), width=w, height=h),
            depth_image=vision_pb2.Image(data=depth_bytes.tobytes(), width=depth_w, height=depth_h)
        )
        cv2.waitKey(1)

def run_client():
    """Initializes and runs the gRPC client."""
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    cam = Camera()
    if not cam.start():
        logging.error("Failed to start camera.")
        return

    intrinsics, dist_coeffs = cam.load_intrinsics(CALIB_FILE)
    if intrinsics is None:
        logging.error(f"Could not load camera calibration from {CALIB_FILE}. Exiting.")
        cam.stop()
        return

    # --- UI Smoothing Deques ---
    frame_queue = deque(maxlen=20)
    mean_curv_history = deque(maxlen=SMOOTHING_WINDOW)
    max_curv_history = deque(maxlen=SMOOTHING_WINDOW)

    try:
        with grpc.insecure_channel(SERVER_ADDRESS) as channel:
            stub = vision_pb2_grpc.VisionAnalysisServiceStub(channel)
            logging.info(f"✅ Client connected to server at {SERVER_ADDRESS}")

            responses = stub.AnalyzeActuatorPerformance(generate_requests(cam, frame_queue))

            for response in responses:
                if not frame_queue:
                    continue
                
                display_image = frame_queue.popleft().copy()
                
                # --- Visualize Mask ---
                if response.mask:
                    mask_buffer = np.frombuffer(response.mask, dtype=np.uint8)
                    mask_image = cv2.imdecode(mask_buffer, cv2.IMREAD_GRAYSCALE)
                    if mask_image is not None:
                        red_overlay = np.zeros_like(display_image)
                        red_overlay[mask_image > 0] = [0, 0, 255] # BGR for red
                        display_image = cv2.addWeighted(display_image, 1.0, red_overlay, 0.5, 0)

                # --- Visualize Spline ---
                if response.spline_points:
                    points_3d = np.array([(p.x, p.y, p.z) for p in response.spline_points], dtype=np.float32)
                    if points_3d.shape[0] > 0:
                        image_points, _ = cv2.projectPoints(points_3d, (0,0,0), (0,0,0), intrinsics, dist_coeffs)
                        if image_points is not None:
                            image_points = np.int32(image_points).reshape(-1, 2)
                            cv2.polylines(display_image, [image_points], isClosed=False, color=(0, 255, 0), thickness=2)

                # --- Update and Smooth Metrics for Display ---
                mean_curv_history.append(response.mean_curvature)
                max_curv_history.append(response.max_curvature)
                
                smoothed_mean_curv = np.mean(mean_curv_history) if mean_curv_history else 0.0
                smoothed_max_curv = np.mean(max_curv_history) if max_curv_history else 0.0

                # --- Display Text ---
                cv2.putText(display_image, f"Mean Curvature: {smoothed_mean_curv:.4f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                cv2.putText(display_image, f"Max Curvature: {smoothed_max_curv:.4f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

                cv2.imshow("Vision Analysis Client", display_image)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

    except grpc.RpcError as e:
        logging.error(f"❌ Could not connect to server: {e.details()}. Is it running?")
    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}", exc_info=True)
    finally:
        cam.stop()
        cv2.destroyAllWindows()
        logging.info("Client shut down.")

if __name__ == '__main__':
    run_client()
