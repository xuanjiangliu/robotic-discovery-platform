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

# scripts/02_collect_segmentation_data.py
#
# Description:
# This script is a simple tool for capturing synchronized color and depth
# frames from the RealSense camera. It's used to collect raw data for
# building new training datasets.

import cv2
import os
import time
import numpy as np
import sys
import logging

# --- Path Setup ---
# This is the standard way to make a script runnable from anywhere in the project
# and ensure it can find the 'pkg' directory.
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Assuming this script is run from the project root directory
from pkg.camera import Camera

# --- Configuration ---
SAVE_INTERVAL_S = 0.5  # Seconds between each save operation
CAPTURE_WIDTH = 640
CAPTURE_HEIGHT = 480

# --- Main Application Logic ---
def main():
    """Main function to run the data collection process."""
    print("--- EvoFab Vision: 02 Raw Data Collector ---")

    # Define paths relative to the project root
    output_dir = os.path.join("ml", "raw_data", f"capture_{int(time.time())}")
    color_dir = os.path.join(output_dir, "color")
    depth_dir = os.path.join(output_dir, "depth")

    os.makedirs(color_dir, exist_ok=True)
    os.makedirs(depth_dir, exist_ok=True)
    print(f"✅ Saving data to: {output_dir}")

    # Initialize the camera
    cam = Camera(width=CAPTURE_WIDTH, height=CAPTURE_HEIGHT)
    if not cam.start():
        return

    # Main loop variables
    is_saving = False
    last_save_time = 0
    saved_count = 0

    print("\n" + "="*30)
    print("Instructions:")
    print("  - Position the actuator in the desired environment.")
    print("  - Press 's' to START or STOP automatically saving frame pairs.")
    print("  - Press 'q' to quit.")
    print("="*30 + "\n")

    try:
        while True:
            depth_frame, color_image = cam.get_frames()
            if color_image is None or depth_frame is None:
                continue

            display_image = color_image.copy()

            # Save frames if toggled
            if is_saving and (time.time() - last_save_time > SAVE_INTERVAL_S):
                timestamp = f"{time.time():.2f}".replace('.', '_')
                color_filename = os.path.join(color_dir, f"color_{timestamp}.png")
                depth_filename = os.path.join(depth_dir, f"depth_{timestamp}.npy")

                cv2.imwrite(color_filename, color_image)
                np.save(depth_filename, np.asanyarray(depth_frame.get_data()))

                saved_count += 1
                last_save_time = time.time()
                print(f"📸 Saved frame pair #{saved_count}", end='\r')


            # --- Visualization ---
            status_text = "SAVING" if is_saving else "PAUSED"
            status_color = (0, 255, 0) if is_saving else (0, 165, 255)
            cv2.putText(display_image, f"Status: {status_text}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
            cv2.putText(display_image, f"Saved Pairs: {saved_count}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

            cv2.imshow("Raw Data Collector", display_image)
            key = cv2.waitKey(1)

            if key & 0xFF == ord('q'):
                print(f"\n✅ Total pairs saved: {saved_count}")
                break
            if key & 0xFF == ord('s'):
                is_saving = not is_saving
                last_save_time = time.time()
                if is_saving:
                    print("\n▶️ Resumed saving...")
                else:
                    print("\n⏸️ Paused saving.")

    finally:
        print("\nShutting down.")
        cam.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
