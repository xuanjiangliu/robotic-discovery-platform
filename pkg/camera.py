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

# pkg/camera.py
#
# Description:
# This module provides a high-level, thread-safe wrapper for an Intel
# RealSense camera. It uses a background thread for frame grabbing to ensure
# the main application thread never blocks, providing maximum performance.

import pyrealsense2 as rs
import numpy as np
import os
import logging
import threading
import time

class Camera:
    """
    A thread-safe wrapper class for the Intel RealSense D4XX camera series
    that uses a background thread for non-blocking frame acquisition.
    """

    def __init__(self, width=640, height=480, fps=30):
        """Initializes the Camera object and the background frame-reading thread."""
        self.width = width
        self.height = height
        self.fps = fps
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.profile = None
        self.align = None
        self.depth_scale = None

        # --- Threading Members ---
        # This lock ensures that the latest_frame is not accessed by one
        # thread while being written to by another.
        self.frame_lock = threading.Lock()
        
        # This event is used to signal the background thread to stop.
        self.stopped = threading.Event()
        
        # The latest frame received from the camera is stored here.
        self.latest_frame = None
        
        # The background thread that will continuously read frames.
        # daemon=True ensures the thread will exit when the main program does.
        self.thread = threading.Thread(target=self._read_loop, daemon=True)

        # Configure the camera streams
        self.config.enable_stream(rs.stream.depth, self.width, self.height, rs.format.z16, self.fps)
        self.config.enable_stream(rs.stream.color, self.width, self.height, rs.format.bgr8, self.fps)

    def start(self):
        """Starts the RealSense pipeline and the background frame-reading thread."""
        logging.info("Starting RealSense pipeline...")
        try:
            self.profile = self.pipeline.start(self.config)
            self.align = rs.align(rs.stream.color)
            self.depth_scale = self.profile.get_device().first_depth_sensor().get_depth_scale()
            
            # Start the background thread
            self.thread.start()
            logging.info("✅ Camera started and background frame-reading thread is running.")
            return True
        except Exception as e:
            logging.error(f"❌ ERROR: Failed to start RealSense pipeline: {e}")
            return False

    def stop(self):
        """Signals the background thread to stop and then stops the pipeline."""
        logging.info("Stopping camera...")
        self.stopped.set()  # Signal the thread to exit its loop
        self.thread.join()  # Wait for the thread to finish cleanly
        self.pipeline.stop()
        logging.info("✅ RealSense pipeline stopped.")

    def _read_loop(self):
        """
        The main loop of the background thread.
        Continuously waits for frames from the camera and updates the
        'latest_frame' instance variable.
        """
        while not self.stopped.is_set():
            try:
                frames = self.pipeline.wait_for_frames()
                aligned_frames = self.align.process(frames)

                depth_frame = aligned_frames.get_depth_frame()
                color_frame = aligned_frames.get_color_frame()

                if not depth_frame or not color_frame:
                    continue

                color_image = np.asanyarray(color_frame.get_data())
                
                # Acquire the lock to safely update the latest frame
                with self.frame_lock:
                    self.latest_frame = (depth_frame, color_image)

            except RuntimeError as e:
                # This can happen if the camera is disconnected
                logging.warning(f"Frame reading error in background thread: {e}")
                time.sleep(0.1) # Prevent busy-looping on error

    def get_frames(self):
        """
        Retrieves the most recent set of aligned frames from the camera's
        internal buffer without blocking.

        Returns:
            tuple: A tuple containing (depth_frame, color_image).
                   Returns (None, None) if no frames are available yet.
        """
        with self.frame_lock:
            if self.latest_frame is None:
                return None, None
            
            # Return a copy of the data to prevent race conditions if the
            # caller modifies the image array while the background thread
            # is writing to it.
            depth_frame, color_image = self.latest_frame
            return depth_frame, color_image.copy()

    def load_intrinsics(self, calib_path):
        """
        Loads camera intrinsics from a calibration file.

        Args:
            calib_path (str): The path to the 'calibration_data.npz' file.

        Returns:
            tuple: A tuple containing (intrinsics_matrix, distortion_coeffs).
                   Returns (None, None) if the file is not found.
        """
        if not os.path.exists(calib_path):
            logging.error(f"❌ FATAL: Calibration file not found at '{calib_path}'.")
            logging.error("   -> Please run the calibration script first.")
            return None, None

        with np.load(calib_path) as data:
            mtx, dist = data['mtx'], data['dist']
            logging.info("✅ Camera intrinsics loaded successfully.")
            return mtx, dist

    def get_depth_scale(self):
        """Returns the depth scale of the camera."""
        if self.depth_scale is None:
            logging.warning("Depth scale not available. Is the camera started?")
        return self.depth_scale
