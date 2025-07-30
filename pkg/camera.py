# pkg/camera.py
#
# Description:
# This module provides a high-level wrapper class for interacting with
# an Intel RealSense camera. It handles initialization, frame grabbing,
# and providing calibration data.

import pyrealsense2 as rs
import numpy as np
import os
import logging
from collections import deque

class Camera:
    """A wrapper class for the Intel RealSense D4XX camera series."""

    def __init__(self, width=640, height=480, fps=30, buffer_size=5):
        """
        Initializes the Camera object.

        Args:
            width (int): The desired width of the camera stream.
            height (int): The desired height of the camera stream.
            fps (int): The desired framerate of the camera stream.
            buffer_size (int): The number of frames to keep in the buffer.
        """
        self.width = width
        self.height = height
        self.fps = fps
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.profile = None
        self.align = None
        self.depth_scale = None
        self.frame_buffer = deque(maxlen=buffer_size)

        # Configure the streams
        self.config.enable_stream(rs.stream.depth, self.width, self.height, rs.format.z16, self.fps)
        self.config.enable_stream(rs.stream.color, self.width, self.height, rs.format.bgr8, self.fps)

    def start(self):
        """Starts the RealSense pipeline and sets up alignment."""
        logging.info("Starting RealSense pipeline...")
        try:
            self.profile = self.pipeline.start(self.config)
            self.align = rs.align(rs.stream.color)
            self.depth_scale = self.profile.get_device().first_depth_sensor().get_depth_scale()
            return True
        except Exception as e:
            logging.error(f"❌ ERROR: Failed to start RealSense pipeline: {e}")
            return False

    def stop(self):
        """Stops the RealSense pipeline."""
        self.pipeline.stop()
        logging.info("✅ RealSense pipeline stopped.")

    def get_frames(self, get_most_recent=False):
        """
        Waits for and retrieves a new, aligned set of frames from the camera.

        Args:
            get_most_recent (bool): If True, returns the latest frame from the
                                    buffer without waiting. Otherwise, waits
                                    for a new frame.

        Returns:
            tuple: A tuple containing (depth_frame, color_image).
                   Returns (None, None) if frames are not available.
        """
        try:
            if get_most_recent and self.frame_buffer:
                return self.frame_buffer[-1]

            frames = self.pipeline.wait_for_frames()
            aligned_frames = self.align.process(frames)

            depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()

            if not depth_frame or not color_frame:
                return None, None

            color_image = np.asanyarray(color_frame.get_data())
            
            latest_frames = (depth_frame, color_image)
            self.frame_buffer.append(latest_frames)
            
            return latest_frames

        except Exception as e:
            logging.warning(f"Could not get frames: {e}")
            return None, None

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
