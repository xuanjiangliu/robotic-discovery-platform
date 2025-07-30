# scripts/01_calibrate_camera.py
#
# Description:
# This script performs intrinsic camera calibration using a checkerboard pattern.
# It uses the Camera class from the refactored 'pkg' library.

import numpy as np
import cv2
import os
import sys
import logging

# --- Path Setup ---
# This is the standard way to make a script runnable from anywhere in the project
# and ensure it can find the 'pkg' directory.
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from pkg.camera import Camera

# --- Configuration ---
CHECKERBOARD = (9, 7) 
SQUARE_SIZE_M = 0.027 
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Script Initialization ---
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
objp = objp * SQUARE_SIZE_M

objpoints = []
imgpoints = []

# --- Main Calibration Logic ---
def main():
    """Main function to run the camera calibration process."""
    data_dir = os.path.join("ml", "data")
    os.makedirs(data_dir, exist_ok=True)
    output_path = os.path.join(data_dir, 'calibration_data.npz')

    cam = Camera(width=640, height=480, fps=30)
    if not cam.start():
        return

    logging.info("Camera connected. Starting capture loop...")
    logging.info("\n--- INSTRUCTIONS ---\n"
                 "1. Show the checkerboard to the camera from various angles.\n"
                 "2. Press 'c' to capture a good view (at least 10 are recommended).\n"
                 "3. Press 'q' to perform calibration and quit.\n"
                 "--------------------")

    try:
        while True:
            _, color_image = cam.get_frames()
            if color_image is None:
                continue

            gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
            cv2.imshow('Camera Feed', color_image)
            key = cv2.waitKey(1)

            if key == ord('c'):
                ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, None)
                if ret:
                    objpoints.append(objp)
                    corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                    imgpoints.append(corners2)
                    
                    cv2.drawChessboardCorners(color_image, CHECKERBOARD, corners2, ret)
                    cv2.imshow('Camera Feed', color_image)
                    cv2.waitKey(500)
                    logging.info(f"✅ Frame captured! Total captures: {len(objpoints)}")
                else:
                    logging.warning("Checkerboard not found. Please try a different angle.")

            elif key == ord('q'):
                if len(objpoints) < 5:
                    logging.error("\nNot enough captures to perform calibration. Need at least 5.")
                    break
                
                logging.info(f"\nQuitting. Performing calibration on {len(objpoints)} images...")
                cv2.destroyAllWindows()
                
                ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

                if ret:
                    logging.info("\n✅ Calibration successful!")
                    np.savez(output_path, mtx=mtx, dist=dist, rvecs=rvecs, tvecs=tvecs)
                    logging.info(f'Calibration data saved to "{output_path}"')

                    mean_error = 0
                    for i in range(len(objpoints)):
                        imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
                        error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
                        mean_error += error
                    logging.info(f"Total re-projection error: {mean_error / len(objpoints):.4f}")
                else:
                    logging.error("\n❌ Calibration failed.")
                break

    finally:
        logging.info("Shutting down.")
        cam.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
