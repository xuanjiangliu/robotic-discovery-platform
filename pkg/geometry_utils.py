# pkg/geometry_utils.py
#
# Description:
# This module contains core, high-performance functions for geometric analysis
# of 3D point clouds.

import numpy as np
from scipy.interpolate import splprep, splev
from dataclasses import dataclass, field
from typing import List

# A simple dataclass to represent a 3D point.
@dataclass
class Point:
    """A simple data class for a 3D point."""
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0

@dataclass
class CurvatureResult:
    """A data class to hold the results of a curvature analysis."""
    mean_curvature: float = 0.0
    max_curvature: float = 0.0
    # he result will now be a list of Point objects
    spline_points: List[Point] = field(default_factory=list)

def compute_curvature_profile(mask: np.ndarray, depth_image: np.ndarray,
                              intrinsics: np.ndarray, depth_scale: float) -> CurvatureResult:
    """
    Computes the curvature profile of an object identified by a mask.

    This is a high-level function that orchestrates the entire geometry pipeline:
    1. Generates a point cloud from the masked depth image.
    2. Finds the top edge of the point cloud.
    3. Fits a spline to the edge.
    4. Calculates the curvature of the spline.

    Args:
        mask (np.ndarray): The 2D segmentation mask (H, W).
        depth_image (np.ndarray): The 2D depth image (H, W).
        intrinsics (np.ndarray): The 3x3 camera intrinsic matrix.
        depth_scale (float): The scale factor to convert depth units to meters.

    Returns:
        CurvatureResult: A dataclass object containing the analysis results.
    """
    pcd = _get_pcd_from_mask(mask, depth_image, intrinsics, depth_scale)

    if pcd.shape[0] < 100:
        return CurvatureResult()

    edge_points = _find_point_cloud_edge(pcd)

    if edge_points.shape[0] < 20:
        return CurvatureResult()

    try:
        # Sort points along the primary axis (x-axis) for a stable spline fit
        sorted_points = edge_points[np.argsort(edge_points[:, 0])]

        # This creates a smoother spline, which will make the calculated
        # curvature values more stable and less jittery between frames.
        tck, _ = splprep([sorted_points[:, 0], sorted_points[:, 1], sorted_points[:, 2]], s=0.1, k=3)
        
        mean_k, max_k = _calculate_spline_curvature(tck)
        
        # Generate points along the fitted spline for visualization
        u_fine = np.linspace(0, 1, 100)
        spline_points_np = np.array(splev(u_fine, tck)).T
        
        # Convert the numpy array into a list of Point dataclass instances.
        spline_points_list = [Point(x=p[0], y=p[1], z=p[2]) for p in spline_points_np]
        
        return CurvatureResult(
            mean_curvature=mean_k,
            max_curvature=max_k,
            spline_points=spline_points_list
        )

    except (TypeError, ValueError):
        # Catches errors from splprep if there aren't enough points
        return CurvatureResult()

# --- Helper Functions (intended for internal use within this module) ---

def _get_pcd_from_mask(mask: np.ndarray, depth_image: np.ndarray,
                       intrinsics: np.ndarray, depth_scale: float) -> np.ndarray:
    """Efficiently generates a 3D point cloud from a depth frame filtered by a mask."""
    v, u = np.where(mask > 0)
    z = depth_image[v, u] * depth_scale
    
    # Filter out invalid depth points (z <= 0)
    valid_depth_mask = z > 0
    v, u, z = v[valid_depth_mask], u[valid_depth_mask], z[valid_depth_mask]
    
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]
    
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy
    
    return np.vstack((x, y, z)).transpose()

def _find_point_cloud_edge(pcd: np.ndarray, num_bins: int = 50, top_k_percent: float = 0.05) -> np.ndarray:
    """Finds the approximate top edge of a point cloud oriented along the x-axis."""
    if pcd.shape[0] < num_bins:
        return np.array([])

    x_min, x_max = pcd[:, 0].min(), pcd[:, 0].max()
    bin_width = (x_max - x_min) / num_bins
    
    if bin_width <= 0:
        return np.array([])

    bin_indices = np.floor((pcd[:, 0] - x_min) / bin_width).astype(int)
    # Clamp indices to handle edge cases
    bin_indices = np.clip(bin_indices, 0, num_bins - 1)

    edge_points = []
    for i in range(num_bins):
        points_in_bin = pcd[bin_indices == i]
        if points_in_bin.shape[0] > 0:
            k = max(1, int(points_in_bin.shape[0] * top_k_percent))
            top_indices = np.argpartition(points_in_bin[:, 1], -k)[-k:]
            edge_points.append(points_in_bin[top_indices])

    return np.vstack(edge_points) if edge_points else np.array([])

def _calculate_spline_curvature(tck: tuple, num_points: int = 100) -> tuple[float, float]:
    """Calculates the mean and max curvature of a 3D parametric spline."""
    u_fine = np.linspace(0, 1, num_points)
    
    r_prime = np.array(splev(u_fine, tck, der=1))
    r_double_prime = np.array(splev(u_fine, tck, der=2))

    cross_product = np.cross(r_prime.T, r_double_prime.T)
    norm_cross_product = np.linalg.norm(cross_product, axis=1)
    norm_r_prime = np.linalg.norm(r_prime, axis=0)
    
    non_zero_mask = norm_r_prime > 1e-6
    curvature = np.zeros_like(norm_r_prime)
    
    if np.any(non_zero_mask):
        curvature[non_zero_mask] = norm_cross_product[non_zero_mask] / (norm_r_prime[non_zero_mask]**3)
        return np.mean(curvature[non_zero_mask]), np.max(curvature[non_zero_mask])
    
    return 0.0, 0.0
