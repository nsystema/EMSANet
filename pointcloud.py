import pyrealsense2 as rs
import numpy as np


def create_point_cloud_rs(depth_frame, color_frame):
    """
    # Create point cloud object and map to color frame to align depth data with color data

    Args:
        depth_frame: The depth frame from the RealSense camera.
        color_frame: The color frame from the RealSense camera.

    Returns:
        A tuple containing:
        - points_3d: A numpy array of 3D points.
        - colors: A numpy array of RGB color values corresponding to the 3D points.
    """

    # Create point cloud object and map to color frame
    pc = rs.pointcloud()
    pc.map_to(color_frame)
    points = pc.calculate(depth_frame)

    # Retrieve vertices and convert to numpy array
    vtx = np.asanyarray(points.get_vertices()).view(np.float32).reshape(-1, 3)
    # Flip the y-axis to match the coordinate system of the color image
    vtx[:, 1] *= -1
    points_3d = vtx

    # Retrieve texture coordinates
    tex_coords = (
        np.asanyarray(points.get_texture_coordinates()).view(np.float32).reshape(-1, 2)
    )

    # Get color data from color frame
    color_image = np.asanyarray(color_frame.get_data())
    h, w = color_image.shape[:2]

    # Map texture coordinates to color values
    u = (tex_coords[:, 0] * w).astype(np.int32)
    v = (tex_coords[:, 1] * h).astype(np.int32)
    u = np.clip(u, 0, w - 1)
    v = np.clip(v, 0, h - 1)
    colors = color_image[v, u]

    return points_3d, colors


def filter_invalid_points(points_3d, color_3d, object_mask, z_max):
    """
    Filters out invalid 3D points based on depth values and an object mask.

    Args:
        points_3d: A numpy array of 3D points.
        color_3d: A numpy array of RGB color values corresponding to the 3D points.
        object_mask: A mask indicating the object of interest.
        z_max: The maximum depth value to consider valid.

    Returns:
        A tuple containing:
        - valid_points: A numpy array of valid 3D points.
        - valid_color: A numpy array of RGB color values corresponding to the valid 3D points.
        - valid_object_mask: A mask indicating the valid points of the object of interest.
    """
    # Create a boolean mask for valid depth values
    mask = (points_3d[:, 2] > 0) & (points_3d[:, 2] < z_max)

    # Apply the mask to filter points, colors, and object_mask
    valid_points, valid_color, valid_object_mask = (
        points_3d[mask],
        color_3d[mask],
        object_mask[mask],
    )

    print(f"Total valid points: {valid_points.shape[0]}")

    return valid_points, valid_color, valid_object_mask
