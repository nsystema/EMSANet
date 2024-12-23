import time

import numpy as np
import plotly.graph_objects as go


class OccupancyGrid:
    """
    A class to represent and manage an occupancy grid for spatial data analysis.

    Attributes:
        height (float): The height of the grid in units.
        width (float): The width of the grid in units.
        cell_size (float): The size of each cell in the grid.
        point_num_threshold (int): The threshold number of points to determine occupancy.
        num_rows (int): Number of rows in the grid.
        num_cols (int): Number of columns in the grid.
        grid (np.ndarray): 2D array representing the occupancy grid.
        user_row (int): Row index of the user's initial position.
        user_col (int): Column index of the user's initial position.
        point_counts (np.ndarray): 2D array storing the count of points per cell.
    """

    def __init__(self, height, width, cell_size, point_num_threshold):
        """
        Initializes the OccupancyGrid with specified dimensions and parameters.

        Args:
            height (float): The height of the grid in units.
            width (float): The width of the grid in units.
            cell_size (float): The size of each cell in the grid.
            point_num_threshold (int): The threshold number of points to determine occupancy.

        Raises:
            AssertionError: If the number of rows or columns is not odd.
        """
        self.height = height
        self.width = width
        self.cell_size = cell_size
        self.point_num_threshold = point_num_threshold

        # Calculate number of rows and columns
        self.num_rows = int(height / cell_size)
        self.num_cols = int(width / cell_size)

        # Ensure that num_rows and num_cols are odd
        assert self.num_rows % 2 == 1, "Number of rows must be odd."
        assert self.num_cols % 2 == 1, "Number of columns must be odd."

        # Initialize the occupancy grid with zeros
        # -1: Unknown, 0: Free, 1: Occupied
        self.grid = np.full((self.num_rows, self.num_cols), -1)

        # The user's initial position is at the bottom center cell
        self.user_row = self.num_rows - 1  # Bottom row index
        self.user_col = self.num_cols // 2  # Center column index

    def process_pointcloud(self, pointcloud, verbose=True):
        """
        Processes a point cloud to update the occupancy grid based on point density.

        Args:
            pointcloud (iterable): A collection of points, each with at least x and z coordinates.
            verbose (bool): Whether to print the processing time for counting and thresholding.

        Performance:
            - Counts the number of points in each cell efficiently using NumPy operations.
            - Labels cells as occupied or free based on the point count threshold.

        Outputs:
            Prints the time taken for counting points, thresholding, and total processing.
        """
        t0 = time.time()

        # Convert pointcloud to a NumPy array for vectorized operations
        pointcloud_np = np.asarray(pointcloud)

        # Extract x and z coordinates
        x = pointcloud_np[:, 0]
        z = pointcloud_np[:, 2]

        # Compute column and row indices
        col = ((x + self.width / 2) / self.cell_size).astype(int)
        row = self.num_rows - (z / self.cell_size).astype(int) - 1

        # Create a mask for valid indices
        valid_mask = (
            (row >= 0) & (row < self.num_rows) & (col >= 0) & (col < self.num_cols)
        )

        # Apply the mask to filter valid rows and columns
        row_valid = row[valid_mask]
        col_valid = col[valid_mask]

        # Compute flat indices for efficient counting
        flat_indices = row_valid * self.num_cols + col_valid

        # Use np.bincount to count the number of points per cell
        point_counts_flat = np.bincount(
            flat_indices, minlength=self.num_rows * self.num_cols
        )
        point_counts = point_counts_flat.reshape((self.num_rows, self.num_cols))

        # Store the point counts
        self.point_counts = point_counts

        t1 = time.time()
        if verbose:
            print(f"Counting points took {(t1 - t0) * 1000:.4f} ms")

        # Label the cells based on the counts and threshold
        self.grid = np.where(point_counts >= self.point_num_threshold, 1, 0)

        if verbose:
            print("Thresholding took {} ms".format((time.time() - t1) * 1000))
            print("Processing pointcloud took {} ms".format((time.time() - t0) * 1000))

    def get_grid(self):
        """
        Retrieves the current occupancy grid.

        Returns:
            np.ndarray: The 2D occupancy grid array.
        """
        return self.grid

    def get_grid_as_rgb(self):
        """
        Converts the occupancy grid to an RGB image using a predefined color map.

        Color Mapping:
            - Unknown (-1): Gray [128, 128, 128]
            - Free (0): Green [0, 255, 0]
            - Occupied (1): Red [255, 0, 0]

        Returns:
            np.ndarray: A 3D array representing the RGB image of the occupancy grid.
        """
        # Define the color map for different cell states
        color_map = np.array(
            [
            [169, 169, 169],  # Unknown (Dark Gray)
            [91, 227, 169],      # Free (Green)
            [227, 101, 91]      # Occupied (Orange Red)
            ],
            dtype=np.uint8,
        )
        # Map the grid values to RGB colors
        grid_rgb = color_map[self.grid + 1]
        return grid_rgb
    
    
