"""Represent a 2D lidar scan with an associated pose and free-space maps.

This module defines `LocalizedScan`, a container that keeps the original scan,
its current pose, a lazily updated localized scan (scan in world frame), and
free-space maps at two resolutions for multi-resolution matching.

The module also provides optimized implementations of Bresenham's line algorithm
and free-space map creation using Numba for performance-critical operations.

Author: Nantha Kumar Sunder
"""

import numpy as np
from math_utils import transform_scan
from numba import njit, prange


class LocalizedScan:
    """A lidar scan with pose and precomputed free-space maps.

    This class represents a single lidar scan with its associated pose
    and provides efficient access to the scan data in both sensor frame
    and world frame coordinates. It also maintains free-space maps at
    multiple resolutions for use in scan matching algorithms.

    Attributes
    ----------
    pose : numpy.ndarray
        Current pose [x, y, theta] in meters and radians.
    id : int
        Unique identifier for this scan.
    free_space_maps : dict
        Dictionary containing low and high resolution free-space maps
        transformed to world frame.

    Parameters
    ----------
    id : int
        Unique identifier for this scan.
    pose : numpy.ndarray
        Initial pose [x, y, theta] in meters and radians.
    scan : numpy.ndarray
        2xN array of scan points in sensor frame.
    low_resolution : float, optional
        Resolution for coarse free-space map in meters (default: 0.1).
    high_resolution : float, optional
        Resolution for fine free-space map in meters (default: 0.05).

    Notes
    -----
    The free-space maps are created using Bresenham's line algorithm
    to efficiently compute all points along rays from the sensor origin
    to each scan point. This is used for scan matching algorithms
    that need to know which areas are known to be free.
    """

    def __init__(
        self,
        id: int,
        pose: np.ndarray,
        scan: np.ndarray,
        low_resolution: float = 0.1,
        high_resolution: float = 0.05,
    ):
        """Initialize a localized scan with pose and free-space maps.

        Creates a new LocalizedScan instance with the given pose and
        scan data. Automatically computes the localized scan (world frame)
        and creates free-space maps at both resolutions.

        Parameters
        ----------
        id : int
            Unique identifier for this scan.
        pose : numpy.ndarray
            Initial pose [x, y, theta] in meters and radians.
        scan : numpy.ndarray
            2xN array of scan points in sensor frame.
        low_resolution : float, optional
            Resolution for coarse free-space map in meters (default: 0.1).
        high_resolution : float, optional
            Resolution for fine free-space map in meters (default: 0.05).
        """
        # Store scan metadata and pose
        self._id = id
        self._pose = pose
        self._original_scan = scan
        # Transform scan to world frame using current pose
        self._localized_scan = transform_scan(self._original_scan, self._pose)
        # Store resolution parameters for free-space maps
        self._low_resolution = low_resolution
        self._high_resolution = high_resolution
        # Create free-space maps at both resolutions
        self._low_free_space_map = create_free_space_map(scan, self._low_resolution)
        self._high_free_space_map = create_free_space_map(scan, self._high_resolution)

    @property
    def pose(self):
        """Return the current pose of the localized scan.

        Returns
        -------
        numpy.ndarray
            Current pose as [x, y, theta] in meters and radians.
        """
        return self._pose

    @property
    def id(self):
        """Return the unique identifier for this scan.

        Returns
        -------
        int
            Unique scan identifier.
        """
        return self._id

    @property
    def free_space_maps(self):
        """Return free-space maps transformed to world frame.

        Provides access to both low and high resolution free-space maps
        transformed into the world coordinate frame. These maps contain
        all points along rays from the sensor origin to each scan point,
        excluding the actual obstacle points.

        Returns
        -------
        dict
            Dictionary with keys "low" and "high", each containing:
            - "points": 2xN array of free-space points in world frame
            - "resolution": map resolution in meters

        Notes
        -----
        The free-space maps are used by scan matching algorithms to
        determine which areas are known to be free of obstacles, helping
        to improve matching accuracy and robustness.
        """
        return {
            "low": {
                "points": transform_scan(self._low_free_space_map, self._pose),
                "resolution": self._low_resolution,
            },
            "high": {
                "points": transform_scan(self._high_free_space_map, self._pose),
                "resolution": self._high_resolution,
            },
        }

    def get_localized_scan(self):
        """Return the scan points transformed into the world frame.

        Returns
        -------
        numpy.ndarray
            2xN array of scan points in world coordinates.
        """
        return self._localized_scan

    def get_original_scan(self):
        """Return the original scan points in the sensor frame.

        Returns
        -------
        numpy.ndarray
            2xN array of scan points in sensor coordinates.
        """
        return self._original_scan

    def update(self, pose: np.ndarray):
        """Update pose and recompute localized scan.

        Updates the scan's pose and recomputes the localized scan
        (world frame coordinates) using the new pose. This is typically
        called after graph optimization when poses are corrected.

        Parameters
        ----------
        pose : numpy.ndarray
            New pose [x, y, theta] in meters and radians.
        """
        # Update pose and recompute world frame coordinates
        self._pose = pose
        self._localized_scan = transform_scan(self._original_scan, self._pose)

    def get_id(self):
        """Return the localized scan identifier.

        Returns
        -------
        int
            Unique identifier for this scan.

        Note
        ----
        This method is redundant with the `id` property and is provided
        for backward compatibility.
        """
        return self._id


@njit
def create_free_space_map(original_scan, resolution):
    """Create a free-space point set from a scan using Bresenham rays.

    This function creates a set of free-space points by tracing rays
    from the sensor origin to each scan point using Bresenham's line
    algorithm. The resulting points represent areas known to be free
    of obstacles, which is useful for scan matching.

    Parameters
    ----------
    original_scan : numpy.ndarray
        2xN array of scan points in the sensor frame.
    resolution : float
        Step size in meters for sampling along each ray.
        Determines the density of free-space points.

    Returns
    -------
    numpy.ndarray
        2xM array of free-space points in sensor coordinates.

    Notes
    -----
    The algorithm excludes the last 2 points of each ray to avoid
    including the obstacle points themselves in the free-space map.
    This function is optimized with Numba for performance.
    """
    # First pass: calculate total number of points needed for allocation
    total_points = 0
    for i in prange(original_scan.shape[1]):
        # Convert scan point to grid coordinates
        end_x = np.rint(original_scan[0, i] / resolution)
        end_y = np.rint(original_scan[1, i] / resolution)
        # Get Bresenham line points from origin to scan point
        line_points = bresenham((0, 0), (int(end_x), int(end_y)))
        # Exclude last 2 points to avoid including obstacle points
        total_points += max(0, line_points.shape[0] - 1)

    # Pre-allocate result array for efficiency
    result = np.zeros((2, total_points), dtype=np.float32)

    # Second pass: fill the result array with free-space points
    point_idx = 0
    for i in prange(original_scan.shape[1]):
        # Convert scan point to grid coordinates
        end_x = np.rint(original_scan[0, i] / resolution)
        end_y = np.rint(original_scan[1, i] / resolution)
        # Get Bresenham line points from origin to scan point
        free_space_coords = bresenham((0, 0), (int(end_x), int(end_y)))

        # Exclude last 2 points to avoid including obstacle points
        total_points = max(0, free_space_coords.shape[0] - 1)

        # Convert grid coordinates back to metric coordinates
        for j in range(total_points):
            result[0, point_idx] = free_space_coords[j, 0] * resolution
            result[1, point_idx] = free_space_coords[j, 1] * resolution
            point_idx += 1

    return result


@njit
def bresenham(start: tuple, end: tuple):
    """Bresenham's line algorithm for integer coordinate generation.

    Implements Bresenham's line algorithm to generate all integer
    coordinates along a line between two points. This is used for
    creating free-space maps by tracing rays from the sensor origin
    to each scan point.

    Inspired by Atsushi Sakai's implementation.
    https://github.com/AtsushiSakai/PythonRobotics

    Parameters
    ----------
    start : tuple
        Starting point (x1, y1) as integer coordinates.
    end : tuple
        Ending point (x2, y2) as integer coordinates.

    Returns
    -------
    numpy.ndarray
        Array of shape (N, 2) with integer pixel coordinates
        along the line from start to end.

    Notes
    -----
    The algorithm handles all octants and ensures that lines are
    drawn correctly regardless of direction. This function is
    optimized with Numba for performance.
    """
    # Extract coordinates and calculate differences
    x1, y1 = start
    x2, y2 = end
    dx = x2 - x1
    dy = y2 - y1
    # Determine if line is steep (slope > 1) for octant handling
    is_steep = abs(dy) > abs(dx)
    if is_steep:  # Rotate line to handle steep slopes
        x1, y1 = y1, x1
        x2, y2 = y2, x2
    # Ensure line goes from left to right for algorithm consistency
    swapped = False
    if x1 > x2:
        x1, x2 = x2, x1
        y1, y2 = y2, y1
        swapped = True
    # Recalculate differences after potential swaps
    dx = x2 - x1
    dy = y2 - y1
    error = int(dx / 2.0)  # Initialize error term
    y_step = 1 if y1 < y2 else -1  # Determine y direction

    # Calculate maximum number of points for pre-allocation
    max_points = abs(x2 - x1) + 1

    # Pre-allocate coordinate arrays for efficiency
    x_coords = np.zeros(max_points, dtype=np.int64)
    y_coords = np.zeros(max_points, dtype=np.int64)

    # Generate points along the line using Bresenham's algorithm
    y = y1
    point_count = 0
    for x in range(x1, x2 + 1):
        # Handle coordinate system based on line steepness
        if is_steep:
            x_coords[point_count] = y
            y_coords[point_count] = x
        else:
            x_coords[point_count] = x
            y_coords[point_count] = y
        point_count += 1
        # Update error term and y coordinate
        error -= abs(dy)
        if error < 0:
            y += y_step
            error += dx

    # Create result array with actual number of points
    result = np.zeros((point_count, 2), dtype=np.int64)
    result[:, 0] = x_coords[:point_count]
    result[:, 1] = y_coords[:point_count]

    # Reverse array if coordinates were swapped earlier
    if swapped:
        result = result[::-1]

    return result
