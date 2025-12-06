# Copyright (C) 2025  Nantha Kumar Sunder
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""
Class for representing a 2D lidar scan with an associated pose and free-space maps.

This module defines LocalizedScan, a class that keeps the original scan,
its current pose, a localized scan (scan in world frame), and
free-space maps at two resolutions for multi-resolution scan matching.

Author: Nantha Kumar Sunder
"""

import numpy as np
from numba import njit, prange

from csm_slam.utils.math_utils import transform_scan


class LocalizedScan:
    """
    A lidar scan with pose and precomputed free-space maps.

    Parameters
    ----------
    scan_id : int
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

    def __init__(
        self,
        scan_id: int,
        pose: np.ndarray,
        scan: np.ndarray,
        low_resolution: float = 0.1,
        high_resolution: float = 0.05,
    ):
        self._id = scan_id
        self._pose = pose
        self._original_scan = scan
        self._localized_scan = transform_scan(self._original_scan, self._pose)

        self._low_resolution = low_resolution
        self._high_resolution = high_resolution

        self._low_free_space_map = create_free_space_map(scan, self._low_resolution)
        self._high_free_space_map = create_free_space_map(scan, self._high_resolution)

    #########################################################
    # Properties                                            #
    #########################################################

    @property
    def free_space_maps(self) -> dict:
        """
        Return free-space maps transformed to world frame.

        Returns
        -------
        dict
            Dictionary with keys "low" and "high", each containing:
            - "points": 2xN array of free-space points in world frame
            - "resolution": map resolution in meters

        """
        return {
            'low': {
                'points': transform_scan(self._low_free_space_map, self._pose),
                'resolution': self._low_resolution,
            },
            'high': {
                'points': transform_scan(self._high_free_space_map, self._pose),
                'resolution': self._high_resolution,
            },
        }

    @property
    def scan_id(self) -> int:
        """
        Return the unique identifier for this scan.

        Returns
        -------
        int
            Unique scan identifier.

        """
        return self._id

    @property
    def pose(self) -> np.ndarray:
        """
        Return the current pose of the localized scan.

        Returns
        -------
        numpy.ndarray
            Current pose as [x, y, theta] in meters and radians.

        """
        return self._pose

    #########################################################
    # Public methods                                        #
    #########################################################

    def get_original_scan(self) -> np.ndarray:
        """
        Return the original scan points in the sensor frame.

        Returns
        -------
        numpy.ndarray
            2xN array of scan points in sensor coordinates.

        """
        return self._original_scan

    def get_localized_scan(self) -> np.ndarray:
        """
        Return the scan points transformed into the world frame.

        Returns
        -------
        numpy.ndarray
            2xN array of scan points in world coordinates.

        """
        return self._localized_scan

    def update(self, pose: np.ndarray) -> None:
        """
        Update pose and recompute localized scan.

        Parameters
        ----------
        pose : numpy.ndarray
            New pose [x, y, theta] in meters and radians.

        """
        self._pose = pose
        self._localized_scan = transform_scan(self._original_scan, self._pose)


#########################################################
# Public methods                                       #
#########################################################


@njit
def create_free_space_map(original_scan: np.ndarray, resolution: float) -> np.ndarray:
    """
    Create a free-space point set from a scan using Bresenham's line algorithm.

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

    """
    # calculate total number of points needed for allocation
    # Preallocating the result array makes numba faster by several times.
    total_points = 0
    for i in prange(original_scan.shape[1]):
        end_x = np.trunc(original_scan[0, i] / resolution)
        end_y = np.trunc(original_scan[1, i] / resolution)
        line_points = bresenham((0, 0), (int(end_x), int(end_y)))
        total_points += max(0, line_points.shape[0] - 1)

    # Pre-allocate result array for efficiency
    result = np.zeros((2, total_points), dtype=np.float32)

    point_idx = 0
    for i in prange(original_scan.shape[1]):
        end_x = np.trunc(original_scan[0, i] / resolution)
        end_y = np.trunc(original_scan[1, i] / resolution)

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
def bresenham(start: tuple, end: tuple) -> np.ndarray:
    """
    Bresenham's line algorithm for integer coordinate generation.

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
