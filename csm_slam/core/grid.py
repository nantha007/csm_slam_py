"""
Occupancy grid construction and multi-resolution grid utilities.

This module provides helper routines to accumulate hit/no-hit evidence from
localized scans into occupancy grids and a `MultiResolutionGrid` class
maintains coarse and fine grids for multi-resolution scan matching.

Author: Nantha Kumar Sunder
"""

import numpy as np
from numba import njit
from typing import List
from enum import Enum

from csm_slam.core.localized_scan import LocalizedScan


class CellValue(Enum):
    OCCUPIED = 0
    FREE = 255
    UNKNOWN = 128


class Grid:
    """
    This class is used to store a grid, origin, and resolution.

    Parameters
    ----------
    grid : np.ndarray
        The grid data.
    origin : np.ndarray
        The origin of the grid in world frame.
    resolution : float
        The resolution of the grid.

    """

    def __init__(self, grid: np.ndarray, origin: np.ndarray, resolution: float):
        self.grid = grid
        self.origin = origin
        self.resolution = resolution


class MultiResolutionGrid:
    """
    Class for maintaining coarse and fine occupancy grids.

    This class maintains two occupancy grids at different
    resolutions from a list of localized scans.

    Attributes
    ----------
    coarse_grid : Grid
        Coarse resolution occupancy grid.
    fine_grid : Grid
        Fine resolution occupancy grid.

    Parameters
    ----------
    low_resolution : float
        Resolution for the coarse occupancy grid in meters.
    high_resolution : float
        Resolution for the fine occupancy grid in meters.
    localized_scans : List[LocalizedScan]
        List of LocalizedScan objects providing free/occupied points.

    """

    def __init__(
        self,
        low_resolution: float,
        high_resolution: float,
        localized_scans: List[LocalizedScan],
    ):
        """
        Initialize and build coarse and fine occupancy grids.

        Parameters
        ----------
        low_resolution : float
            Resolution for the coarse occupancy grid in meters.
        high_resolution : float
            Resolution for the fine occupancy grid in meters.
        localized_scans : List[LocalizedScan]
            List of LocalizedScan objects providing free/occupied points.

        """
        self._low_resolution = low_resolution
        self._high_resolution = high_resolution
        self._localized_scans = localized_scans

        self.coarse_grid = create_occupancy_grid(localized_scans, low_resolution)
        self.fine_grid = create_occupancy_grid(localized_scans, high_resolution)


#########################################################
# Public methods                                        #
#########################################################


def create_occupancy_grid(
    localized_scans: List[LocalizedScan],
    resolution: float,
) -> Grid:
    """
    Create an occupancy grid from hit/no-hit evidence across scans.

    This function processes a list of localized scans to create an occupancy
    grid by accumulating hit (occupied) and no-hit (free space) evidence.

    Parameters
    ----------
    localized_scans : List[LocalizedScan]
        List of LocalizedScan objects contributing free and occupied points.
    resolution : float
        Grid resolution in meters.

    Returns
    -------
    Grid
        Grid object

    """
    # Collect all points from all scans to determine grid bounds
    all_points = []
    for localized_scan in localized_scans:
        # Get appropriate resolution free-space map
        free_maps = localized_scan.free_space_maps
        if free_maps["low"]["resolution"] == resolution:
            free_points = free_maps["low"]["points"]
        else:
            free_points = free_maps["high"]["points"]
        occupied_points = localized_scan.get_localized_scan()

        # Add points to collection if they exist
        if free_points.shape[1] > 0:
            all_points.append(free_points)
        if occupied_points.shape[1] > 0:
            all_points.append(occupied_points)

    # Calculate grid bounds from all collected points
    combined_all = np.hstack(all_points)
    min_x, min_y = np.min(combined_all, axis=1)
    max_x, max_y = np.max(combined_all, axis=1)

    # Add margin around the grid bounds
    margin = 2
    min_x -= margin
    min_y -= margin
    max_x += margin
    max_y += margin

    # Calculate grid dimensions based on bounds and resolution
    width = int(np.floor((max_x - min_x) / resolution))
    height = int(np.floor((max_y - min_y) / resolution))

    # Initialize evidence grids for hit/no-hit accumulation
    hits_grid = np.zeros((height, width), dtype=np.int32)
    no_hits_grid = np.zeros((height, width), dtype=np.int32)

    # Accumulate evidence from all scans
    for localized_scan in localized_scans:
        # Add hits (occupied points) to evidence grid
        occupied_points = localized_scan.get_localized_scan()
        _accumulate_hits(hits_grid, occupied_points, min_x, min_y, resolution)

        # Add no-hits (free space points)
        free_maps = localized_scan.free_space_maps
        if free_maps["low"]["resolution"] == resolution:
            free_points = free_maps["low"]["points"]
        else:
            free_points = free_maps["high"]["points"]
        _accumulate_no_hits(no_hits_grid, free_points, min_x, min_y, resolution)

    occupancy_grid = _count_to_occupancy_grid(hits_grid, no_hits_grid)
    return Grid(occupancy_grid, np.array([min_x, min_y]), resolution)


#########################################################
# Private methods                                       #
#########################################################


@njit
def _accumulate_hits(
    hits_grid: np.ndarray,
    points: np.ndarray,
    min_x: float,
    min_y: float,
    resolution: float,
):
    """
    Accumulate hits (occupied evidence) in the grid.

    Parameters
    ----------
    hits_grid : np.ndarray
        2D grid to accumulate hits
    points : np.ndarray
        2xN array of occupied points
    min_x, min_y : float
        Grid origin coordinates
    resolution : float
        Grid resolution

    """
    height, width = hits_grid.shape

    for i in range(points.shape[1]):
        x, y = points[0, i], points[1, i]

        # Convert to grid coordinates (match scan_matcher rasterization)
        grid_x = int(np.rint((x - min_x) / resolution))
        grid_y_cart = int(np.rint((y - min_y) / resolution))
        grid_y = height - 1 - grid_y_cart

        # Check bounds and increment hit count
        if 0 <= grid_x < width and 0 <= grid_y < height:
            hits_grid[grid_y, grid_x] += 1


@njit
def _accumulate_no_hits(
    no_hits_grid: np.ndarray,
    points: np.ndarray,
    min_x: float,
    min_y: float,
    resolution: float,
):
    """
    Accumulate no-hits (free space evidence) in the grid.

    Parameters
    ----------
    no_hits_grid : np.ndarray
        2D grid to accumulate no-hits
    points : np.ndarray
        2xN array of free space points
    min_x, min_y : float
        Grid origin coordinates
    resolution : float
        Grid resolution

    """
    height, width = no_hits_grid.shape

    for i in range(points.shape[1]):
        x, y = points[0, i], points[1, i]

        # Convert to grid coordinates (match scan_matcher rasterization)
        grid_x = int(np.rint((x - min_x) / resolution))
        grid_y_cart = int(np.rint((y - min_y) / resolution))
        grid_y = height - 1 - grid_y_cart

        # Check bounds and increment no-hit count
        if 0 <= grid_x < width and 0 <= grid_y < height:
            no_hits_grid[grid_y, grid_x] += 1


def _count_to_occupancy_grid(
    hits_grid: np.ndarray,
    no_hits_grid: np.ndarray,
) -> np.ndarray:
    """
    Convert hit and no-hit counts to an occupancy grid.

    Parameters
    ----------
    hits_grid : np.ndarray
        Grid with hit counts.
    no_hits_grid : np.ndarray
        Grid with no-hit counts.

    Returns
    -------
    np.ndarray
        Grid with values indicating free space, occupied space, and unknown areas.

    """
    height, width = hits_grid.shape
    occupancy_grid = np.full((height, width), CellValue.UNKNOWN.value, dtype=np.uint8)

    count_grid = hits_grid - no_hits_grid
    occupancy_grid[count_grid >= 1] = CellValue.OCCUPIED.value
    occupancy_grid[count_grid <= -1] = CellValue.FREE.value

    return occupancy_grid
