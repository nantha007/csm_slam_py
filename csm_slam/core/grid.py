"""Occupancy grid construction and multi-resolution grid utilities.

This module provides helper routines to accumulate hit/no-hit evidence from
localized scans into occupancy grids and a `MultiResolutionGrid` container that
maintains coarse and fine grids for multi-resolution scan matching.

The module includes optimized implementations using Numba for performance-critical
operations like evidence accumulation and occupancy grid conversion.

Author: Nantha Kumar Sunder
"""

import numpy as np
from numba import njit
from typing import Tuple, List
from localized_scan import LocalizedScan


class MultiResolutionGrid:
    """Container for coarse and fine occupancy grids.

    This class generates and maintains two occupancy grids at different
    resolutions from a list of localized scans. The coarse grid is used
    for initial broad search, while the fine grid provides refined
    alignment capabilities.

    Attributes
    ----------
    coarse_grid : MinGrid
        Coarse resolution occupancy grid.
    fine_grid : MinGrid
        Fine resolution occupancy grid.

    Parameters
    ----------
    low_resolution : float
        Resolution for the coarse occupancy grid in meters.
    high_resolution : float
        Resolution for the fine occupancy grid in meters.
    localized_scans : List[LocalizedScan]
        List of LocalizedScan objects providing free/occupied points.

    Notes
    -----
    Given a list of localized scans and two target resolutions, this class
    generates two occupancy grids: a coarse grid at `low_resolution` and a
    fine grid at `high_resolution`. These are useful for multi-stage scan
    matching where a broad search is followed by a refined alignment.
    """

    def __init__(
        self,
        low_resolution: float,
        high_resolution: float,
        localized_scans: List[LocalizedScan],
    ):
        """Initialize and build coarse and fine occupancy grids.

        Parameters
        ----------
        low_resolution : float
            Resolution for the coarse occupancy grid in meters.
        high_resolution : float
            Resolution for the fine occupancy grid in meters.
        localized_scans : List[LocalizedScan]
            List of LocalizedScan objects providing free/occupied points.
        """

        class MinGrid:
            """Lightweight structure holding a grid, origin, and resolution.

            This class is used to store a grid, origin, and resolution.

            Attributes
            ----------
            grid : np.ndarray
                The grid data.
            origin : Tuple[float, float]
                The origin of the grid.
            resolution : float
                The resolution of the grid.

            Parameters
            ----------
            grid : np.ndarray
                The grid data.
            origin : Tuple[float, float]
                The origin of the grid.
            resolution : float
                The resolution of the grid.
            """

            def __init__(
                self, grid: np.ndarray, origin: Tuple[float, float], resolution: float
            ):
                self.grid = grid
                self.origin = origin
                self.resolution = resolution

        # Store resolution parameters
        self._low_resolution = low_resolution
        self._high_resolution = high_resolution
        self._localized_scans = localized_scans
        # Create occupancy grids at both resolutions
        lr_grid, lr_origin = create_occupancy_grid(
            localized_scans, low_resolution, 1, 1
        )
        hr_grid, hr_origin = create_occupancy_grid(
            localized_scans, high_resolution, 1, 1
        )
        # Store grids in MinGrid containers
        self.coarse_grid = MinGrid(lr_grid, lr_origin, low_resolution)
        self.fine_grid = MinGrid(hr_grid, hr_origin, high_resolution)


def create_occupancy_grid(
    localized_scans: List[LocalizedScan],
    resolution: float,
    min_hits: int = 3,
    min_no_hits: int = 3,
    method: str = "ratio_based",
) -> Tuple[np.ndarray, Tuple[float, float]]:
    """Create an occupancy grid from hit/no-hit evidence across scans.

    This function processes a list of localized scans to create an occupancy
    grid by accumulating hit (occupied) and no-hit (free space) evidence.
    The resulting grid uses standard occupancy grid values: 0=occupied,
    125=unknown, 255=free.

    Parameters
    ----------
    localized_scans : List[LocalizedScan]
        List of LocalizedScan objects contributing free and occupied points.
    resolution : float
        Grid resolution in meters.
    min_hits : int, optional
        Minimum number of hits required to classify a cell as occupied
        (default: 3).
    min_no_hits : int, optional
        Minimum number of no-hits required to classify a cell as free
        (default: 3).
    method : str, optional
        Evidence-to-occupancy conversion method. Options:
        - "logodds": Log-odds based conversion
        - "standard": Simple threshold-based conversion
        - "ratio_based": Ratio-based conversion considering hit ratios
        (default: "ratio_based").

    Returns
    -------
    Tuple[numpy.ndarray, Tuple[float, float]]
        Tuple containing:
        - occupancy_grid: 2D occupancy grid where 0=occupied, 125=unknown, 255=free
        - origin: Origin coordinates (min_x, min_y) of the grid in world frame

    Raises
    ------
    ValueError
        If an invalid method is specified.
    """
    # Handle empty scan list
    if not localized_scans:
        return np.array([[125]], dtype=np.uint8), (0.0, 0.0)

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

    # Handle case where no points are available
    if not all_points:
        return np.array([[125]], dtype=np.uint8), (0.0, 0.0)

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
        if occupied_points.shape[1] > 0:
            _accumulate_hits(hits_grid, occupied_points, min_x, min_y, resolution)

        # Add no-hits (free space points)
        free_maps = localized_scan.free_space_maps
        if free_maps["low"]["resolution"] == resolution:
            free_points = free_maps["low"]["points"]
        else:
            free_points = free_maps["high"]["points"]
        if free_points.shape[1] > 0:
            _accumulate_no_hits(no_hits_grid, free_points, min_x, min_y, resolution)

    # Convert evidence to occupancy grid
    if method == "logodds":
        occupancy_grid = _evidence_to_occupancy_logodds(hits_grid, no_hits_grid)
    elif method == "standard":
        occupancy_grid = _evidence_to_occupancy(
            hits_grid, no_hits_grid, min_hits, min_no_hits
        )
    elif method == "ratio_based":
        occupancy_grid = _evidence_to_occupancy_ratio_based(
            hits_grid, no_hits_grid, min_hits, min_no_hits
        )
    else:
        raise ValueError(f"Invalid method: {method}")
    return occupancy_grid, np.array([min_x, min_y])


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

    Parameters:
    -----------
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

    Parameters:
    -----------
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


@njit
def _evidence_to_occupancy(
    hits_grid: np.ndarray, no_hits_grid: np.ndarray, min_hits: int, min_no_hits: int
) -> np.ndarray:
    """
    Convert evidence grids to occupancy grid.

    Parameters:
    -----------
    hits_grid : np.ndarray
        2D grid with hit counts
    no_hits_grid : np.ndarray
        2D grid with no-hit counts
    min_hits : int
        Minimum hits to mark as occupied
    min_no_hits : int
        Minimum no-hits to mark as free

    Returns:
    --------
    occupancy_grid : np.ndarray
        2D occupancy grid (0=occupied, 125=unknown, 255=free)
    """
    height, width = hits_grid.shape
    occupancy_grid = np.full((height, width), 125, dtype=np.uint8)

    # Use explicit loops to avoid Numba boolean indexing issues
    for i in range(height):
        for j in range(width):
            if hits_grid[i, j] >= min_hits:
                occupancy_grid[i, j] = 0  # occupied
            elif no_hits_grid[i, j] >= min_no_hits:
                occupancy_grid[i, j] = 255  # free
    return occupancy_grid


@njit
def _logit(p: float) -> float:
    """Compute the logit function."""
    return np.log(p / (1 - p))


@njit
def _evidence_to_occupancy_logodds(
    hits_grid: np.ndarray,
    no_hits_grid: np.ndarray,
    p_hit: float = 0.55,
    p_miss: float = 0.45,
    occ_threshold: float = 0.55,
    free_threshold: float = 0.45,
) -> np.ndarray:
    """
    Convert hit/miss evidence grids to occupancy grid using log-odds update.

    Parameters:
    -----------
    hits_grid : np.ndarray
        2D grid with hit counts
    no_hits_grid : np.ndarray
        2D grid with no-hit counts
    p_hit : float
        Probability of occupancy given a hit
    p_miss : float
        Probability of occupancy given a miss
    occ_threshold : float
        Probability threshold to classify as occupied
    free_threshold : float
        Probability threshold to classify as free

    Returns:
    --------
    occupancy_grid : np.ndarray
        2D occupancy grid (0=occupied, 125=unknown, 255=free)
    """
    height, width = hits_grid.shape
    occupancy_grid = np.full((height, width), 125, dtype=np.uint8)

    # Precompute log-odds increments
    logit_hit = _logit(p_hit)
    logit_miss = _logit(p_miss)

    for i in range(height):
        for j in range(width):
            # Initial log-odds = 0 (p=0.5)
            log_odds = 0.0
            # Update from hit/miss counts
            log_odds += hits_grid[i, j] * logit_hit
            log_odds += no_hits_grid[i, j] * logit_miss

            # Convert log-odds → probability
            prob = 1.0 / (1.0 + np.exp(-log_odds))

            # Classify
            if prob >= occ_threshold:
                occupancy_grid[i, j] = 0  # occupied
            elif prob <= free_threshold:
                occupancy_grid[i, j] = 255  # free
            else:
                occupancy_grid[i, j] = 125  # unknown

    return occupancy_grid


@njit
def _evidence_to_occupancy_ratio_based(
    hits_grid: np.ndarray,
    no_hits_grid: np.ndarray,
    min_hits: int,
    min_no_hits: int,
    min_total_obs: int = 5,
) -> np.ndarray:
    """
    Ratio-based approach: considers hit ratio when sufficient observations exist

    Parameters:
    ----------
    hits_grid
        Grid with hit counts.
    no_hits_grid
        Grid with no-hit counts.
    min_hits
        Minimum number of hits required to classify a cell as occupied.
    min_no_hits
        Minimum number of no-hits required to classify a cell as free.
    min_total_obs
        Minimum number of total observations required to classify a cell as occupied or free.

    Returns
    -------
    occupancy_grid
        2D occupancy grid where 0=occupied, 125=unknown, 255=free.
    """
    height, width = hits_grid.shape
    occupancy_grid = np.full((height, width), 125, dtype=np.uint8)

    for i in range(height):
        for j in range(width):
            hits = hits_grid[i, j]
            no_hits = no_hits_grid[i, j]
            total_obs = hits + no_hits

            if total_obs >= min_total_obs:
                hit_ratio = hits / total_obs
                if hits >= min_hits or hit_ratio > 0.7:
                    occupancy_grid[i, j] = 0  # occupied
                elif no_hits >= min_no_hits or hit_ratio < 0.3:
                    occupancy_grid[i, j] = 255  # free
            else:
                # Not enough observations - check absolute thresholds
                if hits >= min_hits:
                    occupancy_grid[i, j] = 0  # occupied
                elif no_hits >= min_no_hits:
                    occupancy_grid[i, j] = 255  # free

    return occupancy_grid
