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
Coarse-to-fine Multi-Resolution Grid 2D correlative scan matching.

This module implements a scan matching algorithm that uses distance-transform-based
likelihood fields for robust 2D laser scan alignment.

Author: Nantha Kumar Sunder
"""

import numpy as np
from numba import njit, prange
from scipy.ndimage import distance_transform_edt

from csm_slam.core.grid import Grid, MultiResolutionGrid
from csm_slam.core.math_utils import wrap_to_pi

VALID_SCAN_RATIO = 0.8


class ScanMatcher:
    """
    Class for performing coarse-to-fine correlative scan matching.

    Parameters
    ----------
    resolution_low : float
        Resolution for coarse grid matching in meters.
    resolution_high : float
        Resolution for fine grid matching in meters.
    search_window : list
        3-element list [dx, dy, dtheta] defining search ranges.
    smear_factor : float, optional
        Factor for smearing the likelihood field (default: 10.0).

    """

    def __init__(
        self,
        resolution_low: float,
        resolution_high: float,
        search_window: np.ndarray,
        smear_factor: float = 10.0,
    ):
        self.resolution_low = resolution_low
        self.resolution_high = resolution_high
        self.search_window = search_window
        self.smear_factor = smear_factor

    #########################################################
    # Private methods                                       #
    #########################################################

    def _build_loglikelihood(self, grid: Grid) -> dict:
        """
        Build a lookup table proportional to obstacle proximity.

        Creates a log-likelihood lookup table from an occupancy grid using
        distance transform. The table provides proximity-based scores for
        scan matching.

        Parameters
        ----------
        grid : Grid
            An object with attributes `grid` (2D array), `origin` (2-vector),
            and `resolution` (float).

        Returns
        -------
        dict
            Dictionary with keys `log_lookup_table`, `resolution`, and `origin`.

        """
        obstacle_grid = np.zeros_like(grid.grid).astype(np.bool_)
        obstacle_grid[grid.grid == 0] = 1
        log_likelihood = distance_transform_edt(~obstacle_grid).astype(np.float32)
        log_likelihood = self.smear_factor - np.minimum(
            log_likelihood, np.float32(self.smear_factor)
        )
        max_val = np.max(log_likelihood)
        if max_val > 0:
            log_likelihood /= max_val
        return Grid(log_likelihood.astype(np.float64), grid.origin, grid.resolution)

    def _build_pose_grid(
        self, dx_vals: np.ndarray, dy_vals: np.ndarray, dtheta_vals: np.ndarray
    ) -> np.ndarray:
        """
        Create a grid of candidate pose offsets.

        Generates a 3D grid of candidate pose offsets for exhaustive search
        over the specified ranges.

        Parameters
        ----------
        dx_vals, dy_vals, dtheta_vals : numpy.ndarray
            1D arrays defining the search range along each axis.

        Returns
        -------
        numpy.ndarray
            Array of shape (N, 3) where rows are [dx, dy, dtheta].

        """
        # Create 3D meshgrid and reshape to 2D array
        dx, dy, dtheta = np.meshgrid(dx_vals, dy_vals, dtheta_vals, indexing='ij')
        xi_grid = np.stack([dx, dy, dtheta], axis=-1).reshape(-1, 3)  # shape (N, 3)
        return xi_grid

    def _search(
        self,
        grid: Grid,
        scan: np.ndarray,
        initial_pose: np.ndarray,
        search_window: np.ndarray,
        res_array: np.ndarray,
    ) -> tuple:
        """
        Evaluate candidate poses and return the best-scoring one.

        Performs exhaustive search over a grid of candidate poses and returns
        the pose with the highest likelihood score along with covariance estimation.

        Parameters
        ----------
        grid : Grid
            Lookup table dict from `_build_loglikelihood`.
        scan : numpy.ndarray
            2xN scan points.
        initial_pose : numpy.ndarray
            Initial pose [x, y, theta].
        search_window : numpy.ndarray
            3-vector of maximum absolute deltas [dx, dy, dtheta].
        res_array : numpy.ndarray
            3-vector of step sizes for each axis.

        Returns
        -------
        tuple
            Tuple containing (best_pose, best_score, scores, cov):
            - best_pose: Best pose (3,) as numpy array
            - best_score: Best scalar score
            - scores: Full flat scores array
            - cov: Covariance matrix (3x3)

        """
        # Generate search ranges for each dimension
        x_range = np.arange(
            -search_window[0], search_window[0] + res_array[0], res_array[0]
        )
        y_range = np.arange(
            -search_window[1], search_window[1] + res_array[1], res_array[1]
        )
        theta_range = np.arange(
            -search_window[2], search_window[2] + res_array[2], res_array[2]
        )
        # Create pose grid for exhaustive search
        pose_grid = self._build_pose_grid(x_range, y_range, theta_range)

        # Extract grid parameters for Numba kernel
        lookup_table = grid.grid
        resolution = grid.resolution
        origin = grid.origin

        best_pose, best_score, scores = _search_scores_kernel(
            pose_grid,
            scan,
            initial_pose,
            lookup_table,
            resolution,
            origin,
        )

        # Grid sizes
        nx = x_range.shape[0]
        ny = y_range.shape[0]
        na = theta_range.shape[0]
        scores3d = scores.reshape(nx, ny, na)

        # Responses: normalize to [0,1]; invalid scores -> 0
        xy_scores = np.max(scores3d, axis=2)  # max over angles per (x,y)
        xy_resp = np.clip(xy_scores / 100.0, 0.0, 1.0)
        xy_resp[~np.isfinite(xy_scores)] = 0.0

        # Best response for scaling and thresholds
        best_response = float(np.clip(best_score / 100.0, 0.0, 1.0))
        thr = max(0.0, best_response - 0.1)

        # Best indices to take angular slice at best (x,y)
        bix, biy, bia = np.unravel_index(int(np.argmax(scores)), (nx, ny, na))

        # Deltas wrt search center (initial_pose)
        dx_best = best_pose[0] - initial_pose[0]
        dy_best = best_pose[1] - initial_pose[1]
        dth_best = wrap_to_pi(best_pose[2] - initial_pose[2])

        # Positional covariance
        Xg, Yg = np.meshgrid(x_range, y_range, indexing='ij')
        mask_xy = xy_resp >= thr
        w_xy = np.where(mask_xy, xy_resp, 0.0)
        norm_xy = float(w_xy.sum())

        cov = np.zeros((3, 3), dtype=np.float64)
        if norm_xy > 1e-9:
            dx_rel = Xg - dx_best
            dy_rel = Yg - dy_best
            var_xx = float((w_xy * (dx_rel**2)).sum() / norm_xy)
            var_xy = float((w_xy * (dx_rel * dy_rel)).sum() / norm_xy)
            var_yy = float((w_xy * (dy_rel**2)).sum() / norm_xy)

            # Lower bounds tied to search resolution
            min_xx = 0.1 * (res_array[0] ** 2)
            min_yy = 0.1 * (res_array[1] ** 2)
            var_xx = max(var_xx, min_xx)
            var_yy = max(var_yy, min_yy)

            # Inverse-of-best scaling
            mult = 1.0 / max(best_response, 1e-6) * 1000.0
            cov[0, 0] = var_xx * mult
            cov[0, 1] = var_xy * mult
            cov[1, 0] = var_xy * mult
            cov[1, 1] = var_yy * mult
        else:
            # Max variance fallback when no positional support
            cov[0, 0] = 500.0
            cov[1, 1] = 500.0

        # Initialize angle variance with coarse bound; override below
        cov[2, 2] = 4.0 * (res_array[2] ** 2)

        # Angular covariance at best (x,y)
        ang_scores = scores3d[bix, biy, :]
        ang_resp = np.clip(ang_scores / 100.0, 0.0, 1.0)
        ang_resp[~np.isfinite(ang_scores)] = 0.0
        mask_th = ang_resp >= thr
        w_th = np.where(mask_th, ang_resp, 0.0)
        norm_th = float(w_th.sum())

        if norm_th > 1e-9:
            dths = theta_range - dth_best
            dths = (dths + np.pi) % (2 * np.pi) - np.pi  # wrap differences
            var_th = float((w_th * (dths**2)).sum() / norm_th)
            if var_th < 1e-12:
                var_th = res_array[2] ** 2
        else:
            var_th = 1000.0 * (res_array[2] ** 2)

        cov[2, 2] = var_th

        # Return best pose; mean_pose unused by caller
        return best_pose, best_score, best_pose.copy(), cov

    #########################################################
    # Public methods                                        #
    #########################################################

    def match(
        self, grid: MultiResolutionGrid, scan: np.ndarray, initial_pose: np.ndarray
    ) -> tuple:
        """
        Coarse-to-fine matching over multi-resolution grids.

        Performs two-stage scan matching: first coarse search over low-resolution
        grid, then fine search around the best coarse result using high-resolution
        grid.

        Parameters
        ----------
        grid : MultiResolutionGrid
            A MultiResolutionGrid instance containing coarse and fine maps.
        scan : numpy.ndarray
            2xN scan points.
        initial_pose : numpy.ndarray
            Initial pose estimate [x, y, theta].

        Returns
        -------
        tuple
            Tuple containing (best_pose, best_score, mean_pose, cov):
            - best_pose: The best refined pose
            - best_score: Score of the best pose
            - mean_pose: Mean pose (unused by caller)
            - cov: 3x3 covariance estimate

        """
        grid_map_low = self._build_loglikelihood(grid.coarse_grid)
        grid_map_high = self._build_loglikelihood(grid.fine_grid)

        factor = 10
        # Extract search parameters
        search_x = self.search_window[0]
        search_y = self.search_window[1]
        search_theta = self.search_window[2]

        # Define the coarse search resolution
        coarse_x_res = search_x / factor
        coarse_y_res = search_y / factor
        coarse_theta_res = search_theta / factor

        coarse_best_pose, _, _, _ = self._search(
            grid_map_low,
            scan,
            initial_pose,
            [search_x, search_y, search_theta],
            [coarse_x_res, coarse_y_res, coarse_theta_res],
        )
        # Define fine search window around coarse best pose
        fine_search_x = coarse_x_res
        fine_search_y = coarse_y_res
        fine_search_theta = coarse_theta_res

        # Define fine search resolution
        fine_x_res = fine_search_x / factor
        fine_y_res = fine_search_y / factor
        fine_theta_res = fine_search_theta / factor

        fine_best_pose, fine_best_score, mean_pose, cov = self._search(
            grid_map_high,
            scan,
            coarse_best_pose,
            [fine_search_x, fine_search_y, fine_search_theta],
            [fine_x_res, fine_y_res, fine_theta_res],
        )
        return fine_best_pose, fine_best_score, mean_pose, cov


#########################################################
# Private methods                                       #
#########################################################


@njit(parallel=True, fastmath=True, cache=True)
def _search_scores_kernel(
    pose_grid: np.ndarray,
    scan: np.ndarray,
    initial_pose: np.ndarray,
    lookup_table: np.ndarray,
    resolution: float,
    origin: np.ndarray,
) -> tuple:
    """
    Compute matching scores for each candidate pose in parallel.

    This Numba-optimized function evaluates scan matching scores for all
    candidate poses in parallel. It transforms scan points according to
    each candidate pose and looks up likelihood values from the distance
    transform table.

    Parameters
    ----------
    pose_grid : numpy.ndarray
        Array of candidate pose offsets (N, 3).
    scan : numpy.ndarray
        2xN array of scan points.
    initial_pose : numpy.ndarray
        Initial pose [x, y, theta].
    lookup_table : numpy.ndarray
        Distance transform lookup table.
    resolution : float
        Grid resolution.
    origin : numpy.ndarray
        Grid origin coordinates.

    Returns
    -------
    tuple
        Tuple containing (best_pose, best_score, scores):
        - best_pose: Best pose found
        - best_score: Score of best pose
        - scores: Array of scores for all poses

    """
    m = pose_grid.shape[0]
    height = lookup_table.shape[0]
    width = lookup_table.shape[1]

    scores = np.full(m, np.inf, dtype=np.float64)
    xs = scan[0]
    ys = scan[1]

    for i in prange(m):
        dx = pose_grid[i, 0]
        dy = pose_grid[i, 1]
        dtheta = pose_grid[i, 2]

        cx = initial_pose[0] + dx
        cy = initial_pose[1] + dy
        ct = initial_pose[2] + dtheta

        c = np.cos(ct)
        s = np.sin(ct)

        num_valid = 0
        s_accum = 0.0

        for k in range(xs.shape[0]):
            xw = c * xs[k] - s * ys[k] + cx
            yw = s * xs[k] + c * ys[k] + cy

            ix = int(np.rint((xw - origin[0]) / resolution))
            iy_cart = int(np.rint((yw - origin[1]) / resolution))
            iy = height - 1 - iy_cart

            if 0 <= ix < width and 0 <= iy < height:
                num_valid += 1
                s_accum += lookup_table[iy, ix]

        if (num_valid / xs.shape[0]) < VALID_SCAN_RATIO:
            scores[i] = -1e30
        elif num_valid > 0:
            scores[i] = (s_accum * 100.0) / num_valid
        else:
            scores[i] = -1e30

    # Best pose reconstruction
    best_idx = 0
    best_score = -1e30
    for i in range(m):
        s = scores[i]
        if np.isfinite(s) and s > best_score:
            best_score = s
            best_idx = i

    best_pose = initial_pose.copy()
    best_pose[0] += pose_grid[best_idx, 0]
    best_pose[1] += pose_grid[best_idx, 1]
    best_pose[2] += pose_grid[best_idx, 2]
    return best_pose, best_score, scores
