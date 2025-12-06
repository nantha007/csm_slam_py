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
Tests for coarse-to-fine multi-resolution grid 2D correlative scan matching.

Author: Nantha Kumar Sunder
"""


import numpy as np

from csm_slam.mapping.grid import Grid, MultiResolutionGrid
from csm_slam.sensors.localized_scan import LocalizedScan
from csm_slam.frontend.scan_matcher import ScanMatcher


class TestScanMatcher:
    """Test cases for the ScanMatcher class."""

    def test_scan_matcher_initialization(self):
        """Test scan matcher initialization."""
        resolution_low = 0.1
        resolution_high = 0.05
        search_window = np.array([1.0, 1.0, 0.5])

        matcher = ScanMatcher(resolution_low, resolution_high, search_window)

        assert matcher.resolution_low == resolution_low
        assert matcher.resolution_high == resolution_high
        assert np.array_equal(matcher.search_window, search_window)
        assert matcher.smear_factor == 10.0

    def test_build_pose_grid(self):
        """Test building pose grid for search."""
        matcher = ScanMatcher(0.1, 0.05, np.array([1.0, 1.0, 0.5]))
        dx_vals = np.array([-0.5, 0.0, 0.5])
        dy_vals = np.array([-0.5, 0.0, 0.5])
        dtheta_vals = np.array([-0.25, 0.0, 0.25])

        pose_grid = matcher._build_pose_grid(dx_vals, dy_vals, dtheta_vals)

        assert pose_grid.shape[1] == 3
        assert pose_grid.shape[0] == len(dx_vals) * len(dy_vals) * len(dtheta_vals)

    def test_build_loglikelihood(self):
        """Test building log likelihood lookup table."""
        matcher = ScanMatcher(0.1, 0.05, np.array([1.0, 1.0, 0.5]))
        # Create a simple grid with some occupied cells
        grid_data = np.full((10, 10), 128, dtype=np.uint8)  # UNKNOWN
        grid_data[5, 5] = 0  # OCCUPIED
        grid = Grid(grid_data, np.array([0.0, 0.0]), 0.1)

        log_likelihood = matcher._build_loglikelihood(grid)

        assert isinstance(log_likelihood, Grid)
        assert log_likelihood.resolution == 0.1
        assert log_likelihood.grid.shape == grid_data.shape

    def test_match(self):
        """Test scan matching with multi-resolution grid."""
        # Create a simple scan
        scan = np.array([[1.0, 2.0], [0.0, 1.0]])
        pose = np.array([0.0, 0.0, 0.0])
        localized_scan = LocalizedScan(0, pose, scan, low_resolution=0.1, high_resolution=0.05)

        # Create multi-resolution grid
        multi_grid = MultiResolutionGrid(0.1, 0.05, [localized_scan])

        # Create scan matcher
        search_window = np.array([0.5, 0.5, 0.2])
        matcher = ScanMatcher(0.1, 0.05, search_window)

        # Perform matching
        initial_pose = np.array([0.0, 0.0, 0.0])
        best_pose, best_score, mean_pose, cov = matcher.match(
            multi_grid, scan, initial_pose
        )

        assert len(best_pose) == 3
        assert isinstance(best_score, (int, float))
        assert len(mean_pose) == 3
        assert cov.shape == (3, 3)

    def test_match_with_translation(self):
        """Test scan matching with initial pose offset."""
        scan = np.array([[1.0], [0.0]])
        pose = np.array([0.0, 0.0, 0.0])
        localized_scan = LocalizedScan(0, pose, scan, low_resolution=0.1, high_resolution=0.05)

        multi_grid = MultiResolutionGrid(0.1, 0.05, [localized_scan])
        search_window = np.array([0.5, 0.5, 0.2])
        matcher = ScanMatcher(0.1, 0.05, search_window)

        # Try matching with offset initial pose
        initial_pose = np.array([0.1, 0.1, 0.0])
        best_pose, best_score, mean_pose, cov = matcher.match(
            multi_grid, scan, initial_pose
        )

        assert len(best_pose) == 3
        assert cov.shape == (3, 3)

