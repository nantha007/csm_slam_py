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
Tests for occupancy grid construction and multi-resolution grid utilities.

Author: Nantha Kumar Sunder
"""

import numpy as np

from csm_slam.mapping.grid import (
    Grid,
    MultiResolutionGrid,
    create_occupancy_grid,
)
from csm_slam.sensors.localized_scan import LocalizedScan


class TestGrid:
    """Test cases for the Grid class."""

    def test_grid_initialization(self):
        """Test grid initialization."""
        grid_data = np.zeros((10, 10), dtype=np.uint8)
        origin = np.array([0.0, 0.0])
        resolution = 0.1

        grid = Grid(grid_data, origin, resolution)

        assert np.array_equal(grid.grid, grid_data)
        assert np.array_equal(grid.origin, origin)
        assert grid.resolution == resolution


class TestMultiResolutionGrid:
    """Test cases for the MultiResolutionGrid class."""

    def test_multi_resolution_grid_initialization(self):
        """Test multi-resolution grid initialization."""
        scan = np.array([[1.0, 2.0], [0.0, 1.0]])
        pose = np.array([0.0, 0.0, 0.0])
        localized_scan = LocalizedScan(0, pose, scan, low_resolution=0.1, high_resolution=0.05)

        multi_grid = MultiResolutionGrid(0.1, 0.05, [localized_scan])

        assert multi_grid.coarse_grid is not None
        assert multi_grid.fine_grid is not None
        assert multi_grid.coarse_grid.resolution == 0.1
        assert multi_grid.fine_grid.resolution == 0.05


class TestCreateOccupancyGrid:
    """Test cases for create_occupancy_grid function."""

    def test_create_occupancy_grid(self):
        """Test creating occupancy grid from localized scans."""
        scan = np.array([[1.0, 2.0], [0.0, 1.0]])
        pose = np.array([0.0, 0.0, 0.0])
        localized_scan = LocalizedScan(0, pose, scan, low_resolution=0.1, high_resolution=0.05)

        grid = create_occupancy_grid([localized_scan], 0.1)

        assert isinstance(grid, Grid)
        assert grid.resolution == 0.1
        assert grid.grid.shape[0] > 0
        assert grid.grid.shape[1] > 0
        assert len(grid.origin) == 2

    def test_multiple_scans(self):
        """Test creating occupancy grid from multiple scans."""
        scan1 = np.array([[1.0], [0.0]])
        scan2 = np.array([[2.0], [1.0]])
        pose = np.array([0.0, 0.0, 0.0])
        localized_scan1 = LocalizedScan(0, pose, scan1, low_resolution=0.1, high_resolution=0.05)
        localized_scan2 = LocalizedScan(1, pose, scan2, low_resolution=0.1, high_resolution=0.05)

        grid = create_occupancy_grid([localized_scan1, localized_scan2], 0.1)

        assert isinstance(grid, Grid)
        assert grid.resolution == 0.1

