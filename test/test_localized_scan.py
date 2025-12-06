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
Tests for localized scan class representing 2D lidar scan with pose.

Author: Nantha Kumar Sunder
"""

import numpy as np

from csm_slam.sensors.localized_scan import LocalizedScan, bresenham, create_free_space_map


class TestLocalizedScan:
    """Test cases for the LocalizedScan class."""

    def test_localized_scan_initialization(self):
        """Test localized scan initialization and properties."""
        scan_id = 0
        pose = np.array([1.0, 2.0, 0.5])
        scan = np.array([[1.0, 2.0], [0.0, 1.0]])

        localized_scan = LocalizedScan(scan_id, pose, scan)

        assert localized_scan.scan_id == scan_id
        assert np.array_equal(localized_scan.pose, pose)
        assert np.array_equal(localized_scan.get_original_scan(), scan)

    def test_get_localized_scan(self):
        """Test getting scan transformed to world frame."""
        scan = np.array([[1.0], [0.0]])
        pose = np.array([0.0, 0.0, 0.0])
        localized_scan = LocalizedScan(0, pose, scan)

        # With zero pose, localized scan should match original
        localized = localized_scan.get_localized_scan()
        assert np.array_equal(localized, scan)

    def test_update_pose(self):
        """Test updating pose and recomputing localized scan."""
        scan = np.array([[1.0], [0.0]])
        pose = np.array([0.0, 0.0, 0.0])
        localized_scan = LocalizedScan(0, pose, scan)

        new_pose = np.array([1.0, 2.0, 0.0])
        localized_scan.update(new_pose)

        assert np.array_equal(localized_scan.pose, new_pose)
        # Localized scan should be translated
        localized = localized_scan.get_localized_scan()
        assert np.allclose(localized[0, 0], 2.0)  # 1.0 (scan) + 1.0 (pose)
        assert np.allclose(localized[1, 0], 2.0)  # 0.0 (scan) + 2.0 (pose)

    def test_free_space_maps(self):
        """Test free space maps property."""
        scan = np.array([[1.0, 2.0], [0.0, 1.0]])
        pose = np.array([0.0, 0.0, 0.0])
        localized_scan = LocalizedScan(
            0, pose, scan, low_resolution=0.1, high_resolution=0.05
        )

        free_maps = localized_scan.free_space_maps

        assert "low" in free_maps
        assert "high" in free_maps
        assert free_maps["low"]["resolution"] == 0.1
        assert free_maps["high"]["resolution"] == 0.05
        assert "points" in free_maps["low"]
        assert "points" in free_maps["high"]


class TestBresenham:
    """Test cases for bresenham function."""

    def test_horizontal_line(self):
        """Test Bresenham for horizontal line."""
        result = bresenham((0, 0), (5, 0))
        assert result.shape[1] == 2
        assert result[0, 0] == 0
        assert result[0, 1] == 0
        assert result[-1, 0] == 5
        assert result[-1, 1] == 0
        assert len(result) >= 6

    def test_vertical_line(self):
        """Test Bresenham for vertical line."""
        result = bresenham((0, 0), (0, 5))
        assert result[0, 0] == 0
        assert result[0, 1] == 0
        assert result[-1, 0] == 0
        assert result[-1, 1] == 5

    def test_diagonal_line(self):
        """Test Bresenham for diagonal line."""
        result = bresenham((0, 0), (3, 3))
        assert result[0, 0] == 0
        assert result[0, 1] == 0
        assert result[-1, 0] == 3
        assert result[-1, 1] == 3

    def test_single_point(self):
        """Test Bresenham for single point."""
        result = bresenham((5, 5), (5, 5))
        assert len(result) == 1
        assert result[0, 0] == 5
        assert result[0, 1] == 5


class TestCreateFreeSpaceMap:
    """Test cases for create_free_space_map function."""

    def test_create_free_space_map(self):
        """Test creating free space map from scan."""
        scan = np.array([[1.0, 2.0], [0.0, 1.0]])
        resolution = 0.1

        free_space = create_free_space_map(scan, resolution)

        assert free_space.shape[0] == 2
        assert free_space.shape[1] > 0

    def test_empty_scan(self):
        """Test free space map with empty scan."""
        scan = np.array([[], []])
        resolution = 0.1

        free_space = create_free_space_map(scan, resolution)

        assert free_space.shape[0] == 2
        assert free_space.shape[1] == 0
