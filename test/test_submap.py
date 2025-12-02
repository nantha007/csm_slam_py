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

"""Tests for submap container used to group scans and maintain a pose estimate."""

import numpy as np

from csm_slam.core.submap import Submap


class TestSubmap:
    """Test cases for the Submap class."""

    def test_submap_initialization(self):
        """Test submap initialization and properties."""
        submap = Submap(5, np.array([1.0, 2.0, 0.5]), 10)

        assert submap.submap_id == 5
        assert np.array_equal(submap.pose, np.array([1.0, 2.0, 0.5]))
        assert submap.first_scan_id == 10
        assert submap.scan_ids == [10]

    def test_pose_setter(self):
        """Test updating submap pose."""
        submap = Submap(0, np.array([0.0, 0.0, 0.0]), 0)
        new_pose = np.array([1.5, 2.5, 0.785])
        submap.pose = new_pose

        assert np.array_equal(submap.pose, new_pose)

    def test_add_scan_id(self):
        """Test adding scan IDs to submap."""
        submap = Submap(0, np.array([0.0, 0.0, 0.0]), 0)

        assert submap.scan_ids == [0]
        assert submap.first_scan_id == 0

        submap.add_scan_id(1)
        assert submap.scan_ids == [0, 1]
        assert submap.first_scan_id == 0

        submap.add_scan_id(2)
        assert submap.scan_ids == [0, 1, 2]

