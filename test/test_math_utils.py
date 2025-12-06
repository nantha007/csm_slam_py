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
Tests for math and geometry helpers for 2D poses and scans.

Author: Nantha Kumar Sunder
"""

import os
import sys

import numpy as np

# Add test directory to path to allow importing test_helpers
test_dir = os.path.dirname(os.path.abspath(__file__))
if test_dir not in sys.path:
    sys.path.insert(0, test_dir)

from test_helpers import load_settings  # noqa: E402

load_settings()

from csm_slam.utils.math_utils import (  # noqa: E402
    get_relative_pose,
    matrix_to_pose,
    move_to_pose,
    movement_threshold,
    pose_to_matrix,
    theta_to_rot_mat,
    transform_scan,
    wrap_to_pi,
)


class TestThetaToRotMat:
    """Test cases for theta_to_rot_mat function."""

    def test_zero_angle(self):
        """Test rotation matrix for zero angle."""
        R = theta_to_rot_mat(0.0)
        expected = np.eye(2)
        assert np.allclose(R, expected)

    def test_pi_over_two(self):
        """Test rotation matrix for 90 degrees."""
        R = theta_to_rot_mat(np.pi / 2)
        expected = np.array([[0.0, -1.0], [1.0, 0.0]])
        assert np.allclose(R, expected)

    def test_rotation_orthogonality(self):
        """Test that rotation matrix is orthogonal."""
        R = theta_to_rot_mat(0.5)
        # R^T * R should be identity
        assert np.allclose(R.T @ R, np.eye(2))


class TestWrapToPi:
    """Test cases for wrap_to_pi function."""

    def test_within_range(self):
        """Test angle already within [-pi, pi]."""
        assert abs(wrap_to_pi(0.5) - 0.5) < 1e-9
        assert abs(wrap_to_pi(-0.5) - (-0.5)) < 1e-9

    def test_above_pi(self):
        """Test angle above pi."""
        assert abs(wrap_to_pi(2 * np.pi) - 0.0) < 1e-9
        assert abs(wrap_to_pi(3 * np.pi / 2) - (-np.pi / 2)) < 1e-9

    def test_below_neg_pi(self):
        """Test angle below -pi."""
        assert abs(wrap_to_pi(-2 * np.pi) - 0.0) < 1e-9
        assert abs(wrap_to_pi(-3 * np.pi / 2) - (np.pi / 2)) < 1e-9


class TestTransformScan:
    """Test cases for transform_scan function."""

    def test_identity_transform(self):
        """Test transform with zero pose."""
        scan = np.array([[1.0, 2.0], [3.0, 4.0]])
        pose = np.array([0.0, 0.0, 0.0])
        result = transform_scan(scan, pose)
        assert np.allclose(result, scan)

    def test_translation_only(self):
        """Test transform with translation only."""
        scan = np.array([[1.0], [2.0]])
        pose = np.array([5.0, 10.0, 0.0])
        result = transform_scan(scan, pose)
        assert np.allclose(result, np.array([[6.0], [12.0]]))

    def test_rotation_only(self):
        """Test transform with rotation only."""
        scan = np.array([[1.0], [0.0]])
        pose = np.array([0.0, 0.0, np.pi / 2])
        result = transform_scan(scan, pose)
        assert np.allclose(result, np.array([[0.0], [1.0]]), atol=1e-9)


class TestGetRelativePose:
    """Test cases for get_relative_pose function."""

    def test_identity(self):
        """Test relative pose when poses are identical."""
        pose_a = np.array([0.0, 0.0, 0.0])
        pose_b = np.array([0.0, 0.0, 0.0])
        relative = get_relative_pose(pose_a, pose_b)
        assert np.allclose(relative, np.array([0.0, 0.0, 0.0]))

    def test_translation_only(self):
        """Test relative pose with translation only."""
        pose_a = np.array([0.0, 0.0, 0.0])
        pose_b = np.array([1.0, 2.0, 0.0])
        relative = get_relative_pose(pose_a, pose_b)
        assert np.allclose(relative, np.array([1.0, 2.0, 0.0]))

    def test_rotation_only(self):
        """Test relative pose with rotation only."""
        pose_a = np.array([0.0, 0.0, 0.0])
        pose_b = np.array([0.0, 0.0, np.pi / 2])
        relative = get_relative_pose(pose_a, pose_b)
        assert abs(relative[2] - np.pi / 2) < 1e-9


class TestPoseMatrixConversion:
    """Test cases for pose_to_matrix and matrix_to_pose functions."""

    def test_round_trip(self):
        """Test round-trip conversion between pose and matrix."""
        original_pose = np.array([1.5, 2.5, 0.785])
        matrix = pose_to_matrix(original_pose)
        recovered_pose = matrix_to_pose(matrix)
        assert np.allclose(recovered_pose, original_pose)

    def test_zero_pose(self):
        """Test conversion of zero pose."""
        pose = np.array([0.0, 0.0, 0.0])
        matrix = pose_to_matrix(pose)
        assert np.allclose(matrix[:2, :2], np.eye(2))
        assert np.allclose(matrix[:2, 2], [0.0, 0.0])

    def test_matrix_structure(self):
        """Test that matrix has correct structure."""
        pose = np.array([1.0, 2.0, np.pi / 4])
        matrix = pose_to_matrix(pose)
        assert matrix.shape == (3, 3)
        assert abs(matrix[2, 0]) < 1e-9
        assert abs(matrix[2, 1]) < 1e-9
        assert abs(matrix[2, 2] - 1.0) < 1e-9


class TestMoveToPose:
    """Test cases for move_to_pose function."""

    def test_no_movement(self):
        """Test with zero delta."""
        pose = np.array([1.0, 2.0, 0.5])
        delta = np.array([0.0, 0.0, 0.0])
        result = move_to_pose(pose, delta)
        assert np.allclose(result, pose)

    def test_translation_only(self):
        """Test with translation delta only."""
        pose = np.array([0.0, 0.0, 0.0])
        delta = np.array([1.0, 2.0, 0.0])
        result = move_to_pose(pose, delta)
        assert np.allclose(result, np.array([1.0, 2.0, 0.0]))

    def test_rotation_only(self):
        """Test with rotation delta only."""
        pose = np.array([0.0, 0.0, 0.0])
        delta = np.array([0.0, 0.0, np.pi / 2])
        result = move_to_pose(pose, delta)
        assert abs(result[2] - np.pi / 2) < 1e-9


class TestMovementThreshold:
    """Test cases for movement_threshold function."""

    def test_below_threshold(self):
        """Test when movement is below threshold."""
        pose = np.array([0.1, 0.1, 0.05])
        last_pose = np.array([0.0, 0.0, 0.0])
        threshold = np.array([0.2, 0.1])
        assert movement_threshold(pose, last_pose, threshold) is True

    def test_above_position_threshold(self):
        """Test when position change exceeds threshold."""
        pose = np.array([0.3, 0.0, 0.0])
        last_pose = np.array([0.0, 0.0, 0.0])
        threshold = np.array([0.2, 0.1])
        assert movement_threshold(pose, last_pose, threshold) is False

    def test_above_angle_threshold(self):
        """Test when angle change exceeds threshold."""
        pose = np.array([0.0, 0.0, 0.2])
        last_pose = np.array([0.0, 0.0, 0.0])
        threshold = np.array([0.2, 0.1])
        assert movement_threshold(pose, last_pose, threshold) is False

