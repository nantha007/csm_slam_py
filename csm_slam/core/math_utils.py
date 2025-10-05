"""Math and geometry helpers for 2D poses and scans.

Author: Nantha Kumar Sunder
"""

import numpy as np
from numba import njit


@njit
def transform_scan(scan: np.ndarray, pose: np.ndarray):
    """Transform a 2D scan by a SE(2) pose.

    Parameters
    ----------
    scan
        Array of shape (2, N) with scan points in the sensor frame.
    pose
        Array of shape (3,) representing `[x, y, theta]`.

    Returns
    -------
    transformed_scan
        Array of shape (2, N) with points transformed into the world frame.
    """
    R = theta_to_rot_mat(pose[2])
    t = pose[:2].reshape(2, 1)

    # Perform matrix multiplication manually to avoid Numba type inference issues
    result = np.empty_like(scan)
    for i in range(scan.shape[1]):
        result[0, i] = R[0, 0] * scan[0, i] + R[0, 1] * scan[1, i] + t[0, 0]
        result[1, i] = R[1, 0] * scan[0, i] + R[1, 1] * scan[1, i] + t[1, 0]

    return result


@njit
def theta_to_rot_mat(theta: float):
    """Convert a scalar angle to a 2x2 rotation matrix.

    Parameters
    ----------
    theta
        Rotation angle in radians.

    Returns
    -------
    R
        Array of shape (2, 2) representing the rotation matrix.
    """

    return np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])


@njit
def wrap_to_pi(angle: float):
    """Wrap an angle to the interval [-pi, pi].

    Parameters
    ----------
    angle
        Angle in radians to wrap.

    Returns
    -------
    wrapped_angle
        Wrapped angle in radians.
    """
    return ((angle + np.pi) % (2 * np.pi)) - np.pi


@njit
def get_relative_pose(pose_a: np.ndarray, pose_b: np.ndarray):
    """Compute the pose of `pose_b` expressed in the frame of `pose_a`.

    Parameters
    ----------
    pose_a
        Array of shape (3,) for the reference pose `[x, y, theta]`.
    pose_b
        Array of shape (3,) for the target pose `[x, y, theta]`.

    Returns
    -------
    relative_pose
        Array of shape (3,) for the relative pose of B in A's frame.
    """
    theta1 = pose_a[2]
    theta2 = pose_b[2]

    # Delta in global frame
    dx = pose_b[0] - pose_a[0]
    dy = pose_b[1] - pose_a[1]

    # Rotate into A's frame
    x_r = np.cos(theta1) * dx + np.sin(theta1) * dy
    y_r = -np.sin(theta1) * dx + np.cos(theta1) * dy

    # Relative orientation
    theta_r = wrap_to_pi(theta2 - theta1)

    return np.array([x_r, y_r, theta_r])


def pose_to_matrix(pose: np.ndarray):
    """Convert a pose `[x, y, theta]` to a homogeneous transform.

    Parameters
    ----------
    pose
        Array of shape (3,) representing `[x, y, theta]`.

    Returns
    -------
    matrix
        Array of shape (3, 3) representing the SE(2) transform.
    """
    return np.array(
        [
            [np.cos(pose[2]), -np.sin(pose[2]), pose[0]],
            [np.sin(pose[2]), np.cos(pose[2]), pose[1]],
            [0, 0, 1],
        ]
    )


@njit
def matrix_to_pose(matrix: np.ndarray):
    """Convert a homogeneous transform to pose `[x, y, theta]`.

    Parameters
    ----------
    matrix
        Array of shape (3, 3) representing the SE(2) transform.

    Returns
    -------
    pose
        Array of shape (3,) representing `[x, y, theta]`.
    """
    return np.array(
        [matrix[0, 2], matrix[1, 2], wrap_to_pi(np.arctan2(matrix[1, 0], matrix[0, 0]))]
    )


@njit
def move_to_pose(pose: np.ndarray, delta_pose: np.ndarray):
    """Apply a local delta to a pose in the pose's frame.

    Parameters
    ----------
    pose
        Array of shape (3,) for the current pose `[x, y, theta]`.
    delta_pose
        Array of shape (3,) for the local motion `[dx, dy, dtheta]`.

    Returns
    -------
    new_pose
        Array of shape (3,) for the updated pose in the world frame.
    """
    x_new = pose[0] + np.cos(pose[2]) * delta_pose[0] - np.sin(pose[2]) * delta_pose[1]
    y_new = pose[1] + np.sin(pose[2]) * delta_pose[0] + np.cos(pose[2]) * delta_pose[1]
    theta_new = pose[2] + delta_pose[2]
    return np.array([x_new, y_new, theta_new])


@njit(cache=True)
def movement_threshold(
    pose: np.ndarray, last_pose: np.ndarray, movement_threshold: np.ndarray
):
    """Check if motion between poses is below a threshold.

    Parameters
    ----------
    pose
        Array of shape (3,) for the current pose.
    last_pose
        Array of shape (3,) for the previous pose.
    movement_threshold
        Array of shape (2,) for position and angle thresholds.

    Returns
    -------
    bool
        True if both position and angular changes are below thresholds.
    """
    position_diff = np.sqrt(
        (pose[0] - last_pose[0]) ** 2 + (pose[1] - last_pose[1]) ** 2
    )
    angle_diff = np.abs(wrap_to_pi(pose[2] - last_pose[2]))
    return position_diff < movement_threshold[0] and angle_diff < movement_threshold[1]
