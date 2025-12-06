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
ROS2 utility functions for CSM SLAM nodes.

This module provides common utility functions used by both online and offline
CSM SLAM nodes for message conversion, coordinate transformations, and data processing.

Author: Nantha Kumar Sunder
"""

import numpy as np

from geometry_msgs.msg import Point, PoseStamped, TransformStamped
from nav_msgs.msg import MapMetaData, OccupancyGrid, Path
from rclpy.time import Time
from sensor_msgs.msg import LaserScan, MultiEchoLaserScan
from visualization_msgs.msg import Marker

from csm_slam.mapping.grid import Grid

#########################################################
# Public functions                                      #
#########################################################


def create_base_to_map_transform(
    current_pose: np.ndarray,
    map_frame_name: str,
    base_link_name: str,
    stamp: Time,
) -> TransformStamped:
    """
    Create the transform from base_link to map frame.

    Parameters
    ----------
    current_pose : numpy.ndarray
        Current robot pose [x, y, theta] in meters and radians.
    map_frame_name : str
        Name of the map frame.
    base_link_name : str
        Name of the base_link frame.
    stamp : Time
        Timestamp for the transform.

    Returns
    -------
    TransformStamped
        ROS2 TransformStamped message.

    """
    transform = TransformStamped()
    transform.header.stamp = stamp.to_msg()
    transform.header.frame_id = map_frame_name
    transform.child_frame_id = base_link_name

    # Set translation
    transform.transform.translation.x = current_pose[0]
    transform.transform.translation.y = current_pose[1]
    transform.transform.translation.z = 0.0

    # Set rotation (convert theta to quaternion)
    z, w = theta_to_quaternion(current_pose[2])
    transform.transform.rotation.x = 0.0
    transform.transform.rotation.y = 0.0
    transform.transform.rotation.z = z
    transform.transform.rotation.w = w

    return transform


def graph_edges_to_marker(
    edges: list[tuple[np.ndarray, np.ndarray]],
    frame_id: str,
    stamp: Time,
    line_width: float = 0.03,
    color=(0.0, 0.7, 1.0, 1.0),
) -> Marker:
    """
    Convert pose-graph edges to a Marker for RViz visualization.

    Parameters
    ----------
    edges : list[tuple[np.ndarray, np.ndarray]]
        List of tuples where each tuple contains the from and to poses as `[x, y, theta]`.
    frame_id : str
        Frame ID for the marker header.
    stamp : Time
        Timestamp for the marker header.
    line_width : float, optional
        Width of the rendered lines (default 0.03).
    color : tuple, optional
        RGBA tuple for the line color (default cyan).

    Returns
    -------
    Marker
        Marker message ready for publication.

    """
    marker = Marker()
    marker.header.frame_id = frame_id
    marker.header.stamp = stamp.to_msg()
    marker.ns = 'slam_edges'
    marker.id = 0
    marker.type = Marker.LINE_LIST
    marker.action = Marker.ADD
    marker.pose.orientation.w = 1.0
    marker.scale.x = line_width
    marker.color.r = float(color[0])
    marker.color.g = float(color[1])
    marker.color.b = float(color[2])
    marker.color.a = float(color[3])

    marker.points.clear()
    for from_pose, to_pose in edges:
        start = Point()
        start.x = float(from_pose[0])
        start.y = float(from_pose[1])
        start.z = 0.0

        end = Point()
        end.x = float(to_pose[0])
        end.y = float(to_pose[1])
        end.z = 0.0

        marker.points.append(start)
        marker.points.append(end)

    return marker


def grid_to_occ_grid(
    grid: Grid,
    frame_id: str,
    stamp: Time,
) -> OccupancyGrid:
    """
    Convert numpy array to ROS2 OccupancyGrid message.

    Parameters
    ----------
    grid : Grid
        Grid object containing the occupancy grid map with values
        indicating free space, occupied space, and unknown areas,
        along with origin and resolution information.
    frame_id : str
        Frame ID for the occupancy grid header.
    stamp : Time
        Timestamp for the occupancy grid header.

    Returns
    -------
    OccupancyGrid
        ROS2 OccupancyGrid message.

    """
    meta_data = MapMetaData()
    meta_data.resolution = grid.resolution
    meta_data.width = grid.grid.shape[1]
    meta_data.height = grid.grid.shape[0]
    meta_data.origin.position.x = grid.origin[0]
    meta_data.origin.position.y = grid.origin[1]
    meta_data.origin.orientation.w = 1.0

    occ_grid = OccupancyGrid()
    occ_grid.info = meta_data
    unknown = grid.grid == 128
    occupied = grid.grid == 0
    free = grid.grid == 255
    grid.grid = grid.grid.astype(np.int8)
    grid.grid[unknown] = -1
    grid.grid[occupied] = 100
    grid.grid[free] = 0
    grid_ros = np.flipud(grid.grid)
    occ_grid.data = grid_ros.flatten(order='C').tolist()
    occ_grid.header.stamp = stamp.to_msg()
    occ_grid.header.frame_id = frame_id
    return occ_grid


def laser_to_cart(msg: LaserScan) -> np.ndarray:
    """
    Convert LaserScan message to Cartesian coordinates.

    Parameters
    ----------
    msg : LaserScan
        ROS2 LaserScan message.

    Returns
    -------
    numpy.ndarray
        2xN array of Cartesian coordinates.

    """
    # Extract ranges and filter
    ranges = np.array(msg.ranges, dtype=np.float32)
    n = ranges.shape[0]

    angles = msg.angle_min + np.arange(n, dtype=np.float32) * msg.angle_increment

    # Validity mask
    rmin = max(0.05, float(getattr(msg, 'range_min', 0.0)))
    rmax = float(getattr(msg, 'range_max', 20.0))
    mask = np.isfinite(ranges)
    mask &= ranges >= rmin
    mask &= ranges <= min(rmax, 20.0)

    ranges = ranges[mask]
    angles = angles[mask]

    xs = ranges * np.cos(angles)
    ys = ranges * np.sin(angles)
    return np.vstack((xs, ys)).astype(np.float32)


def multi_echo_to_cart(msg: MultiEchoLaserScan) -> np.ndarray:
    """
    Convert MultiEchoLaserScan message to Cartesian coordinates.

    Parameters
    ----------
    msg : MultiEchoLaserScan
        ROS2 MultiEchoLaserScan message.

    Returns
    -------
    numpy.ndarray
        2xN array of Cartesian coordinates.

    """
    num = len(msg.ranges)

    ranges_list = []
    for i in range(num):
        echoes = getattr(msg.ranges[i], 'echoes', [])
        arr = np.array(echoes, dtype=np.float32)
        arr = arr[np.isfinite(arr)]
        arr = arr[arr > 0.0]
        ranges_list.append(float(np.min(arr)))

    ranges = np.array(ranges_list, dtype=np.float32)
    angles = msg.angle_min + np.arange(num, dtype=np.float32) * msg.angle_increment

    rmin = max(0.05, float(getattr(msg, 'range_min', 0.0)))
    rmax = float(getattr(msg, 'range_max', 20.0))
    mask = np.isfinite(ranges)
    mask &= ranges >= rmin
    mask &= ranges <= min(rmax, 20.0)

    ranges = ranges[mask]
    angles = angles[mask]

    xs = ranges * np.cos(angles)
    ys = ranges * np.sin(angles)
    return np.vstack((xs, ys)).astype(np.float32)


def poses_to_path(
    poses: np.ndarray,
    frame_id: str,
    stamp: Time,
) -> Path:
    """
    Convert poses to ROS2 Path message.

    Parameters
    ----------
    poses : numpy.ndarray
        3xN array of poses [x, y, theta] in meters and radians.
    frame_id : str
        Frame ID for the path header.
    stamp : Time
        Timestamp for the path header.

    Returns
    -------
    Path
        ROS2 Path message.

    """
    path = Path()
    path.header.stamp = stamp.to_msg()
    path.header.frame_id = frame_id
    for i in range(poses.shape[1]):
        z, w = theta_to_quaternion(poses[2, i])
        pose = PoseStamped()
        pose.header.stamp = stamp.to_msg()
        pose.header.frame_id = frame_id
        pose.pose.position.x = poses[0, i]
        pose.pose.position.y = poses[1, i]
        pose.pose.position.z = 0.0
        z, w = theta_to_quaternion(poses[2, i])
        pose.pose.orientation.z = z
        pose.pose.orientation.w = w
        path.poses.append(pose)
    return path


def theta_to_quaternion(theta: float) -> list:
    """
    Convert angle to quaternion representation.

    Parameters
    ----------
    theta : float
        Rotation angle in radians.

    Returns
    -------
    list
        Quaternion as [z, w] components.

    """
    z = np.sin(theta / 2.0)
    w = np.cos(theta / 2.0)
    return [z, w]
