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
CSM SLAM online processing node for real-time SLAM.

This module provides an online SLAM processing node that subscribes to laser scan
and odometry data from ROS2 topics and processes them using the CSM SLAM algorithm.
It supports both LaserScan and MultiEchoLaserScan message types and publishes the
resulting map, trajectory, and odometry data in real-time.

Author: Nantha Kumar Sunder
"""

import os
import sys
import threading

import numpy as np
import yaml

import rclpy
from nav_msgs.msg import OccupancyGrid, Odometry, Path
from rclpy.node import Node
from sensor_msgs.msg import LaserScan, MultiEchoLaserScan
from tf2_ros import TransformBroadcaster
from visualization_msgs.msg import Marker


def _load_settings():
    """Load settings from config/settings.yaml file."""
    from ament_index_python.packages import get_package_share_directory

    config_dir = os.path.join(get_package_share_directory('csm_slam'), 'config')
    settings_file = os.path.join(config_dir, 'venv_settings.yaml')

    if os.path.exists(settings_file):
        with open(settings_file, 'r') as f:
            settings = yaml.safe_load(f)
            venv_path = settings.get('venv_path', '')
            if venv_path:
                venv_path = os.path.expanduser(venv_path)
                if os.path.exists(venv_path):
                    sys.path.append(venv_path)
    else:
        raise FileNotFoundError(f'Settings file not found: {settings_file}')


_load_settings()

from csm_slam.core.grid import Grid  # noqa: E402
from .core.graph_slam import GraphSlam  # noqa: E402
from . import ros_utils  # noqa: E402


class CSMSlamNode(Node):
    """
    CSM SLAM online processing node for real-time SLAM.

    This node subscribes to laser scan and odometry data from ROS2 topics and
    processes them using the CSM SLAM algorithm. It supports both LaserScan and
    MultiEchoLaserScan message types and publishes the resulting map, trajectory,
    and odometry data in real-time.

    """

    def __init__(self):
        # Initialize ROS2 node
        super().__init__('csm_slam_node')
        self._params = {}
        self._initialize()

        self._skip_scan_interval = max(
            1, int(self._params.get('skip_scan_interval', 1))
        )
        self._scan_counter = 0

        # SLAM trajectory
        self._slam = GraphSlam(self.get_logger(), self._params)

        # Publishers
        self._map_publisher = self.create_publisher(
            OccupancyGrid, self._params['map_topic'], 10
        )
        self._odom_publisher = self.create_publisher(
            Odometry, self._params['pub_odom_topic'], 10
        )
        self._trajectory_publisher = self.create_publisher(Path, '/slam_trajectory', 10)
        self._edges_publisher = self.create_publisher(Marker, '/slam_edges', 10)

        # Transform broadcaster
        self._tf_broadcaster = TransformBroadcaster(self)

        # Data storage for latest messages
        self._latest_scan = None
        self._latest_odom = None
        self._scan_lock = threading.Lock()
        self._odom_lock = threading.Lock()

        # Shutdown flag for graceful termination
        self._shutdown_requested = False

        # Create subscribers
        self._create_subscribers()

    #########################################################
    # Private methods                                       #
    #########################################################

    def _initialize(self):
        """
        Initialize ROS2 parameters and SLAM algorithm.

        Declares all ROS2 parameters, builds the parameter dictionary,
        initializes the SLAM algorithm, and creates publishers.

        """
        # Parameters (removed bag_path for online operation)

        # SLAM parameters
        self.declare_parameter('enable_movement_threshold', True)
        self.declare_parameter('enable_odom', False)
        self.declare_parameter('enable_loop_closure', True)
        self.declare_parameter('movement_threshold_distance', 0.2)
        self.declare_parameter('movement_threshold_angle', 15)
        self.declare_parameter('sequence_queue_len', 100)
        self.declare_parameter('skip_scan_interval', 1)

        # Grid resolution parameters
        self.declare_parameter('fine_resolution', 0.05)
        self.declare_parameter('coarse_resolution', 0.1)

        # Publish topic parameters
        self.declare_parameter('map_topic', '/map')
        self.declare_parameter('pub_odom_topic', '/slam_odom')

        # Transform parameters
        self.declare_parameter('base_link_name', 'base_link')
        self.declare_parameter('map_frame_name', 'map')
        self.declare_parameter('odom_frame_name', 'odom')

        # Publish parameters
        self.declare_parameter('publish_base_to_map_transform', True)
        self.declare_parameter('publish_map', True)

        # Subscribe topic parameters
        self.declare_parameter('odom_topic', '/odom')
        self.declare_parameter('lidar_topic', '/lidar')

        # Subscribe topic type parameters
        self.declare_parameter('lidar_type', 'LaserScan')

        # Sequence scan matcher parameters
        self.declare_parameter('sequence_match_distance', 0.3)
        self.declare_parameter('sequence_match_angle', 45)
        self.declare_parameter('sequence_match_factor', 10)

        # Loop scan matcher parameters
        self.declare_parameter('loop_match_distance', 4.0)
        self.declare_parameter('loop_match_angle', 45)
        self.declare_parameter('loop_match_factor', 10)

        # Slam parameters
        self.declare_parameter('submap_distance_threshold', 6.0)
        self.declare_parameter('loop_closure_search_distance', 10.0)
        self.declare_parameter('loop_closure_score_threshold', 85.0)

        self._params = {
            'enable_movement_threshold': self.get_parameter('enable_movement_threshold')
            .get_parameter_value()
            .bool_value,
            'enable_odom': self.get_parameter('enable_odom')
            .get_parameter_value()
            .bool_value,
            'enable_loop_closure': self.get_parameter('enable_loop_closure')
            .get_parameter_value()
            .bool_value,
            'movement_threshold_distance': self.get_parameter(
                'movement_threshold_distance'
            )
            .get_parameter_value()
            .double_value,
            'movement_threshold_angle': self.get_parameter('movement_threshold_angle')
            .get_parameter_value()
            .integer_value,
            'sequence_queue_len': self.get_parameter('sequence_queue_len')
            .get_parameter_value()
            .integer_value,
            'skip_scan_interval': self.get_parameter('skip_scan_interval')
            .get_parameter_value()
            .integer_value,
            'fine_resolution': self.get_parameter('fine_resolution')
            .get_parameter_value()
            .double_value,
            'coarse_resolution': self.get_parameter('coarse_resolution')
            .get_parameter_value()
            .double_value,
            'map_topic': self.get_parameter('map_topic')
            .get_parameter_value()
            .string_value,
            'pub_odom_topic': self.get_parameter('pub_odom_topic')
            .get_parameter_value()
            .string_value,
            'base_link_name': self.get_parameter('base_link_name')
            .get_parameter_value()
            .string_value,
            'map_frame_name': self.get_parameter('map_frame_name')
            .get_parameter_value()
            .string_value,
            'odom_frame_name': self.get_parameter('odom_frame_name')
            .get_parameter_value()
            .string_value,
            'publish_base_to_map_transform': self.get_parameter(
                'publish_base_to_map_transform'
            )
            .get_parameter_value()
            .bool_value,
            'publish_map': self.get_parameter('publish_map')
            .get_parameter_value()
            .bool_value,
            'odom_topic': self.get_parameter('odom_topic')
            .get_parameter_value()
            .string_value,
            'lidar_topic': self.get_parameter('lidar_topic')
            .get_parameter_value()
            .string_value,
            'lidar_type': self.get_parameter('lidar_type')
            .get_parameter_value()
            .string_value,
            'sequence_match_distance': self.get_parameter('sequence_match_distance')
            .get_parameter_value()
            .double_value,
            'sequence_match_angle': self.get_parameter('sequence_match_angle')
            .get_parameter_value()
            .integer_value,
            'sequence_match_factor': self.get_parameter('sequence_match_factor')
            .get_parameter_value()
            .integer_value,
            'loop_match_distance': self.get_parameter('loop_match_distance')
            .get_parameter_value()
            .double_value,
            'loop_match_angle': self.get_parameter('loop_match_angle')
            .get_parameter_value()
            .integer_value,
            'loop_match_factor': self.get_parameter('loop_match_factor')
            .get_parameter_value()
            .integer_value,
            'submap_distance_threshold': self.get_parameter('submap_distance_threshold')
            .get_parameter_value()
            .double_value,
            'loop_closure_search_distance': self.get_parameter(
                'loop_closure_search_distance'
            )
            .get_parameter_value()
            .double_value,
            'loop_closure_score_threshold': self.get_parameter(
                'loop_closure_score_threshold'
            )
            .get_parameter_value()
            .double_value,
        }

        # Get and log all parameters
        self.get_logger().info('=== CSM SLAM Parameters ===')
        self.get_logger().info('SLAM parameters:')
        self.get_logger().info(
            f"  enable_movement_threshold: {self._params['enable_movement_threshold']}"
        )
        self.get_logger().info(f"  enable_odom: {self._params['enable_odom']}")
        self.get_logger().info(
            f"  enable_loop_closure: {self._params['enable_loop_closure']}"
        )
        self.get_logger().info(
            f"  movement_threshold_distance: {self._params['movement_threshold_distance']}"
        )
        self.get_logger().info(
            f"  movement_threshold_angle: {self._params['movement_threshold_angle']}"
        )
        self.get_logger().info(
            f"  sequence_queue_len: {self._params['sequence_queue_len']}"
        )
        self.get_logger().info(
            f"  skip_scan_interval: {self._params['skip_scan_interval']}"
        )

        self.get_logger().info('Grid resolution parameters:')
        self.get_logger().info(f"  fine_resolution: {self._params['fine_resolution']}")
        self.get_logger().info(
            f"  coarse_resolution: {self._params['coarse_resolution']}"
        )

        self.get_logger().info('Publish topic parameters:')
        self.get_logger().info(f"  map_topic: {self._params['map_topic']}")
        self.get_logger().info(f"  pub_odom_topic: {self._params['pub_odom_topic']}")

        self.get_logger().info('Transform parameters:')
        self.get_logger().info(f"  base_link_name: {self._params['base_link_name']}")
        self.get_logger().info(f"  map_frame_name: {self._params['map_frame_name']}")
        self.get_logger().info(f"  odom_frame_name: {self._params['odom_frame_name']}")

        self.get_logger().info('Publish parameters:')
        self.get_logger().info(
            f"  publish_base_to_map_transform: {self._params['publish_base_to_map_transform']}"
        )
        self.get_logger().info(f"  publish_map: {self._params['publish_map']}")

        self.get_logger().info('Subscribe topic parameters:')
        self.get_logger().info(f"  odom_topic: {self._params['odom_topic']}")
        self.get_logger().info(f"  lidar_topic: {self._params['lidar_topic']}")

        self.get_logger().info('Subscribe topic type parameters:')
        self.get_logger().info(f"  lidar_type: {self._params['lidar_type']}")

        self.get_logger().info('Sequence scan matcher parameters:')
        self.get_logger().info(
            f"  sequence_match_distance: {self._params['sequence_match_distance']}"
        )
        self.get_logger().info(
            f"  sequence_match_angle: {self._params['sequence_match_angle']}"
        )
        self.get_logger().info(
            f"  sequence_match_factor: {self._params['sequence_match_factor']}"
        )

        self.get_logger().info('Loop scan matcher parameters:')
        self.get_logger().info(
            f"  loop_match_distance: {self._params['loop_match_distance']}"
        )
        self.get_logger().info(
            f"  loop_match_angle: {self._params['loop_match_angle']}"
        )
        self.get_logger().info(
            f"  loop_match_factor: {self._params['loop_match_factor']}"
        )
        self.get_logger().info('==========================')

    def _create_subscribers(self):
        """Create subscribers for laser scan and odometry data."""
        # Laser scan subscriber
        if self._params['lidar_type'] == 'LaserScan':
            self._scan_subscriber = self.create_subscription(
                LaserScan, self._params['lidar_topic'], self._laser_scan_callback, 10
            )
        else:  # MultiEchoLaserScan
            self._scan_subscriber = self.create_subscription(
                MultiEchoLaserScan,
                self._params['lidar_topic'],
                self._multi_echo_scan_callback,
                10,
            )

        # Odometry subscriber (only if enabled)
        if self._params['enable_odom']:
            self._odom_subscriber = self.create_subscription(
                Odometry, self._params['odom_topic'], self._odom_callback, 10
            )

        self.get_logger().info(
            f"Subscribed to laser scan topic: {self._params['lidar_topic']}"
        )
        if self._params['enable_odom']:
            self.get_logger().info(
                f"Subscribed to odometry topic: {self._params['odom_topic']}"
            )

    def _laser_scan_callback(self, msg: LaserScan):
        """
        Handle LaserScan messages.

        Parameters
        ----------
        msg : LaserScan
            LaserScan message.

        """
        with self._scan_lock:
            scan_xy = ros_utils.laser_to_cart(msg)
            self._latest_scan = scan_xy
            self._process_latest_data()

    def _multi_echo_scan_callback(self, msg: MultiEchoLaserScan):
        """
        Handle MultiEchoLaserScan messages.

        Parameters
        ----------
        msg : MultiEchoLaserScan
            MultiEchoLaserScan message.

        """
        with self._scan_lock:
            scan_xy = ros_utils.multi_echo_to_cart(msg)
            self._latest_scan = scan_xy
            self._process_latest_data()

    def _odom_callback(self, msg: Odometry):
        """
        Handle Odometry messages.

        Parameters
        ----------
        msg : Odometry
            Odometry message.

        """
        with self._odom_lock:
            self._latest_odom = msg

    def _process_latest_data(self):
        """Process the latest laser scan and odometry data."""
        if self._latest_scan is None:
            return

        self._scan_counter += 1
        if self._skip_scan_interval > 1 and (
            (self._scan_counter - 1) % self._skip_scan_interval != 0
        ):
            return

        # Process scan with latest odometry (if available)
        odom = self._latest_odom if self._params['enable_odom'] else None
        self._slam.process_scan(self._latest_scan, odom)

        # Update and publish current state
        current_pose = self._slam.current_pose
        map_grid = self._slam.occupancy_map
        self._publish_data(map_grid, current_pose)

    def _publish_transforms(self, current_pose):
        """
        Publish the transform from base_link to map frame.

        Parameters
        ----------
        current_pose : numpy.ndarray
            Current robot pose [x, y, theta] in meters and radians.

        """
        transform = ros_utils.create_base_to_map_transform(
            current_pose,
            self._params['map_frame_name'],
            self._params['base_link_name'],
            self.get_clock().now(),
        )
        # Broadcast the transform
        self._tf_broadcaster.sendTransform(transform)

    def _publish_data(self, grid: Grid, current_pose: np.ndarray):
        """
        Publish the current map, odometry, and trajectory data.

        Parameters
        ----------
        grid : Grid
            Grid object containing the occupancy grid map with values
            indicating free space, occupied space, and unknown areas,
            along with origin and resolution information.
        current_pose : numpy.ndarray
            Current robot pose [x, y, theta] in meters and radians.

        """
        stamp = self.get_clock().now()

        occ_grid = ros_utils.grid_to_occ_grid(
            grid,
            self._params['map_frame_name'],
            stamp,
        )
        self._map_publisher.publish(occ_grid)
        odom = Odometry()
        odom.header.stamp = stamp.to_msg()
        odom.header.frame_id = self._params['map_frame_name']
        odom.child_frame_id = self._params['base_link_name']
        odom.pose.pose.position.x = current_pose[0]
        odom.pose.pose.position.y = current_pose[1]
        z, w = ros_utils.theta_to_quaternion(current_pose[2])
        odom.pose.pose.orientation.z = z
        odom.pose.pose.orientation.w = w
        cov = np.eye(6) * 0.01
        odom.pose.covariance = cov.flatten().tolist()
        self._odom_publisher.publish(odom)
        trajectory_poses = ros_utils.poses_to_path(
            self._slam.poses,
            self._params['map_frame_name'],
            stamp,
        )
        self._trajectory_publisher.publish(trajectory_poses)

        edges_marker = ros_utils.graph_edges_to_marker(
            self._slam.get_graph_edges(),
            self._params['map_frame_name'],
            stamp,
        )
        self._edges_publisher.publish(edges_marker)

        # Publish base_link to map transform if enabled
        if self._params['publish_base_to_map_transform']:
            self._publish_transforms(current_pose)


def main(args=None):
    """
    Run the main entry point for the CSM SLAM online node.

    """
    rclpy.init(args=args)
    node = CSMSlamNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
