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
CSM SLAM localization node for real-time localization.

This module provides a localization node that subscribes to laser scan
and initial pose data from ROS2 topics and processes them using the CSM
localization algorithm. It supports both LaserScan and MultiEchoLaserScan
message types and publishes the resulting map, odometry, and transform data
in real-time.

Author: Nantha Kumar Sunder
"""

import os
import sys
import threading
import time

import numpy as np
import yaml

import rclpy
from geometry_msgs.msg import PoseWithCovarianceStamped
from nav_msgs.msg import OccupancyGrid, Odometry
from rclpy.node import Node
from sensor_msgs.msg import LaserScan, MultiEchoLaserScan
from tf2_ros import TransformBroadcaster


def _load_settings():
    """Load settings from config/settings.yaml file."""
    from ament_index_python.packages import get_package_share_directory

    config_dir = os.path.join(get_package_share_directory("csm_slam"), "config")
    settings_file = os.path.join(config_dir, "venv_settings.yaml")

    if os.path.exists(settings_file):
        with open(settings_file, "r") as f:
            settings = yaml.safe_load(f)
            venv_path = settings.get("venv_path", "")
            if venv_path:
                venv_path = os.path.expanduser(venv_path)
                if os.path.exists(venv_path):
                    sys.path.append(venv_path)
    else:
        raise FileNotFoundError(f"Settings file not found: {settings_file}")


_load_settings()

from csm_slam.mapping.grid import Grid  # noqa: E402
from .core.localization import Localization  # noqa: E402
from .ros_utils import ros_utils  # noqa: E402


class CSMLocalizationNode(Node):
    """
    CSM SLAM localization node for real-time localization.

    This node subscribes to laser scan and initial pose data from ROS2 topics
    and processes them using the CSM localization algorithm. It supports both
    LaserScan and MultiEchoLaserScan message types and publishes the resulting
    map, odometry, and transform data in real-time.

    """

    def __init__(self):
        # Initialize ROS2 node
        super().__init__("csm_localization_node")
        self._params = {}
        self._initialize()

        # Localization system
        self._localization = Localization(self.get_logger(), self._params)
        # Cache occupancy grid once after localization is initialized to avoid
        # repeatedly recreating it on the publishing thread.
        self._map_grid = None

        # Load pose graph (required for localization)
        if self._params.get("pose_graph_input_path"):
            pg_path = self._params["pose_graph_input_path"]
            self.get_logger().info(f"Loading pose graph from {pg_path}")
            self._localization.load_pose_graph(pg_path)
            self.get_logger().info(f"Loaded pose graph from {pg_path}")
            self._map_grid = self._localization.occupancy_map
            if self._map_grid is None:
                self.get_logger().error("Failed to initialize occupancy grid")
                raise ValueError("Occupancy grid not available after loading pose graph")
        else:
            self.get_logger().error("Pose graph input path is not set")
            raise ValueError("Pose graph input path is not set")

        # Publishers
        self._map_publisher = self.create_publisher(
            OccupancyGrid, self._params["map_topic"], 10
        )
        self._odom_publisher = self.create_publisher(
            Odometry, self._params["pub_odom_topic"], 10
        )

        # Transform broadcaster
        self._tf_broadcaster = TransformBroadcaster(self)

        # Data storage for latest messages
        self._latest_scan = None
        self._scan_lock = threading.Lock()

        # Shutdown flag for graceful termination
        self._shutdown_requested = False

        # Publishing thread
        self._publish_thread = None
        self._publish_lock = threading.Lock()

        # Create subscribers
        self._create_subscribers()

        # Start publishing thread
        self._start_publish_thread()

    #########################################################
    # Private methods                                       #
    #########################################################

    def _initialize(self):
        """
        Initialize ROS2 parameters and localization algorithm.

        Declares all ROS2 parameters, builds the parameter dictionary,
        and initializes the localization algorithm.

        """
        # Grid resolution parameters
        self.declare_parameter("fine_resolution", 0.05)
        self.declare_parameter("coarse_resolution", 0.1)

        # Publish topic parameters
        self.declare_parameter("map_topic", "/map")
        self.declare_parameter("pub_odom_topic", "/slam_odom")

        # Transform parameters
        self.declare_parameter("base_link_name", "base_link")
        self.declare_parameter("map_frame_name", "map")
        self.declare_parameter("odom_frame_name", "odom")

        # Publish parameters
        self.declare_parameter("publish_base_to_map_transform", True)
        self.declare_parameter("publish_map", True)
        self.declare_parameter("publish_frequency", 10.0)

        # Subscribe topic parameters
        self.declare_parameter("lidar_topic", "/lidar")
        self.declare_parameter("initial_pose_topic", "initialpose")

        # Subscribe topic type parameters
        self.declare_parameter("lidar_type", "LaserScan")

        # Localization parameters
        self.declare_parameter("pose_graph_input_path", "")

        # Scan to map matching parameters for localization
        self.declare_parameter("search_distance", 0.2)
        self.declare_parameter("search_angle", 15)
        self.declare_parameter("search_factor", 5)

        self._params = {
            "fine_resolution": self.get_parameter("fine_resolution")
            .get_parameter_value()
            .double_value,
            "coarse_resolution": self.get_parameter("coarse_resolution")
            .get_parameter_value()
            .double_value,
            "map_topic": self.get_parameter("map_topic")
            .get_parameter_value()
            .string_value,
            "pub_odom_topic": self.get_parameter("pub_odom_topic")
            .get_parameter_value()
            .string_value,
            "base_link_name": self.get_parameter("base_link_name")
            .get_parameter_value()
            .string_value,
            "map_frame_name": self.get_parameter("map_frame_name")
            .get_parameter_value()
            .string_value,
            "odom_frame_name": self.get_parameter("odom_frame_name")
            .get_parameter_value()
            .string_value,
            "publish_base_to_map_transform": self.get_parameter(
                "publish_base_to_map_transform"
            )
            .get_parameter_value()
            .bool_value,
            "publish_map": self.get_parameter("publish_map")
            .get_parameter_value()
            .bool_value,
            "publish_frequency": self.get_parameter("publish_frequency")
            .get_parameter_value()
            .double_value,
            "lidar_topic": self.get_parameter("lidar_topic")
            .get_parameter_value()
            .string_value,
            "lidar_type": self.get_parameter("lidar_type")
            .get_parameter_value()
            .string_value,
            "initial_pose_topic": self.get_parameter("initial_pose_topic")
            .get_parameter_value()
            .string_value,
            "pose_graph_input_path": self.get_parameter("pose_graph_input_path")
            .get_parameter_value()
            .string_value,
            "search_distance": self.get_parameter("search_distance")
            .get_parameter_value()
            .double_value,
            "search_angle": self.get_parameter("search_angle")
            .get_parameter_value()
            .integer_value,
            "search_factor": self.get_parameter("search_factor")
            .get_parameter_value()
            .integer_value,
        }

        # Get and log all parameters
        self.get_logger().info("=== CSM Localization Parameters ===")
        self.get_logger().info("Grid resolution parameters:")
        self.get_logger().info(f"  fine_resolution: {self._params['fine_resolution']}")
        self.get_logger().info(
            f"  coarse_resolution: {self._params['coarse_resolution']}"
        )

        self.get_logger().info("Publish topic parameters:")
        self.get_logger().info(f"  map_topic: {self._params['map_topic']}")
        self.get_logger().info(f"  pub_odom_topic: {self._params['pub_odom_topic']}")

        self.get_logger().info("Transform parameters:")
        self.get_logger().info(f"  base_link_name: {self._params['base_link_name']}")
        self.get_logger().info(f"  map_frame_name: {self._params['map_frame_name']}")
        self.get_logger().info(f"  odom_frame_name: {self._params['odom_frame_name']}")

        self.get_logger().info("Publish parameters:")
        self.get_logger().info(
            f"  publish_base_to_map_transform: {self._params['publish_base_to_map_transform']}"
        )
        self.get_logger().info(f"  publish_map: {self._params['publish_map']}")
        self.get_logger().info(
            f"  publish_frequency: {self._params['publish_frequency']} Hz"
        )

        self.get_logger().info("Subscribe topic parameters:")
        self.get_logger().info(f"  lidar_topic: {self._params['lidar_topic']}")
        self.get_logger().info(
            f"  initial_pose_topic: {self._params['initial_pose_topic']}"
        )

        self.get_logger().info("Subscribe topic type parameters:")
        self.get_logger().info(f"  lidar_type: {self._params['lidar_type']}")

        self.get_logger().info("Localization parameters:")
        self.get_logger().info(
            f"  pose_graph_input_path: {self._params['pose_graph_input_path']}"
        )

        self.get_logger().info("Scan to map matching parameters:")
        self.get_logger().info(
            f"  search_distance: {self._params['search_distance']}"
        )
        self.get_logger().info(f"  search_angle: {self._params['search_angle']}")
        self.get_logger().info(f"  search_factor: {self._params['search_factor']}")
        self.get_logger().info("==========================")

    def _create_subscribers(self):
        """Create subscribers for laser scan and initial pose data."""
        # Laser scan subscriber
        if self._params["lidar_type"] == "LaserScan":
            self._scan_subscriber = self.create_subscription(
                LaserScan, self._params["lidar_topic"], self._laser_scan_callback, 10
            )
        else:  # MultiEchoLaserScan
            self._scan_subscriber = self.create_subscription(
                MultiEchoLaserScan,
                self._params["lidar_topic"],
                self._multi_echo_scan_callback,
                10,
            )

        # Initial pose subscriber
        self._initial_pose_subscriber = self.create_subscription(
            PoseWithCovarianceStamped,
            self._params["initial_pose_topic"],
            self._initial_pose_callback,
            10,
        )

        self.get_logger().info(
            f"Subscribed to laser scan topic: {self._params['lidar_topic']}"
        )
        self.get_logger().info(
            f"Subscribed to initial pose topic: {self._params['initial_pose_topic']}"
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

    def _initial_pose_callback(self, msg: PoseWithCovarianceStamped):
        """
        Handle initial pose messages.

        Parameters
        ----------
        msg : PoseWithCovarianceStamped
            PoseWithCovarianceStamped message.

        """
        self._localization.set_initial_pose(ros_utils.initial_pose_to_numpy(msg))

    def _process_latest_data(self):
        """Process the latest laser scan data."""
        if self._latest_scan is None:
            return

        # Process scan
        self._localization.process_scan(self._latest_scan)

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
            self._params["map_frame_name"],
            self._params["base_link_name"],
            self.get_clock().now(),
        )
        # Broadcast the transform
        self._tf_broadcaster.sendTransform(transform)

    def _publish_data(self, grid: Grid, current_pose: np.ndarray):
        """
        Publish the current map, odometry, and transform data.

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
            self._params["map_frame_name"],
            stamp,
        )
        self._map_publisher.publish(occ_grid)

        odom = Odometry()
        odom.header.stamp = stamp.to_msg()
        odom.header.frame_id = self._params["map_frame_name"]
        odom.child_frame_id = self._params["base_link_name"]
        odom.pose.pose.position.x = current_pose[0]
        odom.pose.pose.position.y = current_pose[1]
        z, w = ros_utils.theta_to_quaternion(current_pose[2])
        odom.pose.pose.orientation.z = z
        odom.pose.pose.orientation.w = w
        cov = np.eye(6) * 0.01
        odom.pose.covariance = cov.flatten().tolist()
        self._odom_publisher.publish(odom)

        # Publish base_link to map transform if enabled
        if self._params["publish_base_to_map_transform"]:
            self._publish_transforms(current_pose)

    def _start_publish_thread(self):
        """Start the publishing thread at the configured frequency."""
        publish_freq = self._params.get("publish_frequency", 10.0)
        self.get_logger().info(f"Starting publishing thread at {publish_freq} Hz")
        self._publish_thread = threading.Thread(
            target=self._publish_thread_loop, daemon=True
        )
        self._publish_thread.start()

    def _publish_thread_loop(self):
        """Publish thread loop that publishes data at a fixed frequency."""
        publish_freq = self._params.get("publish_frequency", 10.0)
        publish_period = 1.0 / publish_freq if publish_freq > 0 else 0.1

        while not self._shutdown_requested:
            try:
                # Get current state from localization
                with self._publish_lock:
                    current_pose = self._localization.current_pose
                self._publish_data(self._map_grid, current_pose)

            except Exception as exc:
                self.get_logger().error(
                    f"Error in publishing thread: {exc}", exc_info=True
                )

            # Sleep for the publish period
            time.sleep(publish_period)

    def destroy_node(self):
        """Clean up resources and stop publishing thread before destroying node."""
        self.get_logger().info("Shutting down publishing thread...")
        self._shutdown_requested = True
        if self._publish_thread is not None and self._publish_thread.is_alive():
            self._publish_thread.join(timeout=2.0)
            if self._publish_thread.is_alive():
                self.get_logger().warning(
                    "Publishing thread did not terminate gracefully"
                )
        super().destroy_node()


def main(args=None):
    """Run the main entry point for the CSM localization node."""
    rclpy.init(args=args)
    node = CSMLocalizationNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
