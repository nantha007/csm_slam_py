"""CSM SLAM offline processing node for ROS2 bag files.

This module provides an offline SLAM processing node that reads laser scan data
from ROS2 bag files and processes them using the CSM SLAM algorithm. It supports
both LaserScan and MultiEchoLaserScan message types and publishes the resulting
map, trajectory, and odometry data.

The node processes bag files sequentially and publishes results in real-time,
making it suitable for offline analysis and visualization of SLAM performance.

Author: Nantha Kumar Sunder
"""

import sys
import os

HOME_DIR = os.path.expanduser("~")
sys.path.append(f"{HOME_DIR}/anaconda3/envs/rospy/lib/python3.12/site-packages")

import os
import sys
from typing import Optional, Dict

import numpy as np
import matplotlib.pyplot as plt

import rclpy
from rclpy.node import Node
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message
from sensor_msgs.msg import LaserScan, MultiEchoLaserScan
import rosbag2_py
from nav_msgs.msg import OccupancyGrid, MapMetaData, Odometry, Path
from geometry_msgs.msg import PoseStamped, TransformStamped
from tf2_ros import TransformBroadcaster

# Ensure core modules with local imports (e.g., from math_utils import ...) resolve
CORE_DIR = os.path.join(os.path.dirname(__file__), "core")
if CORE_DIR not in sys.path:
    sys.path.insert(0, CORE_DIR)

from graph_slam import GraphSlam  # noqa: E402


class CSMSlamNode(Node):
    """CSM SLAM offline processing node for ROS2 bag files.

    This node processes laser scan data from ROS2 bag files using the CSM SLAM
    algorithm. It supports both LaserScan and MultiEchoLaserScan message types
    and publishes the resulting map, trajectory, and odometry data.

    The node reads bag files sequentially and processes scans in real-time,
    making it suitable for offline analysis and visualization of SLAM performance.
    """

    def __init__(self):
        """Initialize the CSM SLAM offline processing node.

        Sets up ROS2 parameters, initializes the SLAM algorithm, and creates
        publishers for map, odometry, and trajectory data.
        """
        # Initialize ROS2 node
        super().__init__("csm_slam_node")
        self._params = {}
        self._initialize()

        self._bag_path: str = (
            self.get_parameter("bag_path").get_parameter_value().string_value
        )

        if not self._bag_path:
            self.get_logger().error(
                "Parameter 'bag_path' is required. Use --ros-args -p bag_path:=/path/to/bag"
            )
            raise SystemExit(2)

        if not os.path.exists(self._bag_path):
            self.get_logger().error(f"Bag path does not exist: {self._bag_path}")
            raise SystemExit(2)

        self.get_logger().info(f"Reading bag: {self._bag_path}")

        # SLAM trajectory
        self._slam = GraphSlam(self.get_logger(), self._params)

        # Publisher
        self._map_publisher = self.create_publisher(
            OccupancyGrid, self._params["map_topic"], 10
        )
        self._odom_publisher = self.create_publisher(
            Odometry, self._params["pub_odom_topic"], 10
        )
        self._trajectory_publisher = self.create_publisher(Path, "/slam_trajectory", 10)
        
        # Transform broadcaster
        self._tf_broadcaster = TransformBroadcaster(self)

    def _initialize(self):
        """Initialize ROS2 parameters and SLAM algorithm.

        Declares all ROS2 parameters, builds the parameter dictionary,
        initializes the SLAM algorithm, and creates publishers.
        """
        # Parameters
        self.declare_parameter("bag_path", "")

        # SLAM parameters
        self.declare_parameter("enable_movement_threshold", True)
        self.declare_parameter("enable_odom", False)
        self.declare_parameter("enable_imu", False)
        self.declare_parameter("enable_loop_closure", True)
        self.declare_parameter("movement_threshold_distance", 0.2)
        self.declare_parameter("movement_threshold_angle", 15)
        self.declare_parameter("sequence_queue_len", 100)

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

        # Subscribe topic parameters
        self.declare_parameter("odom_topic", "/odom")
        self.declare_parameter("lidar_topic", "/lidar")
        self.declare_parameter("imu_topic", "/imu")

        # Subscribe topic type parameters
        self.declare_parameter("lidar_type", "LaserScan")

        # Sequence scan matcher parameters
        self.declare_parameter("sequence_match_distance", 0.3)
        self.declare_parameter("sequence_match_angle", 45)
        self.declare_parameter("sequence_match_factor", 10)

        # Loop scan matcher parameters
        self.declare_parameter("loop_match_distance", 4.0)
        self.declare_parameter("loop_match_angle", 45)
        self.declare_parameter("loop_match_factor", 10)

        # Slam parameters
        self.declare_parameter("submap_distance_threshold", 6.0)
        self.declare_parameter("loop_closure_search_distance", 10.0)
        self.declare_parameter("loop_closure_score_threshold", 85.0)

        self._params = {
            "enable_movement_threshold": self.get_parameter("enable_movement_threshold")
            .get_parameter_value()
            .bool_value,
            "enable_odom": self.get_parameter("enable_odom")
            .get_parameter_value()
            .bool_value,
            "enable_imu": self.get_parameter("enable_imu")
            .get_parameter_value()
            .bool_value,
            "enable_loop_closure": self.get_parameter("enable_loop_closure")
            .get_parameter_value()
            .bool_value,
            "movement_threshold_distance": self.get_parameter(
                "movement_threshold_distance"
            )
            .get_parameter_value()
            .double_value,
            "movement_threshold_angle": self.get_parameter("movement_threshold_angle")
            .get_parameter_value()
            .integer_value,
            "sequence_queue_len": self.get_parameter("sequence_queue_len")
            .get_parameter_value()
            .integer_value,
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
            "odom_topic": self.get_parameter("odom_topic")
            .get_parameter_value()
            .string_value,
            "lidar_topic": self.get_parameter("lidar_topic")
            .get_parameter_value()
            .string_value,
            "imu_topic": self.get_parameter("imu_topic")
            .get_parameter_value()
            .string_value,
            "lidar_type": self.get_parameter("lidar_type")
            .get_parameter_value()
            .string_value,
            "sequence_match_distance": self.get_parameter("sequence_match_distance")
            .get_parameter_value()
            .double_value,
            "sequence_match_angle": self.get_parameter("sequence_match_angle")
            .get_parameter_value()
            .integer_value,
            "sequence_match_factor": self.get_parameter("sequence_match_factor")
            .get_parameter_value()
            .integer_value,
            "loop_match_distance": self.get_parameter("loop_match_distance")
            .get_parameter_value()
            .double_value,
            "loop_match_angle": self.get_parameter("loop_match_angle")
            .get_parameter_value()
            .integer_value,
            "loop_match_factor": self.get_parameter("loop_match_factor")
            .get_parameter_value()
            .integer_value,
            "submap_distance_threshold": self.get_parameter("submap_distance_threshold")
            .get_parameter_value()
            .double_value,
            "loop_closure_search_distance": self.get_parameter(
                "loop_closure_search_distance"
            )
            .get_parameter_value()
            .double_value,
            "loop_closure_score_threshold": self.get_parameter(
                "loop_closure_score_threshold"
            )
            .get_parameter_value()
            .double_value,
        }

        # Get and log all parameters
        self.get_logger().info("=== CSM SLAM Parameters ===")
        self.get_logger().info(f"SLAM parameters:")
        self.get_logger().info(
            f"  enable_movement_threshold: {self._params['enable_movement_threshold']}"
        )
        self.get_logger().info(f"  enable_odom: {self._params['enable_odom']}")
        self.get_logger().info(f"  enable_imu: {self._params['enable_imu']}")
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

        self.get_logger().info(f"Grid resolution parameters:")
        self.get_logger().info(f"  fine_resolution: {self._params['fine_resolution']}")
        self.get_logger().info(
            f"  coarse_resolution: {self._params['coarse_resolution']}"
        )

        self.get_logger().info(f"Publish topic parameters:")
        self.get_logger().info(f"  map_topic: {self._params['map_topic']}")
        self.get_logger().info(f"  pub_odom_topic: {self._params['pub_odom_topic']}")

        self.get_logger().info(f"Transform parameters:")
        self.get_logger().info(f"  base_link_name: {self._params['base_link_name']}")
        self.get_logger().info(f"  map_frame_name: {self._params['map_frame_name']}")
        self.get_logger().info(f"  odom_frame_name: {self._params['odom_frame_name']}")

        self.get_logger().info(f"Publish parameters:")
        self.get_logger().info(
            f"  publish_base_to_map_transform: {self._params['publish_base_to_map_transform']}"
        )
        self.get_logger().info(f"  publish_map: {self._params['publish_map']}")

        self.get_logger().info(f"Subscribe topic parameters:")
        self.get_logger().info(f"  odom_topic: {self._params['odom_topic']}")
        self.get_logger().info(f"  lidar_topic: {self._params['lidar_topic']}")
        self.get_logger().info(f"  imu_topic: {self._params['imu_topic']}")

        self.get_logger().info(f"Subscribe topic type parameters:")
        self.get_logger().info(f"  lidar_type: {self._params['lidar_type']}")

        self.get_logger().info(f"Sequence scan matcher parameters:")
        self.get_logger().info(
            f"  sequence_match_distance: {self._params['sequence_match_distance']}"
        )
        self.get_logger().info(
            f"  sequence_match_angle: {self._params['sequence_match_angle']}"
        )
        self.get_logger().info(
            f"  sequence_match_factor: {self._params['sequence_match_factor']}"
        )

        self.get_logger().info(f"Loop scan matcher parameters:")
        self.get_logger().info(
            f"  loop_match_distance: {self._params['loop_match_distance']}"
        )
        self.get_logger().info(
            f"  loop_match_angle: {self._params['loop_match_angle']}"
        )
        self.get_logger().info(
            f"  loop_match_factor: {self._params['loop_match_factor']}"
        )
        self.get_logger().info("==========================")

    def run(self):
        """Process the ROS2 bag file and run SLAM algorithm.

        Opens the bag file, processes all laser scan messages sequentially,
        and publishes the resulting map, trajectory, and odometry data.
        """
        reader, topic_type_map, type_class_map = self._open_bag(self._bag_path)

        # Stream all messages
        num_scans = 0
        while reader.has_next():
            topic, serialized, _ = reader.read_next()
            typ = topic_type_map.get(topic)
            if typ is None:
                continue
            if typ not in (
                "sensor_msgs/msg/LaserScan",
                "sensor_msgs/msg/MultiEchoLaserScan",
                "nav_msgs/msg/Odometry",
            ):
                continue

            msg_cls = type_class_map[typ]
            msg = deserialize_message(serialized, msg_cls)

            scan_xy = None
            if topic == self._params["lidar_topic"]:
                if self._params["lidar_type"] == "LaserScan":
                    scan_xy = self._laser_to_cart(msg)
                    # uncomment this only for revo dataset from cartographer project
                    scan_xy = self._filter_scan(scan_xy)
                else:
                    scan_xy = self._multi_echo_to_cart(msg)

            odom = None
            if topic == self._params["odom_topic"] and self._params["enable_odom"]:
                odom = msg

            if scan_xy is None:
                continue

            # Process without odometry; Trajectory internally does keyframe checks
            self._slam.process_scan(scan_xy, odom)
            current_pose = self._slam.current_pose
            current_map, origin = self._slam.map
            self.publish_data(current_map, origin, current_pose)
            num_scans += 1

        self.get_logger().info(f"Finished processing {num_scans} scans")
        plt.show(block=True)

    def np_to_occ_grid(self, current_map, origin):
        """Convert numpy array to ROS2 OccupancyGrid message.

        Converts a numpy occupancy grid array to ROS2 OccupancyGrid message
        format with proper metadata and coordinate system.

        Parameters
        ----------
        current_map : numpy.ndarray
            2D occupancy grid array.
        origin : numpy.ndarray
            Origin coordinates [x, y] of the grid.

        Returns
        -------
        OccupancyGrid
            ROS2 OccupancyGrid message.
        """
        meta_data = MapMetaData()
        meta_data.resolution = self._params["fine_resolution"]
        meta_data.width = current_map.shape[1]
        meta_data.height = current_map.shape[0]
        meta_data.origin.position.x = origin[0]
        meta_data.origin.position.y = origin[1]
        meta_data.origin.orientation.w = 1.0

        occ_grid = OccupancyGrid()
        occ_grid.info = meta_data
        unknown = current_map == 125
        occupied = current_map == 0
        free = current_map == 255
        current_map = current_map.astype(np.int8)
        current_map[unknown] = -1
        current_map[occupied] = 100
        current_map[free] = 0
        grid_ros = np.flipud(current_map)
        occ_grid.data = grid_ros.flatten(order="C").tolist()
        occ_grid.header.stamp = self.get_clock().now().to_msg()
        occ_grid.header.frame_id = self._params["map_frame_name"]
        return occ_grid

    def theta_to_quaternion(self, theta):
        """Convert angle to quaternion representation.

        Converts a 2D rotation angle to quaternion representation
        for ROS2 message compatibility.

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

    def poses_to_path(self, poses):
        """Convert poses to ROS2 Path message.

        Converts a numpy array of poses to ROS2 Path message format.

        Parameters
        ----------
        poses : numpy.ndarray
            3xN array of poses [x, y, theta] in meters and radians.

        Returns
        -------
        Path
            ROS2 Path message.
        """
        path = Path()
        path.header.stamp = self.get_clock().now().to_msg()
        path.header.frame_id = self._params["map_frame_name"]
        for i in range(poses.shape[1]):
            z, w = self.theta_to_quaternion(poses[2, i])
            pose = PoseStamped()
            pose.header.stamp = self.get_clock().now().to_msg()
            pose.header.frame_id = self._params["map_frame_name"]
            pose.pose.position.x = poses[0, i]
            pose.pose.position.y = poses[1, i]
            pose.pose.position.z = 0.0
            z, w = self.theta_to_quaternion(poses[2, i])
            pose.pose.orientation.z = z
            pose.pose.orientation.w = w
            path.poses.append(pose)
        return path

    def _publish_base_to_map_transform(self, current_pose):
        """Publish the transform from base_link to map frame.
        
        Publishes the transform from base_link to map frame based on the
        current robot pose estimated by SLAM.
        
        Parameters
        ----------
        current_pose : numpy.ndarray
            Current robot pose [x, y, theta] in meters and radians.
        """
        transform = TransformStamped()
        transform.header.stamp = self.get_clock().now().to_msg()
        transform.header.frame_id = self._params["map_frame_name"]
        transform.child_frame_id = self._params["base_link_name"]
        
        # Set translation
        transform.transform.translation.x = current_pose[0]
        transform.transform.translation.y = current_pose[1]
        transform.transform.translation.z = 0.0
        
        # Set rotation (convert theta to quaternion)
        z, w = self.theta_to_quaternion(current_pose[2])
        transform.transform.rotation.x = 0.0
        transform.transform.rotation.y = 0.0
        transform.transform.rotation.z = z
        transform.transform.rotation.w = w
        
        # Broadcast the transform
        self._tf_broadcaster.sendTransform(transform)

    def publish_data(self, current_map, origin, current_pose):
        """Publish the current map, odometry, and trajectory data.

        Publishes the current map, odometry, and trajectory data to the ROS2 topics.

        Parameters
        ----------
        current_map : numpy.ndarray
            2D occupancy grid array.
        origin : numpy.ndarray
            Origin coordinates [x, y] of the grid.
        current_pose : numpy.ndarray
            Current robot pose [x, y, theta] in meters and radians.
        """
        occ_grid = self.np_to_occ_grid(current_map, origin)
        self._map_publisher.publish(occ_grid)
        odom = Odometry()
        odom.header.stamp = self.get_clock().now().to_msg()
        odom.header.frame_id = self._params["map_frame_name"]
        odom.child_frame_id = self._params["base_link_name"]
        odom.pose.pose.position.x = current_pose[0]
        odom.pose.pose.position.y = current_pose[1]
        z, w = self.theta_to_quaternion(current_pose[2])
        odom.pose.pose.orientation.z = z
        odom.pose.pose.orientation.w = w
        cov = np.eye(6) * 0.01
        odom.pose.covariance = cov.flatten().tolist()
        self._odom_publisher.publish(odom)
        trajectory_poses = self.poses_to_path(self._slam.poses)
        self._trajectory_publisher.publish(trajectory_poses)
        
        # Publish base_link to map transform if enabled
        if self._params["publish_base_to_map_transform"]:
            self._publish_base_to_map_transform(current_pose)

    def _open_bag(self, bag_path: str):
        """Open the ROS2 bag file and return the reader, topic type map, and type class map.

        Opens the ROS2 bag file and returns the reader, topic type map, and type class map.

        Parameters
        ----------
        bag_path : str
            Path to the ROS2 bag file to open.

        Returns
        -------
        reader : SequentialReader
            ROS2 SequentialReader object.
        topic_type_map : dict
            Dictionary mapping topic names to their types.
        type_class_map : dict
            Dictionary mapping message types to their classes.
        """
        reader = rosbag2_py.SequentialReader()
        storage_options = rosbag2_py.StorageOptions(uri=bag_path, storage_id="sqlite3")
        converter_options = rosbag2_py.ConverterOptions(
            input_serialization_format="cdr",
            output_serialization_format="cdr",
        )
        reader.open(storage_options, converter_options)

        # Build topic->type and type->class maps
        topic_type_map: Dict[str, str] = {}
        type_class_map: Dict[str, type] = {}
        for t in reader.get_all_topics_and_types():
            topic_type_map[t.name] = t.type
            # Lazily get class only for supported types
            if t.type in (
                "sensor_msgs/msg/LaserScan",
                "sensor_msgs/msg/MultiEchoLaserScan",
            ):
                try:
                    type_class_map[t.type] = get_message(t.type)
                except Exception:
                    pass

        if not any(
            ty in ("sensor_msgs/msg/LaserScan", "sensor_msgs/msg/MultiEchoLaserScan")
            for ty in topic_type_map.values()
        ):
            self.get_logger().error(
                "No LaserScan or MultiEchoLaserScan topics found in bag"
            )
            raise SystemExit(2)

        return reader, topic_type_map, type_class_map

    def _laser_to_cart(self, msg) -> Optional[np.ndarray]:
        """Convert LaserScan message to Cartesian coordinates.

        Converts a LaserScan message to 2D Cartesian coordinates,
        filtering out invalid ranges and applying range limits.

        Parameters
        ----------
        msg : LaserScan
            ROS2 LaserScan message.

        Returns
        -------
        Optional[numpy.ndarray]
            2xN array of Cartesian coordinates, or None if no valid points.
        """
        # Extract ranges and filter
        ranges = np.array(msg.ranges, dtype=np.float32)
        n = ranges.shape[0]
        if n == 0:
            return None

        angles = msg.angle_min + np.arange(n, dtype=np.float32) * msg.angle_increment

        # Validity mask
        rmin = max(0.05, float(getattr(msg, "range_min", 0.0)))
        rmax = float(getattr(msg, "range_max", 20.0))
        mask = np.isfinite(ranges)
        mask &= ranges >= rmin
        mask &= ranges <= min(rmax, 20.0)

        if not np.any(mask):
            return None

        ranges = ranges[mask]
        angles = angles[mask]

        xs = ranges * np.cos(angles)
        ys = ranges * np.sin(angles)
        return np.vstack((xs, ys)).astype(np.float32)

    def _multi_echo_to_cart(self, msg) -> Optional[np.ndarray]:
        """Convert MultiEchoLaserScan message to Cartesian coordinates.

        Converts a MultiEchoLaserScan message to 2D Cartesian coordinates
        by taking the minimum valid echo per beam and applying range filtering.

        Parameters
        ----------
        msg : MultiEchoLaserScan
            ROS2 MultiEchoLaserScan message.

        Returns
        -------
        Optional[numpy.ndarray]
            2xN array of Cartesian coordinates, or None if no valid points.
        """
        # Convert MultiEchoLaserScan to single-echo ranges by taking the minimum valid echo per beam
        try:
            num = len(msg.ranges)
        except Exception:
            return None
        if num == 0:
            return None

        # Build ranges array
        ranges_list = []
        for i in range(num):
            echoes = getattr(msg.ranges[i], "echoes", [])
            if len(echoes) == 0:
                ranges_list.append(np.nan)
                continue
            arr = np.array(echoes, dtype=np.float32)
            arr = arr[np.isfinite(arr)]
            arr = arr[arr > 0.0]
            if arr.size == 0:
                ranges_list.append(np.nan)
            else:
                ranges_list.append(float(np.min(arr)))

        ranges = np.array(ranges_list, dtype=np.float32)
        angles = msg.angle_min + np.arange(num, dtype=np.float32) * msg.angle_increment

        rmin = max(0.05, float(getattr(msg, "range_min", 0.0)))
        rmax = float(getattr(msg, "range_max", 20.0))
        mask = np.isfinite(ranges)
        mask &= ranges >= rmin
        mask &= ranges <= min(rmax, 20.0)

        if not np.any(mask):
            return None

        ranges = ranges[mask]
        angles = angles[mask]

        xs = ranges * np.cos(angles)
        ys = ranges * np.sin(angles)
        return np.vstack((xs, ys)).astype(np.float32)

    def _filter_scan(self, scan_xy: np.ndarray) -> np.ndarray:
        """Filter scan points to remove robot body and close obstacles.

        Removes scan points that are likely from the robot body or very
        close obstacles that could interfere with SLAM processing.

        Parameters
        ----------
        scan_xy : numpy.ndarray
            2xN array of scan points in Cartesian coordinates.

        Returns
        -------
        numpy.ndarray
            Filtered 2xN array of scan points.
        """
        if scan_xy.size == 0:
            return scan_xy

        xs = scan_xy[0]
        ys = scan_xy[1]

        radii = np.hypot(xs, ys)
        # Angle of each point relative to robot forward (x-axis)
        angles = np.arctan2(ys, xs)
        # Smallest absolute difference to the backward direction (pi radians)
        angle_diff = np.abs((angles - np.pi + np.pi) % (2 * np.pi) - np.pi)

        mask = ~((radii <= 2.0) & (angle_diff <= np.deg2rad(60.0)))
        return scan_xy[:, mask]


def main(args=None):
    """Main entry point for the CSM SLAM offline node.

    Initializes ROS2, creates the SLAM node, runs the bag processing,
    and handles cleanup.
    """
    rclpy.init(args=args)
    node = None
    try:
        node = CSMSlamNode()
        node.run()
    finally:
        if node is not None:
            node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
