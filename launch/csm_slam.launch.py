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
Launch file for CSM SLAM online node with rosbag playback.

Author: Nantha Kumar Sunder
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def launch_setup(context, *args, **kwargs):
    """Set up the launch configuration."""
    # Get configuration directory
    config_dir = os.path.join(get_package_share_directory('csm_slam'), 'config')
    config_file = os.path.join(config_dir, 'csm_slam_params.yaml')
    rviz_config = os.path.join(config_dir, 'rviz_config.rviz')

    # Launch configuration parameters
    bag_path = LaunchConfiguration('bag_path')

    # CSM SLAM Node (online node - no bag_path parameter needed)
    csm_slam_node = Node(
        package='csm_slam',
        executable='csm_slam_node',
        name='csm_slam_node',
        parameters=[config_file],
        output='log',
        emulate_tty=True,
    )

    # Rosbag2 Play Node
    rosbag_play_node = ExecuteProcess(
        cmd=['ros2', 'bag', 'play', bag_path], output='screen'
    )

    # RViz2 Node
    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        arguments=['-d', rviz_config],
        output='log',
    )

    return [csm_slam_node, rosbag_play_node, rviz_node]


def generate_launch_description():
    """Generate launch description."""
    return LaunchDescription(
        [
            # Declare launch arguments
            DeclareLaunchArgument(
                'bag_path',
                description='Path to the rosbag2 file to play',
                default_value='',
            ),
            DeclareLaunchArgument(
                'config_file',
                description='Path to config file (optional override)',
                default_value='',
            ),
            # Set up nodes
            OpaqueFunction(function=launch_setup),
        ]
    )
