#!/usr/bin/env python3

"""
Launch file for CSM SLAM online node with rosbag playback.
This launch file loads configuration parameters, starts the CSM SLAM online node,
and plays a rosbag2 file to provide sensor data.
"""

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction, ExecuteProcess
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
            cmd=['ros2', 'bag', 'play', bag_path],
            output='screen'
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
    
    return LaunchDescription([
        # Declare launch arguments
        DeclareLaunchArgument(
            'bag_path',
            description='Path to the rosbag2 file to play',
            default_value=''
        ),
        
        DeclareLaunchArgument(
            'config_file',
            description='Path to config file (optional override)',
            default_value=''
        ),
        
        # Set up nodes
        OpaqueFunction(function=launch_setup)
    ])
