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
Graph-SLAM pipeline orchestrating scan processing and optimization.

This module implements a complete Graph-SLAM system that orchestrates
scan acquisition, scan matching, submap management, and pose graph
optimization to produce a globally consistent trajectory and map.

The GraphSlam class serves as the main coordinator, managing the
interaction between various components including scan matchers,
submap creation, loop closure detection, and graph optimization.

Author: Nantha Kumar Sunder
"""

import os
import time
from collections import deque

import numpy as np

from csm_slam.backend.graph import Graph
from csm_slam.core.loop_closure import LoopClosure
from csm_slam.io.pg_export import save_pose_graph_hdf5
from csm_slam.mapping.grid import create_occupancy_grid, MultiResolutionGrid
from csm_slam.sensors.localized_scan import LocalizedScan
from csm_slam.utils.math_utils import (
    get_relative_pose,
    movement_threshold,
)
from csm_slam.backend.optimizer import Optimizer
from csm_slam.frontend.scan_matcher import ScanMatcher
from csm_slam.mapping.submap import Submap


class GraphSlam:
    """
    Graph-SLAM algorithm implementation for 2D lidar mapping.

    This class implements a complete Graph-SLAM pipeline that processes
    laser scan data to build a globally consistent map and trajectory.
    It manages scan matching, submap creation, loop closure detection,
    and pose graph optimization.

    Attributes
    ----------
    current_pose : numpy.ndarray
        Current robot pose as [x, y, theta] in meters and radians.
    map : numpy.ndarray
        Occupancy grid map built from all localized scans.
    poses : numpy.ndarray
        3xN array containing poses for all localized scans.

    Parameters
    ----------
    logger : logging.Logger
        Logger instance for outputting debug and info messages.
    params : dict
        Dictionary containing algorithm parameters including
        resolution settings, thresholds, and matching parameters.

    """

    def __init__(self, logger, params):
        """
        Initialize the Graph-SLAM system.

        Sets up the pose graph, optimizer, and all necessary data
        structures for scan processing and map building.

        Parameters
        ----------
        logger : logging.Logger
            Logger instance for outputting messages.
        params : dict
            Configuration parameters including:
            - example config is in config/csm_slam_params.yaml

        """
        self._graph = Graph()
        self._optimizer = Optimizer()
        self._logger = logger
        self._params = params

        # Initialize data structures for trajectory and map building
        self._trajectory = {}
        self._localized_scans = {}
        self._submaps = {}

        # Initialize system state
        self._is_initialized = False

        # Current scan and submap tracking
        self._current_scan = None
        self._current_submap = None
        self._current_submap_id = 1
        self._scan_id = 1
        self._current_pose = np.array([0.0, 0.0, 0.0])
        self._last_odom_pose = np.array([0.0, 0.0, 0.0])
        # Movement threshold configuration for scan processing
        self._movement_threshold = np.array(
            [
                self._params['movement_threshold_distance'],
                np.deg2rad(self._params['movement_threshold_angle']),
            ]
        )
        # Algorithm configuration flags
        self._seq_running_scans_len = self._params['sequence_queue_len']
        self._enable_movement_threshold = True
        self._enable_odom = False
        self._enable_loop_closure = True

        # Initialize scan matchers for sequence and loop closure matching
        self._seq_matcher = ScanMatcher(
            self._params['coarse_resolution'],
            self._params['fine_resolution'],
            [
                self._params['sequence_match_distance'],
                self._params['sequence_match_distance'],
                np.deg2rad(self._params['sequence_match_angle']),
            ],
            smear_factor=self._params['sequence_match_factor'],
        )
        self._loop_matcher = ScanMatcher(
            self._params['coarse_resolution'],
            self._params['fine_resolution'],
            [
                self._params['loop_match_distance'],
                self._params['loop_match_distance'],
                np.deg2rad(self._params['loop_match_angle']),
            ],
            smear_factor=self._params['loop_match_factor'],
        )
        self._submap_distance_threshold = self._params['submap_distance_threshold']

        # Loop closure detection parameters
        self._loop_closure_search_distance = self._params[
            'loop_closure_search_distance'
        ]
        self._loop_closure_score_threshold = self._params[
            'loop_closure_score_threshold'
        ]
        # Performance optimization caches
        self._recent_scan_ids = deque(maxlen=self._seq_running_scans_len)

        # Optimization timing
        self._last_optimization_time = 0.0
        self._optimization_interval = self._params.get('optimization_interval', 5.0)

        # Loop-closure helper for search and matching
        self._loop_closure = LoopClosure(
            self._logger,
            self._loop_matcher,
            self._submaps,
            self._localized_scans,
            self._graph,
            self._loop_closure_search_distance,
            self._loop_closure_score_threshold,
        )

    @property
    def current_pose(self):
        """
        Return the current robot pose.

        Returns
        -------
        numpy.ndarray
            Current pose as [x, y, theta] in meters and radians.

        """
        return np.array(
            [self._current_pose[0], self._current_pose[1], self._current_pose[2]]
        )

    @property
    def occupancy_map(self):
        """
        Return an occupancy grid map.

        Creates and returns an occupancy grid map built from all
        localized scans using the fine resolution parameter.

        Returns
        -------
        Grid
            Grid object containing the occupancy grid map with values
            indicating free space, occupied space, and unknown areas,
            along with origin and resolution information.

        """
        scans = list(self._localized_scans.values())
        return create_occupancy_grid(scans, self._params['fine_resolution'])

    @property
    def poses(self):
        """
        Return poses for all localized scans.

        Returns
        -------
        numpy.ndarray
            3xN array where each column contains [x, y, theta]
            pose for a localized scan. Returns empty array if
            no scans are available.

        """
        poses = [
            np.array(
                [
                    self._localized_scans[scan_id].pose[0],
                    self._localized_scans[scan_id].pose[1],
                    self._localized_scans[scan_id].pose[2],
                ]
            )
            for scan_id in self._localized_scans.keys()
        ]
        if not poses:
            return np.empty((3, 0))
        poses_array = np.vstack(poses)
        return poses_array.T

    def get_graph_edges(self):
        """
        Return absolute pose pairs for every edge currently in the graph.

        Returns
        -------
        list of tuple
            Each entry is ``(from_pose, to_pose)`` where pose is a 3-vector
            ``[x, y, theta]`` describing the absolute pose of the vertex.

        """
        vertices = self._graph.get_vertices()
        edges = list(self._graph.get_edges().values())
        edge_pairs = []

        for edge in edges:
            from_vertex = vertices.get(edge.from_submap_id)
            to_vertex = vertices.get(edge.to_submap_id)
            if from_vertex is None or to_vertex is None:
                continue
            edge_pairs.append(
                (
                    np.array(from_vertex.pose, copy=True),
                    np.array(to_vertex.pose, copy=True),
                )
            )

        return edge_pairs

    def get_graph(self) -> Graph:
        """
        Return the underlying pose graph object.

        Returns
        -------
        Graph
            Graph containing all vertices and edges.

        """
        return self._graph

    def get_localized_scans(self):
        """
        Return a copy of the localized scans mapping keyed by scan_id.

        Returns
        -------
        dict
            Mapping of scan_id to LocalizedScan objects.

        """
        return dict(self._localized_scans)

    def export_pose_graph(
        self,
        output_path: str = '',
        bag_path: str | None = None,
        extra_meta: dict | None = None,
    ) -> str:
        """
        Export the pose graph and scans to an HDF5 `.pg` file.

        Parameters
        ----------
        output_path : str, optional
            Destination path. If empty, a default is chosen.
        bag_path : str, optional
            Bag path used to derive a default output name when output_path is empty.
        extra_meta : dict, optional
            Additional metadata to store in the file.

        Returns
        -------
        str
            Absolute path of the written file.

        """
        resolved_path = output_path
        if not resolved_path:
            if bag_path:
                bag_dir = os.path.dirname(bag_path)
                bag_base = os.path.splitext(os.path.basename(bag_path))[0]
                resolved_path = os.path.join(bag_dir, f'{bag_base}_pose_graph.pg')
            else:
                resolved_path = 'pose_graph.pg'
        elif not resolved_path.endswith('.pg'):
            resolved_path = f'{resolved_path}.pg'

        meta = {
            'map_frame': self._params.get('map_frame_name', 'map'),
            'base_link': self._params.get('base_link_name', 'base_link'),
        }
        if bag_path:
            meta['bag_path'] = bag_path
        if extra_meta:
            meta.update(extra_meta)

        try:
            graph = self.get_graph()
            scans = self.get_localized_scans().values()
            save_pose_graph_hdf5(graph, scans, resolved_path, meta=meta)
            self._logger.info(f'Saved pose graph to {resolved_path}')
        except Exception as exc:
            self._logger.error(f'Failed to save pose graph: {exc}')
            raise

        return os.path.abspath(resolved_path)

    def _check_movement_threshold(self, pose: np.ndarray):
        """
        Check if motion since last pose is below configured thresholds.

        Determines whether the robot has moved enough since the last
        pose to warrant processing a new scan.

        Parameters
        ----------
        pose : numpy.ndarray
            Current pose to check against the last odometry pose.

        Returns
        -------
        bool
            True if movement is below threshold (skip processing),
            False if movement is significant (process scan).

        """
        return movement_threshold(pose, self._last_odom_pose, self._movement_threshold)

    def _record_recent_scan(self, scan_id: int):
        """
        Record a scan ID in the recent scans queue.

        Adds the scan ID to the deque of recent scans.

        Parameters
        ----------
        scan_id : int
            ID of the scan to record.

        """
        self._recent_scan_ids.append(scan_id)

    def _get_recent_scan_ids(self):
        """
        Return the list of recent scan IDs for sequence grid building.

        Returns the most recent scan IDs up to the configured queue
        length. If no recent scans are recorded, falls back to the
        last N scans from all localized scans.

        Returns
        -------
        list
            List of scan IDs to use for sequence grid construction.

        """
        if self._recent_scan_ids:
            return list(self._recent_scan_ids)
        return list(self._localized_scans.keys())[-self._seq_running_scans_len:]

    def _get_sequence_grid(self):
        """
        Return a newly built multi-resolution grid for recent scans.

        Creates a multi-resolution occupancy grid from recent scans
        for use in scan matching.

        Returns
        -------
        MultiResolutionGrid or None
            Multi-resolution grid for recent scans, or None if
            no recent scans are available.

        """
        scan_ids = self._get_recent_scan_ids()
        if not scan_ids:
            return None
        scans = [self._localized_scans[scan_id] for scan_id in scan_ids]
        grid = MultiResolutionGrid(0.1, 0.05, scans)
        return grid

    def _check_new_submap(self, current_pose: np.ndarray):
        """
        Check if current pose is far enough to create a new submap.

        Determines whether the robot has moved far enough from the
        current submap's origin to warrant creating a new submap.

        Parameters
        ----------
        current_pose : numpy.ndarray
            Current robot pose [x, y, theta].

        Returns
        -------
        bool
            True if a new submap should be created, False otherwise.

        """
        submap_pose = self._submaps[self._current_submap_id].pose
        if (
            np.linalg.norm(current_pose[:2] - submap_pose[:2])
            > self._submap_distance_threshold
        ):
            return True
        return False

    def _optimize(self):
        """
        Optimize the pose graph and update scans and submaps.

        Performs pose graph optimization to correct accumulated drift
        and improve global consistency. Updates all poses based on
        the optimized graph vertices.

        """
        self._logger.info('Optimizing graph...')
        self._optimizer.optimize(self._graph)

        vertices = self._graph.get_vertices()

        for scan_id, loc_scan in self._localized_scans.items():
            if scan_id in vertices:
                loc_scan.update(vertices[scan_id].pose)

        for submap_id, submap in self._submaps.items():
            first_scan_id = submap.first_scan_id
            if first_scan_id in vertices:
                submap.pose = vertices[first_scan_id].pose
            elif submap_id in vertices:
                submap.pose = vertices[submap_id].pose

        last_scan_id = self._scan_id - 1
        if last_scan_id in vertices:
            self._current_pose = vertices[last_scan_id].pose

        self._logger.info('Graph optimization completed')

    def process_scan(self, scan: np.ndarray, odom_pose: np.ndarray = None):
        """
        Process a new scan, update the graph, and manage submaps.

        Main entry point for processing laser scan data. Handles
        initialization, scan matching, graph updates, submap management,
        and loop closure detection.

        Parameters
        ----------
        scan : numpy.ndarray
            2xN array of scan points in Cartesian coordinates.
        odom_pose : numpy.ndarray, optional
            Odometry pose [x, y, theta] in meters and radians.
            Used for movement threshold checking when enabled.

        """
        if not self._is_initialized:
            # Initialize system with first scan at origin
            self._is_initialized = True
            localized_scan = LocalizedScan(
                self._scan_id, np.array([0.0, 0.0, 0.0]), scan
            )
            self._current_scan = localized_scan.get_localized_scan()
            self._localized_scans[self._scan_id] = localized_scan

            # Create initial submap and add to graph
            self._current_submap = Submap(
                self._current_submap_id, np.array([0.0, 0.0, 0.0]), self._scan_id
            )
            self._graph.add_vertex(
                self._current_submap_id, np.array([0.0, 0.0, 0.0])
            )
            self._submaps[self._current_submap_id] = self._current_submap
            self._record_recent_scan(localized_scan.scan_id)

            # Increment scan counter and return after initialization
            self._scan_id += 1
            return

        # Perform scan matching against recent scans
        initial_pose = self._current_pose.copy()
        grid = self._get_sequence_grid()
        best_pose, _, _, seq_cov = self._seq_matcher.match(grid, scan, initial_pose)

        # Check if movement is significant enough to process scan
        if (
            self._enable_movement_threshold
            and not self._enable_odom
            and self._check_movement_threshold(best_pose)
        ):
            self._current_pose = best_pose
            return

        # Add scan to trajectory and update current pose
        self._logger.info(f'Processing scan {self._scan_id}')
        localized_scan = LocalizedScan(self._scan_id, best_pose, scan)
        self._current_scan = localized_scan.get_localized_scan()
        self._localized_scans[self._scan_id] = localized_scan
        self._record_recent_scan(self._scan_id)

        # Update pose tracking and add vertex to graph
        self._current_pose = best_pose
        self._last_odom_pose = odom_pose if self._enable_odom else best_pose

        # Add vertex to graph and create odometry edge
        self._graph.add_vertex(self._scan_id, self._current_pose)
        rel_pose = get_relative_pose(
            self._localized_scans[self._scan_id - 1].pose, self._current_pose
        )
        self._graph.add_edge(
            self._scan_id - 1,
            self._scan_id,
            rel_pose,
            seq_cov,
        )

        # Create new submap if distance threshold exceeded
        if self._check_new_submap(self._current_pose):
            # Run loop closure detection before creating new submap
            self.loop_closure()
            # Initialize new submap
            self._current_submap_id += 1
            self._current_submap = Submap(
                self._current_submap_id, self._current_pose, self._scan_id
            )
            self._submaps[self._current_submap_id] = self._current_submap

        else:
            # Add scan to current submap
            self._current_submap.add_scan_id(self._scan_id)
        self._scan_id += 1

    def loop_closure(self):
        """Run loop-closure detection asynchronously and optimize the graph."""
        if not self._enable_loop_closure:
            return

        submap_id = self._current_submap_id
        pose = np.array(self._current_pose, copy=True)
        self._loop_closure.add_submap_to_queue(pose, submap_id)
        
        # Optimize only if enough time has passed since last optimization
        current_time = time.time()
        if current_time - self._last_optimization_time >= self._optimization_interval:
            self._optimize()
            self._last_optimization_time = current_time

    def shutdown(self):
        """Clean stop background helpers such as loop-closure search."""
        self._loop_closure.shutdown()
