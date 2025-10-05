"""Graph-SLAM pipeline orchestrating scan processing and optimization.

This module implements a complete Graph-SLAM system that orchestrates
scan acquisition, scan matching, submap management, and pose graph
optimization to produce a globally consistent trajectory and map.

The GraphSlam class serves as the main coordinator, managing the
interaction between various components including scan matchers,
submap creation, loop closure detection, and graph optimization.

Author: Nantha Kumar Sunder
"""

import numpy as np
from scipy.spatial import cKDTree
from collections import deque

from math_utils import (
    get_relative_pose,
    movement_threshold,
)
from graph import Graph, Optimizer, EdgeType
from localized_scan import LocalizedScan
from scan_matcher import ScanMatcher
from submap import Submap
from grid import create_occupancy_grid, MultiResolutionGrid


class GraphSlam:
    """Graph-SLAM algorithm implementation for 2D lidar mapping.

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
        """Initialize the Graph-SLAM system.

        Sets up the pose graph, optimizer, and all necessary data
        structures for scan processing and map building.

        Parameters
        ----------
        logger : logging.Logger
            Logger instance for outputting messages.
        params : dict
            Configuration parameters including:
            - movement_threshold_distance: Minimum distance for processing
            - movement_threshold_angle: Minimum angle for processing
            - sequence_queue_len: Number of recent scans for matching
            - coarse_resolution: Coarse grid resolution for matching
            - fine_resolution: Fine grid resolution for matching
            - sequence_match_distance: Search distance for sequence matching
            - sequence_match_angle: Search angle for sequence matching
            - sequence_match_factor: Smear factor for sequence matching
            - loop_match_distance: Search distance for loop closure
            - loop_match_angle: Search angle for loop closure
            - loop_match_factor: Smear factor for loop closure
            - submap_distance_threshold: Distance threshold for new submaps
            - loop_closure_search_distance: Search radius for loop closure
            - loop_closure_score_threshold: Score threshold for loop closure
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
                self._params["movement_threshold_distance"],
                np.deg2rad(self._params["movement_threshold_angle"]),
            ]
        )
        # Algorithm configuration flags
        self._seq_running_scans_len = self._params["sequence_queue_len"]
        self._enable_movement_threshold = True
        self._enable_odom = False
        self._enable_loop_closure = True

        # Initialize scan matchers for sequence and loop closure matching
        self._seq_matcher = ScanMatcher(
            self._params["coarse_resolution"],
            self._params["fine_resolution"],
            [
                self._params["sequence_match_distance"],
                self._params["sequence_match_distance"],
                np.deg2rad(self._params["sequence_match_angle"]),
            ],
            smear_factor=self._params["sequence_match_factor"],
        )
        self._loop_matcher = ScanMatcher(
            self._params["coarse_resolution"],
            self._params["fine_resolution"],
            [
                self._params["loop_match_distance"],
                self._params["loop_match_distance"],
                np.deg2rad(self._params["loop_match_angle"]),
            ],
            smear_factor=self._params["loop_match_factor"],
        )
        self._submap_distance_threshold = self._params["submap_distance_threshold"]

        # Loop closure detection parameters
        self._loop_closure_search_distance = self._params[
            "loop_closure_search_distance"
        ]
        self._loop_closure_score_threshold = self._params[
            "loop_closure_score_threshold"
        ]

        # Performance optimization caches
        self._recent_scan_ids = deque(maxlen=self._seq_running_scans_len)
        self._seq_grid_cache = None
        self._submap_grids = {}
        self._submap_grid_dirty = set()
        self._submap_kd_tree = None
        self._submap_kd_tree_dirty = True
        self._submap_positions = None
        self._submap_ids = None

    @property
    def current_pose(self):
        """Return the current robot pose.

        Returns
        -------
        numpy.ndarray
            Current pose as [x, y, theta] in meters and radians.
        """
        return np.array(
            [self._current_pose[0], self._current_pose[1], self._current_pose[2]]
        )

    @property
    def map(self):
        """Return an occupancy grid map.

        Creates and returns an occupancy grid map built from all
        localized scans using the fine resolution parameter.

        Returns
        -------
        numpy.ndarray
            Occupancy grid map with values indicating free space,
            occupied space, and unknown areas.
        """
        scans = list(self._localized_scans.values())
        return create_occupancy_grid(scans, self._params["fine_resolution"])

    @property
    def poses(self):
        """Return poses for all localized scans.

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

    def _check_movement_threshold(self, pose: np.ndarray):
        """Check if motion since last pose is below configured thresholds.

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
        if pose is None:
            return True
        return movement_threshold(pose, self._last_odom_pose, self._movement_threshold)

    def _record_recent_scan(self, scan_id: int):
        """Record a scan ID in the recent scans queue.

        Adds the scan ID to the deque of recent scans and invalidates
        the sequence grid cache to force regeneration.

        Parameters
        ----------
        scan_id : int
            ID of the scan to record.
        """
        self._recent_scan_ids.append(scan_id)
        self._seq_grid_cache = None

    def _get_recent_scan_ids(self):
        """Return the list of recent scan IDs for sequence grid building.

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
        return list(self._localized_scans.keys())[-self._seq_running_scans_len :]

    def _get_sequence_grid(self):
        """Return a cached or newly built multi-resolution grid for recent scans.

        Creates a multi-resolution occupancy grid from recent scans
        for use in scan matching. Uses caching to avoid rebuilding
        the same grid multiple times.

        Returns
        -------
        MultiResolutionGrid or None
            Multi-resolution grid for recent scans, or None if
            no recent scans are available.
        """
        scan_ids = self._get_recent_scan_ids()
        if not scan_ids:
            return None
        cache_key = tuple(scan_ids)
        if self._seq_grid_cache and self._seq_grid_cache[0] == cache_key:
            return self._seq_grid_cache[1]
        scans = [self._localized_scans[scan_id] for scan_id in scan_ids]
        grid = MultiResolutionGrid(0.1, 0.05, scans)
        self._seq_grid_cache = (cache_key, grid)
        return grid

    def _mark_submap_grid_dirty(self, submap_id: int):
        """Mark the occupancy grids of a submap as outdated.

        Adds the submap ID to the dirty set, indicating that its
        cached occupancy grid needs to be regenerated.

        Parameters
        ----------
        submap_id : int
            ID of the submap whose grid cache should be invalidated.
        """
        self._submap_grid_dirty.add(submap_id)

    def _get_submap_grid(self, submap_id: int):
        """Return a cached or rebuilt grid for a given submap ID.

        Retrieves the occupancy grid for a submap, rebuilding it
        if the cache is marked as dirty.

        Parameters
        ----------
        submap_id : int
            ID of the submap whose grid to retrieve.

        Returns
        -------
        MultiResolutionGrid
            Multi-resolution occupancy grid for the specified submap.
        """
        if submap_id in self._submap_grids and submap_id not in self._submap_grid_dirty:
            return self._submap_grids[submap_id]
        scan_ids = self._submaps[submap_id].scan_ids
        scans = [self._localized_scans[scan_id] for scan_id in scan_ids]
        grid = MultiResolutionGrid(0.1, 0.05, scans)
        self._submap_grids[submap_id] = grid
        self._submap_grid_dirty.discard(submap_id)
        return grid

    def _ensure_submap_kd_tree(self):
        """Ensure KD-tree over submap positions exists and return it with data arrays.

        Creates or updates a KD-tree for efficient spatial search over
        submap positions. Used for loop closure detection.

        Returns
        -------
        tuple or None
            Tuple containing (kd_tree, positions, ids) if submaps exist,
            None if loop closure is disabled or no submaps available.

        Notes
        -----
        The KD-tree excludes the current submap to avoid self-matching
        during loop closure detection.
        """
        if not self._enable_loop_closure:
            return None
        if self._submap_kd_tree_dirty:
            positions = []
            ids = []
            for submap_id, submap in self._submaps.items():
                if submap_id == self._current_submap_id:
                    continue
                positions.append(submap.pose[:2])
                ids.append(submap_id)
            if positions:
                self._submap_positions = np.array(positions)
                self._submap_ids = np.array(ids)
                self._submap_kd_tree = cKDTree(self._submap_positions)
            else:
                self._submap_positions = None
                self._submap_ids = None
                self._submap_kd_tree = None
            self._submap_kd_tree_dirty = False
        if self._submap_kd_tree is None:
            return None
        return self._submap_kd_tree, self._submap_positions, self._submap_ids

    def _check_new_submap(self, current_pose: np.ndarray):
        """Check if current pose is far enough to create a new submap.

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
        """Optimize the pose graph and update scans, submaps, and caches.

        Performs pose graph optimization to correct accumulated drift
        and improve global consistency. Updates all poses based on
        the optimized graph vertices and marks caches as dirty.

        Notes
        -----
        This method updates:
        1. All localized scan poses from optimized graph vertices
        2. Submap poses using their first scan pose when available
        3. Current pose to the latest scan's optimized pose
        4. Marks all caches as dirty to force regeneration
        """
        self._logger.info("Optimizing graph...")
        self._graph = self._optimizer.optimize(self._graph)

        vertices = self._graph.get_vertices()

        # 1) Update all localized scan poses from optimized graph vertices
        for scan_id, loc_scan in self._localized_scans.items():
            if scan_id in vertices:
                loc_scan.update(vertices[scan_id].pose)

        # 2) Update submap poses using their first scan pose when available
        for submap_id, submap in self._submaps.items():
            first_scan_id = submap.first_scan_id
            if first_scan_id in vertices:
                submap.pose = vertices[first_scan_id].pose
            elif submap_id in vertices:
                submap.pose = vertices[submap_id].pose

        # 3) Set current pose to the latest scan's optimized pose if present
        last_scan_id = self._scan_id - 1
        if last_scan_id in vertices:
            self._current_pose = vertices[last_scan_id].pose

        # Mark caches as dirty after optimization adjusts poses
        self._submap_grid_dirty.update(self._submaps.keys())
        self._submap_kd_tree_dirty = True
        self._seq_grid_cache = None

        self._logger.info("Graph optimization completed")

    def process_scan(self, scan: np.ndarray, odom_pose: np.ndarray = None):
        """Process a new scan, update the graph, and manage submaps.

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

        Notes
        -----
        The processing pipeline:
        1. Initialize system with first scan if not already initialized
        2. Match scan against recent scans using sequence matcher
        3. Check movement threshold to skip processing if motion is small
        4. Add scan to graph with odometry edge
        5. Create new submap if distance threshold exceeded
        6. Perform loop closure detection if enabled
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
            self._graph.add_vertex(self._current_submap_id, np.array([0.0, 0.0, 0.0]))
            self._submaps[self._current_submap_id] = self._current_submap
            self._mark_submap_grid_dirty(self._current_submap_id)
            self._record_recent_scan(localized_scan.id)

            # Increment scan counter and return after initialization
            self._scan_id += 1
            return

        # Check if the movement is too small
        # Note: Movement threshold checking is currently disabled
        # if self._enable_movement_threshold and self._enable_odom and self.check_movement_threshold(odom_pose):
        #     return

        # Perform scan matching against recent scans
        initial_pose = self._current_pose.copy()
        grid = self._get_sequence_grid()
        best_pose, score, _, seq_cov = self._seq_matcher.match(grid, scan, initial_pose)

        # Check if movement is significant enough to process scan
        if (
            self._enable_movement_threshold
            and not self._enable_odom
            and self._check_movement_threshold(best_pose)
        ):
            self._current_pose = best_pose
            return

        # Add scan to trajectory and update current pose
        self._logger.info(f"Processing scan {self._scan_id}")
        # Add localized scan to the trajectory
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
            EdgeType.ODOM,
        )

        # Create new submap if distance threshold exceeded
        if self._check_new_submap(self._current_pose):
            # Perform loop closure detection before creating new submap
            if self._enable_loop_closure:
                self.loop_close()
            # Initialize new submap
            self._current_submap_id += 1
            self._current_submap = Submap(
                self._current_submap_id, self._current_pose, self._scan_id
            )
            self._submaps[self._current_submap_id] = self._current_submap
            self._mark_submap_grid_dirty(self._current_submap_id)
            self._submap_kd_tree_dirty = True

        else:
            # Add scan to current submap and mark grid as dirty
            self._current_submap.add_scan_id(self._scan_id)
        # Increment scan counter for next iteration
        self._scan_id += 1

    def loop_close(self):
        """Search for loop closures around the current submap and update graph.

        Performs loop closure detection by searching for nearby submaps
        and attempting to match scans from the current submap against
        them. Adds loop closure edges to the graph when successful
        matches are found.

        Notes
        -----
        Loop closure process:
        1. Use KD-tree to find nearby submaps efficiently
        2. For each nearby submap, try to match all scans in current submap
        3. Add loop closure edge if match score exceeds threshold
        4. Trigger graph optimization after adding loop closures

        The method automatically triggers graph optimization after
        processing all potential loop closures.
        """
        if len(self._submaps) <= 1:
            return

        # Use KD-tree for efficient spatial search over submap positions
        kd_data = self._ensure_submap_kd_tree()
        if kd_data is None:
            self._logger.info("No submaps available for KD-tree")
            return
        kd, _, submap_ids = kd_data
        current_xy = self._current_pose[:2]

        # Find nearby submaps within search radius
        nearby_indices = kd.query_ball_point(
            current_xy, r=self._loop_closure_search_distance
        )
        if len(nearby_indices) == 0:
            self._logger.info("No nearby submaps found")
            return
        else:
            self._logger.info(f"KD-tree submaps: {len(submap_ids)}")

        # For each nearby candidate submap, match every scan in current submap
        for idx in nearby_indices:
            candidate_submap_id = int(submap_ids[idx])
            grid = self._get_submap_grid(candidate_submap_id)
            # Target to connect: first scan of the candidate submap
            target_first_scan_id = self._submaps[candidate_submap_id].first_scan_id
            target_first_scan_pose = self._localized_scans[target_first_scan_id].pose

            # Iterate over all scans in current submap and attempt matching
            for scan_id in self._current_submap.scan_ids:
                original_scan = self._localized_scans[scan_id].get_original_scan()
                initial_pose = self._localized_scans[scan_id].pose

                # Attempt scan matching against candidate submap
                matched_pose, score, _, loop_cov = self._loop_matcher.match(
                    grid, original_scan, initial_pose
                )

                self._logger.info(
                    f"Loop try: cand_submap={candidate_submap_id} scan_id={scan_id} score={score}"
                )

                # Add loop closure edge if match score exceeds threshold
                if score > self._loop_closure_score_threshold:
                    # Calculate relative transform from matched scan to target scan
                    relative_pose = get_relative_pose(
                        matched_pose, target_first_scan_pose
                    )

                    # Add loop closure edge to graph
                    self._graph.add_edge(
                        scan_id,
                        target_first_scan_id,
                        relative_pose,
                        loop_cov,
                        EdgeType.LOOP,
                    )
        # Trigger graph optimization after adding loop closures
        self._optimize()
