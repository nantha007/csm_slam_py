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
Localization pipeline for scan-to-map matching.

This module implements a localization system that performs scan-to-map
matching to localize a robot within a pre-built map. It loads a pose graph
with scans and submaps, then matches incoming scans against the map to
determine the robot's pose.

Author: Nantha Kumar Sunder
"""

import numpy as np

from csm_slam.frontend.scan_matcher import ScanMatcher
from csm_slam.io.pg_import import load_pose_graph_msgpack
from csm_slam.mapping.grid import MultiResolutionGrid
from csm_slam.sensors.localized_scan import LocalizedScan


class Localization:
    """
    This class implements a simple Localization pipeline.

    It processes initial pose data and scan to map matching to localize the robot.

    Parameters
    ----------
    logger : logging.Logger
        Logger instance for outputting debug and info messages.
    params : dict
        Dictionary containing localization parameters including
        resolution settings, thresholds, and matching parameters.

    """

    def __init__(self, logger, params):
        self._logger = logger
        self._params = params

        # Initialize system state
        self._is_initialized = False
        self._is_initial_pose_set = False

        # Current pose tracking
        self._current_pose = np.array([0.0, 0.0, 0.0])

        # Initialize scan matcher for scan to map matching
        self._scan_matcher = ScanMatcher(
            self._params["coarse_resolution"],
            self._params["fine_resolution"],
            [
                self._params["search_distance"],
                self._params["search_distance"],
                np.deg2rad(self._params["search_angle"]),
            ],
            smear_factor=self._params["search_factor"],
        )
        self._grid = None

    #########################################################
    # Properties                                            #
    #########################################################

    @property
    def current_pose(self):
        """
        Return the current robot pose.

        Returns
        -------
        numpy.ndarray
            Current pose as [x, y, theta] in meters and radians.

        """
        return self._current_pose

    @property
    def occupancy_map(self):
        """
        Return an occupancy grid map.

        Creates and returns an occupancy grid map built from all
        scans in the loaded pose graph using the fine resolution parameter.

        Returns
        -------
        Grid or None
            Grid object containing the occupancy grid map with values
            indicating free space, occupied space, and unknown areas,
            along with origin and resolution information. Returns None
            if the pose graph has not been loaded yet.

        """
        if self._grid is None:
            return None
        return self._grid.fine_grid

    #########################################################
    # Public methods                                        #
    #########################################################

    def load_pose_graph(self, file_path: str):
        """
        Load a pose graph with scans from a MessagePack `.pg` file.

        Parameters
        ----------
        file_path : str
            Path to the pose graph file to load.

        Raises
        ------
        ValueError
            If the pose graph file did not contain scans.
        Exception
            If loading the pose graph fails.
        
        """
        try:
            _, scans, _, _ = load_pose_graph_msgpack(file_path)
            if scans is None:
                raise ValueError("Pose graph file did not contain scans")

            # Recreate LocalizedScan objects with correct resolutions
            # to ensure free space maps match the grid resolutions
            scans_list = []
            for sid in sorted(scans.keys()):
                scan = scans[sid]
                new_scan = LocalizedScan(
                    scan_id=scan.scan_id,
                    pose=scan.pose,
                    scan=scan.get_original_scan(),
                    coarse_resolution=self._params["coarse_resolution"],
                    fine_resolution=self._params["fine_resolution"],
                )
                scans_list.append(new_scan)
            
            self._grid = MultiResolutionGrid(
                self._params["coarse_resolution"],
                self._params["fine_resolution"],
                scans_list,
            )
            self._is_initialized = True
            self._logger.info(f"Loaded pose graph from {file_path}")
        except Exception as e:
            self._logger.error(f"Failed to load pose graph from {file_path}: {e}")
            raise e

    def set_initial_pose(self, initial_pose: np.ndarray):
        """
        Set the initial pose of the robot.

        Parameters
        ----------
        initial_pose : numpy.ndarray
            Initial pose [x, y, theta] in meters and radians.
        
        """
        self._current_pose = initial_pose
        self._is_initial_pose_set = True
        self._logger.info(f"Set initial pose to {initial_pose}")

    def process_scan(self, scan: np.ndarray):
        """
        Process a new scan, update the localization.

        Parameters
        ----------
        scan : numpy.ndarray
            2xN array of scan points in Cartesian coordinates.
        
        """
        if not self._is_initialized or not self._is_initial_pose_set:
            return

        # Perform scan matching against map
        best_pose, _, _, _ = self._scan_matcher.match(
            self._grid, scan, self._current_pose
        )
        self._current_pose = best_pose
