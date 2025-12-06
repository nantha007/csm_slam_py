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
Loop-closure helper for Graph-SLAM.

Encapsulates loop-closure search and edge creation so GraphSlam can
delegate this responsibility while keeping optimization separate.

Author: Nantha Kumar Sunder
"""

import numpy as np
import queue
import time
import threading
from scipy.spatial import cKDTree

from csm_slam.mapping.grid import MultiResolutionGrid
from csm_slam.utils.math_utils import get_relative_pose


class LoopClosure:
    """
    Manage loop-closure detection and edge creation for Graph-SLAM.

    Parameters
    ----------
    logger : logging.Logger
        Logger used for informational output.
    loop_matcher : ScanMatcher
        Scan matcher configured for loop-closure matching.
    submaps : dict[int, Submap]
        Mapping of submap id to Submap objects (shared with GraphSlam).
    localized_scans : dict[int, LocalizedScan]
        Mapping of scan id to LocalizedScan objects (shared with GraphSlam).
    graph : Graph
        Pose graph to which loop-closure edges are added.
    search_distance : float
        Maximum distance to search for candidate submaps.
    score_threshold : float
        Minimum matcher score required to accept a loop closure.

    """

    def __init__(
        self,
        logger,
        loop_matcher,
        submaps,
        localized_scans,
        graph,
        search_distance: float,
        score_threshold: float,
    ):
        self._logger = logger
        self._loop_matcher = loop_matcher
        self._submaps = submaps
        self._localized_scans = localized_scans
        self._graph = graph
        self._search_distance = search_distance
        self._score_threshold = score_threshold
        self._ok = True
        self._search_thread = threading.Thread(target=self._search_and_match_loop)
        self._submap_queue_lock = threading.Lock()
        self._submap_queue = queue.Queue()
        self._search_thread.start()

    def __del__(self):
        try:
            self.shutdown()
        except Exception:
            # Avoid raising during garbage collection
            pass

    #########################################################
    # Private methods                                       #
    #########################################################
    def _get_submap_grid(self, submap_id: int):
        """
        Return a newly built multi-resolution grid for a submap.

        Parameters
        ----------
        submap_id : int
            The ID of the submap to get the grid for.

        Returns
        -------
        MultiResolutionGrid
            The multi-resolution grid for the submap.

        """
        scan_ids = self._submaps[submap_id].scan_ids
        scans = [self._localized_scans[scan_id] for scan_id in scan_ids]
        grid = MultiResolutionGrid(0.1, 0.05, scans)
        return grid

    def _build_submap_kd_tree(self, current_submap_id: int):
        """
        Build and return KD-tree over submap positions with data arrays.

        Excludes the current submap from candidates to avoid self-matching.

        Parameters
        ----------
        current_submap_id : int
            The ID of the current submap.

        Returns
        -------
        tuple or None
            Tuple containing (kd_tree, positions, ids) if submaps exist,
            None if no submaps available.

        """
        positions = []
        ids = []
        for submap_id, submap in self._submaps.items():
            if submap_id == current_submap_id:
                continue
            positions.append(submap.pose[:2])
            ids.append(submap_id)

        if not positions:
            return None

        submap_positions = np.array(positions)
        submap_ids = np.array(ids)
        submap_kd_tree = cKDTree(submap_positions)
        return submap_kd_tree, submap_positions, submap_ids

    def _search_and_match_loop(self):
        """Search for loop closures around the current submap and add edges."""
        while self._ok:
            if self._submap_queue.empty():
                time.sleep(0.01)
                continue
            with self._submap_queue_lock:
                pose, submap_id = self._submap_queue.get(0)
            self._search_and_match(pose, submap_id)

    def _search_and_match(
        self, current_pose: np.ndarray, current_submap_id: int
    ) -> None:
        """
        Search for loop closures around the current submap and add edges.

        Parameters
        ----------
        current_pose : np.ndarray
            The pose of the current submap.
        current_submap_id : int
            The ID of the current submap.

        """
        if len(self._submaps) <= 1:
            return

        kd_data = self._build_submap_kd_tree(current_submap_id)
        if kd_data is None:
            self._logger.info("No submaps available for KD-tree")
            return

        kd, _, submap_ids = kd_data
        current_xy = current_pose[:2]
        nearby_indices = kd.query_ball_point(current_xy, r=self._search_distance)
        if len(nearby_indices) == 0:
            self._logger.info("No nearby submaps found")
            return
        else:
            self._logger.info(f"KD-tree submaps: {len(submap_ids)}")

        for idx in nearby_indices:
            candidate_submap_id = int(submap_ids[idx])
            grid = self._get_submap_grid(candidate_submap_id)
            target_first_scan_id = self._submaps[candidate_submap_id].first_scan_id
            target_first_scan_pose = self._localized_scans[target_first_scan_id].pose

            for scan_id in self._submaps[current_submap_id].scan_ids:
                original_scan = self._localized_scans[scan_id].get_original_scan()
                initial_pose = self._localized_scans[scan_id].pose

                matched_pose, score, _, loop_cov = self._loop_matcher.match(
                    grid, original_scan, initial_pose
                )

                self._logger.info(
                    f"Loop try: cand_submap={candidate_submap_id} "
                    f"scan_id={scan_id} score={score}"
                )

                if score > self._score_threshold:
                    relative_pose = get_relative_pose(
                        matched_pose, target_first_scan_pose
                    )
                    self._graph.add_edge(
                        scan_id,
                        target_first_scan_id,
                        relative_pose,
                        loop_cov,
                    )

    #########################################################
    # Public methods                                        #
    #########################################################

    def add_submap_to_queue(self, pose: np.ndarray, submap_id: int):
        """
        Add a submap to the queue for loop-closure detection.

        Parameters
        ----------
        pose : np.ndarray
            The pose of the submap.
        submap_id : int
            The ID of the submap.

        """
        with self._submap_queue_lock:
            self._submap_queue.put((pose, submap_id))

    def shutdown(self):
        """Stop the search thread and clean up resources."""
        self._ok = False
        with self._submap_queue_lock:
            self._submap_queue.put((None, None))
        if self._search_thread.is_alive():
            self._search_thread.join(timeout=1.0)
