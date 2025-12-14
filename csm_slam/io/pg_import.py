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
Utilities to import pose graphs, scans, and submaps from a MessagePack `.pg` file.

Author: Nantha Kumar Sunder
"""

from __future__ import annotations

import os

import msgpack
import numpy as np

from csm_slam.backend.graph import Graph
from csm_slam.sensors.localized_scan import LocalizedScan
from csm_slam.mapping.submap import Submap


def load_pose_graph_msgpack(
    file_path: str,
) -> tuple[
    Graph | None, dict[int, LocalizedScan] | None, dict[int, Submap] | None, dict
]:
    """
    Load a pose graph, scans, and submaps from a MessagePack `.pg` file.

    Parameters
    ----------
    file_path : str
        Path to the MessagePack `.pg` file to read.

    Returns
    -------
    tuple
        Tuple containing:
        - graph : Graph | None
            Reconstructed Graph object.
        - scans : dict[int, LocalizedScan] | None
            Dictionary mapping scan_id to LocalizedScan objects.
        - submaps : dict[int, Submap] | None
            Dictionary mapping submap_id to Submap objects, or None if
            no submaps stored.
        - meta : dict
            Dictionary containing metadata from the file.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.

    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Pose graph file not found: {file_path}")

    # Deserialize MessagePack file
    with open(file_path, "rb") as f:
        data = msgpack.unpackb(f.read(), raw=False)

    if (
        "vertices" not in data
        or "edges" not in data
        or "scans" not in data
        or "submaps" not in data
    ):
        raise ValueError("Pose graph file is missing required data")

    # Extract metadata
    meta = data.get("meta", {})

    # Reconstruct graph
    graph = Graph()

    # Load vertices
    vertices_data = data["vertices"]
    if "ids" in vertices_data and "poses" in vertices_data:
        vertex_ids = vertices_data["ids"]
        vertex_poses = vertices_data["poses"]

        for vid, pose in zip(vertex_ids, vertex_poses):
            graph.add_vertex(int(vid), np.array(pose, dtype=np.float64))

    # Load edges
    edges_data = data["edges"]
    if "ids" in edges_data and "relative_poses" in edges_data:
        edge_ids = edges_data["ids"]
        edge_rel_poses = edges_data["relative_poses"]
        edge_covs = edges_data.get("covariances", [])

        for i, (edge_id_row, rel_pose) in enumerate(zip(edge_ids, edge_rel_poses)):
            from_id = int(edge_id_row[1])
            to_id = int(edge_id_row[2])
            pose_array = np.array(rel_pose, dtype=np.float64)

            cov = None
            if i < len(edge_covs) and edge_covs[i] is not None:
                cov = np.array(edge_covs[i], dtype=np.float64)

            graph.add_edge(from_id, to_id, pose_array, cov)

    # Reconstruct scans
    scans = {}
    for scan_data in data["scans"]:
        scan_id = int(scan_data["scan_id"])
        vertex_id = int(scan_data.get("vertex_id", scan_id))
        pose_list = scan_data.get("pose", None)

        if pose_list is not None:
            pose = np.array(pose_list, dtype=np.float64)
        else:
            # Fallback: try to get pose from vertex if graph was loaded
            if vertex_id in graph.get_vertices():
                pose = graph.get_vertices()[vertex_id].pose
            else:
                # Default to origin if no pose available
                pose = np.array([0.0, 0.0, 0.0], dtype=np.float64)

        # Convert scan_data from list to numpy array
        scan_array = np.array(scan_data["scan_data"], dtype=np.float32)

        # Create LocalizedScan object
        # Note: We use default resolutions since they're not stored
        localized_scan = LocalizedScan(scan_id=scan_id, pose=pose, scan=scan_array)
        scans[scan_id] = localized_scan

    # Reconstruct submaps
    submaps = {}
    for sm_data in data["submaps"]:
        submap_id = int(sm_data["submap_id"])
        first_scan_id = int(sm_data["first_scan_id"])
        pose_list = sm_data.get("pose", None)

        if pose_list is not None:
            pose = np.array(pose_list, dtype=np.float64)
        else:
            # Fallback: use vertex pose if available
            if submap_id in graph.get_vertices():
                pose = graph.get_vertices()[submap_id].pose
            else:
                pose = np.array([0.0, 0.0, 0.0], dtype=np.float64)

        scan_ids = [int(sid) for sid in sm_data.get("scan_ids", [])]

        # Construct Submap
        submap = Submap(submap_id, pose, first_scan_id)
        for sid in scan_ids:
            if sid != first_scan_id:
                submap.add_scan_id(int(sid))

        submaps[submap_id] = submap

    return graph, scans, submaps, meta
