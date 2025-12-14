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
Utilities to export pose graphs, scans, and submaps to a MessagePack `.pg` file.

Author: Nantha Kumar Sunder
"""

from __future__ import annotations

import os
from typing import Optional

import msgpack
import numpy as np

from csm_slam.backend.graph import Graph
from csm_slam.sensors.localized_scan import LocalizedScan
from csm_slam.mapping.submap import Submap


def _collect_vertices(graph: Graph) -> tuple[np.ndarray, np.ndarray]:
    vertices = graph.get_vertices()
    vertex_ids = np.array([vid for vid in vertices.keys()], dtype=np.int64)
    poses = np.vstack([v.pose for v in vertices.values()]).astype(np.float64, copy=False)
    return vertex_ids, poses


def _collect_edges(graph: Graph) -> tuple[np.ndarray, np.ndarray, list]:
    edges = graph.get_edges()
    sorted_edges = sorted(edges.values(), key=lambda e: e.edge_id)
    ids = np.stack(
        [
            np.array([e.edge_id, e.from_submap_id, e.to_submap_id], dtype=np.int64)
            for e in sorted_edges
        ],
        axis=0,
    )
    rel_poses = np.stack([e.pose for e in sorted_edges], axis=0).astype(
        np.float64, copy=False
    )

    covs = []
    for e in sorted_edges:
        covs.append(np.array(e.cov, dtype=np.float64, copy=False))
    return ids, rel_poses, covs


def _iter_scans(scans: dict[int, LocalizedScan]):
    return sorted(scans.values(), key=lambda s: s.scan_id)


def _iter_submaps(submaps: dict[int, Submap]):
    """Normalize and sort submaps by submap_id."""
    return sorted(submaps.values(), key=lambda sm: sm.submap_id)


def save_pose_graph_msgpack(
    graph: Graph,
    scans: dict[int, LocalizedScan],
    submaps: dict[int, Submap],
    file_path: str,
    meta: Optional[dict] = None,
) -> None:
    """
    Save the pose graph and original scans into a MessagePack `.pg` file.

    Parameters
    ----------
    graph : Graph
        Pose graph containing vertices and edges.
    scans : dict[int, LocalizedScan]
        Collection of localized scans; each must expose `scan_id`, `pose`,
        and `get_original_scan()`.
    submaps : dict[int, Submap]
        Collection of submaps to store; each must expose `submap_id`,
        `first_scan_id`, `pose`, and `scan_ids`.
    meta : dict, optional
        Additional metadata to store in the file.
    file_path : str
        Destination file path (should end with `.pg`).

    """
    os.makedirs(os.path.dirname(file_path) or ".", exist_ok=True)

    vertex_ids, vertex_poses = _collect_vertices(graph)
    edge_ids, edge_rel_poses, edge_covs = _collect_edges(graph)
    sorted_scans = _iter_scans(scans)
    sorted_submaps = _iter_submaps(submaps)

    # Build MessagePack dictionary structure
    data = {
        "version": 1,
        "meta": {
            "num_vertices": int(vertex_ids.shape[0]),
            "num_edges": int(edge_ids.shape[0]),
            "num_scans": int(len(sorted_scans)),
            "num_submaps": int(len(sorted_submaps)),
        },
        "vertices": {
            "ids": vertex_ids.tolist(),
            "poses": vertex_poses.tolist(),
        },
        "edges": {
            "ids": edge_ids.tolist(),
            "relative_poses": edge_rel_poses.tolist(),
            "covariances": [
                cov.tolist() if cov is not None else None for cov in edge_covs
            ],
        },
        "scans": [
            {
                "scan_id": int(scan.scan_id),
                "vertex_id": int(scan.scan_id),
                "pose": np.array(scan.pose, dtype=np.float64, copy=False).tolist(),
                "scan_data": np.array(
                    scan.get_original_scan(), dtype=np.float32, copy=False
                ).tolist(),
            }
            for scan in sorted_scans
        ],
    }

    data["submaps"] = [
        {
            "submap_id": int(sm.submap_id),
            "first_scan_id": int(sm.first_scan_id),
            "pose": np.array(sm.pose, dtype=np.float64, copy=False).tolist(),
            "scan_ids": [int(sid) for sid in sm.scan_ids],
        }
        for sm in sorted_submaps
    ]

    # Add additional metadata if provided
    if meta:
        data["meta"].update(meta)

    # Serialize and write to file
    with open(file_path, "wb") as f:
        f.write(msgpack.packb(data))
