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
Utilities to export pose graphs and scans to an HDF5 `.pg` file.

The exported layout is intentionally simple and self-describing:
- /vertices/ids          : int64 [N]         vertex IDs
- /vertices/poses        : float64 [N,3]     [x, y, theta] per vertex
- /edges/ids             : int64 [E,3]       [edge_id, from_id, to_id]
- /edges/relative_poses  : float64 [E,3]     [dx, dy, dtheta] per edge
- /edges/covariance      : float64 [E,3,3]   covariance matrices; identity if missing
- /scans/scan_<id>       : float32 [2,N]     original scan points in sensor frame
                          attrs: scan_id (int), vertex_id (int), pose (float64[3])
- /meta (attrs)          : file_format, created_utc, plus any user metadata

This keeps the file easy to inspect and consume from other tools while retaining
all required information to reconstruct the trajectory and measurements.
"""

from __future__ import annotations

import datetime as _dt
import os
from typing import Iterable, Mapping, Optional, Sequence

import h5py
import numpy as np

from csm_slam.backend.graph import Graph
from csm_slam.sensors.localized_scan import LocalizedScan


def _collect_vertices(graph: Graph) -> tuple[np.ndarray, np.ndarray]:
    vertices = graph.get_vertices()
    if not vertices:
        return np.zeros((0,), dtype=np.int64), np.zeros((0, 3), dtype=np.float64)

    sorted_items = sorted(vertices.items(), key=lambda kv: kv[0])
    vertex_ids = np.array([vid for vid, _ in sorted_items], dtype=np.int64)
    poses = np.vstack([v.pose for _, v in sorted_items]).astype(np.float64, copy=False)
    return vertex_ids, poses


def _collect_edges(graph: Graph) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    edges = graph.get_edges()
    if not edges:
        return (
            np.zeros((0, 3), dtype=np.int64),
            np.zeros((0, 3), dtype=np.float64),
            np.zeros((0, 3, 3), dtype=np.float64),
        )

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
        if e.cov is None:
            covs.append(np.eye(3, dtype=np.float64))
        else:
            covs.append(np.array(e.cov, dtype=np.float64, copy=False))
    cov_stack = np.stack(covs, axis=0) if covs else np.zeros((0, 3, 3))
    return ids, rel_poses, cov_stack


def _iter_scans(scans: Sequence[LocalizedScan] | Mapping | Iterable[LocalizedScan]):
    if isinstance(scans, Mapping):
        scan_iter = scans.values()
    else:
        scan_iter = scans
    return sorted(scan_iter, key=lambda s: s.scan_id)


def save_pose_graph_hdf5(
    graph: Graph,
    scans: Sequence[LocalizedScan] | Mapping | Iterable[LocalizedScan],
    file_path: str,
    meta: Optional[dict] = None,
) -> None:
    """
    Save the pose graph and original scans into an HDF5 `.pg` file.

    Parameters
    ----------
    graph : Graph
        Pose graph containing vertices and edges.
    scans : Sequence[LocalizedScan] | Mapping | Iterable[LocalizedScan]
        Collection of localized scans; each must expose `scan_id`, `pose`,
        and `get_original_scan()`.
    file_path : str
        Destination file path (should end with `.pg`).
    meta : dict, optional
        Additional metadata to store as attributes under `/meta`.

    """
    os.makedirs(os.path.dirname(file_path) or ".", exist_ok=True)

    vertex_ids, vertex_poses = _collect_vertices(graph)
    edge_ids, edge_rel_poses, edge_covs = _collect_edges(graph)
    sorted_scans = _iter_scans(scans)

    with h5py.File(file_path, "w") as f:
        # Vertices
        vgrp = f.create_group("vertices")
        vgrp.create_dataset("ids", data=vertex_ids, dtype="i8")
        vgrp.create_dataset("poses", data=vertex_poses, dtype="f8")

        # Edges
        egrp = f.create_group("edges")
        egrp.create_dataset("ids", data=edge_ids, dtype="i8")
        egrp.create_dataset("relative_poses", data=edge_rel_poses, dtype="f8")
        egrp.create_dataset("covariance", data=edge_covs, dtype="f8")

        # Scans
        sgrp = f.create_group("scans")
        for scan in sorted_scans:
            dataset = sgrp.create_dataset(
                f"scan_{scan.scan_id}",
                data=np.array(scan.get_original_scan(), dtype=np.float32, copy=False),
                dtype="f4",
            )
            dataset.attrs["scan_id"] = int(scan.scan_id)
            dataset.attrs["vertex_id"] = int(scan.scan_id)
            dataset.attrs["pose"] = np.array(scan.pose, dtype=np.float64, copy=False)

        # Metadata
        mgrp = f.create_group("meta")
        mgrp.attrs["file_format"] = "csm_slam_pg_hdf5_v1"
        mgrp.attrs["created_utc"] = _dt.datetime.utcnow().isoformat() + "Z"
        mgrp.attrs["num_vertices"] = int(vertex_ids.shape[0])
        mgrp.attrs["num_edges"] = int(edge_ids.shape[0])
        mgrp.attrs["num_scans"] = int(len(sorted_scans))
        if meta:
            for key, value in meta.items():
                mgrp.attrs[key] = value

