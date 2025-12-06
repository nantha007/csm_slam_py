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
Utilities to import pose graphs and scans from an HDF5 `.pg` file.

This module provides functions to read back the data structures exported
by `pg_export.py`. The file format is:
- /vertices/ids          : int64 [N]         vertex IDs
- /vertices/poses        : float64 [N,3]     [x, y, theta] per vertex
- /edges/ids             : int64 [E,3]       [edge_id, from_id, to_id]
- /edges/relative_poses  : float64 [E,3]     [dx, dy, dtheta] per edge
- /edges/covariance      : float64 [E,3,3]   covariance matrices
- /scans/scan_<id>       : float32 [2,N]     original scan points in sensor frame
                          attrs: scan_id (int), vertex_id (int), pose (float64[3])
- /meta (attrs)          : file_format, created_utc, plus any user metadata

Author: Nantha Kumar Sunder
"""

from __future__ import annotations

import os

import h5py
import numpy as np

from csm_slam.backend.graph import Graph
from csm_slam.sensors.localized_scan import LocalizedScan


def load_pose_graph_hdf5(
    file_path: str,
    reconstruct_graph: bool = True,
    reconstruct_scans: bool = True,
) -> tuple[Graph | None, dict[int, LocalizedScan] | None, dict]:
    """
    Load a pose graph and scans from an HDF5 `.pg` file.

    Parameters
    ----------
    file_path : str
        Path to the HDF5 `.pg` file to read.
    reconstruct_graph : bool, optional
        If True, reconstruct and return a Graph object. If False, returns None.
        Default is True.
    reconstruct_scans : bool, optional
        If True, reconstruct and return LocalizedScan objects. If False, returns None.
        Default is True.

    Returns
    -------
    tuple
        Tuple containing:
        - graph : Graph | None
            Reconstructed Graph object, or None if reconstruct_graph is False.
        - scans : dict[int, LocalizedScan] | None
            Dictionary mapping scan_id to LocalizedScan objects, or None if
            reconstruct_scans is False.
        - meta : dict
            Dictionary containing metadata from the file.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    ValueError
        If the file format is invalid or unsupported.

    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f'Pose graph file not found: {file_path}')

    graph = None
    scans = None

    with h5py.File(file_path, 'r') as f:
        # Validate file format
        if 'meta' not in f:
            raise ValueError('Invalid pose graph file: missing meta group')
        meta_attrs = f['meta'].attrs
        file_format = meta_attrs.get('file_format', '')
        if not file_format.startswith('csm_slam_pg_hdf5'):
            raise ValueError(
                f'Unsupported file format: {file_format}. '
                'Expected format starting with "csm_slam_pg_hdf5"'
            )

        # Load metadata
        meta = {}
        for key in meta_attrs.keys():
            value = meta_attrs[key]
            # Convert numpy types to Python types for JSON-like serialization
            if isinstance(value, (np.integer, np.floating)):
                meta[key] = value.item()
            elif isinstance(value, np.ndarray):
                meta[key] = value.tolist()
            elif isinstance(value, bytes):
                meta[key] = value.decode('utf-8')
            else:
                meta[key] = value

        # Reconstruct graph if requested
        if reconstruct_graph:
            graph = Graph()

            # Load vertices
            if 'vertices' in f:
                vgrp = f['vertices']
                if 'ids' in vgrp and 'poses' in vgrp:
                    vertex_ids = vgrp['ids'][...]
                    vertex_poses = vgrp['poses'][...]

                    for vid, pose in zip(vertex_ids, vertex_poses):
                        graph.add_vertex(int(vid), np.array(pose, dtype=np.float64))

            # Load edges
            if 'edges' in f:
                egrp = f['edges']
                if 'ids' in egrp and 'relative_poses' in egrp:
                    edge_ids = egrp['ids'][...]
                    edge_rel_poses = egrp['relative_poses'][...]
                    edge_covs = None

                    if 'covariance' in egrp:
                        edge_covs = egrp['covariance'][...]

                    for i, (edge_id_row, rel_pose) in enumerate(
                        zip(edge_ids, edge_rel_poses)
                    ):
                        from_id = int(edge_id_row[1])
                        to_id = int(edge_id_row[2])
                        pose_array = np.array(rel_pose, dtype=np.float64)

                        cov = None
                        if edge_covs is not None:
                            cov = np.array(edge_covs[i], dtype=np.float64)
                            # Check if it's an identity matrix (default from export)
                            if np.allclose(cov, np.eye(3)):
                                cov = None

                        graph.add_edge(from_id, to_id, pose_array, cov)

        # Reconstruct scans if requested
        if reconstruct_scans:
            scans = {}

            if 'scans' in f:
                sgrp = f['scans']
                for scan_name in sgrp.keys():
                    scan_dataset = sgrp[scan_name]
                    scan_data = np.array(scan_dataset[...], dtype=np.float32)

                    # Get attributes
                    scan_id = int(scan_dataset.attrs.get('scan_id', 0))
                    vertex_id = int(scan_dataset.attrs.get('vertex_id', scan_id))
                    pose_attr = scan_dataset.attrs.get('pose', None)

                    if pose_attr is not None:
                        pose = np.array(pose_attr, dtype=np.float64)
                    else:
                        # Fallback: try to get pose from vertex if graph was loaded
                        if graph is not None and vertex_id in graph.get_vertices():
                            pose = graph.get_vertices()[vertex_id].pose
                        else:
                            # Default to origin if no pose available
                            pose = np.array([0.0, 0.0, 0.0], dtype=np.float64)

                    # Create LocalizedScan object
                    # Note: We use default resolutions since they're not stored
                    localized_scan = LocalizedScan(
                        scan_id=scan_id, pose=pose, scan=scan_data
                    )
                    scans[scan_id] = localized_scan

    return graph, scans, meta


def load_pose_graph_metadata(file_path: str) -> dict:
    """
    Load only the metadata from an HDF5 `.pg` file.

    Loads metadata without reconstructing the full graph or scans.

    Parameters
    ----------
    file_path : str
        Path to the HDF5 `.pg` file to read.

    Returns
    -------
    dict
        Dictionary containing metadata from the file.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    ValueError
        If the file format is invalid.

    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f'Pose graph file not found: {file_path}')

    with h5py.File(file_path, 'r') as f:
        if 'meta' not in f:
            raise ValueError('Invalid pose graph file: missing meta group')

        meta_attrs = f['meta'].attrs
        meta = {}
        for key in meta_attrs.keys():
            value = meta_attrs[key]
            # Convert numpy types to Python types
            if isinstance(value, (np.integer, np.floating)):
                meta[key] = value.item()
            elif isinstance(value, np.ndarray):
                meta[key] = value.tolist()
            elif isinstance(value, bytes):
                meta[key] = value.decode('utf-8')
            else:
                meta[key] = value

    return meta


def list_scan_ids(file_path: str) -> list[int]:
    """
    List all scan IDs stored in the pose graph file.

    Parameters
    ----------
    file_path : str
        Path to the HDF5 `.pg` file to read.

    Returns
    -------
    list[int]
        Sorted list of scan IDs found in the file.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.

    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f'Pose graph file not found: {file_path}')

    scan_ids = []
    with h5py.File(file_path, 'r') as f:
        if 'scans' in f:
            sgrp = f['scans']
            for scan_name in sgrp.keys():
                scan_dataset = sgrp[scan_name]
                scan_id = int(scan_dataset.attrs.get('scan_id', 0))
                scan_ids.append(scan_id)

    return sorted(scan_ids)

