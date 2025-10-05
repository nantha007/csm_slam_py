"""Submap container used to group scans and maintain a pose estimate.

Author: Nantha Kumar Sunder
"""

import numpy as np


class Submap:
    """A submap groups one or more scans under a common 2D pose."""

    def __init__(self, id: int, pose: np.ndarray, origin_scan_id: int):
        self._id = id
        self._pose = pose
        self._scan_ids = [origin_scan_id]

    @property
    def id(self):
        """Return the unique identifier of the submap."""
        return self._id

    @property
    def pose(self):
        """Return the current pose of the submap as `[x, y, theta]`."""
        return self._pose

    @property
    def first_scan_id(self):
        """Return the identifier of the first scan that created this submap."""
        return self._scan_ids[0]

    @property
    def scan_ids(self):
        """Return the list of scan identifiers associated with this submap."""
        return self._scan_ids

    @pose.setter
    def pose(self, pose: np.ndarray):
        """Update the current pose of the submap."""
        self._pose = pose

    def add_scan_id(self, scan_id: int):
        """Append a scan identifier to this submap."""
        self._scan_ids.append(scan_id)
