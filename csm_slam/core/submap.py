#!/usr/bin/env python3
"""Submap container used to group scans and maintain a pose estimate for a 2D map.

Author: Nantha Kumar Sunder
"""

import numpy as np


class Submap:
    """A submap groups one or more scans under a common 2D pose.

    Parameters
    ----------
    id : int
        Unique identifier for this submap.
    pose : np.ndarray
        Current pose of the submap as `[x, y, theta]`.
    origin_scan_id : int
        Identifier of the first scan that created this submap.
    """

    def __init__(self, id: int, pose: np.ndarray, origin_scan_id: int):
        self._id = id
        self._pose = pose
        self._scan_ids = [origin_scan_id]

    #########################################################
    # Properties                                            #
    #########################################################

    @property
    def first_scan_id(self) -> int:
        """Return the identifier of the first scan that created this submap.

        Returns
        -------
        int
            Identifier of the first scan that created this submap.
        """
        return self._scan_ids[0]

    @property
    def id(self) -> int:
        """Return the unique identifier of the submap.

        Returns
        -------
        int
            Unique identifier for this submap.
        """
        return self._id

    @property
    def pose(self) -> np.ndarray:
        """Return the current pose of the submap as `[x, y, theta]`.

        Returns
        -------
        np.ndarray
            Current pose of the submap as `[x, y, theta]`.
        """
        return self._pose

    @property
    def scan_ids(self) -> list:
        """Return the list of scan identifiers associated with this submap.

        Returns
        -------
        list
            List of scan identifiers associated with this submap.
        """
        return self._scan_ids

    #########################################################
    # setters                                               #
    #########################################################

    @pose.setter
    def pose(self, pose: np.ndarray) -> None:
        """Update the current pose of the submap.
        
        Parameters
        ----------
        pose : np.ndarray
            New pose of the submap as `[x, y, theta]`.
        """
        self._pose = pose

    #########################################################
    # Public methods                                        #
    #########################################################

    def add_scan_id(self, scan_id: int) -> None:
        """Append a scan identifier to this submap.
        
        Parameters
        ----------
        scan_id : int
            Identifier of the scan to add to this submap.
        """
        self._scan_ids.append(scan_id)
