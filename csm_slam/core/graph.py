"""Graph structures and optimization utilities for 2D pose SLAM.

- `Vertex`: A single 2D pose node.
- `Edge`: A relative pose constraint between two vertices with an covariance
  matrix and a semantic type (odometry or loop-closure).
- `Graph`: A storage for vertices and edges with helper methods to mutate the
  set.

Author: Nantha Kumar Sunder
"""

import numpy as np
import gtsam
from gtsam import Pose2


class Vertex:
    """A node in the graph representing a single 2D pose.

    This class represents a vertex in the pose graph, storing a 2D pose
    and providing conversion methods to/from GTSAM Pose2 objects.

    Parameters
    ----------
    id : int
        Unique identifier for this vertex.
    pose : numpy.ndarray
        Initial pose [x, y, theta] in meters and radians.
    """

    def __init__(self, id: int, pose: np.ndarray):
        """Initialize a vertex with ID and pose.

        Parameters
        ----------
        id : int
            Unique identifier for this vertex.
        pose : numpy.ndarray
            Initial pose [x, y, theta] in meters and radians.
        """
        # Store vertex metadata and pose
        self._id = id
        self._pose = pose

    #########################################################
    # Properties                                            #
    #########################################################

    @property
    def id(self):
        """Return the vertex identifier.

        Returns
        -------
        int
            Unique identifier for this vertex.
        """
        return self._id

    @property
    def pose(self):
        """Return the pose as a NumPy array.

        Returns
        -------
        numpy.ndarray
            Pose as [x, y, theta] in meters and radians.
        """
        return self._pose

    #########################################################
    # Public methods                                        #
    #########################################################

    def from_pose2(self, pose: gtsam.Pose2) -> None:
        """Update the stored pose from a GTSAM Pose2 object.

        Parameters
        ----------
        pose : gtsam.Pose2
            GTSAM Pose2 object to extract pose from.
        """
        self._pose = np.array([pose.x(), pose.y(), pose.theta()])

    def to_pose2(self) -> gtsam.Pose2:
        """Convert the stored pose to a GTSAM Pose2 object.

        Returns
        -------
        gtsam.Pose2
            GTSAM Pose2 object representing this vertex's pose.
        """
        return gtsam.Pose2(self._pose[0], self._pose[1], self._pose[2])


class Edge:
    """A relative pose constraint connecting two vertices.

    This class represents an edge in the pose graph, storing a relative
    pose measurement between two vertices along with uncertainty covariance
    and semantic type classification.

    Parameters
    ----------
    id : int
        Unique identifier for this edge.
    from_id : int
        Source vertex identifier.
    to_id : int
        Destination vertex identifier.
    pose : numpy.ndarray
        Relative pose measurement [dx, dy, dtheta].
    cov : numpy.ndarray
        Covariance matrix for this constraint.
    type : EdgeType
        Semantic type of this edge.
    """

    def __init__(
        self, id: int, from_id: int, to_id: int, pose: np.ndarray, cov: np.ndarray
    ):
        self._id = id
        self._from_submap_id = from_id
        self._to_submap_id = to_id
        self._pose = pose
        self._cov = cov

    #########################################################
    # Properties                                            #
    #########################################################

    @property
    def cov(self):
        """Return the covariance matrix for this constraint.

        Returns
        -------
        numpy.ndarray
            Covariance matrix for this constraint.
        """
        return self._cov

    @property
    def from_submap_id(self):
        """Return the source vertex identifier.

        Returns
        -------
        int
            Source vertex identifier for this edge.
        """
        return self._from_submap_id

    @property
    def id(self):
        """Return the edge identifier.

        Returns
        -------
        int
            Unique identifier for this edge.
        """
        return self._id

    @property
    def pose(self):
        """Return the relative pose measurement.

        Returns
        -------
        numpy.ndarray
            Relative pose measurement [dx, dy, dtheta].
        """
        return self._pose

    @property
    def to_submap_id(self):
        """Return the destination vertex identifier.

        Returns
        -------
        int
            Destination vertex identifier for this edge.
        """
        return self._to_submap_id

    #########################################################
    # Public methods                                        #
    #########################################################

    def from_pose2(self, pose: gtsam.Pose2) -> None:
        """Update the stored relative pose from a GTSAM Pose2 object.

        Parameters
        ----------
        pose : gtsam.Pose2
            GTSAM Pose2 object to extract relative pose from.
        """
        self._pose = np.array([pose.x(), pose.y(), pose.theta()])

    def to_pose2(self) -> gtsam.Pose2:
        """Convert the stored relative pose to a GTSAM Pose2 object.

        Returns
        -------
        gtsam.Pose2
            GTSAM Pose2 object representing the relative pose.
        """
        return gtsam.Pose2(self._pose[0], self._pose[1], self._pose[2])


class Graph:
    """Class for managing vertices and edges for graph-based SLAM.


    Parameters
    ----------
    None
    """

    def __init__(self):
        self._vertices = {}
        self._edges = {}
        self._edge_id = 0

    #########################################################
    # Public methods                                        #
    #########################################################

    def add_vertex(self, id: int, pose: np.ndarray) -> None:
        """Add a vertex with identifier and pose.

        Parameters
        ----------
        id : int
            Unique identifier for the vertex.
        pose : numpy.ndarray
            Pose [x, y, theta] in meters and radians.
        """
        # Create and store new vertex
        self._vertices[id] = Vertex(id, pose)

    def add_edge(
        self,
        from_id: int,
        to_id: int,
        pose: np.ndarray,
        cov: np.ndarray = None,
    ) -> None:
        """Add a relative pose constraint between two vertices.

        Parameters
        ----------
        from_id : int
            Identifier of the source vertex.
        to_id : int
            Identifier of the destination vertex.
        pose : numpy.ndarray
            Relative pose measurement [dx, dy, dtheta].
        cov : numpy.ndarray, optional
            Covariance matrix associated with the measurement.
            If None, a default covariance matrix is used.
        """
        # Create and store new edge with auto-generated ID
        self._edges[self._edge_id] = Edge(self._edge_id, from_id, to_id, pose, cov)
        self._edge_id += 1

    def get_edges(self) -> dict:
        """Return the internal dictionary of edges.

        Returns
        -------
        dict
            Dictionary mapping edge IDs to Edge objects.
        """
        return self._edges

    def get_vertices(self) -> dict:
        """Return the internal dictionary of vertices.

        Returns
        -------
        dict
            Dictionary mapping vertex IDs to Vertex objects.
        """
        return self._vertices
