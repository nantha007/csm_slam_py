"""Graph structures and optimization utilities for 2D pose SLAM.

This module defines lightweight containers for graph-based SLAM:

- `Vertex`: A single 2D pose node.
- `Edge`: A relative pose constraint between two vertices with an covariance
  matrix and a semantic type (odometry or loop-closure).
- `Graph`: A storage for vertices and edges with helper methods to mutate the
  set.
- `Optimizer`: A thin wrapper over GTSAM to build a factor graph from a
  `Graph`, run nonlinear optimization, and write back optimized poses.

The code targets clarity and separation of concerns between data structures
and the optimizer backend. The optimizer uses GTSAM's Levenberg-Marquardt
algorithm for efficient pose graph optimization.

Author: Nantha Kumar Sunder
"""

import gtsam
import numpy as np
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

    def to_pose2(self):
        """Convert the stored pose to a GTSAM Pose2 object.

        Returns
        -------
        gtsam.Pose2
            GTSAM Pose2 object representing this vertex's pose.
        """
        return gtsam.Pose2(self._pose[0], self._pose[1], self._pose[2])

    def from_pose2(self, pose: gtsam.Pose2):
        """Update the stored pose from a GTSAM Pose2 object.

        Parameters
        ----------
        pose : gtsam.Pose2
            GTSAM Pose2 object to extract pose from.
        """
        self._pose = np.array([pose.x(), pose.y(), pose.theta()])


class EdgeType:
    """Enumeration for edge semantics in the graph.

    This class defines the semantic types of edges in the pose graph,
    distinguishing between odometry constraints (sequential poses) and
    loop closure constraints (non-sequential poses).

    Attributes
    ----------
    ODOM : str
        Odometry constraint between successive poses.
    LOOP : str
        Loop-closure constraint between non-consecutive poses.
    """

    # Define edge type constants
    ODOM = "odom"
    LOOP = "loop"


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
        self,
        id: int,
        from_id: int,
        to_id: int,
        pose: np.ndarray,
        cov: np.ndarray,
        type: EdgeType,
    ):
        """Initialize an edge with all required parameters.

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
        # Store edge metadata and constraint covariance matrix
        self._id = id
        self._from_submap_id = from_id
        self._to_submap_id = to_id
        self._pose = pose
        self._cov = cov
        self._type = type

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
    def from_submap_id(self):
        """Return the source vertex identifier.

        Returns
        -------
        int
            Source vertex identifier for this edge.
        """
        return self._from_submap_id

    @property
    def to_submap_id(self):
        """Return the destination vertex identifier.

        Returns
        -------
        int
            Destination vertex identifier for this edge.
        """
        return self._to_submap_id

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
    def type(self):
        """Return the semantic type of this edge.

        Returns
        -------
        EdgeType
            Semantic type of this edge (odometry or loop closure).
        """
        return self._type

    def to_pose2(self):
        """Convert the stored relative pose to a GTSAM Pose2 object.

        Returns
        -------
        gtsam.Pose2
            GTSAM Pose2 object representing the relative pose.
        """
        return gtsam.Pose2(self._pose[0], self._pose[1], self._pose[2])

    def from_pose2(self, pose: gtsam.Pose2):
        """Update the stored relative pose from a GTSAM Pose2 object.

        Parameters
        ----------
        pose : gtsam.Pose2
            GTSAM Pose2 object to extract relative pose from.
        """
        self._pose = np.array([pose.x(), pose.y(), pose.theta()])

    @property
    def cov(self):
        """Return the covariance matrix for this constraint.

        Returns
        -------
        numpy.ndarray
            Covariance matrix for this constraint.
        """
        return self._cov


class Graph:
    """Container managing vertices and edges for graph-based SLAM.

    This class provides a lightweight container for managing the pose graph,
    including vertices (poses) and edges (constraints). It offers methods to
    add vertices and edges, retrieve the graph structure, and manage
    loop closure edges.
    """

    def __init__(self):
        """Initialize an empty graph.

        Creates a new graph with empty vertex and edge collections
        and initializes the edge ID counter.
        """
        # Initialize graph data structures
        self._vertices = {}
        self._edges = {}
        self._edge_id = 0

    def add_vertex(self, id: int, pose: np.ndarray):
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
        type: EdgeType = EdgeType.ODOM,
    ):
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
        type : EdgeType, optional
            Edge semantic type (odometry or loop-closure). Defaults to ODOM.
        """
        # Create and store new edge with auto-generated ID
        self._edges[self._edge_id] = Edge(
            self._edge_id, from_id, to_id, pose, cov, type
        )
        self._edge_id += 1

    def get_vertices(self):
        """Return the internal dictionary of vertices.

        Returns
        -------
        dict
            Dictionary mapping vertex IDs to Vertex objects.
        """
        return self._vertices

    def get_edges(self):
        """Return the internal dictionary of edges.

        Returns
        -------
        dict
            Dictionary mapping edge IDs to Edge objects.
        """
        return self._edges

    def clear_loop_edges(self):
        """Remove all loop-closure edges from the graph.

        This method removes all edges with type EdgeType.LOOP, which is
        useful for debugging or when loop closures need to be disabled.
        """
        # Find all loop closure edges to remove
        clear_ids = []
        for edge in self._edges.values():
            if edge.type == EdgeType.LOOP:
                clear_ids.append(edge.id)
        # Remove loop closure edges
        for id in clear_ids:
            self._edges.pop(id)


class Optimizer:
    """GTSAM-backed optimizer for 2D pose graphs.

    This class provides a wrapper around GTSAM's optimization capabilities
    for 2D pose graphs. It handles factor graph construction, optimization
    using Levenberg-Marquardt, and updating the original graph with
    optimized poses.

    Attributes
    ----------
    graph : gtsam.NonlinearFactorGraph
        GTSAM factor graph for optimization.
    initial_estimate : gtsam.Values
        Initial pose estimates for optimization.
    max_iterations : int
        Maximum number of optimization iterations.
    params : gtsam.LevenbergMarquardtParams
        Optimization parameters.
    odom_noise_default : gtsam.noiseModel
        Default noise model for odometry edges.
    loop_noise_default : gtsam.noiseModel
        Default noise model for loop closure edges.
    prior_noise : gtsam.noiseModel
        Strong prior noise model for the first vertex.
    first_vertex_added : bool
        Flag indicating if the first vertex has been added.

    Parameters
    ----------
    max_iterations : int, optional
        Maximum number of optimization iterations (default: 100).
    """

    def __init__(self, max_iterations=100):
        """Initialize the optimizer with default parameters.

        Parameters
        ----------
        max_iterations : int, optional
            Maximum number of optimization iterations (default: 100).
        """
        # Initialize GTSAM components
        self.graph = gtsam.NonlinearFactorGraph()
        self.initial_estimate = gtsam.Values()
        self.max_iterations = max_iterations
        self.params = gtsam.LevenbergMarquardtParams()
        self.odom_noise_default = gtsam.noiseModel.Diagonal.Covariance(np.eye(3))
        self.loop_noise_default = gtsam.noiseModel.Gaussian.Information(np.eye(3))
        self.prior_noise = gtsam.noiseModel.Diagonal.Sigmas(
            np.array([1e-6, 1e-6, 1e-6])
        )
        self.first_vertex_added = False

    def add_vertex(self, id: int, pose: Pose2):
        """Insert a prior (for the first vertex) and initial pose estimate.

        The first inserted vertex receives a strong prior to anchor the graph
        and prevent gauge freedom during optimization.

        Parameters
        ----------
        id : int
            Vertex identifier.
        pose : gtsam.Pose2
            Initial pose estimate for this vertex.
        """
        # Add vertex to initial estimate
        self.initial_estimate.insert(id, pose)

        # Add strong prior factor for the first vertex to anchor the graph
        if not self.first_vertex_added:
            self.graph.add(
                gtsam.PriorFactorPose2(
                    id,
                    pose,
                    self.prior_noise,
                )
            )
            self.first_vertex_added = True

    def add_edge(
        self,
        id1: int,
        id2: int,
        pose: Pose2,
        cov: np.ndarray = None,
        type: EdgeType = EdgeType.ODOM,
    ):
        """Add a `BetweenFactorPose2` for the relative pose measurement.

        If `cov` is not provided, a default noise is chosen depending on the
        edge type (odometry or loop-closure). When provided, `cov` is treated
        as a covariance matrix for diagonal noise construction.
        """
        if cov is None:
            noise = (
                self.odom_noise_default
                if type == EdgeType.ODOM
                else self.loop_noise_default
            )
        else:
            # noise = gtsam.noiseModel.Diagonal.Covariance(np.eye(3) *  0.01)
            noise = gtsam.noiseModel.Diagonal.Covariance(cov)

        self.graph.add(gtsam.BetweenFactorPose2(id1, id2, pose, noise))

    def get_vertices(self):
        """Return the list of vertex ids currently in the initial estimate."""
        return list(self.initial_estimate.keys())

    def optimize(self, graph: Graph):
        """Optimize the given Graph and write back the optimized state.

        This method rebuilds the GTSAM factor graph from the provided graph,
        runs Levenberg-Marquardt optimization, and updates both vertex
        absolute poses and edge relative poses to reflect the optimized
        configuration.

        Parameters
        ----------
        graph : Graph
            The graph to optimize. This graph will be modified in-place
            with optimized poses.

        Returns
        -------
        Graph
            The same graph object with updated poses.

        Notes
        -----
        The optimization process:
        1. Rebuilds the GTSAM factor graph from the input graph
        2. Runs Levenberg-Marquardt optimization
        3. Updates vertex poses with optimized values
        4. Updates edge relative poses using optimized vertex poses
        """
        # Initialize the GTSAM factor graph and estimates
        self.graph = gtsam.NonlinearFactorGraph()
        self.initial_estimate = gtsam.Values()
        self.first_vertex_added = False
        # Insert current graph state into GTSAM
        for vertex in graph.get_vertices().values():
            self.add_vertex(vertex.id, vertex.to_pose2())
        for edge in graph.get_edges().values():
            id1, id2 = edge.from_submap_id, edge.to_submap_id
            self.add_edge(id1, id2, edge.to_pose2(), edge.cov, edge.type)

        # Run Levenberg-Marquardt optimization
        optimizer = gtsam.LevenbergMarquardtOptimizer(
            self.graph, self.initial_estimate, self.params
        )
        result = optimizer.optimize()

        # Update vertex poses with optimized values
        for vertex_id, vertex in graph.get_vertices().items():
            optimized_pose: gtsam.Pose2 = result.atPose2(vertex_id)
            vertex.from_pose2(optimized_pose)

        # Update edge relative poses using optimized vertex poses
        for edge in graph.get_edges().values():
            from_pose: gtsam.Pose2 = result.atPose2(edge.from_submap_id)
            to_pose: gtsam.Pose2 = result.atPose2(edge.to_submap_id)
            relative_opt: gtsam.Pose2 = from_pose.between(to_pose)
            edge.from_pose2(relative_opt)

        return graph
