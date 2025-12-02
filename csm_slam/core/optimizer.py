#!/usr/bin/env python3
"""
GTSAM-backed optimizer for 2D pose graphs.

Author: Nantha Kumar Sunder
"""

import gtsam
import numpy as np
from gtsam import Pose2

from csm_slam.core.graph import Graph


class Optimizer:
    """
    GTSAM-backed optimizer for 2D pose graphs.

    Parameters
    ----------
    max_iterations : int, optional
        Maximum number of optimization iterations (default: 100).

    """

    def __init__(self, max_iterations=100):
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

    #########################################################
    # Public methods                                        #
    #########################################################

    def add_edge(self, id1: int, id2: int, pose: Pose2, cov: np.ndarray = None) -> None:
        """
        Add a between factor for the relative pose measurement between two vertices.

        Parameters
        ----------
        id1 : int
            Identifier of the source vertex.
        id2 : int
            Identifier of the destination vertex.
        pose : gtsam.Pose2
            Relative pose measurement between the two vertices.
        cov : numpy.ndarray, optional
            Covariance matrix associated with the measurement.
            If None, a default noise is used.

        """
        if cov is None:
            noise = self.odom_noise_default
        else:
            noise = gtsam.noiseModel.Diagonal.Covariance(cov)

        self.graph.add(gtsam.BetweenFactorPose2(id1, id2, pose, noise))

    def add_vertex(self, id: int, pose: Pose2) -> None:
        """
        Add a vertex to the initial estimate.

        Parameters
        ----------
        id : int
            Vertex identifier.
        pose : gtsam.Pose2
            Initial pose estimate for this vertex.

        """
        self.initial_estimate.insert(id, pose)

        if not self.first_vertex_added:
            self.graph.add(
                gtsam.PriorFactorPose2(
                    id,
                    pose,
                    self.prior_noise,
                )
            )
            self.first_vertex_added = True

    def get_vertices(self) -> list:
        """
        Return the list of vertex ids currently in the initial estimate.

        Returns
        -------
        list
            List of vertex ids currently in the initial estimate.

        """
        return list(self.initial_estimate.keys())

    def optimize(self, graph: Graph) -> Graph:
        """
        Optimize the given Graph and write back the optimized state.

        Parameters
        ----------
        graph : Graph
            The graph to optimize. This graph will be modified in-place
            with optimized poses.

        Returns
        -------
        Graph
            The same graph object with updated poses.

        """
        # Initialize the GTSAM factor graph and initial estimate
        self.graph = gtsam.NonlinearFactorGraph()
        self.initial_estimate = gtsam.Values()
        self.first_vertex_added = False

        # Add current graph state to GTSAM
        for vertex in graph.get_vertices().values():
            self.add_vertex(vertex.id, vertex.to_pose2())
        for edge in graph.get_edges().values():
            id1, id2 = edge.from_submap_id, edge.to_submap_id
            self.add_edge(id1, id2, edge.to_pose2(), edge.cov)

        # Run Levenberg-Marquardt optimization
        optimizer = gtsam.LevenbergMarquardtOptimizer(
            self.graph, self.initial_estimate, self.params
        )
        result = optimizer.optimize()

        # Update vertex poses with optimized poses
        for vertex_id, vertex in graph.get_vertices().items():
            optimized_pose = result.atPose2(vertex_id)
            vertex.from_pose2(optimized_pose)

        # Update edge relative poses with optimized vertex poses
        for edge in graph.get_edges().values():
            from_pose = result.atPose2(edge.from_submap_id)
            to_pose = result.atPose2(edge.to_submap_id)
            relative_opt = from_pose.between(to_pose)
            edge.from_pose2(relative_opt)

        return graph
