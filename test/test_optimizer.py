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
Tests for GTSAM-backed optimizer for 2D pose graphs.

Author: Nantha Kumar Sunder
"""

import os
import sys

import numpy as np

# Add test directory to path to allow importing test_helpers
test_dir = os.path.dirname(os.path.abspath(__file__))
if test_dir not in sys.path:
    sys.path.insert(0, test_dir)

from test_helpers import load_settings  # noqa: E402

load_settings()

from gtsam import Pose2  # noqa: E402

from csm_slam.backend.graph import Graph  # noqa: E402
from csm_slam.backend.optimizer import Optimizer  # noqa: E402


class TestOptimizer:
    """Test cases for the Optimizer class."""

    def test_optimizer_initialization(self):
        """Test optimizer initialization."""
        optimizer = Optimizer(max_iterations=50)

        assert optimizer.max_iterations == 50
        assert optimizer.first_vertex_added is False

    def test_add_vertex(self):
        """Test adding vertices to optimizer."""
        optimizer = Optimizer()
        pose = Pose2(1.0, 2.0, 0.5)

        optimizer.add_vertex(0, pose)

        assert 0 in optimizer.get_vertices()
        assert optimizer.first_vertex_added is True

    def test_add_vertex_first_gets_prior(self):
        """Test that first vertex gets a prior factor."""
        optimizer = Optimizer()
        pose = Pose2(0.0, 0.0, 0.0)

        optimizer.add_vertex(0, pose)
        assert optimizer.first_vertex_added is True

        # Add second vertex - should not add prior
        optimizer.add_vertex(1, pose)
        assert optimizer.first_vertex_added is True

    def test_add_edge(self):
        """Test adding edges to optimizer."""
        optimizer = Optimizer()
        pose1 = Pose2(0.0, 0.0, 0.0)
        pose2 = Pose2(1.0, 0.0, 0.0)
        relative_pose = Pose2(1.0, 0.0, 0.0)

        optimizer.add_vertex(0, pose1)
        optimizer.add_vertex(1, pose2)
        optimizer.add_edge(0, 1, relative_pose)

        # Check that edge was added (graph should have factors)
        assert optimizer.graph.size() > 0

    def test_add_edge_with_covariance(self):
        """Test adding edge with custom covariance."""
        optimizer = Optimizer()
        pose1 = Pose2(0.0, 0.0, 0.0)
        pose2 = Pose2(1.0, 0.0, 0.0)
        relative_pose = Pose2(1.0, 0.0, 0.0)
        cov = np.eye(3) * 0.1

        optimizer.add_vertex(0, pose1)
        optimizer.add_vertex(1, pose2)
        optimizer.add_edge(0, 1, relative_pose, cov)

        assert optimizer.graph.size() > 0

    def test_get_vertices(self):
        """Test getting list of vertex IDs."""
        optimizer = Optimizer()
        pose = Pose2(0.0, 0.0, 0.0)

        optimizer.add_vertex(0, pose)
        optimizer.add_vertex(1, pose)
        optimizer.add_vertex(2, pose)

        vertices = optimizer.get_vertices()
        assert len(vertices) == 3
        assert 0 in vertices
        assert 1 in vertices
        assert 2 in vertices

    def test_optimize(self):
        """Test optimizing a graph."""
        # Create a simple graph
        graph = Graph()
        graph.add_vertex(0, np.array([0.0, 0.0, 0.0]))
        graph.add_vertex(1, np.array([1.0, 0.0, 0.0]))
        graph.add_edge(0, 1, np.array([1.0, 0.0, 0.0]), np.eye(3))

        # Optimize
        optimizer = Optimizer()
        optimizer.optimize(graph)

        # Check that graph was modified
        assert len(graph.get_vertices()) == 2
        assert len(graph.get_edges()) == 1

        # Check that poses were updated
        vertices = graph.get_vertices()
        assert vertices[0].pose is not None
        assert vertices[1].pose is not None
        # Verify poses are numpy arrays
        assert isinstance(vertices[0].pose, np.ndarray)
        assert isinstance(vertices[1].pose, np.ndarray)
        assert len(vertices[0].pose) == 3
        assert len(vertices[1].pose) == 3

    def test_optimize_with_loop_closure(self):
        """Test optimizing graph with loop closure."""
        # Create a simple loop: 0 -> 1 -> 2 -> 0
        graph = Graph()
        graph.add_vertex(0, np.array([0.0, 0.0, 0.0]))
        graph.add_vertex(1, np.array([1.0, 0.0, 0.0]))
        graph.add_vertex(2, np.array([2.0, 0.0, 0.0]))

        # Add odometry edges
        graph.add_edge(0, 1, np.array([1.0, 0.0, 0.0]), np.eye(3))
        graph.add_edge(1, 2, np.array([1.0, 0.0, 0.0]), np.eye(3))

        # Add loop closure
        graph.add_edge(2, 0, np.array([-2.0, 0.0, 0.0]), np.eye(3) * 0.1)

        optimizer = Optimizer()
        optimizer.optimize(graph)

        assert len(graph.get_vertices()) == 3
        assert len(graph.get_edges()) == 3

