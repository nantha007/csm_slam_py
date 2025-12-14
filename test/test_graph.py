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

"""Tests for graph structures and optimization utilities for 2D pose SLAM."""

import os
import sys

import numpy as np

# Add test directory to path to allow importing test_helpers
test_dir = os.path.dirname(os.path.abspath(__file__))
if test_dir not in sys.path:
    sys.path.insert(0, test_dir)

from test_helpers import load_settings  # noqa: E402

load_settings()

import gtsam  # noqa: E402

from csm_slam.backend.graph import Vertex, Edge, Graph  # noqa: E402


class TestVertex:
    """Test cases for the Vertex class."""

    def test_vertex_initialization(self):
        """Test vertex initialization and properties."""
        vertex = Vertex(5, np.array([1.0, 2.0, 0.5]))

        assert vertex.vertex_id == 5
        assert np.array_equal(vertex.pose, np.array([1.0, 2.0, 0.5]))

    def test_vertex_pose2_conversion(self):
        """Test conversion to/from GTSAM Pose2."""
        pose = np.array([1.5, 2.5, 0.785])
        vertex = Vertex(0, pose)

        # Convert to Pose2
        pose2 = vertex.to_pose2()
        assert isinstance(pose2, gtsam.Pose2)
        assert np.allclose([pose2.x(), pose2.y(), pose2.theta()], pose)

        # Convert back from Pose2
        new_pose2 = gtsam.Pose2(3.14, 2.71, 1.0)
        vertex.from_pose2(new_pose2)
        assert np.allclose(vertex.pose, [3.14, 2.71, 1.0])


class TestEdge:
    """Test cases for the Edge class."""

    def test_edge_initialization(self):
        """Test edge initialization and properties."""
        edge = Edge(5, 10, 20, np.array([1.0, 2.0, 0.5]), np.eye(3))

        assert edge.edge_id == 5
        assert edge.from_submap_id == 10
        assert edge.to_submap_id == 20
        assert np.array_equal(edge.pose, np.array([1.0, 2.0, 0.5]))
        assert np.array_equal(edge.cov, np.eye(3))

    def test_edge_pose2_conversion(self):
        """Test conversion to/from GTSAM Pose2."""
        pose = np.array([0.1, 0.2, 0.3])
        edge = Edge(0, 1, 2, pose, np.eye(3))

        # Convert to Pose2
        pose2 = edge.to_pose2()
        assert isinstance(pose2, gtsam.Pose2)
        assert np.allclose([pose2.x(), pose2.y(), pose2.theta()], pose)

        # Convert back from Pose2
        new_pose2 = gtsam.Pose2(0.5, 0.6, 0.7)
        edge.from_pose2(new_pose2)
        assert np.allclose(edge.pose, [0.5, 0.6, 0.7])


class TestGraph:
    """Test cases for the Graph class."""

    def test_graph_initialization(self):
        """Test graph initialization."""
        graph = Graph()

        assert len(graph.get_vertices()) == 0
        assert len(graph.get_edges()) == 0

    def test_add_vertex(self):
        """Test adding vertices to the graph."""
        graph = Graph()
        graph.add_vertex(0, np.array([1.0, 2.0, 0.5]))
        graph.add_vertex(1, np.array([3.0, 4.0, 1.0]))

        vertices = graph.get_vertices()
        assert len(vertices) == 2
        assert vertices[0].vertex_id == 0
        assert np.array_equal(vertices[1].pose, np.array([3.0, 4.0, 1.0]))

    def test_add_edge(self):
        """Test adding edges to the graph."""
        graph = Graph()
        graph.add_vertex(0, np.array([0.0, 0.0, 0.0]))
        graph.add_vertex(1, np.array([1.0, 0.0, 0.0]))

        # Add edge with covariance
        graph.add_edge(0, 1, np.array([1.0, 0.0, 0.0]), np.eye(3))
        edges = graph.get_edges()
        assert len(edges) == 1
        edge = list(edges.values())[0]
        assert edge.from_submap_id == 0
        assert edge.to_submap_id == 1
        assert np.array_equal(edge.cov, np.eye(3))

        # Add edge without covariance
        graph.add_edge(1, 0, np.array([-1.0, 0.0, 0.0]))
        edges = graph.get_edges()
        assert len(edges) == 2
        assert list(edges.values())[1].cov is None

    def test_edge_id_auto_increment(self):
        """Test that edge IDs are auto-incremented."""
        graph = Graph()
        for i in range(3):
            graph.add_vertex(i, np.array([float(i), 0.0, 0.0]))

        graph.add_edge(0, 1, np.array([1.0, 0.0, 0.0]))
        graph.add_edge(1, 2, np.array([1.0, 0.0, 0.0]))

        edges = graph.get_edges()
        assert sorted(edges.keys()) == [0, 1]
        assert edges[0].edge_id == 0
        assert edges[1].edge_id == 1

