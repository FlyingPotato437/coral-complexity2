import pytest
import math
import pdb
from mesh3d.vertex import Vertex


class TestVertex:

    def test_vertex_contructor(self):
        vertex = Vertex(1, 2, 3)
        assert vertex is not None

    def test_vertex_x(self):
        vertex = Vertex(1, 2, 3)
        assert vertex.x == 1.0

    def test_vertex_y(self):
        vertex = Vertex(1, 2, 3)
        assert vertex.y == 2.0

    def test_vertex_z(self):
        vertex = Vertex(1, 2, 3)
        assert vertex.z == 3.0

    def test_vertex_distance_to_xyz(self):
        vertex = Vertex(1, 1, 1)
        distance = vertex.distance_to_xyz(Vertex(3, 3, 3))
        result = math.sqrt(12)
        assert distance == result

    def test_vertex_distance_to_xy(self):
        vertex = Vertex(1, 1, 1)
        distance = vertex.distance_to_xy(Vertex(3, 3, 3))
        result = math.sqrt(8)
        assert distance == result

    def test_vertex_eq(self):
        vertex1 = Vertex(1, 1, 1)
        vertex2 = Vertex(1, 1, 1)
        assert vertex1 == vertex2

    def test_vertex_not_eq(self):
        vertex1 = Vertex(1, 1, 1)
        vertex2 = Vertex(1, 1, 2)
        assert not vertex1.__eq__(vertex2)

    def test_vertex_str(self):
        vertex = Vertex(1, 1, 1)
        v_correct = "x=" + str(vertex.x) + ", y=" + str(vertex.y) + ", z=" + str(vertex.z)
        v_result = vertex.__str__()
        assert v_result == v_correct

    def test_vertex_lt(self):
        vertex1 = Vertex(1, 1, 1)
        vertex2 = Vertex(2, 2, 2)
        result = vertex1 < vertex2
        assert result is True

    def test_vertex_not_lt(self):
        vertex1 = Vertex(1, 1, 1)
        vertex2 = Vertex(0, 0, 0)
        assert vertex1 > vertex2

    def test_vertex_make_xy(self):
        vertex = Vertex(1, 1, 1)
        result = vertex.make_xy()
        assert result.z == 0
