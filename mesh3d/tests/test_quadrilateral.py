import math
from mesh3d.quadrilateral import Quadrilateral
from mesh3d.vertex import Vertex
import pytest


class TestQuadrilateral:

    def test_quad_contructor_not_none(self):
        my_quad = Quadrilateral(Vertex(0, 0, 0), Vertex(
            1, 1, 0), Vertex(1, 3, 0), Vertex(0, 3, 0))
        assert my_quad is not None

    def test_quad_contructor_vertices(self):
        my_quad = Quadrilateral(Vertex(0, 0, 0), Vertex(
            1, 1, 0), Vertex(1, 3, 0), Vertex(0, 3, 0))
        assert my_quad.vertices[0] == Vertex(0, 0, 0)
        assert my_quad.vertices[1] == Vertex(1, 1, 0)
        assert my_quad.vertices[2] == Vertex(1, 3, 0)
        assert my_quad.vertices[3] == Vertex(0, 3, 0)

    def test_quadrat_constructer_midpoint_is_vertex(self):
        my_quad = Quadrilateral(Vertex(0, 0, 0), Vertex(
            1, 1, 0), Vertex(1, 3, 0), Vertex(0, 3, 0))
        assert isinstance(my_quad.centroid, Vertex)

    def test_quadrat_constructer_midpoint_false(self):
        my_quad = Quadrilateral(Vertex(0, 0, 0), Vertex(
            1, 1, 0), Vertex(1, 3, 0), Vertex(0, 3, 0))
        point = Vertex(0.5, 0.5, 0)
        assert my_quad.centroid != point

    def test_quadrat_constructer_midpoint_true(self):
        my_quad = Quadrilateral(Vertex(0, 0, 0), Vertex(
            1, 0, 0), Vertex(1, 1, 0), Vertex(0, 1, 0))
        point = Vertex(0.5, 0.5, 0)
        assert my_quad.centroid == point

    def test_quad_contains_point(self):
        my_quad = Quadrilateral(Vertex(0, 0, 0), Vertex(
            1, 0, 0), Vertex(1, 1, 0), Vertex(0, 1, 0))
        point = Vertex(0.5, 0.5, 0)
        assert my_quad.contains(point) is True

    def test_quad_does_not_contain_point(self):
        my_quad = Quadrilateral(Vertex(0, 0, 0), Vertex(
            1, 0, 0), Vertex(1, 1, 0), Vertex(0, 1, 0))
        point = Vertex(1.5, 0.5, 0)
        assert my_quad.contains(point) is False

    def test_quad_does_contains_point_on_line(self):
        my_quad = Quadrilateral(Vertex(0, 0, 0), Vertex(
            1, 0, 0), Vertex(1, 1, 0), Vertex(0, 1, 0))
        point = Vertex(0, 0.5, 0)
        assert my_quad.contains(point) is True
