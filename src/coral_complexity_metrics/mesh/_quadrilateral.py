import math
from ._vertex import Vertex
import sys
from ._helpers import get_midpoint_of_edge


class Quadrilateral(object):

    def __init__(self, vertex_1, vertex_2, vertex_3, vertex_4):
        self.vertices = (vertex_1, vertex_2, vertex_3, vertex_4)
        # self.vertices_array = ([[vertex_1.x, vertex_1.y],
        #                  [vertex_2.x, vertex_2.y],
        #                  [vertex_3.x, vertex_3.y],
        #                  [vertex_4.x, vertex_4.y]])
        self.centroid = get_midpoint_of_edge(
            get_midpoint_of_edge(vertex_1, vertex_3),
            get_midpoint_of_edge(vertex_2, vertex_4))

    def contains(self, vertex):
        if self._within(vertex) is True:
            return True
        elif self._bump(vertex) is True:
            return True
        else:
            return False

    def _within(self, vertex):
        #     inside_x = vertex.x > self.vertices[0].x and vertex.x < self.vertices[1].x
        #     inside_y = vertex.y > self.vertices[0].y and vertex.y < self.vertices[3].y
        #     inside = inside_x and inside_y
        xl = [v.x for v in self.vertices]
        yl = [v.y for v in self.vertices]
        if vertex.x < min(xl) or vertex.x > max(xl) or vertex.y < min(yl) or vertex.y > max(yl):
            return False
        return True

    def _bump(self, vertex):
        _epsilon = 0.00001
        vertex_bumped = Vertex(vertex.x + _epsilon,
                               vertex.y + _epsilon,
                               0)
        return self._within(vertex)
