import math
from ._vertex import Vertex
import pdb


class Face(object):
    """Class representing a face."""

    def __init__(self, vertex1, vertex2, vertex3):
        """Constructor."""
        self.vertices = (vertex1, vertex2, vertex3)
        self.centroid = Vertex((vertex1.x + vertex2.x + vertex3.x) / 3,
                               (vertex1.y + vertex2.y + vertex3.y) / 3,
                               (vertex1.z + vertex2.z + vertex3.z) / 3)
        self.area3d = self._area_3d()
        self.area2d = self._area_2d()

    def _area_3d(self):
        edge1 = self.vertices[0].distance_to_xyz(self.vertices[1])
        edge2 = self.vertices[1].distance_to_xyz(self.vertices[2])
        edge3 = self.vertices[2].distance_to_xyz(self.vertices[0])
        return self._get_area(edge1, edge2, edge3)

    def _area_2d(self):
        edge1_xy = self.vertices[0].distance_to_xy(self.vertices[1])
        edge2_xy = self.vertices[1].distance_to_xy(self.vertices[2])
        edge3_xy = self.vertices[2].distance_to_xy(self.vertices[0])
        return self._get_area(edge1_xy, edge2_xy, edge3_xy)

    def _get_area(self, edge1, edge2, edge3):
        semiperimeter = (edge1 + edge2 + edge3) * 0.5
        result = 0.0
        try:
            result = math.sqrt(semiperimeter *
                               (semiperimeter - edge1) *
                               (semiperimeter - edge2) *
                               (semiperimeter - edge3))
        except ValueError:
            """ Added to account for issue when semiperimeter and
            edge round to the same number."""
            semiperimeter = semiperimeter + 0.0001
            result = math.sqrt(semiperimeter *
                               (semiperimeter - edge1) *
                               (semiperimeter - edge2) *
                               (semiperimeter - edge3))
        return result
