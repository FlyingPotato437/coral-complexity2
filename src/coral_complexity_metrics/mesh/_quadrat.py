from ._quadrilateral import Quadrilateral
from ._vertex import Vertex
from ._metric import Metric


class Quadrat(Quadrilateral):

    def __init__(self, id, size, midpoint):
        self.id = id
        self.size = size
        self.midpoint = midpoint
        # self.vertices_inside = []
        Quadrilateral.__init__(self,
                               Vertex(midpoint.x - (size / 2),
                                      midpoint.y - (size / 2), midpoint.z),
                               Vertex(midpoint.x + (size / 2),
                                      midpoint.y - (size / 2), midpoint.z),
                               Vertex(midpoint.x + (size / 2),
                                      midpoint.y + (size / 2), midpoint.z),
                               Vertex(midpoint.x - (size / 2), midpoint.y + (size / 2), midpoint.z))

    def area(self):
        return self.size * self.size

    def __eq__(self, other):
        return self.midpoint == other.midpoint and self.id == other.id

    def __hash__(self):
        return hash(self.id)
