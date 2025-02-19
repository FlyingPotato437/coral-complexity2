"""Class for vertex."""
import math


class Vertex(object):
    """Vertex class."""

    def __init__(self, x, y, z):
        """Create a vertex with three points, x, y & z."""
        self.x = float(x)
        self.y = float(y)
        self.z = float(z)

    def distance_to_xyz(self, to_vertex):
        """Get the distance between vertexs in XYZ plane."""
        return math.sqrt(math.pow(to_vertex.x - self.x, 2) +
                         math.pow(to_vertex.y - self.y, 2) +
                         math.pow(to_vertex.z - self.z, 2))

    def distance_to_xy(self, to_vertex):
        """Get the distance between vertexs in XY plane."""
        return math.sqrt(math.pow(to_vertex.x - self.x, 2) +
                         math.pow(to_vertex.y - self.y, 2))

    def make_xy(self):
        return Vertex(self.x, self.y, 0)

    def __str__(self):
        """Pretty prints the vertex."""
        return "x=" + str(self.x) + ", y=" + str(self.y) + ", z=" + str(self.z)

    def __eq__(self, other):
        """Check the vertex equals another, returns a Boolean."""
        if type(other) is type(self):
            return (self.x == other.x and self.y == other.y and self.z == other.z)
        return False

    def __lt__(self, other):
        """Check the vertex equals another, returns a Boolean."""
        if type(other) is type(self):
            return (self.x < other.x and self.y < other.y and self.z < other.z)
        return False

    def __hash__(self):
        """Hash code for vertex."""
        return hash(self.x) + hash(self.y) + hash(self.z)

    def displacement(self, vertex):
        return Vertex(vertex.x - self.x, vertex.y - self.y, vertex.z - self.z)
