from ._vertex import Vertex
from ._quadrat import Quadrat
from ._helpers import get_midpoint_of_edge


class QuadratBuilder(object):
    """A class that builds quadrats on a mesh, starting from the centre position."""

    def build(self, box, size):
        """
            Method to contruct a list of quadrats using a given bounding box.
            @param box the bounding box of the meshes to measure
            @param size the size of the square quadrilateral (in the units of the provided mesh)
            @return list of the quadrats that fit the bounding box
        """
        centroid = box.centroid
        edge1_midpoint = get_midpoint_of_edge(
            box.vertices[0], box.vertices[1])
        distance_to_edge1 = centroid.distance_to_xyz(edge1_midpoint)
        quadrat_indexes1 = range(int(distance_to_edge1 / size) + 1)

        edge2_midpoint = get_midpoint_of_edge(
            box.vertices[0], box.vertices[3])
        distance_to_edge2 = centroid.distance_to_xyz(edge2_midpoint)
        quadrat_indexes2 = range(int(distance_to_edge2 / size) + 1)

        quadrats = list()
        for i in quadrat_indexes2:
            for j in quadrat_indexes1:
                if i == 0 and j == 0:
                    quadrats.append(Quadrat((0, 0), size, centroid))
                else:
                    top_right = Vertex(centroid.x + i * size,
                                       centroid.y + (j * size), centroid.z)
                    top_left = Vertex(centroid.x + i * size,
                                      centroid.y - (j * size), centroid.z)
                    bottom_right = Vertex(
                        centroid.x - i * size, centroid.y + (j * size), centroid.z)
                    bottom_left = Vertex(
                        centroid.x - i * size, centroid.y - (j * size), centroid.z)
                    four_quadrats = [Quadrat((i, j), size, top_right),
                                     Quadrat((i, -1 * j), size, top_left),
                                     Quadrat((-1 * i, j), size, bottom_right),
                                     Quadrat((-1 * i, -1 * j), size, bottom_left)]
                    quadrats_inside = list(
                        filter(lambda x: box.contains(x.midpoint), four_quadrats))
                    quadrats += quadrats_inside
        # print("There are this many quadrats: " + str(len(quadrats)))
        return tuple(set(quadrats))
