from .vertex import Vertex


class DimensionOrder(object):
    X = 0
    Y = 1
    Z = 2

    def __init__(self, order):
        self.order = str(order)

    def get_first(self):
        return self._get_value(self.order[0])

    def get_second(self):
        return self._get_value(self.order[1])

    def get_third(self):
        return self._get_value(self.order[2])

    def _get_value(self, c):
        result = None
        if c == 'X':
            result = self.X
        elif c == 'Y':
            result = self.Y
        else:
            result = self.Z
        return result

    def get_vertex(self, x, y, z):
        index = self.get_third()
        vertex = None
        if index == 0:
            vertex = Vertex(z, y, x)
        elif index == 1:
            vertex = Vertex(x, z, y)
        else:
            vertex = Vertex(x, y, z)
        return vertex
