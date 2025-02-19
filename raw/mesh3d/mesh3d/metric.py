import math


class Metric(object):
    """A class to store the metrics to be measured"""

    def __init__(self):
        self.quadrat_id = list()
        self.quadrat_midpoint = None
        self.area3d = 0
        self.area2d = 0
        self.face_count = 0
        self.vertices_count = 0
        self.relative_z_mean = 0
        self.relative_z_sd = 0

    def surface_rugosity(self):
        """Returns the rugosity of the area"""
        if self.area2d == 0:
            return float('nan')
        return float(self.area3d) / float(self.area2d)
