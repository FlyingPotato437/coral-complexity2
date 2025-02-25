from ._vertex import Vertex
from ._face import Face
from ._metric import Metric
from ._helpers import get_z_value, mean, sd
import pdb


class Mesh(object):

    def __init__(self, faces, name):
        self.faces = faces
        self.name = name

    def calculate_metrics(self, quadrats):
        metrics = list()
        for quadrat in quadrats:
            metric = Metric()
            quadrat_vertices = list()
            metric.quadrat_id = quadrat.id
            metric.quadrat_midpoint = quadrat.midpoint
            for face in self.faces:
                if quadrat.contains(face.centroid):
                    metric.area3d += face.area3d
                    metric.area2d += face.area2d
                    metric.face_count += 1
                    quadrat_vertices += face.vertices
                else:
                    pass
            quadrat_vertices = list(set(quadrat_vertices))
            metric.vertices_count = len(quadrat_vertices)
            z_values = list(map(get_z_value, quadrat_vertices))
            if len(z_values) > 0:
                metric.relative_z_mean = mean(z_values)
                metric.relative_z_sd = sd(z_values, sample=False)
            metrics.append(metric)
        return metrics
