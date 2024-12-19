from mesh3d.metric import Metric
import math
import pytest


class TestMetric:

    def test_metric_constructor(self):
        my_metric = Metric()
        assert my_metric is not None
        assert my_metric.quadrat_id == list()
        assert my_metric.quadrat_midpoint is None
        assert my_metric.area3d == 0
        assert my_metric.area2d == 0
        assert my_metric.face_count == 0
        assert my_metric.vertices_count == 0
        assert my_metric.relative_z_mean == 0
        assert my_metric.relative_z_sd == 0

    def test_metric_rugosity_zero_area_returns_nan(self):
        my_metric = Metric()
        my_rugosity = my_metric.surface_rugosity()
        assert math.isnan(my_rugosity)

    def test_metric_rugosity(self):
        my_metric = Metric()
        my_metric.area3d = 3.0
        my_metric.area2d = 2.0
        my_rugosity = my_metric.surface_rugosity()
        assert my_rugosity == 1.5
