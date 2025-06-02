from .shading import Shading
from .geometric_measures import GeometricMeasures
from .quadrat_metrics import QuadratMetrics

# Added exports for broader library usability
from .mesh_utils import clip_mesh_by_polygon, calculate_projected_area_convex_hull, calculate_bounding_box_area
from ._metric import (
    BaseMetric, SurfaceMetric, VolumeMetric, ComplexityMetric, ShadingMetric, 
    Rugosity, FractalDimension, MeshClosure, 
    MetricRegistry, get_metric_registry, register_metric, list_available_metrics, calculate_all_metrics
)
from .mesh_validator import MeshValidator, MeshValidationResult
from .complexity_metrics import SlopeMetric # Assuming SlopeMetric is a key one to export, add others if necessary
# Consider exporting other specific complexity metrics if they are commonly used directly
# e.g., from .complexity_metrics import HeightRangeMetric, PlaneOfBestFitMetric etc.
