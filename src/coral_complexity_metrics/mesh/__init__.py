"""
Mesh processing and analysis module for coral complexity metrics.

This module provides comprehensive mesh processing capabilities including
validation, geometric measurements, complexity calculations, and 
shapefile-based processing.
"""

# Import core validation functionality
from ._metric import BaseMetric, register_metric, MetricRegistry

# Import basic utilities that don't require heavy dependencies
from .mesh_utils import (
    calculate_projected_area_convex_hull,
    calculate_bounding_box_area, 
    calculate_mesh_coverage_quality
)

# Import unified metrics (these handle their own dependency checking)
from .unified_metrics import *

# Import shading modules (these handle their own dependency checking)  
from .shading_modules import *

# Optional imports - these require heavy dependencies
try:
    from .mesh_validator import MeshValidator, MeshValidationResult, batch_validate_meshes
    from .quadrat_metrics import quadrat_metrics_on_mesh
    from .complexity_metrics import *
    from .geometric_measures import *
    
    # Set availability flags
    _VALIDATION_AVAILABLE = True
    _QUADRAT_METRICS_AVAILABLE = True
    _COMPLEXITY_METRICS_AVAILABLE = True
    _GEOMETRIC_MEASURES_AVAILABLE = True
    
except ImportError as e:
    # Create placeholder functions that warn about missing dependencies
    def _missing_dependency_warning(module_name):
        def wrapper(*args, **kwargs):
            raise ImportError(f"{module_name} requires additional dependencies. Install with: pip install coral-complexity-metrics[full]")
        return wrapper
    
    MeshValidator = _missing_dependency_warning("MeshValidator")
    MeshValidationResult = _missing_dependency_warning("MeshValidationResult") 
    batch_validate_meshes = _missing_dependency_warning("batch_validate_meshes")
    quadrat_metrics_on_mesh = _missing_dependency_warning("quadrat_metrics_on_mesh")
    
    # Set availability flags
    _VALIDATION_AVAILABLE = False
    _QUADRAT_METRICS_AVAILABLE = False
    _COMPLEXITY_METRICS_AVAILABLE = False
    _GEOMETRIC_MEASURES_AVAILABLE = False

# Optional shapefile processing
try:
    from .shapefile_processor import ShapefileMeshProcessor
    _SHAPEFILE_PROCESSING_AVAILABLE = True
except ImportError:
    ShapefileMeshProcessor = _missing_dependency_warning("ShapefileMeshProcessor")
    _SHAPEFILE_PROCESSING_AVAILABLE = False

# Import shading (optional)
try:
    from .shading import Shading
    ShadingClass = Shading
except ImportError:
    ShadingClass = _missing_dependency_warning("Shading")

# Shapefile processing
from .shapefile_processor import (
    process_mesh_shapefile_batch
)

__all__ = [
    # Core metrics framework
    'BaseMetric', 'MetricRegistry', 'register_metric',
    'SurfaceMetric', 'VolumeMetric', 'ComplexityMetric', 'ShadingMetric',
    'list_available_metrics', 'calculate_all_metrics',
    'Rugosity', 'FractalDimension', 'MeshClosure',
    
    # Unified metrics (recommended for new code)
    'Watertight', 'Volume', 'SurfaceArea', 'ConvexHullVolume',
    'ProportionOccupied', 'AbsoluteSpatialRefuge', 'ShelterSizeFactor',
    'Diameter', 'Height', 'QuadratMetrics', 'MeshCounts', 'SurfaceRugosity',
    'Slope', 'PlaneOfBestFit', 'HeightRange', 'FractalDimension', 'ShadingPercentage',
    
    # USYD complexity metrics
    'SlopeMetric', 'FractalDimensionBox', 'SurfaceComplexityIndex', 'VectorDispersion',
    
    # Shading functionality
    'ShadingClass',
    'ModularShadingCalculator', 'create_default_shading_pipeline',
    'LightingModel', 'SolarLightingModel', 'DirectionalLightingModel', 'AmbientLightingModel',
    'EnvironmentalFactor', 'SlopeAspectFactor', 'DepthAttenuationFactor',
    'ShadingMetric', 'PercentageShadingMetric', 'SpatialShadingMetric',
    
    # Mesh processing and validation
    'MeshValidator', 'MeshValidationResult', 'batch_validate_meshes',
    'calculate_projected_area_convex_hull', 'calculate_bounding_box_area',
    'calculate_mesh_coverage_quality', 'is_mesh_watertight',
    'validate_mesh_for_metrics', 'prepare_mesh_data_for_metrics',
    'clip_mesh_by_polygon', 'extract_triangular_faces_from_pv',
    
    # Shapefile processing
    'ShapefileMeshProcessor', 'process_mesh_shapefile_batch',
    
    # Legacy components
    'GeometricMeasures', 'LegacyQuadratMetrics',
    'quadrat_metrics_on_mesh',
]
