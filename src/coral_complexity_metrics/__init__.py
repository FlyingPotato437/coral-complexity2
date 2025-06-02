"""
Coral Complexity Metrics - A comprehensive package for 3D coral analysis.

This package provides tools for:
- Enhanced shading analysis with parametric environmental inputs
- Comprehensive structural complexity metrics
- Mesh validation and repair
- Validation against in-situ measurements
- Automated visualization and quality control

Key Features:
- Structural-only shading analysis with clear scope documentation
- Extensible metric registration system
- Pure Python/NumPy complexity metrics translated from C++
- Automated mesh validation and repair using PyVista and PyMeshLab
- Light logger data comparison and validation
- HTML/PNG preview generation with quality flagging
"""

from .mesh import *
from .mesh.shading import Shading
from .mesh._metric import (
    BaseMetric, SurfaceMetric, VolumeMetric, ComplexityMetric, ShadingMetric,
    MetricRegistry, register_metric, get_metric_registry, 
    list_available_metrics, calculate_all_metrics
)
from .mesh.complexity_metrics import (
    SlopeMetric, PlaneOfBestFit, HeightRange, FractalDimensionBox,
    SurfaceComplexityIndex, VectorDispersion
)
from .mesh.geometric_measures import GeometricMeasures
from .mesh.quadrat_metrics import QuadratMetrics

# Conditional imports for optional dependencies
try:
    from .mesh.mesh_validator import (
        MeshValidator, MeshValidationResult, 
        validate_and_repair_mesh, batch_validate_meshes
    )
    HAS_MESH_VALIDATION = True
except ImportError:
    HAS_MESH_VALIDATION = False

try:
    from .validation.shading_validator import (
        ShadingValidator, LightLoggerData, ShadingComparison, 
        ValidationMetrics, run_validation_study
    )
    HAS_SHADING_VALIDATION = True
except ImportError:
    HAS_SHADING_VALIDATION = False

try:
    from .visualization.mesh_previews import (
        MeshVisualizer, generate_mesh_previews
    )
    HAS_VISUALIZATION = True
except ImportError:
    HAS_VISUALIZATION = False

try:
    from .utils.crop_mesh import crop_mesh_to_segments as crop_mesh
    HAS_CROP_MESH = True
except ImportError:
    HAS_CROP_MESH = False

# Version info
__version__ = "0.1.0"
__author__ = "Hannah White, AIMS"
__email__ = "ha.white@aims.gov.au"

# Core exports (always available)
__all__ = [
    # Core shading
    "Shading",
    
    # Metric system
    "BaseMetric", "SurfaceMetric", "VolumeMetric", "ComplexityMetric", "ShadingMetric",
    "MetricRegistry", "register_metric", "get_metric_registry",
    "list_available_metrics", "calculate_all_metrics",
    
    # Complexity metrics
    "SlopeMetric", "PlaneOfBestFit", "HeightRange", "FractalDimensionBox",
    "SurfaceComplexityIndex", "VectorDispersion",
    
    # Legacy components
    "GeometricMeasures", "QuadratMetrics",
    
    # Package info
    "__version__", "__author__", "__email__",
]

# Conditional exports
if HAS_MESH_VALIDATION:
    __all__.extend([
        "MeshValidator", "MeshValidationResult", 
        "validate_and_repair_mesh", "batch_validate_meshes"
    ])

if HAS_SHADING_VALIDATION:
    __all__.extend([
        "ShadingValidator", "LightLoggerData", "ShadingComparison",
        "ValidationMetrics", "run_validation_study"
    ])

if HAS_VISUALIZATION:
    __all__.extend([
        "MeshVisualizer", "generate_mesh_previews"
    ])

if HAS_CROP_MESH:
    __all__.extend(["crop_mesh"])


def check_dependencies():
    """Check which optional dependencies are available."""
    deps = {
        "mesh_validation": HAS_MESH_VALIDATION,
        "shading_validation": HAS_SHADING_VALIDATION, 
        "visualization": HAS_VISUALIZATION,
        "crop_mesh": HAS_CROP_MESH,
    }
    
    missing = [name for name, available in deps.items() if not available]
    
    if missing:
        print(f"Optional dependencies not available: {', '.join(missing)}")
        print("Install with: pip install coral-complexity-metrics[dev,viz]")
    else:
        print("All optional dependencies available ✓")
    
    return deps


def get_info():
    """Get package information and feature availability."""
    return {
        "version": __version__,
        "author": __author__,
        "features": {
            "enhanced_shading": True,
            "metric_registry": True,
            "complexity_metrics": True,
            "mesh_validation": HAS_MESH_VALIDATION,
            "shading_validation": HAS_SHADING_VALIDATION,
            "visualization": HAS_VISUALIZATION,
            "crop_mesh": HAS_CROP_MESH,
        }
    }


# Initialize metric registry with default metrics
def _initialize_metrics():
    """Initialize the metric registry with all available metrics."""
    try:
        # The complexity metrics are automatically registered when imported
        # Additional metrics can be registered here
        pass
    except Exception as e:
        import warnings
        warnings.warn(f"Failed to initialize some metrics: {e}")

_initialize_metrics()
