"""
EcoRRAP: Enhanced Coral Reef Complexity Metrics Package

A comprehensive Python package for analyzing 3D coral reef structural complexity
using photogrammetry and LiDAR-derived mesh data with support for shading analysis,
geometric measurements, and spatial metrics calculation.

Key features:
- Enhanced shading analysis with CPU percentage control
- Mesh-by-shapefile cropping with data quality assessment  
- Unified metrics system with proper non-closed mesh handling
- USYD complexity metrics (slope, plane of best fit, height range, fractal dimensions)
- Comprehensive data validation and quality scoring
"""

# Core functionality that doesn't require heavy dependencies
from . import utils
from . import validation  
from . import visualization

# Try to import mesh functionality - this now handles dependencies gracefully
try:
    from . import mesh
    from .mesh import (
        BaseMetric, MetricRegistry, register_metric,
        calculate_projected_area_convex_hull, calculate_bounding_box_area,
        calculate_mesh_coverage_quality
    )
    _MESH_AVAILABLE = True
except ImportError as e:
    _MESH_AVAILABLE = False
    
    # Create warning functions
    def _mesh_not_available(*args, **kwargs):
        raise ImportError("Mesh functionality requires additional dependencies. Install with: pip install coral-complexity-metrics[full]")
    
    class _MeshModulePlaceholder:
        def __getattr__(self, name):
            return _mesh_not_available
    
    mesh = _MeshModulePlaceholder()

# Try to import shading functionality
try:
    from .mesh.shading import Shading
    _SHADING_AVAILABLE = True
except ImportError:
    def Shading(*args, **kwargs):
        raise ImportError("Shading analysis requires PyVista. Install with: pip install coral-complexity-metrics[full]")
    _SHADING_AVAILABLE = False

# Package metadata
__version__ = "2.0.0"
__author__ = "EcoRRAP Development Team"
__email__ = "support@ecorap.org"

def get_info():
    """Get package information and feature availability."""
    return {
        'version': __version__,
        'author': __author__,
        'email': __email__,
        'features': {
            'mesh_processing': _MESH_AVAILABLE,
            'shading_analysis': _SHADING_AVAILABLE,
            'shapefile_processing': getattr(mesh, '_SHAPEFILE_PROCESSING_AVAILABLE', False) if _MESH_AVAILABLE else False,
            'validation': getattr(mesh, '_VALIDATION_AVAILABLE', False) if _MESH_AVAILABLE else False,
            'complexity_metrics': getattr(mesh, '_COMPLEXITY_METRICS_AVAILABLE', False) if _MESH_AVAILABLE else False,
            'geometric_measures': getattr(mesh, '_GEOMETRIC_MEASURES_AVAILABLE', False) if _MESH_AVAILABLE else False,
        }
    }

def check_dependencies():
    """Check availability of optional dependencies."""
    deps = {
        'core': {},
        'optional': {}
    }
    
    # Check core dependencies
    for dep in ['numpy']:
        try:
            __import__(dep)
            deps['core'][dep] = {'available': True, 'version': None}
        except ImportError:
            deps['core'][dep] = {'available': False, 'version': None}
    
    # Check optional dependencies
    optional_deps = [
        'pyvista', 'scikit-learn', 'scipy', 'pandas', 
        'geopandas', 'shapely', 'rasterio', 'matplotlib'
    ]
    
    for dep in optional_deps:
        try:
            module = __import__(dep)
            version = getattr(module, '__version__', 'unknown')
            deps['optional'][dep] = {'available': True, 'version': version}
        except ImportError:
            deps['optional'][dep] = {'available': False, 'version': None}
    
    return deps

# Convenience functions that work with available features
def list_available_metrics():
    """List all available metrics."""
    if not _MESH_AVAILABLE:
        return {'error': 'Mesh functionality not available - missing dependencies'}
    
    try:
        return mesh.list_available_metrics()
    except Exception as e:
        return {'error': str(e)}

def get_available_metrics():
    """Get detailed information about available metrics."""
    if not _MESH_AVAILABLE:
        return {'error': 'Mesh functionality not available - missing dependencies'}
    
    try:
        registry = mesh.MetricRegistry()
        categories = {}
        
        for category in ['surface', 'volume', 'complexity', 'shading']:
            metrics = registry.list_metrics(category)
            categories[category] = metrics
        
        return {
            'total_count': len(registry.list_metrics()),
            'categories': categories
        }
    except Exception as e:
        return {'error': str(e)}

def process_mesh_with_shapefile(*args, **kwargs):
    """Convenience function for mesh-shapefile processing."""
    if not _MESH_AVAILABLE:
        raise ImportError("Mesh processing requires additional dependencies")
    
    if not getattr(mesh, '_SHAPEFILE_PROCESSING_AVAILABLE', False):
        raise ImportError("Shapefile processing requires geopandas and shapely")
    
    processor = mesh.ShapefileMeshProcessor()
    return processor.process_mesh_with_shapefile(*args, **kwargs)

def validate_and_repair_mesh(*args, **kwargs):
    """Convenience function for mesh validation and repair."""
    if not _MESH_AVAILABLE:
        raise ImportError("Mesh validation requires additional dependencies")
    
    if not getattr(mesh, '_VALIDATION_AVAILABLE', False):
        raise ImportError("Mesh validation requires PyVista")
    
    validator = mesh.MeshValidator()
    return validator.validate_and_repair(*args, **kwargs)

# Legacy compatibility and test imports
try:
    from .mesh.quadrat_metrics import QuadratMetrics
    from .mesh.complexity_metrics import *
    from .mesh.geometric_measures import GeometricMeasures
    _TEST_CLASSES_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import test classes: {e}")
    _TEST_CLASSES_AVAILABLE = False

# Import AIMS-compatible ComplexityMetrics class for backward compatibility
try:
    from .mesh.complexity_metrics import ComplexityMetrics
    _COMPLEXITY_CLASS_AVAILABLE = True
except ImportError:
    def ComplexityMetrics(*args, **kwargs):
        raise ImportError("ComplexityMetrics requires additional dependencies")
    _COMPLEXITY_CLASS_AVAILABLE = False

# Export main functionality
__all__ = [
    # Package info
    'get_info', 'check_dependencies', '__version__',
    
    # Core functionality (when available)
    'Shading', 'mesh', 'utils', 'validation', 'visualization',
    
    # Convenience functions
    'list_available_metrics', 'get_available_metrics',
    'process_mesh_with_shapefile', 'validate_and_repair_mesh',
    
    # Test classes and AIMS compatibility
    'GeometricMeasures', 'QuadratMetrics', 'ComplexityMetrics',
]
