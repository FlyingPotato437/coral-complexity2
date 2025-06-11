"""
Coral Complexity Metrics

A Python package for calculating structural complexity metrics from 3D coral mesh files.
This tool provides quantitative measures of coral structure using 3D models in OBJ or PLY format,
supporting both plot-level and quadrat-level metrics, and can process individual files or entire directories.
"""

# Try to import mesh functionality
try:
    from . import mesh
    _MESH_AVAILABLE = True
except ImportError as e:
    _MESH_AVAILABLE = False
    
    def _mesh_not_available(*args, **kwargs):
        raise ImportError("Mesh functionality requires additional dependencies. Install with: pip install coral-complexity-metrics[full]")
    
    class _MeshModulePlaceholder:
        def __getattr__(self, name):
            return _mesh_not_available
    
    mesh = _MeshModulePlaceholder()

# Try to import ComplexityMetrics class
try:
    from .mesh.complexity_metrics import ComplexityMetrics
    _COMPLEXITY_CLASS_AVAILABLE = True
except ImportError:
    def ComplexityMetrics(*args, **kwargs):
        raise ImportError("ComplexityMetrics requires additional dependencies")
    _COMPLEXITY_CLASS_AVAILABLE = False

# Package metadata
__version__ = "1.0.0"
__author__ = "Hannah White"
__email__ = "ha.white@aims.gov.au"

def get_info():
    """Get package information and feature availability."""
    return {
        'version': __version__,
        'author': __author__,
        'email': __email__,
        'features': {
            'mesh_processing': _MESH_AVAILABLE,
            'complexity_metrics': _COMPLEXITY_CLASS_AVAILABLE,
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
    optional_deps = ['pyvista', 'scipy', 'pandas']
    
    for dep in optional_deps:
        try:
            module = __import__(dep)
            version = getattr(module, '__version__', 'unknown')
            deps['optional'][dep] = {'available': True, 'version': version}
        except ImportError:
            deps['optional'][dep] = {'available': False, 'version': None}
    
    return deps


# Export main functionality
__all__ = [
    'ComplexityMetrics',
    'get_info', 
    'check_dependencies', 
    '__version__'
]
