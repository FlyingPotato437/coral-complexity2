"""Visualization tools for coral complexity metrics."""

# Optional imports that require additional dependencies
try:
    from .mesh_previews import (
        MeshVisualizer,
        generate_mesh_previews
    )
    _VISUALIZATION_AVAILABLE = True
except ImportError:
    # Create placeholders for missing functionality
    def _visualization_not_available(*args, **kwargs):
        raise ImportError("Visualization functionality requires additional dependencies. Install with: pip install coral-complexity-metrics[full]")
    
    MeshVisualizer = _visualization_not_available
    generate_mesh_previews = _visualization_not_available
    _VISUALIZATION_AVAILABLE = False

__all__ = [
    'MeshVisualizer',
    'generate_mesh_previews'
] 