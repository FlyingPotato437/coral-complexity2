# Optional imports that require additional dependencies
try:
    from .crop_mesh import crop_mesh_to_segments as crop_mesh
    _CROP_MESH_AVAILABLE = True
except ImportError:
    def crop_mesh(*args, **kwargs):
        raise ImportError("Crop mesh functionality requires geopandas. Install with: pip install coral-complexity-metrics[full]")
    _CROP_MESH_AVAILABLE = False

# Core utility functions that don't require heavy dependencies
# (Add other utility imports here as they're created)

__all__ = ['crop_mesh']
