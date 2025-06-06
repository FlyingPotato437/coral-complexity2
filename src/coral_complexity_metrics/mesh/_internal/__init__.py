"""
Internal utility modules for coral complexity metrics.

These modules contain legacy implementation details and helper functions.
They are not part of the public API and may change without notice.
"""

# Legacy internal modules
# NOTE: these modules live one level above this package. Use ``..`` imports so
# they resolve correctly when ``_internal`` is imported.  Using ``.`` relative
# imports here results in ``ImportError`` because Python looks for the modules
# inside the ``_internal`` package.
from .._dimension_order import DimensionOrder
from .._face import Face
from .._helpers import get_z_value, mean, sd, get_midpoint_of_edge
from .._mesh import Mesh
from .._quadrat import Quadrat
from .._quadrat_builder import QuadratBuilder
from .._quadrilateral import Quadrilateral
from .._shading_utils import AABB, BVHNode
from .._vertex import Vertex

__all__ = [
    'DimensionOrder', 'Face', 'get_z_value', 'mean', 'sd', 'get_midpoint_of_edge',
    'Mesh', 'Quadrat', 'QuadratBuilder', 'Quadrilateral',
    'AABB', 'BVHNode', 'Vertex'
]
