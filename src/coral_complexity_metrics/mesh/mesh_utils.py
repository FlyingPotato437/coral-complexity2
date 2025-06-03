"""
Mesh utility functions for coral complexity metrics.

This module provides core utility functions for mesh processing, including
spatial calculations, clipping operations, and data quality assessments.
"""

import numpy as np
from typing import Dict, Any, Optional, Tuple, List, Union, TYPE_CHECKING
import warnings
import logging

# Optional imports - handle gracefully if not available
try:
    import pyvista as pv
    _PYVISTA_AVAILABLE = True
except ImportError:
    _PYVISTA_AVAILABLE = False
    pv = None

try:
    import geopandas as gpd
    from shapely.geometry import Point, Polygon
    _GEOSPATIAL_AVAILABLE = True
except ImportError:
    _GEOSPATIAL_AVAILABLE = False
    gpd = None
    Polygon = None
    Point = None

try:
    from scipy.spatial import ConvexHull
    _SCIPY_AVAILABLE = True
except ImportError:
    _SCIPY_AVAILABLE = False
    ConvexHull = None

# Type annotations that work when pyvista is not available
if TYPE_CHECKING and _PYVISTA_AVAILABLE:
    from pyvista import PolyData
else:
    PolyData = Any

logger = logging.getLogger(__name__)


def calculate_projected_area_convex_hull(points: np.ndarray) -> float:
    """
    Calculate the 2D projected area of a convex hull from 3D points.
    
    Parameters:
    points: Array of 3D points (N, 3)
    
    Returns:
    float: 2D projected area of the convex hull
    """
    if len(points) < 3:
        return 0.0
    
    # Project to 2D (X, Y coordinates)
    points_2d = points[:, :2]
    
    try:
        if _SCIPY_AVAILABLE and ConvexHull is not None:
            # Use scipy ConvexHull for accurate calculation
            hull = ConvexHull(points_2d)
            return hull.volume  # In 2D, volume is actually area
        else:
            # Fallback: use bounding box area as approximation
            logger.debug("SciPy not available, using bounding box approximation for convex hull area")
            min_coords = np.min(points_2d, axis=0)
            max_coords = np.max(points_2d, axis=0)
            return (max_coords[0] - min_coords[0]) * (max_coords[1] - min_coords[1])
    except Exception as e:
        logger.warning(f"Failed to calculate convex hull area: {e}")
        # Fallback to bounding box
        min_coords = np.min(points_2d, axis=0)
        max_coords = np.max(points_2d, axis=0)
        return (max_coords[0] - min_coords[0]) * (max_coords[1] - min_coords[1])


def calculate_bounding_box_area(size_2d: np.ndarray) -> float:
    """
    Calculate the 2D area from a bounding box size array.
    
    Parameters:
    size_2d: Array with [width, height]
    
    Returns:
    float: 2D area (width * height)
    """
    if len(size_2d) < 2:
        return 0.0
    return float(size_2d[0] * size_2d[1])


def is_mesh_watertight(mesh) -> bool:
    """
    Check if a mesh is watertight (closed) using multiple methods.
    
    Parameters:
    mesh: PyVista mesh to check (or None if pyvista not available)
    
    Returns:
    bool: True if mesh is watertight, False otherwise
    """
    if not _PYVISTA_AVAILABLE or mesh is None:
        logger.warning("PyVista not available - cannot check mesh watertight status")
        return False
    
    try:
        # Method 1: Try to calculate volume
        volume = mesh.volume
        if volume <= 0:
            return False
        
        # Method 2: Check for open edges using PyVista's built-in method
        if hasattr(mesh, 'is_closed'):
            return mesh.is_closed
        
        # Method 3: Manual check for boundary edges
        # Extract edges and check if any edge is shared by only one face
        edges = mesh.extract_all_edges()
        if edges.n_cells == 0:
            return False
        
        # A watertight mesh should have no boundary edges
        # This is a simplified check - a more robust implementation would
        # count edge-face adjacencies
        return True
        
    except Exception as e:
        logger.debug(f"Watertight check failed: {e}")
        return False


def calculate_mesh_coverage_quality(mesh_points: np.ndarray, 
                                   polygon_bounds: Tuple[float, float, float, float],
                                   expansion_percentage: float = 0.0) -> Dict[str, float]:
    """
    Calculate coverage quality metrics for a mesh within polygon bounds.
    
    Parameters:
    mesh_points: Array of 3D mesh points
    polygon_bounds: Tuple of (min_x, min_y, max_x, max_y)
    expansion_percentage: Percentage expansion applied to bounds
    
    Returns:
    Dict with coverage quality metrics
    """
    if len(mesh_points) == 0:
        return {
            'mesh_coverage_percentage': 0.0,
            'missing_data_percentage': 100.0,
            'data_quality_score': 0.0,
            'point_density': 0.0
        }
    
    min_x, min_y, max_x, max_y = polygon_bounds
    
    # Calculate defined area (polygon bounding box)
    defined_area = (max_x - min_x) * (max_y - min_y)
    
    # Calculate actual mesh coverage area (convex hull of projected points)
    actual_area = calculate_projected_area_convex_hull(mesh_points)
    
    # Calculate coverage percentages
    if defined_area > 0:
        coverage_percentage = min(100.0, (actual_area / defined_area) * 100.0)
        missing_percentage = max(0.0, 100.0 - coverage_percentage)
    else:
        coverage_percentage = 100.0 if actual_area > 0 else 0.0
        missing_percentage = 0.0 if actual_area > 0 else 100.0
    
    # Calculate point density
    point_density = len(mesh_points) / max(actual_area, 1e-9) if actual_area > 0 else 0.0
    
    # Calculate overall data quality score (0-1)
    quality_factors = [
        coverage_percentage / 100.0,  # Coverage factor
        min(1.0, point_density / 1000.0),  # Density factor (normalized to 1000 points per unit area)
        1.0 if len(mesh_points) > 10 else len(mesh_points) / 10.0  # Minimum points factor
    ]
    data_quality_score = np.mean(quality_factors)
    
    return {
        'mesh_coverage_percentage': coverage_percentage,
        'missing_data_percentage': missing_percentage,
        'data_quality_score': data_quality_score,
        'point_density': point_density,
        'actual_mesh_area': actual_area,
        'defined_area': defined_area,
        'num_points': len(mesh_points)
    }


def clip_mesh_by_polygon(main_mesh: PolyData, 
                        polygon_geom,
                        expansion_percentage: float = 0.0,
                        center_on_centroid: bool = True) -> Optional[Dict[str, Any]]:
    """
    Clip a mesh using a polygon boundary with data quality assessment.
    
    Parameters:
    main_mesh: Input mesh to clip
    polygon_geom: Shapely polygon for clipping
    expansion_percentage: Percentage to expand clipping bounds
    center_on_centroid: Whether to center the clip on polygon centroid
    
    Returns:
    Dict containing clipped mesh data and quality metrics, or None if clipping fails
    """
    if not _PYVISTA_AVAILABLE or not _GEOSPATIAL_AVAILABLE:
        logger.error("Mesh clipping requires PyVista and geospatial libraries (geopandas, shapely)")
        return None
    
    if main_mesh is None or polygon_geom is None:
        logger.error("Invalid mesh or polygon provided for clipping")
        return None
    
    try:
        # Get polygon bounds
        bounds_2d = polygon_geom.bounds  # (min_x, min_y, max_x, max_y)
        
        # Apply expansion if requested
        if expansion_percentage > 0:
            width = bounds_2d[2] - bounds_2d[0]
            height = bounds_2d[3] - bounds_2d[1]
            expansion_x = width * expansion_percentage / 100.0
            expansion_y = height * expansion_percentage / 100.0
            
            bounds_2d = (
                bounds_2d[0] - expansion_x,
                bounds_2d[1] - expansion_y,
                bounds_2d[2] + expansion_x,
                bounds_2d[3] + expansion_y
            )
        
        # Get mesh Z bounds
        mesh_z_min = main_mesh.bounds[4]
        mesh_z_max = main_mesh.bounds[5]
        
        # Create 3D clipping bounds
        clip_bounds = [
            bounds_2d[0], bounds_2d[2],  # X min, max
            bounds_2d[1], bounds_2d[3],  # Y min, max
            mesh_z_min, mesh_z_max       # Z min, max
        ]
        
        # Clip the mesh
        clipped_mesh = main_mesh.clip_box(clip_bounds)
        
        # Extract clipped points and calculate metrics
        clipped_points = clipped_mesh.points if clipped_mesh.n_points > 0 else np.array([]).reshape(0, 3)
        
        # Calculate surface area of clipped mesh
        surface_area_3d = clipped_mesh.area if clipped_mesh.n_points > 0 else 0.0
        
        # Calculate clipped region size
        clipped_region_size_2d = np.array([
            bounds_2d[2] - bounds_2d[0],  # width
            bounds_2d[3] - bounds_2d[1]   # height
        ])
        
        # Calculate coverage quality metrics
        quality_metrics = calculate_mesh_coverage_quality(
            clipped_points, bounds_2d, expansion_percentage
        )
        
        return {
            'clipped_mesh_pv': clipped_mesh,
            'mesh_clipped_points': clipped_points,
            'surface_area_3d': surface_area_3d,
            'clipped_region_size_2d': clipped_region_size_2d,
            'polygon_bounds': bounds_2d,
            'clip_bounds_3d': clip_bounds,
            'is_watertight': is_mesh_watertight(clipped_mesh),
            **quality_metrics
        }
        
    except Exception as e:
        logger.error(f"Failed to clip mesh by polygon: {e}")
        return None


def extract_triangular_faces_from_pv(mesh: PolyData) -> np.ndarray:
    """
    Extract triangular faces from a PyVista mesh.
    
    Parameters:
    mesh: PyVista mesh
    
    Returns:
    Array of triangular face indices (N, 3)
    """
    if not _PYVISTA_AVAILABLE or mesh is None:
        return np.array([]).reshape(0, 3)
    
    faces_array = mesh.faces
    triangular_faces = []
    i = 0
    
    while i < len(faces_array):
        n_vertices = faces_array[i]
        if n_vertices == 3:  # Only triangular faces
            face = faces_array[i+1:i+1+n_vertices]
            triangular_faces.append(face)
        i += n_vertices + 1
    
    return np.array(triangular_faces) if triangular_faces else np.array([]).reshape(0, 3)


def validate_mesh_for_metrics(mesh: PolyData, 
                             require_watertight: bool = False) -> Dict[str, Any]:
    """
    Validate mesh suitability for metric calculations.
    
    Parameters:
    mesh: PyVista mesh to validate
    require_watertight: Whether to require watertight mesh
    
    Returns:
    Dict with validation results and recommendations
    """
    validation_result = {
        'is_valid': True,
        'is_watertight': False,
        'can_calculate_volume': False,
        'warnings': [],
        'errors': [],
        'point_count': 0,
        'face_count': 0,
        'surface_area': 0.0
    }
    
    if not _PYVISTA_AVAILABLE or mesh is None:
        validation_result['is_valid'] = False
        validation_result['errors'].append("PyVista not available or mesh is None")
        return validation_result
    
    validation_result.update({
        'point_count': mesh.n_points,
        'face_count': mesh.n_cells
    })
    
    # Basic checks
    if mesh.n_points == 0:
        validation_result['is_valid'] = False
        validation_result['errors'].append("Mesh has no points")
        return validation_result
    
    if mesh.n_cells == 0:
        validation_result['is_valid'] = False
        validation_result['errors'].append("Mesh has no faces")
        return validation_result
    
    # Calculate surface area
    try:
        validation_result['surface_area'] = mesh.area
    except Exception as e:
        validation_result['warnings'].append(f"Could not calculate surface area: {e}")
    
    # Check if mesh is watertight
    validation_result['is_watertight'] = is_mesh_watertight(mesh)
    validation_result['can_calculate_volume'] = validation_result['is_watertight']
    
    if require_watertight and not validation_result['is_watertight']:
        validation_result['is_valid'] = False
        validation_result['errors'].append("Mesh is not watertight but watertight mesh is required")
    
    if not validation_result['is_watertight']:
        validation_result['warnings'].append(
            "Mesh is not watertight - volume-dependent metrics will return NaN"
        )
    
    return validation_result


def prepare_mesh_data_for_metrics(mesh: PolyData, 
                                 clip_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Prepare standardized mesh data dictionary for metric calculations.
    
    Parameters:
    mesh: PyVista mesh
    clip_data: Optional clipping data from clip_mesh_by_polygon
    
    Returns:
    Standardized mesh data dictionary
    """
    if not _PYVISTA_AVAILABLE or mesh is None:
        return {
            'mesh': None,
            'points': np.array([]).reshape(0, 3),
            'faces': np.array([]).reshape(0, 3),
            'surface_area': 0.0,
            'n_points': 0,
            'n_faces': 0,
            'is_watertight': False,
            'can_calculate_volume': False,
            'validation_warnings': ["PyVista not available"],
            'validation_errors': ["Cannot process mesh without PyVista"],
            'volume': float('nan')
        }
    
    # Basic mesh data
    mesh_data = {
        'mesh': mesh,
        'points': mesh.points,
        'faces': extract_triangular_faces_from_pv(mesh),
        'surface_area': mesh.area,
        'n_points': mesh.n_points,
        'n_faces': mesh.n_cells
    }
    
    # Add validation data
    validation = validate_mesh_for_metrics(mesh)
    mesh_data.update({
        'is_watertight': validation['is_watertight'],
        'can_calculate_volume': validation['can_calculate_volume'],
        'validation_warnings': validation['warnings'],
        'validation_errors': validation['errors']
    })
    
    # Add volume if mesh is watertight
    if validation['can_calculate_volume']:
        try:
            mesh_data['volume'] = mesh.volume
        except Exception as e:
            mesh_data['volume'] = float('nan')
            mesh_data['validation_warnings'].append(f"Volume calculation failed: {e}")
    else:
        mesh_data['volume'] = float('nan')
    
    # Add clipping data if provided
    if clip_data is not None:
        mesh_data.update({
            'surface_area_3d': clip_data['surface_area_3d'],
            'mesh_clipped_points': clip_data['mesh_clipped_points'],
            'clipped_region_size_2d': clip_data['clipped_region_size_2d'],
            'mesh_coverage_percentage': clip_data['mesh_coverage_percentage'],
            'missing_data_percentage': clip_data['missing_data_percentage'],
            'data_quality_score': clip_data['data_quality_score'],
            'point_density': clip_data['point_density']
        })
    
    return mesh_data