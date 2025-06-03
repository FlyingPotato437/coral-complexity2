"""
Unified metrics system for coral complexity analysis.

This module provides a consolidated and standardized approach to calculating
all coral complexity metrics with consistent naming, proper watertight mesh
handling, and support for different application contexts.
"""

import numpy as np
from typing import Dict, Any, Optional, List, Union, Tuple
import warnings
import logging
import math

# Optional imports - handle gracefully if not available
try:
    import pyvista as pv
    _PYVISTA_AVAILABLE = True
except ImportError:
    _PYVISTA_AVAILABLE = False
    pv = None

try:
    from scipy.spatial import ConvexHull
    from scipy import stats
    import scipy.linalg
    _SCIPY_AVAILABLE = True
except ImportError:
    _SCIPY_AVAILABLE = False
    ConvexHull = None

try:
    from sklearn.neighbors import NearestNeighbors
    _SKLEARN_AVAILABLE = True
except ImportError:
    _SKLEARN_AVAILABLE = False
    NearestNeighbors = None

from ._metric import BaseMetric, SurfaceMetric, VolumeMetric, ComplexityMetric, ShadingMetric
from .mesh_utils import (
    calculate_projected_area_convex_hull,
    calculate_bounding_box_area,
    is_mesh_watertight
)

logger = logging.getLogger(__name__)


class Watertight(VolumeMetric):
    """Mesh watertight status and closure quality assessment."""
    
    @property
    def name(self) -> str:
        return "watertight"
    
    @property
    def description(self) -> str:
        return "Assessment of mesh closure status and quality"
    
    def calculate(self, mesh_data: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Calculate watertight status and quality metrics."""
        mesh = mesh_data.get('mesh')
        
        if mesh is None:
            return {
                'is_watertight': False,
                'closure_quality': 0.0,
                'can_calculate_volume': False
            }
        
        is_watertight = is_mesh_watertight(mesh)
        
        # Calculate closure quality score (0-1)
        quality_score = 1.0 if is_watertight else 0.0
        
        # Additional quality factors could be added here
        # (e.g., hole count, boundary edge count, etc.)
        
        return {
            'is_watertight': is_watertight,
            'closure_quality': quality_score,
            'can_calculate_volume': is_watertight
        }


class Volume(VolumeMetric):
    """Mesh volume calculation for watertight meshes."""
    
    @property
    def name(self) -> str:
        return "volume"
    
    @property
    def description(self) -> str:
        return "3D volume of watertight mesh"
    
    def calculate(self, mesh_data: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Calculate mesh volume (only for watertight meshes)."""
        mesh = mesh_data.get('mesh')
        is_watertight = mesh_data.get('is_watertight', False)
        
        if mesh is None or not is_watertight:
            return {'volume': float('nan')}
        
        try:
            volume = mesh.volume
            return {'volume': volume}
        except Exception as e:
            logger.warning(f"Volume calculation failed: {e}")
            return {'volume': float('nan')}


class SurfaceArea(SurfaceMetric):
    """3D surface area calculation."""
    
    @property
    def name(self) -> str:
        return "surface_area"
    
    @property
    def description(self) -> str:
        return "3D surface area of the mesh"
    
    def calculate(self, mesh_data: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Calculate 3D surface area."""
        mesh = mesh_data.get('mesh')
        
        if mesh is None:
            return {'surface_area': float('nan')}
        
        try:
            surface_area = mesh.area
            return {'surface_area': surface_area}
        except Exception as e:
            logger.warning(f"Surface area calculation failed: {e}")
            return {'surface_area': float('nan')}


class ConvexHullVolume(VolumeMetric):
    """Convex hull volume calculation."""
    
    @property
    def name(self) -> str:
        return "convex_hull_volume"
    
    @property
    def description(self) -> str:
        return "Volume of the convex hull"
    
    def calculate(self, mesh_data: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Calculate convex hull volume."""
        points = mesh_data.get('points')
        
        if points is None or len(points) < 4:
            return {'convex_hull_volume': float('nan')}
        
        if not _SCIPY_AVAILABLE or ConvexHull is None:
            logger.warning("SciPy not available - cannot calculate convex hull volume")
            return {'convex_hull_volume': float('nan')}
        
        try:
            hull = ConvexHull(points)
            return {'convex_hull_volume': hull.volume}
        except Exception as e:
            logger.warning(f"Convex hull volume calculation failed: {e}")
            return {'convex_hull_volume': float('nan')}


class ProportionOccupied(VolumeMetric):
    """Proportion of convex hull volume occupied by the mesh."""
    
    @property
    def name(self) -> str:
        return "proportion_occupied"
    
    @property
    def description(self) -> str:
        return "Ratio of mesh volume to convex hull volume"
    
    def calculate(self, mesh_data: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Calculate proportion occupied (mesh volume / convex hull volume)."""
        # Get volume and convex hull volume
        volume_calc = Volume().calculate(mesh_data)
        convex_hull_calc = ConvexHullVolume().calculate(mesh_data)
        
        mesh_volume = volume_calc.get('volume', float('nan'))
        hull_volume = convex_hull_calc.get('convex_hull_volume', float('nan'))
        
        if np.isnan(mesh_volume) or np.isnan(hull_volume) or hull_volume == 0:
            return {'proportion_occupied': float('nan')}
        
        proportion = mesh_volume / hull_volume
        return {'proportion_occupied': proportion}


class AbsoluteSpatialRefuge(VolumeMetric):
    """Absolute spatial refuge (available space ratio)."""
    
    @property
    def name(self) -> str:
        return "absolute_spatial_refuge"
    
    @property
    def description(self) -> str:
        return "Available space within convex hull (convex hull volume - mesh volume)"
    
    def calculate(self, mesh_data: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Calculate absolute spatial refuge."""
        # Get volume and convex hull volume
        volume_calc = Volume().calculate(mesh_data)
        convex_hull_calc = ConvexHullVolume().calculate(mesh_data)
        
        mesh_volume = volume_calc.get('volume', float('nan'))
        hull_volume = convex_hull_calc.get('convex_hull_volume', float('nan'))
        
        if np.isnan(mesh_volume) or np.isnan(hull_volume):
            return {'absolute_spatial_refuge': float('nan')}
        
        refuge = hull_volume - mesh_volume
        return {'absolute_spatial_refuge': refuge}


class ShelterSizeFactor(SurfaceMetric):
    """Shelter size factor (available space ratio / surface area)."""
    
    @property
    def name(self) -> str:
        return "shelter_size_factor"
    
    @property
    def description(self) -> str:
        return "Ratio of available space to surface area"
    
    def calculate(self, mesh_data: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Calculate shelter size factor."""
        # Get absolute spatial refuge and surface area
        refuge_calc = AbsoluteSpatialRefuge().calculate(mesh_data)
        surface_calc = SurfaceArea().calculate(mesh_data)
        
        refuge = refuge_calc.get('absolute_spatial_refuge', float('nan'))
        surface_area = surface_calc.get('surface_area', float('nan'))
        
        if np.isnan(refuge) or np.isnan(surface_area) or surface_area == 0:
            return {'shelter_size_factor': float('nan')}
        
        factor = refuge / surface_area
        return {'shelter_size_factor': factor}


class Diameter(SurfaceMetric):
    """Maximum extent in X-Y plane."""
    
    @property
    def name(self) -> str:
        return "diameter"
    
    @property
    def description(self) -> str:
        return "Maximum extent of mesh in X-Y plane"
    
    def calculate(self, mesh_data: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Calculate diameter (maximum X-Y extent)."""
        points = mesh_data.get('points')
        
        if points is None or len(points) == 0:
            return {'diameter': float('nan')}
        
        # Calculate X and Y ranges
        x_range = np.max(points[:, 0]) - np.min(points[:, 0])
        y_range = np.max(points[:, 1]) - np.min(points[:, 1])
        
        # Diameter is the maximum of X and Y ranges
        diameter = max(x_range, y_range)
        
        return {'diameter': diameter}


class Height(SurfaceMetric):
    """Vertical extent (Z-direction range)."""
    
    @property
    def name(self) -> str:
        return "height"
    
    @property
    def description(self) -> str:
        return "Vertical extent of mesh in Z direction"
    
    def calculate(self, mesh_data: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Calculate height (Z-direction range)."""
        points = mesh_data.get('points')
        
        if points is None or len(points) == 0:
            return {'height': float('nan')}
        
        height = np.max(points[:, 2]) - np.min(points[:, 2])
        return {'height': height}


class QuadratMetrics(SurfaceMetric):
    """Quadrat-specific spatial identifiers and extents."""
    
    @property
    def name(self) -> str:
        return "quadrat_metrics"
    
    @property
    def description(self) -> str:
        return "Spatial extent and position information for quadrat analysis"
    
    def calculate(self, mesh_data: Dict[str, Any], 
                 context: str = 'whole_mesh',
                 quadrat_size: Optional[float] = None,
                 quadrat_position: Optional[Tuple[int, int]] = None,
                 **kwargs) -> Dict[str, Any]:
        """
        Calculate quadrat metrics based on context.
        
        Parameters:
        context: 'whole_mesh', 'quadrat', or 'crop'
        quadrat_size: Size of quadrat (for quadrat context)
        quadrat_position: (x, y) position of quadrat (for quadrat context)
        """
        points = mesh_data.get('points')
        
        if points is None or len(points) == 0:
            return {
                'quadrat_size': float('nan'),
                'quadrat_x': float('nan'),
                'quadrat_y': float('nan'),
                'x_min': float('nan'),
                'y_min': float('nan'),
                'x_max': float('nan'),
                'y_max': float('nan'),
                'x_center': float('nan'),
                'y_center': float('nan'),
                'planar_area': float('nan')
            }
        
        # Calculate spatial extents
        x_min, x_max = np.min(points[:, 0]), np.max(points[:, 0])
        y_min, y_max = np.min(points[:, 1]), np.max(points[:, 1])
        x_center = (x_min + x_max) / 2
        y_center = (y_min + y_max) / 2
        
        # Calculate planar area
        if context == 'crop':
            # For crops, use actual mesh coverage area
            planar_area = mesh_data.get('actual_mesh_2d_area', 
                                      calculate_projected_area_convex_hull(points))
        else:
            # For whole mesh or quadrats, use bounding box area
            planar_area = (x_max - x_min) * (y_max - y_min)
        
        result = {
            'x_min': x_min,
            'y_min': y_min,
            'x_max': x_max,
            'y_max': y_max,
            'x_center': x_center,
            'y_center': y_center,
            'planar_area': planar_area
        }
        
        # Context-specific values
        if context == 'quadrat':
            result.update({
                'quadrat_size': quadrat_size or float('nan'),
                'quadrat_x': quadrat_position[0] if quadrat_position else float('nan'),
                'quadrat_y': quadrat_position[1] if quadrat_position else float('nan')
            })
        elif context == 'crop':
            # For crops, quadrat size represents crop area
            result.update({
                'quadrat_size': planar_area,  # Crop planar area
                'quadrat_x': x_center,       # Crop center X
                'quadrat_y': y_center        # Crop center Y
            })
        else:  # whole_mesh
            result.update({
                'quadrat_size': planar_area,  # Total mesh planar area
                'quadrat_x': x_center,       # Mesh center X
                'quadrat_y': y_center        # Mesh center Y
            })
        
        return result


class MeshCounts(SurfaceMetric):
    """Number of faces and vertices in the mesh."""
    
    @property
    def name(self) -> str:
        return "mesh_counts"
    
    @property
    def description(self) -> str:
        return "Count of faces and vertices in the mesh"
    
    def calculate(self, mesh_data: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Calculate mesh element counts."""
        mesh = mesh_data.get('mesh')
        points = mesh_data.get('points')
        faces = mesh_data.get('faces')
        
        if mesh is not None:
            num_faces = mesh.n_cells
            num_vertices = mesh.n_points
        else:
            num_vertices = len(points) if points is not None else 0
            num_faces = len(faces) if faces is not None else 0
        
        return {
            'num_faces': num_faces,
            'num_vertices': num_vertices
        }


class SurfaceRugosity(SurfaceMetric):
    """Surface rugosity with proper handling of incomplete meshes."""
    
    @property
    def name(self) -> str:
        return "surface_rugosity"
    
    @property
    def description(self) -> str:
        return "Ratio of 3D surface area to 2D projected area"
    
    def calculate(self, mesh_data: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Calculate surface rugosity with proper area handling."""
        # Get 3D surface area
        surface_calc = SurfaceArea().calculate(mesh_data)
        surface_area_3d = surface_calc.get('surface_area', float('nan'))
        
        # Get 2D projected area - use actual mesh coverage for crops
        if 'actual_mesh_2d_area' in mesh_data:
            # For cropped meshes, use actual mesh coverage area
            area_2d = mesh_data['actual_mesh_2d_area']
        else:
            # For whole meshes, calculate projected area
            points = mesh_data.get('points')
            if points is not None and len(points) > 0:
                area_2d = calculate_projected_area_convex_hull(points)
            else:
                area_2d = float('nan')
        
        # Calculate rugosity
        if np.isnan(surface_area_3d) or np.isnan(area_2d) or area_2d == 0:
            rugosity = float('nan')
        else:
            rugosity = surface_area_3d / area_2d
        
        return {
            'surface_rugosity': rugosity,
            'surface_area_3d': surface_area_3d,
            'surface_area_2d': area_2d
        }


# Enhanced USYD metrics with proper implementation
class Slope(ComplexityMetric):
    """Surface slope calculation using local plane fitting."""
    
    @property
    def name(self) -> str:
        return "slope"
    
    @property
    def description(self) -> str:
        return "Local surface slope statistics"
    
    def calculate(self, mesh_data: Dict[str, Any], 
                 neighborhood_radius: float = 0.1,
                 min_neighbors: int = 5,
                 **kwargs) -> Dict[str, Any]:
        """Calculate slope statistics."""
        points = mesh_data.get('points')
        
        if points is None or len(points) < min_neighbors:
            return {
                'slope_mean': float('nan'),
                'slope_std': float('nan'),
                'slope_median': float('nan'),
                'slope_max': float('nan'),
                'slope_min': float('nan')
            }
        
        if not _SKLEARN_AVAILABLE or NearestNeighbors is None:
            logger.warning("Scikit-learn not available - cannot calculate slope statistics")
            return {
                'slope_mean': float('nan'),
                'slope_std': float('nan'),
                'slope_median': float('nan'),
                'slope_max': float('nan'),
                'slope_min': float('nan')
            }
        
        try:
            # Build neighbor finder
            nbrs = NearestNeighbors(n_neighbors=min_neighbors, algorithm='auto')
            nbrs.fit(points)
            
            slopes = []
            for point in points:
                # Find neighbors
                distances, indices = nbrs.kneighbors([point])
                neighbor_points = points[indices[0]]
                
                # Fit plane and calculate slope
                normal = self._fit_plane_normal(neighbor_points)
                if normal is not None:
                    # Calculate slope angle from vertical
                    vertical = np.array([0, 0, 1])
                    cos_angle = np.abs(np.dot(normal, vertical))
                    slope_angle = np.degrees(np.arccos(np.clip(cos_angle, 0, 1)))
                    slopes.append(slope_angle)
            
            if slopes:
                slopes = np.array(slopes)
                return {
                    'slope_mean': np.mean(slopes),
                    'slope_std': np.std(slopes),
                    'slope_median': np.median(slopes),
                    'slope_max': np.max(slopes),
                    'slope_min': np.min(slopes)
                }
            else:
                return {
                    'slope_mean': float('nan'),
                    'slope_std': float('nan'),
                    'slope_median': float('nan'),
                    'slope_max': float('nan'),
                    'slope_min': float('nan')
                }
        except Exception as e:
            logger.warning(f"Slope calculation failed: {e}")
            return {
                'slope_mean': float('nan'),
                'slope_std': float('nan'),
                'slope_median': float('nan'),
                'slope_max': float('nan'),
                'slope_min': float('nan')
            }
    
    def _fit_plane_normal(self, points: np.ndarray) -> Optional[np.ndarray]:
        """Fit plane to points and return normal vector."""
        if len(points) < 3:
            return None
        
        try:
            centroid = np.mean(points, axis=0)
            centered = points - centroid
            _, _, V = np.linalg.svd(centered)
            normal = V[-1]
            return normal / np.linalg.norm(normal)
        except:
            return None


class PlaneOfBestFit(ComplexityMetric):
    """Plane of best fit analysis."""
    
    @property
    def name(self) -> str:
        return "plane_of_best_fit"
    
    @property
    def description(self) -> str:
        return "Global plane fitting analysis"
    
    def calculate(self, mesh_data: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Calculate plane of best fit metrics."""
        points = mesh_data.get('points')
        
        if points is None or len(points) < 3:
            return {
                'global_fit_error': float('nan'),
                'planarity_score': float('nan')
            }
        
        try:
            # Fit global plane
            centroid = np.mean(points, axis=0)
            centered = points - centroid
            
            _, s, V = np.linalg.svd(centered)
            normal = V[-1]
            
            # Calculate fitting error
            distances = np.abs(np.dot(centered, normal))
            fit_error = np.mean(distances**2)
            
            # Calculate planarity score (0-1, where 1 is perfectly planar)
            # Based on ratio of smallest to largest singular values
            if len(s) >= 3:
                planarity_score = 1 - (s[-1] / (s[0] + 1e-10))
            else:
                planarity_score = float('nan')
            
            return {
                'global_fit_error': fit_error,
                'planarity_score': planarity_score
            }
            
        except Exception as e:
            logger.warning(f"Plane of best fit calculation failed: {e}")
            return {
                'global_fit_error': float('nan'),
                'planarity_score': float('nan')
            }


class HeightRange(ComplexityMetric):
    """Height range and vertical distribution analysis."""
    
    @property
    def name(self) -> str:
        return "height_range"
    
    @property
    def description(self) -> str:
        return "Vertical extent and distribution statistics"
    
    def calculate(self, mesh_data: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Calculate height range and distribution metrics."""
        points = mesh_data.get('points')
        
        if points is None or len(points) == 0:
            return {
                'height_range': float('nan'),
                'height_mean': float('nan'),
                'height_std': float('nan')
            }
        
        z_coords = points[:, 2]
        
        return {
            'height_range': np.max(z_coords) - np.min(z_coords),
            'height_mean': np.mean(z_coords),
            'height_std': np.std(z_coords)
        }


class FractalDimension(ComplexityMetric):
    """Box-counting fractal dimension."""
    
    @property
    def name(self) -> str:
        return "fractal_dimension"
    
    @property
    def description(self) -> str:
        return "Box-counting fractal dimension"
    
    def calculate(self, mesh_data: Dict[str, Any], 
                 n_scales: int = 15,
                 **kwargs) -> Dict[str, Any]:
        """Calculate fractal dimension using box counting."""
        points = mesh_data.get('points')
        
        if points is None or len(points) < 10:
            return {'fractal_dimension': float('nan')}
        
        try:
            # Calculate bounding box
            min_coords = np.min(points, axis=0)
            max_coords = np.max(points, axis=0)
            extent = max_coords - min_coords
            max_extent = np.max(extent)
            
            if max_extent == 0:
                return {'fractal_dimension': float('nan')}
            
            # Generate scales
            min_scale = max_extent / 100
            max_scale = max_extent / 2
            scales = np.logspace(np.log10(min_scale), np.log10(max_scale), n_scales)
            
            counts = []
            for scale in scales:
                # Count occupied boxes
                n_boxes = np.ceil(extent / scale).astype(int)
                box_indices = np.floor((points - min_coords) / scale).astype(int)
                
                # Ensure indices are within bounds
                for dim in range(3):
                    box_indices[:, dim] = np.clip(box_indices[:, dim], 0, n_boxes[dim] - 1)
                
                unique_boxes = np.unique(box_indices, axis=0)
                counts.append(len(unique_boxes))
            
            # Fit power law
            log_scales = np.log(scales)
            log_counts = np.log(counts)
            
            slope, _ = np.polyfit(log_scales, log_counts, 1)
            fractal_dim = -slope
            
            return {'fractal_dimension': fractal_dim}
            
        except Exception as e:
            logger.warning(f"Fractal dimension calculation failed: {e}")
            return {'fractal_dimension': float('nan')}


# Shading metrics placeholder for integration
class ShadingPercentage(ShadingMetric):
    """Shading percentage metric wrapper."""
    
    @property
    def name(self) -> str:
        return "shading_percentage"
    
    @property
    def description(self) -> str:
        return "Percentage of mesh surface that is shaded"
    
    def calculate(self, mesh_data: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Calculate shading percentage (placeholder for Shading class integration)."""
        # This would integrate with the existing Shading class
        return {
            'shading_percentage': float('nan'),
            'illuminated_percentage': float('nan')
        }