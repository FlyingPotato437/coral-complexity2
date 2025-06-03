import math
import numpy as np
from typing import Dict, Any, Optional, Callable, List, Union
from abc import ABC, abstractmethod
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
    from scipy.stats import kurtosis, skew
    _SCIPY_AVAILABLE = True
except ImportError:
    _SCIPY_AVAILABLE = False

try:
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    _SKLEARN_AVAILABLE = True
except ImportError:
    _SKLEARN_AVAILABLE = False

from .mesh_utils import calculate_projected_area_convex_hull, calculate_bounding_box_area

logger = logging.getLogger(__name__)


class BaseMetric(ABC):
    """Abstract base class for all coral complexity metrics."""
    
    @abstractmethod
    def calculate(self, mesh_data: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Calculate the metric from mesh data."""
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Return the metric name."""
        pass
    
    @property
    @abstractmethod
    def description(self) -> str:
        """Return a description of what this metric measures."""
        pass
    
    @property
    def requires_closed_mesh(self) -> bool:
        """Return True if this metric requires a closed (watertight) mesh."""
        return False
    
    @property
    def metric_type(self) -> str:
        """Return the type of metric (surface, volume, complexity, shading)."""
        return "unknown"


class SurfaceMetric(BaseMetric):
    """Base class for surface-based metrics that work on open or closed meshes."""
    
    @property
    def metric_type(self) -> str:
        return "surface"


class VolumeMetric(BaseMetric):
    """Base class for volume-based metrics that require closed meshes."""
    
    @property
    def requires_closed_mesh(self) -> bool:
        return True
    
    @property
    def metric_type(self) -> str:
        return "volume"


class ComplexityMetric(BaseMetric):
    """Base class for structural complexity metrics."""
    
    @property
    def metric_type(self) -> str:
        return "complexity"


class ShadingMetric(BaseMetric):
    """Base class for shading/illumination metrics."""
    
    @property
    def metric_type(self) -> str:
        return "shading"


class Rugosity(SurfaceMetric):
    """Surface rugosity metric (3D surface area / 2D projected area) with missing data tracking."""
    
    @property
    def name(self) -> str:
        return "rugosity"
    
    @property
    def description(self) -> str:
        return "Ratio of 3D surface area to actual 2D projected mesh area, with coverage stats."
    
    def calculate(self, mesh_data: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """
        Calculate surface rugosity and mesh coverage statistics.

        Expected keys in mesh_data:
        - 'surface_area_3d': float, The 3D surface area of the mesh within the clipped region.
        - 'mesh_clipped_points': np.ndarray, The 3D points of the mesh within the clipped region.
        - 'clipped_region_size_2d': np.ndarray, The [width, height] of the 2D bounding box of the clip.

        Returns:
        Dict[str, Any]: A dictionary containing:
            - 'rugosity': The calculated rugosity (3D area / actual 2D projected area).
            - 'actual_mesh_2d_area': The 2D area of the convex hull of projected mesh points.
            - 'defined_2d_area': The 2D area of the bounding box of the clipped region.
            - 'missing_data_percentage': Percentage of the clipped region's 2D area not covered by mesh.
            - 'mesh_coverage_percentage': Percentage of the clipped region's 2D area covered by mesh.
        """
        surface_area_3d = mesh_data.get('surface_area_3d')
        mesh_clipped_points = mesh_data.get('mesh_clipped_points')
        clipped_region_size_2d = mesh_data.get('clipped_region_size_2d')

        if surface_area_3d is None:
            logger.warning("'surface_area_3d' not found in mesh_data for Rugosity calculation.")
            return {'rugosity': np.nan, 'error': "Missing surface_area_3d"}
        if mesh_clipped_points is None:
            logger.warning("'mesh_clipped_points' not found in mesh_data for Rugosity calculation.")
            return {'rugosity': np.nan, 'error': "Missing mesh_clipped_points"}
        if clipped_region_size_2d is None:
            logger.warning("'clipped_region_size_2d' not found in mesh_data for Rugosity calculation.")
            return {'rugosity': np.nan, 'error': "Missing clipped_region_size_2d"}

        actual_mesh_2d_area = calculate_projected_area_convex_hull(mesh_clipped_points)
        defined_2d_area = calculate_bounding_box_area(clipped_region_size_2d)

        rugosity_val = np.nan
        missing_data_percentage = np.nan
        mesh_coverage_percentage = np.nan

        if actual_mesh_2d_area > 1e-9: # Avoid division by zero or tiny areas
            rugosity_val = surface_area_3d / actual_mesh_2d_area
        else:
            logger.warning(f"Actual projected 2D mesh area is {actual_mesh_2d_area}, too small for rugosity. Surface area 3D was {surface_area_3d}.")

        if defined_2d_area > 1e-9: # Avoid division by zero for percentages
            missing_data_percentage = (1.0 - (actual_mesh_2d_area / defined_2d_area)) * 100.0
            mesh_coverage_percentage = (actual_mesh_2d_area / defined_2d_area) * 100.0
            # Clamp percentages to a reasonable range, e.g. 0-100, actual_mesh_2d_area might slightly exceed defined_2d_area due to convex hull on edge cases
            missing_data_percentage = max(0.0, min(100.0, missing_data_percentage))
            mesh_coverage_percentage = max(0.0, min(100.0, mesh_coverage_percentage))
        else:
            logger.warning(f"Defined 2D area of clipped region is {defined_2d_area}, too small for coverage percentages.")
            if actual_mesh_2d_area > 1e-9:
                 mesh_coverage_percentage = 100.0 # If defined area is zero but mesh exists, it's 100% of what's definable
                 missing_data_percentage = 0.0
            # else both remain NaN if both areas are zero

        return {
            'rugosity': rugosity_val,
            'actual_mesh_2d_area': actual_mesh_2d_area,
            'defined_2d_area': defined_2d_area,
            'missing_data_percentage': missing_data_percentage,
            'mesh_coverage_percentage': mesh_coverage_percentage
        }


class FractalDimension(ComplexityMetric):
    """Box-counting fractal dimension metric."""
    
    @property
    def name(self) -> str:
        return "fractal_dimension"
    
    @property
    def description(self) -> str:
        return "Box-counting fractal dimension of the surface"
    
    def calculate(self, mesh_data: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Calculate box-counting fractal dimension."""
        points = mesh_data.get('points')
        if points is None:
            return {'fractal_dimension': float('nan')}
        
        # Implement box-counting algorithm
        scales = np.logspace(-3, 0, 20)  # Range of box sizes
        counts = []
        
        for scale in scales:
            # Create a grid at this scale
            min_coords = np.min(points, axis=0)
            max_coords = np.max(points, axis=0)
            
            # Number of boxes needed in each dimension
            n_boxes = np.ceil((max_coords - min_coords) / scale).astype(int)
            
            # Count occupied boxes
            box_indices = np.floor((points - min_coords) / scale).astype(int)
            # Ensure indices are within bounds
            box_indices = np.clip(box_indices, 0, n_boxes - 1)
            
            # Count unique boxes
            unique_boxes = np.unique(box_indices, axis=0)
            counts.append(len(unique_boxes))
        
        # Fit line to log-log plot
        log_scales = np.log(scales)
        log_counts = np.log(counts)
        
        # Linear regression
        A = np.vstack([log_scales, np.ones(len(log_scales))]).T
        slope, _ = np.linalg.lstsq(A, log_counts, rcond=None)[0]
        
        fractal_dim = -slope  # Negative because slope should be negative
        
        return {'fractal_dimension': fractal_dim}


class MeshClosure(VolumeMetric):
    """Mesh closure detection and repair quality metric."""
    
    @property
    def name(self) -> str:
        return "mesh_closure"
    
    @property
    def description(self) -> str:
        return "Assessment of mesh closure quality and volume calculation validity"
    
    def calculate(self, mesh_data: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Calculate mesh closure metrics."""
        mesh = mesh_data.get('mesh')
        if mesh is None:
            return {'is_closed': False, 'open_edges': float('nan'), 'closure_quality': 0.0}
        
        # Check if mesh is closed (simplified check)
        # In a proper implementation, this would check for open edges
        try:
            # Try to calculate volume - if it fails, mesh isn't closed
            volume = mesh.volume
            is_closed = volume > 0
            open_edges = 0 if is_closed else -1  # -1 indicates unknown count
            closure_quality = 1.0 if is_closed else 0.0
        except:
            is_closed = False
            open_edges = -1
            closure_quality = 0.0
        
        return {
            'is_closed': is_closed,
            'open_edges': open_edges,
            'closure_quality': closure_quality
        }


class MetricRegistry:
    """Registry for coral complexity metrics with automatic discovery and validation."""
    
    def __init__(self):
        self._metrics: Dict[str, BaseMetric] = {}
        self._metric_groups: Dict[str, List[str]] = {
            'surface': [],
            'volume': [],
            'complexity': [],
            'shading': []
        }
        self._register_default_metrics()
    
    def _register_default_metrics(self):
        """Register the default set of metrics."""
        default_metrics = [
            Rugosity(),
            FractalDimension(),
            MeshClosure()
        ]
        
        for metric in default_metrics:
            self.register(metric)
    
    def register(self, metric: BaseMetric) -> None:
        """
        Register a new metric.
        
        Parameters:
        metric (BaseMetric): The metric instance to register.
        """
        if not isinstance(metric, BaseMetric):
            raise TypeError("Metric must inherit from BaseMetric")
        
        name = metric.name
        if name in self._metrics:
            warnings.warn(f"Metric '{name}' already registered. Overwriting.", UserWarning)
        
        self._metrics[name] = metric
        
        # Add to appropriate group
        metric_type = metric.metric_type
        if metric_type in self._metric_groups:
            if name not in self._metric_groups[metric_type]:
                self._metric_groups[metric_type].append(name)
    
    def unregister(self, name: str) -> None:
        """
        Unregister a metric.
        
        Parameters:
        name (str): Name of the metric to unregister.
        """
        if name not in self._metrics:
            raise ValueError(f"Metric '{name}' is not registered")
        
        metric = self._metrics[name]
        del self._metrics[name]
        
        # Remove from groups
        for group_metrics in self._metric_groups.values():
            if name in group_metrics:
                group_metrics.remove(name)
    
    def get_metric(self, name: str) -> BaseMetric:
        """
        Get a metric by name.
        
        Parameters:
        name (str): Name of the metric.
        
        Returns:
        BaseMetric: The metric instance.
        """
        if name not in self._metrics:
            raise ValueError(f"Metric '{name}' is not registered")
        return self._metrics[name]
    
    def list_metrics(self, metric_type: Optional[str] = None) -> List[str]:
        """
        List registered metrics.
        
        Parameters:
        metric_type (Optional[str]): Filter by metric type (surface, volume, complexity, shading).
        
        Returns:
        List[str]: List of metric names.
        """
        if metric_type is None:
            return list(self._metrics.keys())
        
        if metric_type not in self._metric_groups:
            raise ValueError(f"Unknown metric type: {metric_type}")
        
        return self._metric_groups[metric_type].copy()
    
    def get_metric_info(self, name: str) -> Dict[str, Any]:
        """
        Get information about a metric.
        
        Parameters:
        name (str): Name of the metric.
        
        Returns:
        Dict[str, Any]: Metric information.
        """
        metric = self.get_metric(name)
        return {
            'name': metric.name,
            'description': metric.description,
            'type': metric.metric_type,
            'requires_closed_mesh': metric.requires_closed_mesh
        }
    
    def calculate_metrics(self, 
                         mesh_data: Dict[str, Any], 
                         metrics: Optional[List[str]] = None,
                         include_types: Optional[List[str]] = None,
                         exclude_types: Optional[List[str]] = None,
                         check_mesh_closure: bool = True) -> Dict[str, Any]:
        """
        Calculate multiple metrics from mesh data.
        
        Parameters:
        mesh_data (Dict[str, Any]): Mesh data dictionary.
        metrics (Optional[List[str]]): Specific metrics to calculate. If None, calculates all applicable.
        include_types (Optional[List[str]]): Metric types to include.
        exclude_types (Optional[List[str]]): Metric types to exclude.
        check_mesh_closure (bool): Whether to check mesh closure before calculating volume metrics.
        
        Returns:
        Dict[str, Any]: Dictionary of calculated metrics.
        """
        results = {}
        
        # Determine which metrics to calculate
        if metrics is None:
            metrics_to_calc = list(self._metrics.keys())
        else:
            metrics_to_calc = metrics
        
        # Filter by type inclusion/exclusion
        if include_types:
            metrics_to_calc = [m for m in metrics_to_calc 
                             if self._metrics[m].metric_type in include_types]
        
        if exclude_types:
            metrics_to_calc = [m for m in metrics_to_calc 
                             if self._metrics[m].metric_type not in exclude_types]
        
        # Check mesh closure if needed
        mesh_is_closed = None
        if check_mesh_closure:
            volume_metrics = [m for m in metrics_to_calc 
                            if self._metrics[m].requires_closed_mesh]
            
            if volume_metrics:
                # Check if mesh is closed
                closure_metric = MeshClosure()
                closure_result = closure_metric.calculate(mesh_data)
                mesh_is_closed = closure_result.get('is_closed', False)
                
                if not mesh_is_closed:
                    warnings.warn(
                        "Mesh is not closed. Volume-dependent metrics will return NaN.",
                        UserWarning
                    )
        
        # Calculate each metric
        for metric_name in metrics_to_calc:
            try:
                metric = self._metrics[metric_name]
                
                # Skip volume metrics if mesh is not closed
                if (metric.requires_closed_mesh and 
                    mesh_is_closed is False and 
                    check_mesh_closure):
                    results[metric_name] = {k: float('nan') 
                                          for k in ['volume', 'convex_hull_volume', 
                                                   'proportion_occupied', 'available_space_ratio']}
                    continue
                
                # Calculate the metric
                metric_result = metric.calculate(mesh_data)
                results[metric_name] = metric_result
                
            except Exception as e:
                warnings.warn(f"Error calculating metric '{metric_name}': {e}", UserWarning)
                results[metric_name] = {'error': str(e)}
        
        return results
    
    def validate_metric(self, metric: BaseMetric) -> List[str]:
        """
        Validate a metric implementation.
        
        Parameters:
        metric (BaseMetric): The metric to validate.
        
        Returns:
        List[str]: List of validation errors (empty if valid).
        """
        errors = []
        
        # Check inheritance
        if not isinstance(metric, BaseMetric):
            errors.append("Metric must inherit from BaseMetric")
            return errors
        
        # Check required properties
        try:
            name = metric.name
            if not isinstance(name, str) or not name:
                errors.append("Metric name must be a non-empty string")
        except Exception as e:
            errors.append(f"Error accessing metric name: {e}")
        
        try:
            description = metric.description
            if not isinstance(description, str) or not description:
                errors.append("Metric description must be a non-empty string")
        except Exception as e:
            errors.append(f"Error accessing metric description: {e}")
        
        try:
            metric_type = metric.metric_type
            if metric_type not in ['surface', 'volume', 'complexity', 'shading', 'unknown']:
                errors.append(f"Invalid metric type: {metric_type}")
        except Exception as e:
            errors.append(f"Error accessing metric type: {e}")
        
        # Test calculate method with dummy data
        try:
            dummy_data = {'points': np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]])}
            result = metric.calculate(dummy_data)
            if not isinstance(result, dict):
                errors.append("calculate() method must return a dictionary")
        except Exception as e:
            errors.append(f"Error in calculate() method: {e}")
        
        return errors


# Global metric registry instance
_global_registry = MetricRegistry()


def register_metric(metric: BaseMetric) -> None:
    """Register a metric with the global registry."""
    _global_registry.register(metric)


def get_metric_registry() -> MetricRegistry:
    """Get the global metric registry."""
    return _global_registry


def list_available_metrics(metric_type: Optional[str] = None) -> List[str]:
    """List available metrics."""
    return _global_registry.list_metrics(metric_type)


def calculate_all_metrics(mesh_data: Dict[str, Any], **kwargs) -> Dict[str, Any]:
    """Calculate all applicable metrics for the given mesh data."""
    return _global_registry.calculate_metrics(mesh_data, **kwargs)


class Metric(object):
    """
    Legacy metric class for backward compatibility.
    
    DEPRECATED: Use the new metric registry system instead.
    """

    def __init__(self):
        warnings.warn(
            "The Metric class is deprecated. Use the new metric registry system instead.",
            DeprecationWarning,
            stacklevel=2
        )
        self.quadrat_id = list()
        self.quadrat_midpoint = None
        self.area3d = 0
        self.area2d = 0
        self.face_count = 0
        self.vertices_count = 0
        self.relative_z_mean = 0
        self.relative_z_sd = 0

    def surface_rugosity(self):
        """Returns the rugosity of the area"""
        if self.area2d == 0:
            return float('nan')
        return float(self.area3d) / float(self.area2d)


# Auto-register unified metrics when module is imported
def register_unified_metrics():
    """Register all unified metrics with the global registry."""
    try:
        from .unified_metrics import (
            Watertight, Volume, SurfaceArea, ConvexHullVolume, 
            ProportionOccupied, AbsoluteSpatialRefuge, ShelterSizeFactor,
            Diameter, Height, QuadratMetrics, MeshCounts, SurfaceRugosity,
            Slope, PlaneOfBestFit, HeightRange, FractalDimension, ShadingPercentage
        )
        
        unified_metrics = [
            Watertight(),
            Volume(),
            SurfaceArea(),
            ConvexHullVolume(),
            ProportionOccupied(),
            AbsoluteSpatialRefuge(),
            ShelterSizeFactor(),
            Diameter(),
            Height(),
            QuadratMetrics(),
            MeshCounts(),
            SurfaceRugosity(),
            Slope(),
            PlaneOfBestFit(),
            HeightRange(),
            FractalDimension(),
            ShadingPercentage()
        ]
        
        for metric in unified_metrics:
            register_metric(metric)
            
        logger.info(f"Registered {len(unified_metrics)} unified metrics")
        
    except ImportError as e:
        logger.warning(f"Could not import unified metrics: {e}")
    except Exception as e:
        logger.error(f"Failed to register unified metrics: {e}")


# Auto-register unified metrics when module is imported
register_unified_metrics()
