"""
Modular shading components for extensible coral complexity analysis.

This module provides a modular architecture for shading calculations that
can be easily extended with new metrics, environmental factors, and 
lighting models.
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Tuple, Union, TYPE_CHECKING
import math
import warnings
import logging

# Optional imports - handle gracefully if not available
try:
    import pyvista as pv
    _PYVISTA_AVAILABLE = True
except ImportError:
    _PYVISTA_AVAILABLE = False
    pv = None

# Type annotations that work when pyvista is not available
if TYPE_CHECKING and _PYVISTA_AVAILABLE:
    from pyvista import PolyData
else:
    PolyData = Any

logger = logging.getLogger(__name__)


class LightingModel(ABC):
    """Abstract base class for lighting models."""
    
    @abstractmethod
    def calculate_light_direction(self, **kwargs) -> np.ndarray:
        """Calculate light direction vector."""
        pass
    
    @abstractmethod
    def get_model_name(self) -> str:
        """Return the name of the lighting model."""
        pass


class SolarLightingModel(LightingModel):
    """Solar position-based lighting model."""
    
    def calculate_light_direction(self, 
                                day_of_year: int,
                                time_of_day: float,
                                latitude: float = 0.0,
                                longitude: float = 0.0,
                                **kwargs) -> np.ndarray:
        """Calculate solar light direction."""
        # Solar declination angle
        declination = 23.45 * math.sin(math.radians(360 * (284 + day_of_year) / 365))
        
        # Hour angle (solar time)
        hour_angle = 15 * (time_of_day - 12)  # degrees
        
        # Convert to radians
        lat_rad = math.radians(latitude)
        dec_rad = math.radians(declination)
        hour_rad = math.radians(hour_angle)
        
        # Solar elevation angle
        elevation = math.asin(
            math.sin(lat_rad) * math.sin(dec_rad) + 
            math.cos(lat_rad) * math.cos(dec_rad) * math.cos(hour_rad)
        )
        
        # Solar azimuth angle
        azimuth = math.atan2(
            math.sin(hour_rad),
            math.cos(hour_rad) * math.sin(lat_rad) - math.tan(dec_rad) * math.cos(lat_rad)
        )
        
        # Convert to light direction vector (pointing toward sun)
        x = math.sin(azimuth) * math.cos(elevation)  # East component
        y = math.cos(azimuth) * math.cos(elevation)  # North component  
        z = math.sin(elevation)  # Up component
        
        # Return normalized vector pointing FROM sun TO surface (for ray casting)
        return -np.array([x, y, z])
    
    def get_model_name(self) -> str:
        return "solar_position"


class DirectionalLightingModel(LightingModel):
    """Simple directional lighting model."""
    
    def __init__(self, direction: np.ndarray):
        """Initialize with a fixed direction."""
        self.direction = direction / np.linalg.norm(direction)
    
    def calculate_light_direction(self, **kwargs) -> np.ndarray:
        """Return the fixed light direction."""
        return self.direction
    
    def get_model_name(self) -> str:
        return "directional"


class AmbientLightingModel(LightingModel):
    """Ambient (diffuse) lighting model."""
    
    def calculate_light_direction(self, **kwargs) -> np.ndarray:
        """Return downward direction for ambient lighting."""
        return np.array([0, 0, -1])
    
    def get_model_name(self) -> str:
        return "ambient"


class EnvironmentalFactor(ABC):
    """Abstract base class for environmental factors."""
    
    @abstractmethod
    def apply_adjustment(self, 
                        light_direction: np.ndarray, 
                        mesh_points: np.ndarray,
                        **kwargs) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Apply environmental adjustment to light direction.
        
        Returns:
        Tuple of (adjusted_light_direction, adjustment_metadata)
        """
        pass
    
    @abstractmethod
    def get_factor_name(self) -> str:
        """Return the name of the environmental factor."""
        pass


class SlopeAspectFactor(EnvironmentalFactor):
    """Slope and aspect environmental adjustment."""
    
    def apply_adjustment(self, 
                        light_direction: np.ndarray,
                        mesh_points: np.ndarray,
                        slope: float,
                        aspect: float,
                        **kwargs) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Apply slope and aspect adjustment to light direction."""
        if not 0 <= slope <= 90:
            raise ValueError("slope must be between 0 and 90 degrees")
        if not 0 <= aspect <= 360:
            raise ValueError("aspect must be between 0 and 360 degrees")
        
        # Convert to radians
        slope_rad = math.radians(slope)
        aspect_rad = math.radians(aspect)
        
        # Create rotation matrix for slope and aspect
        cos_slope = math.cos(slope_rad)
        sin_slope = math.sin(slope_rad)
        cos_aspect = math.cos(aspect_rad)
        sin_aspect = math.sin(aspect_rad)
        
        # Rotation matrix around aspect direction
        rotation_matrix = np.array([
            [cos_aspect * cos_slope, -sin_aspect, cos_aspect * sin_slope],
            [sin_aspect * cos_slope, cos_aspect, sin_aspect * sin_slope],
            [-sin_slope, 0, cos_slope]
        ])
        
        adjusted_direction = rotation_matrix @ light_direction
        
        metadata = {
            'original_direction': light_direction.tolist(),
            'slope_degrees': slope,
            'aspect_degrees': aspect,
            'adjustment_magnitude': np.linalg.norm(adjusted_direction - light_direction)
        }
        
        return adjusted_direction, metadata
    
    def get_factor_name(self) -> str:
        return "slope_aspect"


class DepthAttenuationFactor(EnvironmentalFactor):
    """Depth-based light attenuation factor (placeholder for future implementation)."""
    
    def apply_adjustment(self, 
                        light_direction: np.ndarray,
                        mesh_points: np.ndarray,
                        depth: float,
                        attenuation_coefficient: float = 0.1,
                        **kwargs) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Apply depth-based attenuation (future implementation)."""
        warnings.warn(
            "Depth attenuation factor is not fully implemented. "
            "This is a placeholder for future physics-based underwater lighting models.",
            UserWarning
        )
        
        # Placeholder: return original direction with metadata
        metadata = {
            'depth_meters': depth,
            'attenuation_coefficient': attenuation_coefficient,
            'attenuation_factor': math.exp(-attenuation_coefficient * depth),
            'note': 'Placeholder implementation - requires full underwater optics model'
        }
        
        return light_direction, metadata
    
    def get_factor_name(self) -> str:
        return "depth_attenuation"


class ShadingMetric(ABC):
    """Abstract base class for shading metrics."""
    
    @abstractmethod
    def calculate(self, 
                 mesh: PolyData,
                 lighting_results: Dict[str, Any],
                 **kwargs) -> Dict[str, Any]:
        """Calculate the shading metric."""
        pass
    
    @abstractmethod
    def get_metric_name(self) -> str:
        """Return the name of the metric."""
        pass


class PercentageShadingMetric(ShadingMetric):
    """Standard percentage shading metric."""
    
    def calculate(self, 
                 mesh: PolyData,
                 lighting_results: Dict[str, Any],
                 **kwargs) -> Dict[str, Any]:
        """Calculate shading and illumination percentages."""
        if not _PYVISTA_AVAILABLE or mesh is None:
            logger.warning("PyVista not available - cannot calculate shading metrics")
            return {
                'shaded_percentage': float('nan'),
                'illuminated_percentage': float('nan')
            }
        
        shadowed_points = lighting_results.get('shadowed_points', np.array([]))
        total_points = lighting_results.get('total_points_sampled', 0)
        
        if total_points == 0:
            return {
                'shaded_percentage': float('nan'),
                'illuminated_percentage': float('nan')
            }
        
        shaded_count = np.sum(shadowed_points) if len(shadowed_points) > 0 else 0
        shaded_percentage = (shaded_count / total_points) * 100
        illuminated_percentage = 100 - shaded_percentage
        
        return {
            'shaded_percentage': shaded_percentage,
            'illuminated_percentage': illuminated_percentage
        }
    
    def get_metric_name(self) -> str:
        return "percentage_shading"


class SpatialShadingMetric(ShadingMetric):
    """Spatial distribution of shading (future implementation)."""
    
    def calculate(self, 
                 mesh: PolyData,
                 lighting_results: Dict[str, Any],
                 **kwargs) -> Dict[str, Any]:
        """Calculate spatial shading distribution metrics."""
        warnings.warn(
            "Spatial shading metric is not fully implemented. "
            "This is a placeholder for future spatial analysis features.",
            UserWarning
        )
        
        return {
            'shading_variance': float('nan'),
            'shading_clustering': float('nan'),
            'shadow_length_mean': float('nan'),
            'note': 'Placeholder implementation - requires spatial analysis algorithms'
        }
    
    def get_metric_name(self) -> str:
        return "spatial_shading"


class ModularShadingCalculator:
    """
    Modular shading calculator that combines lighting models, environmental
    factors, and shading metrics in a configurable way.
    """
    
    def __init__(self):
        """Initialize the modular shading calculator."""
        self.lighting_models: Dict[str, LightingModel] = {
            'solar': SolarLightingModel(),
            'ambient': AmbientLightingModel()
        }
        
        self.environmental_factors: Dict[str, EnvironmentalFactor] = {
            'slope_aspect': SlopeAspectFactor(),
            'depth_attenuation': DepthAttenuationFactor()
        }
        
        self.shading_metrics: Dict[str, ShadingMetric] = {
            'percentage': PercentageShadingMetric(),
            'spatial': SpatialShadingMetric()
        }
    
    def register_lighting_model(self, model: LightingModel) -> None:
        """Register a new lighting model."""
        self.lighting_models[model.get_model_name()] = model
    
    def register_environmental_factor(self, factor: EnvironmentalFactor) -> None:
        """Register a new environmental factor."""
        self.environmental_factors[factor.get_factor_name()] = factor
    
    def register_shading_metric(self, metric: ShadingMetric) -> None:
        """Register a new shading metric."""
        self.shading_metrics[metric.get_metric_name()] = metric
    
    def calculate_comprehensive_shading(self,
                                      mesh: PolyData,
                                      lighting_config: Dict[str, Any],
                                      environmental_config: Dict[str, Any],
                                      metrics_config: List[str],
                                      ray_tracing_config: Dict[str, Any],
                                      **kwargs) -> Dict[str, Any]:
        """
        Calculate comprehensive shading analysis with modular components.
        
        Parameters:
        mesh: Input mesh
        lighting_config: Configuration for lighting model
        environmental_config: Configuration for environmental factors
        metrics_config: List of metrics to calculate
        ray_tracing_config: Configuration for ray tracing
        
        Returns:
        Comprehensive shading analysis results
        """
        results = {
            'lighting_model': None,
            'environmental_adjustments': [],
            'ray_tracing_stats': {},
            'metrics': {},
            'warnings': [],
            'processing_time': None
        }
        
        if not _PYVISTA_AVAILABLE or mesh is None:
            results['error'] = "PyVista not available or mesh is None"
            return results
        
        import time
        start_time = time.time()
        
        try:
            # Step 1: Calculate base light direction
            lighting_model_name = lighting_config.get('model', 'solar')
            if lighting_model_name not in self.lighting_models:
                raise ValueError(f"Unknown lighting model: {lighting_model_name}")
            
            lighting_model = self.lighting_models[lighting_model_name]
            base_light_direction = lighting_model.calculate_light_direction(**lighting_config)
            results['lighting_model'] = {
                'name': lighting_model_name,
                'direction': base_light_direction.tolist(),
                'config': lighting_config
            }
            
            # Step 2: Apply environmental factors
            current_light_direction = base_light_direction
            mesh_points = mesh.points
            
            for factor_name, factor_config in environmental_config.items():
                if factor_name in self.environmental_factors:
                    factor = self.environmental_factors[factor_name]
                    adjusted_direction, adjustment_metadata = factor.apply_adjustment(
                        current_light_direction, mesh_points, **factor_config
                    )
                    current_light_direction = adjusted_direction
                    
                    results['environmental_adjustments'].append({
                        'factor': factor_name,
                        'config': factor_config,
                        'metadata': adjustment_metadata
                    })
            
            # Step 3: Perform ray tracing (simplified version)
            # This would integrate with the existing ray tracing logic from the Shading class
            lighting_results = self._perform_ray_tracing(
                mesh, current_light_direction, ray_tracing_config
            )
            results['ray_tracing_stats'] = lighting_results
            
            # Step 4: Calculate requested metrics
            for metric_name in metrics_config:
                if metric_name in self.shading_metrics:
                    metric = self.shading_metrics[metric_name]
                    metric_result = metric.calculate(mesh, lighting_results, **kwargs)
                    results['metrics'][metric_name] = metric_result
                else:
                    results['warnings'].append(f"Unknown metric: {metric_name}")
            
            results['processing_time'] = time.time() - start_time
            
        except Exception as e:
            results['error'] = str(e)
            results['processing_time'] = time.time() - start_time
            logger.error(f"Modular shading calculation failed: {e}")
        
        return results
    
    def _perform_ray_tracing(self, 
                           mesh: PolyData,
                           light_direction: np.ndarray,
                           config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform ray tracing analysis (simplified implementation).
        
        This method provides a placeholder for integration with the full
        ray tracing implementation from the Shading class.
        """
        if not _PYVISTA_AVAILABLE or mesh is None:
            return {
                'total_points_sampled': 0,
                'shadowed_points': np.array([]),
                'light_direction': light_direction.tolist(),
                'sample_method': 'none - PyVista not available'
            }
        
        sample_size = config.get('sample_size', 1000)
        
        # Sample points from mesh
        points = mesh.points
        if len(points) > sample_size:
            indices = np.random.choice(len(points), sample_size, replace=False)
            sampled_points = points[indices]
        else:
            sampled_points = points
        
        # Placeholder for actual ray tracing
        # In a full implementation, this would use the BVH and ray intersection
        # logic from the Shading class
        shadowed = np.random.random(len(sampled_points)) < 0.3  # Placeholder
        
        return {
            'total_points_sampled': len(sampled_points),
            'shadowed_points': shadowed,
            'light_direction': light_direction.tolist(),
            'sample_method': 'random'
        }
    
    def get_available_components(self) -> Dict[str, List[str]]:
        """Get list of available components."""
        return {
            'lighting_models': list(self.lighting_models.keys()),
            'environmental_factors': list(self.environmental_factors.keys()),
            'shading_metrics': list(self.shading_metrics.keys())
        }


def create_default_shading_pipeline() -> ModularShadingCalculator:
    """Create a default modular shading pipeline."""
    calculator = ModularShadingCalculator()
    
    # Add directional lighting for custom directions
    calculator.register_lighting_model(
        DirectionalLightingModel(np.array([0, 0, -1]))
    )
    
    return calculator