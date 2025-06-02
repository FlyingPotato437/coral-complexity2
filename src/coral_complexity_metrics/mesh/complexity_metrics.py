"""
Coral complexity metrics translated from Mitch Bryson's C++ implementations.

This module provides pure Python/NumPy implementations of various structural
complexity measures for coral reef analysis.
"""

import numpy as np
import scipy.spatial
import scipy.linalg
from typing import Dict, Any, Optional, Tuple, List
import warnings
from sklearn.neighbors import NearestNeighbors
from ._metric import ComplexityMetric, SurfaceMetric, register_metric


class SlopeMetric(ComplexityMetric):
    """Surface slope calculation using local plane fitting."""
    
    @property
    def name(self) -> str:
        return "slope"
    
    @property
    def description(self) -> str:
        return "Local surface slope angles calculated via plane fitting"
    
    def calculate(self, mesh_data: Dict[str, Any], 
                 neighborhood_radius: float = 0.1,
                 min_neighbors: int = 5) -> Dict[str, Any]:
        """
        Calculate local slope angles across the mesh surface.
        
        Parameters:
        mesh_data: Dictionary containing mesh data
        neighborhood_radius: Radius for local neighborhood search
        min_neighbors: Minimum number of neighbors required
        
        Returns:
        Dictionary with slope statistics
        """
        points = mesh_data.get('points')
        if points is None:
            return {'slope_mean': float('nan'), 'slope_std': float('nan')}
        
        # Build KD-tree for efficient neighbor search
        tree = scipy.spatial.cKDTree(points)
        slopes = []
        
        for i, point in enumerate(points):
            # Find neighbors within radius
            neighbor_indices = tree.query_ball_point(point, neighborhood_radius)
            
            if len(neighbor_indices) < min_neighbors:
                continue
            
            neighbor_points = points[neighbor_indices]
            
            # Fit plane to local neighborhood
            normal = self._fit_plane_normal(neighbor_points)
            if normal is not None:
                # Calculate slope angle (angle from vertical)
                vertical = np.array([0, 0, 1])
                cos_angle = np.abs(np.dot(normal, vertical))
                slope_angle = np.degrees(np.arccos(np.clip(cos_angle, 0, 1)))
                slopes.append(slope_angle)
        
        if not slopes:
            return {'slope_mean': float('nan'), 'slope_std': float('nan')}
        
        slopes = np.array(slopes)
        return {
            'slope_mean': np.mean(slopes),
            'slope_std': np.std(slopes),
            'slope_median': np.median(slopes),
            'slope_max': np.max(slopes),
            'slope_min': np.min(slopes)
        }
    
    def _fit_plane_normal(self, points: np.ndarray) -> Optional[np.ndarray]:
        """Fit a plane to points and return the normal vector."""
        if len(points) < 3:
            return None
        
        # Center the points
        centroid = np.mean(points, axis=0)
        centered = points - centroid
        
        # SVD to find the normal
        try:
            _, _, V = np.linalg.svd(centered)
            normal = V[-1]  # Last row of V
            return normal / np.linalg.norm(normal)
        except np.linalg.LinAlgError:
            return None


class PlaneOfBestFit(ComplexityMetric):
    """Plane of best fit analysis for surface characterization."""
    
    @property
    def name(self) -> str:
        return "plane_of_best_fit"
    
    @property
    def description(self) -> str:
        return "Global and local plane fitting analysis"
    
    def calculate(self, mesh_data: Dict[str, Any], 
                 grid_size: int = 10) -> Dict[str, Any]:
        """
        Calculate plane of best fit metrics.
        
        Parameters:
        mesh_data: Dictionary containing mesh data
        grid_size: Size of grid for local analysis
        
        Returns:
        Dictionary with plane fitting results
        """
        points = mesh_data.get('points')
        if points is None:
            return {'global_fit_error': float('nan')}
        
        # Global plane of best fit
        global_normal, global_d, global_error = self._fit_global_plane(points)
        
        # Local plane fitting across grid
        local_errors = self._analyze_local_planes(points, grid_size)
        
        return {
            'global_fit_error': global_error,
            'global_normal': global_normal.tolist() if global_normal is not None else None,
            'local_fit_error_mean': np.mean(local_errors) if local_errors else float('nan'),
            'local_fit_error_std': np.std(local_errors) if local_errors else float('nan'),
            'planarity_ratio': global_error / np.mean(local_errors) if local_errors and global_error else float('nan')
        }
    
    def _fit_global_plane(self, points: np.ndarray) -> Tuple[Optional[np.ndarray], float, float]:
        """Fit a single plane to all points."""
        if len(points) < 3:
            return None, 0, float('nan')
        
        # Center points
        centroid = np.mean(points, axis=0)
        centered = points - centroid
        
        # Fit plane using SVD
        try:
            _, s, V = np.linalg.svd(centered)
            normal = V[-1]
            
            # Calculate fitting error (sum of squared distances to plane)
            distances = np.abs(np.dot(centered, normal))
            error = np.sum(distances**2) / len(points)
            
            # Plane equation: normal • (x - centroid) = 0
            d = -np.dot(normal, centroid)
            
            return normal, d, error
        except np.linalg.LinAlgError:
            return None, 0, float('nan')
    
    def _analyze_local_planes(self, points: np.ndarray, grid_size: int) -> List[float]:
        """Analyze plane fitting across a grid."""
        if len(points) == 0:
            return []
        
        # Create bounding box
        min_coords = np.min(points, axis=0)
        max_coords = np.max(points, axis=0)
        
        # Grid spacing
        spacing = (max_coords - min_coords) / grid_size
        
        errors = []
        
        for i in range(grid_size):
            for j in range(grid_size):
                for k in range(grid_size):
                    # Define grid cell bounds
                    cell_min = min_coords + np.array([i, j, k]) * spacing
                    cell_max = cell_min + spacing
                    
                    # Find points in this cell
                    mask = np.all((points >= cell_min) & (points < cell_max), axis=1)
                    cell_points = points[mask]
                    
                    if len(cell_points) >= 3:
                        _, _, error = self._fit_global_plane(cell_points)
                        if not np.isnan(error):
                            errors.append(error)
        
        return errors


class HeightRange(SurfaceMetric):
    """Height range and vertical distribution analysis."""
    
    @property
    def name(self) -> str:
        return "height_range"
    
    @property
    def description(self) -> str:
        return "Vertical extent and distribution characteristics"
    
    def calculate(self, mesh_data: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate height range metrics."""
        points = mesh_data.get('points')
        if points is None:
            return {'height_range': float('nan')}
        
        z_coords = points[:, 2]
        
        return {
            'height_range': np.max(z_coords) - np.min(z_coords),
            'height_mean': np.mean(z_coords),
            'height_std': np.std(z_coords),
            'height_skewness': self._calculate_skewness(z_coords),
            'height_kurtosis': self._calculate_kurtosis(z_coords),
            'height_percentile_90': np.percentile(z_coords, 90),
            'height_percentile_10': np.percentile(z_coords, 10),
            'height_interquartile_range': np.percentile(z_coords, 75) - np.percentile(z_coords, 25)
        }
    
    def _calculate_skewness(self, data: np.ndarray) -> float:
        """Calculate skewness of the distribution."""
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0
        return np.mean(((data - mean) / std) ** 3)
    
    def _calculate_kurtosis(self, data: np.ndarray) -> float:
        """Calculate kurtosis of the distribution."""
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0
        return np.mean(((data - mean) / std) ** 4) - 3


class FractalDimensionBox(ComplexityMetric):
    """Box-counting fractal dimension with multiple algorithms."""
    
    @property
    def name(self) -> str:
        return "fractal_dimension_box"
    
    @property
    def description(self) -> str:
        return "Box-counting fractal dimension using multiple scales"
    
    def calculate(self, mesh_data: Dict[str, Any], 
                 n_scales: int = 20,
                 scale_range: Tuple[float, float] = (1e-3, 1.0)) -> Dict[str, Any]:
        """Calculate box-counting fractal dimension."""
        points = mesh_data.get('points')
        if points is None:
            return {'fractal_dimension': float('nan')}
        
        # Generate logarithmically spaced scales
        min_scale, max_scale = scale_range
        scales = np.logspace(np.log10(min_scale), np.log10(max_scale), n_scales)
        
        # Calculate bounding box
        min_coords = np.min(points, axis=0)
        max_coords = np.max(points, axis=0)
        extent = max_coords - min_coords
        
        # Normalize scales by the largest extent
        max_extent = np.max(extent)
        scales = scales * max_extent
        
        counts = []
        
        for scale in scales:
            # Count occupied boxes at this scale
            n_boxes = np.ceil(extent / scale).astype(int)
            
            # Assign points to boxes
            box_indices = np.floor((points - min_coords) / scale).astype(int)
            
            # Ensure indices are within bounds
            for dim in range(3):
                box_indices[:, dim] = np.clip(box_indices[:, dim], 0, n_boxes[dim] - 1)
            
            # Count unique boxes
            unique_boxes = np.unique(box_indices, axis=0)
            counts.append(len(unique_boxes))
        
        # Fit power law: N(r) = C * r^(-D)
        # log(N) = log(C) - D * log(r)
        log_scales = np.log(scales)
        log_counts = np.log(counts)
        
        # Linear regression
        A = np.vstack([log_scales, np.ones(len(log_scales))]).T
        slope, intercept = np.linalg.lstsq(A, log_counts, rcond=None)[0]
        
        fractal_dim = -slope
        
        # Calculate goodness of fit
        predicted = slope * log_scales + intercept
        r_squared = 1 - np.sum((log_counts - predicted)**2) / np.sum((log_counts - np.mean(log_counts))**2)
        
        return {
            'fractal_dimension': fractal_dim,
            'r_squared': r_squared,
            'scales_used': len(scales),
            'scale_range': [float(np.min(scales)), float(np.max(scales))]
        }


class SurfaceComplexityIndex(ComplexityMetric):
    """Combined surface complexity index using multiple measures."""
    
    @property
    def name(self) -> str:
        return "surface_complexity_index"
    
    @property
    def description(self) -> str:
        return "Composite index combining multiple complexity measures"
    
    def calculate(self, mesh_data: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate composite surface complexity index."""
        # Get individual metrics
        slope_metric = SlopeMetric()
        height_metric = HeightRange()
        fractal_metric = FractalDimensionBox()
        
        slope_results = slope_metric.calculate(mesh_data)
        height_results = height_metric.calculate(mesh_data)
        fractal_results = fractal_metric.calculate(mesh_data)
        
        # Extract key values
        slope_std = slope_results.get('slope_std', 0)
        height_range = height_results.get('height_range', 0)
        fractal_dim = fractal_results.get('fractal_dimension', 2)
        
        # Normalize components (example normalization)
        # In practice, these would be calibrated against known complexity scales
        norm_slope = np.clip(slope_std / 45.0, 0, 1)  # Normalize by max expected slope
        norm_height = np.clip(height_range / np.max(mesh_data.get('points', [[0,0,1]])[:, 2] - 
                                                   np.min(mesh_data.get('points', [[0,0,0]])[:, 2])), 0, 1)
        norm_fractal = np.clip((fractal_dim - 2.0) / 1.0, 0, 1)  # Fractal excess over 2D
        
        # Weighted combination
        complexity_index = (0.4 * norm_slope + 0.3 * norm_height + 0.3 * norm_fractal)
        
        return {
            'complexity_index': complexity_index,
            'normalized_slope': norm_slope,
            'normalized_height': norm_height,
            'normalized_fractal': norm_fractal,
            'component_weights': [0.4, 0.3, 0.3]
        }


class VectorDispersion(ComplexityMetric):
    """Surface normal vector dispersion analysis."""
    
    @property
    def name(self) -> str:
        return "vector_dispersion"
    
    @property
    def description(self) -> str:
        return "Analysis of surface normal vector dispersion"
    
    def calculate(self, mesh_data: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate vector dispersion metrics."""
        mesh = mesh_data.get('mesh')
        if mesh is None:
            return {'vector_dispersion': float('nan')}
        
        # Compute face normals
        mesh.compute_normals(inplace=True)
        
        # Get face normals
        if hasattr(mesh, 'face_normals'):
            normals = mesh.face_normals
        else:
            # Fallback: compute normals manually
            points = mesh.points
            faces = mesh.faces.reshape(-1, 4)[:, 1:4]  # Remove the count column
            normals = self._compute_face_normals(points, faces)
        
        if len(normals) == 0:
            return {'vector_dispersion': float('nan')}
        
        # Calculate dispersion measures
        dispersion_metrics = self._analyze_normal_dispersion(normals)
        
        return dispersion_metrics
    
    def _compute_face_normals(self, points: np.ndarray, faces: np.ndarray) -> np.ndarray:
        """Compute face normals manually."""
        normals = []
        
        for face in faces:
            if len(face) >= 3:
                p0, p1, p2 = points[face[0]], points[face[1]], points[face[2]]
                v1 = p1 - p0
                v2 = p2 - p0
                normal = np.cross(v1, v2)
                norm = np.linalg.norm(normal)
                if norm > 0:
                    normals.append(normal / norm)
        
        return np.array(normals)
    
    def _analyze_normal_dispersion(self, normals: np.ndarray) -> Dict[str, Any]:
        """Analyze the dispersion of normal vectors."""
        # Mean normal direction
        mean_normal = np.mean(normals, axis=0)
        mean_normal = mean_normal / np.linalg.norm(mean_normal)
        
        # Angular deviations from mean
        dot_products = np.dot(normals, mean_normal)
        # Clamp to valid range for arccos
        dot_products = np.clip(dot_products, -1, 1)
        angular_deviations = np.arccos(np.abs(dot_products))
        
        # Spherical variance (1 - |mean resultant length|)
        resultant = np.mean(normals, axis=0)
        resultant_length = np.linalg.norm(resultant)
        spherical_variance = 1 - resultant_length
        
        # Concentration parameter (von Mises-Fisher approximation)
        if resultant_length > 0.999:
            concentration = 1000  # Very concentrated
        elif resultant_length < 0.001:
            concentration = 0  # Completely dispersed
        else:
            # Approximation for 3D
            concentration = resultant_length * (3 - resultant_length**2) / (1 - resultant_length**2)
        
        return {
            'vector_dispersion': spherical_variance,
            'mean_angular_deviation': np.mean(angular_deviations),
            'std_angular_deviation': np.std(angular_deviations),
            'concentration_parameter': concentration,
            'resultant_length': resultant_length,
            'max_angular_deviation': np.max(angular_deviations)
        }


# Register all metrics
def register_complexity_metrics():
    """Register all complexity metrics with the global registry."""
    metrics = [
        SlopeMetric(),
        PlaneOfBestFit(),
        HeightRange(),
        FractalDimensionBox(),
        SurfaceComplexityIndex(),
        VectorDispersion()
    ]
    
    for metric in metrics:
        register_metric(metric)


# Auto-register when module is imported
register_complexity_metrics() 