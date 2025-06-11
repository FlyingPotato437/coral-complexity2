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
    
    @property
    def name(self) -> str:
        return "slope"
    
    @property
    def description(self) -> str:
        return "local surface slope angles calculated via plane fitting"
    
    def calculate(self, mesh_data: Dict[str, Any], 
                 neighborhood_radius: float = 0.1,
                 min_neighbors: int = 5) -> Dict[str, Any]:
        
        points = mesh_data.get('points')
        if points is None:
            return {'slope_mean': float('nan'), 'slope_std': float('nan')}
        
        tree = scipy.spatial.cKDTree(points)
        slopes = []
        
        for i, point in enumerate(points):
            neighbor_indices = tree.query_ball_point(point, neighborhood_radius)
            
            if len(neighbor_indices) < min_neighbors:
                continue
            
            neighbor_points = points[neighbor_indices]
            normal = self._fit_plane_normal(neighbor_points)
            if normal is not None:
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
        if len(points) < 3:
            return None
        
        centroid = np.mean(points, axis=0)
        centered = points - centroid
        
        try:
            _, _, V = np.linalg.svd(centered)
            normal = V[-1]
            return normal / np.linalg.norm(normal)
        except np.linalg.LinAlgError:
            return None


class PlaneOfBestFit(ComplexityMetric):
    
    @property
    def name(self) -> str:
        return "plane_of_best_fit"
    
    @property
    def description(self) -> str:
        return "global and local plane fitting analysis"
    
    def calculate(self, mesh_data: Dict[str, Any], 
                 grid_size: int = 10) -> Dict[str, Any]:
        
        points = mesh_data.get('points')
        if points is None:
            return {'global_fit_error': float('nan')}
        
        global_normal, global_d, global_error = self._fit_global_plane(points)
        local_errors = self._analyze_local_planes(points, grid_size)
        
        return {
            'global_fit_error': global_error,
            'global_normal': global_normal.tolist() if global_normal is not None else None,
            'local_fit_error_mean': np.mean(local_errors) if local_errors else float('nan'),
            'local_fit_error_std': np.std(local_errors) if local_errors else float('nan'),
            'planarity_ratio': global_error / np.mean(local_errors) if local_errors and global_error else float('nan')
        }
    
    def _fit_global_plane(self, points: np.ndarray) -> Tuple[Optional[np.ndarray], float, float]:
        if len(points) < 3:
            return None, 0, float('nan')
        
        centroid = np.mean(points, axis=0)
        centered = points - centroid
        
        try:
            _, s, V = np.linalg.svd(centered)
            normal = V[-1]
            distances = np.abs(np.dot(centered, normal))
            error = np.sum(distances**2) / len(points)
            d = -np.dot(normal, centroid)
            return normal, d, error
        except np.linalg.LinAlgError:
            return None, 0, float('nan')
    
    def _analyze_local_planes(self, points: np.ndarray, grid_size: int) -> List[float]:
        if len(points) == 0:
            return []
        
        min_coords = np.min(points, axis=0)
        max_coords = np.max(points, axis=0)
        spacing = (max_coords - min_coords) / grid_size
        errors = []
        
        for i in range(grid_size):
            for j in range(grid_size):
                for k in range(grid_size):
                    cell_min = min_coords + np.array([i, j, k]) * spacing
                    cell_max = cell_min + spacing
                    mask = np.all((points >= cell_min) & (points < cell_max), axis=1)
                    cell_points = points[mask]
                    
                    if len(cell_points) >= 3:
                        _, _, error = self._fit_global_plane(cell_points)
                        if not np.isnan(error):
                            errors.append(error)
        
        return errors


class HeightRange(SurfaceMetric):
    
    @property
    def name(self) -> str:
        return "height_range"
    
    @property
    def description(self) -> str:
        return "vertical extent and distribution characteristics"
    
    def calculate(self, mesh_data: Dict[str, Any]) -> Dict[str, Any]:
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
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0
        return np.mean(((data - mean) / std) ** 3)
    
    def _calculate_kurtosis(self, data: np.ndarray) -> float:
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0
        return np.mean(((data - mean) / std) ** 4) - 3


class FractalDimensionBox(ComplexityMetric):
    
    @property
    def name(self) -> str:
        return "fractal_dimension_box"
    
    @property
    def description(self) -> str:
        return "box-counting fractal dimension using multiple scales"
    
    def calculate(self, mesh_data: Dict[str, Any], 
                 n_scales: int = 20,
                 scale_range: Tuple[float, float] = (1e-3, 1.0)) -> Dict[str, Any]:
        points = mesh_data.get('points')
        if points is None:
            return {'fractal_dimension': float('nan')}
        
        min_scale, max_scale = scale_range
        scales = np.logspace(np.log10(min_scale), np.log10(max_scale), n_scales)
        min_coords = np.min(points, axis=0)
        max_coords = np.max(points, axis=0)
        extent = max_coords - min_coords
        max_extent = np.max(extent)
        scales = scales * max_extent
        
        counts = []
        
        for scale in scales:
            n_boxes = np.ceil(extent / scale).astype(int)
            box_indices = np.floor((points - min_coords) / scale).astype(int)
            
            for dim in range(3):
                box_indices[:, dim] = np.clip(box_indices[:, dim], 0, n_boxes[dim] - 1)
            
            unique_boxes = np.unique(box_indices, axis=0)
            counts.append(len(unique_boxes))
        
        log_scales = np.log(scales)
        log_counts = np.log(counts)
        A = np.vstack([log_scales, np.ones(len(log_scales))]).T
        slope, intercept = np.linalg.lstsq(A, log_counts, rcond=None)[0]
        fractal_dim = -slope
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


class ComplexityMetrics:
    
    def __init__(self):
        self.mesh_file = None
        self.mesh = None
        
    def load_mesh(self, file, verbose=True):
        import os
        import time
        
        self.mesh_file = file
        if not os.path.exists(file):
            if verbose:
                print(f"3D model file not found: {file}")
            self.mesh = None
            return
        
        if verbose:
            print(f"Loading mesh from {file}...")
            
        start_time = time.time()
        
        try:
            import pyvista as pv
            self.mesh = pv.read(file)
            
            if verbose:
                end_time = time.time()
                print("Mesh loaded in {:.2f} seconds".format(end_time - start_time))
                print(f"Number of points: {self.mesh.n_points}")
                print(f"Number of faces: {self.mesh.n_cells}")
                
        except Exception as e:
            if verbose:
                print(f"Failed to load mesh: {e}")
            self.mesh = None
            
    def calculate(self, mesh_file, shading_metrics=True, shading_light_dir=None, 
                 shading_sample_size=1000000, quadrat_metrics=False, 
                 quadrat_sizes=[1], verbose=True):
        
        import os
        import numpy as np
        from .geometric_measures import GeometricMeasures
        from .shading import Shading
        from .quadrat_metrics import QuadratMetrics
        
        if not os.path.exists(mesh_file):
            if verbose:
                print(f"3D model file not found: {mesh_file}")
            return None
            
        if self.mesh is None or self.mesh_file != mesh_file:
            self.load_mesh(mesh_file, verbose=verbose)
            
        if self.mesh is None:
            return None
            
        if verbose:
            print("Calculating complexity measures for mesh...")
            
        try:
            surface_area = float(self.mesh.area) if hasattr(self.mesh, 'area') else np.nan
            is_watertight = self.mesh.is_manifold if hasattr(self.mesh, 'is_manifold') else False
            
            volume = np.nan
            if is_watertight:
                try:
                    volume = float(self.mesh.volume) if hasattr(self.mesh, 'volume') else np.nan
                except:
                    volume = np.nan
                    is_watertight = False
            
            try:
                convex_hull = self.mesh.convex_hull()
                cvh_volume = float(convex_hull.volume) if hasattr(convex_hull, 'volume') else np.nan
            except:
                cvh_volume = np.nan
            
            points = self.mesh.points
            if len(points) > 0:
                points_2d = points[:, :2]
                try:
                    from scipy.spatial import ConvexHull
                    hull_2d = ConvexHull(points_2d)
                    projected_area = hull_2d.volume
                except:
                    min_xy = np.min(points_2d, axis=0)
                    max_xy = np.max(points_2d, axis=0)
                    projected_area = (max_xy[0] - min_xy[0]) * (max_xy[1] - min_xy[1])
            else:
                projected_area = np.nan
            
            asr = cvh_volume - volume if not np.isnan(cvh_volume) and not np.isnan(volume) else np.nan
            proportion_occupied = volume / cvh_volume if not np.isnan(cvh_volume) and not np.isnan(volume) and cvh_volume > 0 else np.nan
            ssf = asr / surface_area if not np.isnan(asr) and not np.isnan(surface_area) and surface_area > 0 else np.nan
            
            geom_results = {
                'surface_area': surface_area,
                'projected_area': projected_area,
                'volume': volume,
                'CVH_volume': cvh_volume,
                'ASR': asr,
                'proportion_occupied': proportion_occupied,
                'SSF': ssf,
                'is_watertight': is_watertight
            }
            
        except Exception as e:
            if verbose:
                print(f"Error in geometric calculations: {e}")
            geom_results = {}
        
        plot_metrics = {
            'mesh_file': mesh_file,
            'is_watertight': geom_results.get('is_watertight', False),
            'num_faces': self.mesh.n_cells,
            'num_vertices': self.mesh.n_points,
            '3d_surface_area': geom_results.get('surface_area', np.nan),
            '2d_surface_area': geom_results.get('projected_area', np.nan),
            'volume': geom_results.get('volume', np.nan),
            'convex_hull_volume': geom_results.get('CVH_volume', np.nan),
            'absolute_spatial_refuge': geom_results.get('ASR', np.nan),
            'proportion_occupied': geom_results.get('proportion_occupied', np.nan),
            'shelter_size_factor': geom_results.get('SSF', np.nan),
            'surface_rugosity': geom_results.get('surface_area', np.nan) / geom_results.get('projected_area', 1) if geom_results.get('projected_area', 0) > 0 else np.nan,
        }
        
        if shading_metrics:
            try:
                if shading_light_dir is None:
                    shading_light_dir = np.array([0, 0, -1])
                
                shading = Shading(cpu_percentage=50.0)
                shading.mesh = self.mesh
                shading.mesh_file = mesh_file
                
                shading_result = shading.calculate(
                    light_dir=shading_light_dir,
                    sample_size=min(shading_sample_size, 500000),
                    verbose=False
                )
                
                plot_metrics['shaded_percentage'] = shading_result.get('shaded_percentage', np.nan)
                plot_metrics['illuminated_percentage'] = shading_result.get('illuminated_percentage', np.nan)
                
            except Exception as e:
                if verbose:
                    print(f"Error in shading calculations: {e}")
                plot_metrics['shaded_percentage'] = np.nan
                plot_metrics['illuminated_percentage'] = np.nan
        
        result = {'plot_metrics': plot_metrics}
        
        if quadrat_metrics:
            try:
                quadrat_results = []
                
                for size in quadrat_sizes:
                    quadrats = self._generate_quadrats(size)
                    
                    for quadrat in quadrats:
                        filtered_mesh = self._filter_mesh_to_quadrat(quadrat)
                        if len(filtered_mesh.points) > 0:
                            quad_metrics = self._calculate_quadrat_metrics(
                                mesh_file, filtered_mesh, quadrat, shading_metrics, 
                                shading_light_dir, shading_sample_size
                            )
                            quadrat_results.append(quad_metrics)
                
                result['quadrat_metrics'] = quadrat_results
                
            except Exception as e:
                if verbose:
                    print(f"Error in quadrat calculations: {e}")
                result['quadrat_metrics'] = []
        
        return result
    
    def process_directory(self, directory, shading_metrics=True, shading_light_dir=None, 
                         shading_sample_size=1000000, quadrat_metrics=False, 
                         quadrat_sizes=[1], verbose=False, save_results=True, save_dir='.'):
        """
        Process all mesh files in the specified directory using enhanced functionality.
        
        Parameters:
        directory (str): Path to the directory containing 3D model files.
        shading_metrics (bool): Whether to apply shading to the mesh.
        shading_light_dir (np.array): Direction of the light source for shading.
        shading_sample_size (int): Number of samples for shading calculation.
        quadrat_metrics (bool): Whether to calculate quadrat metrics.
        quadrat_sizes (list): List of quadrat sizes to use for quadrat metrics.
        verbose (bool): Whether to print progress messages.
        save_results (bool): Whether to save the results to CSV files.
        save_dir (str): Directory to save the CSV files.
        
        Returns:
        list: List of dictionaries containing complexity measures of each mesh.
        """
        import os
        import pandas as pd
        from tqdm import tqdm
        import numpy as np
        
        if not os.path.exists(directory):
            print(f"Directory not found: {directory}")
            return []
        
        mesh_files = [file for file in os.listdir(directory) 
                     if file.lower().endswith((".obj", ".ply", ".stl", ".vtk"))]
        
        if not mesh_files:
            print(f"No supported mesh files found in {directory}")
            return []
        
        results = []
        
        # Use default light direction if not provided
        if shading_light_dir is None:
            shading_light_dir = np.array([0, 0, -1])
        
        for mesh_file in tqdm(mesh_files, desc="Processing 3D models", disable=not verbose):
            try:
                result = self.calculate(
                    mesh_file=os.path.join(directory, mesh_file),
                    shading_metrics=shading_metrics,
                    shading_light_dir=shading_light_dir,
                    shading_sample_size=shading_sample_size,
                    quadrat_metrics=quadrat_metrics,
                    quadrat_sizes=quadrat_sizes,
                    verbose=False  # Suppress individual file verbose output
                )
                if result:
                    results.append(result)
            except Exception as e:
                if verbose:
                    print(f"Error processing {mesh_file}: {e}")
                continue
        
        # Save results if requested
        if save_results and results:
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
                
            # Plot-level metrics
            plot_data = [r['plot_metrics'] for r in results if 'plot_metrics' in r]
            if plot_data:
                plot_df = pd.DataFrame(plot_data)
                plot_path = os.path.join(save_dir, "plot_complexity_metrics.csv")
                plot_df.to_csv(plot_path, index=False)
                if verbose:
                    print(f"Plot metrics saved to {plot_path}")
            
            # Quadrat-level metrics
            quadrat_data = []
            for r in results:
                if 'quadrat_metrics' in r and r['quadrat_metrics']:
                    quadrat_data.extend(r['quadrat_metrics'])
            
            if quadrat_data:
                quadrat_df = pd.DataFrame(quadrat_data)
                quadrat_path = os.path.join(save_dir, "quadrat_complexity_metrics.csv")
                quadrat_df.to_csv(quadrat_path, index=False)
                if verbose:
                    print(f"Quadrat metrics saved to {quadrat_path}")
        
        return results
    
    def _generate_quadrats(self, quadrat_size):
        import numpy as np
        
        points = self.mesh.points
        x_min, x_max = float(np.min(points[:, 0])), float(np.max(points[:, 0]))
        y_min, y_max = float(np.min(points[:, 1])), float(np.max(points[:, 1]))
        
        centroid_x = (x_min + x_max) / 2
        centroid_y = (y_min + y_max) / 2
        
        dist_x_pos = x_max - centroid_x
        dist_x_neg = centroid_x - x_min
        dist_y_pos = y_max - centroid_y
        dist_y_neg = centroid_y - y_min
        
        num_x_pos = int(np.ceil(dist_x_pos / quadrat_size))
        num_x_neg = int(np.ceil(dist_x_neg / quadrat_size))
        num_y_pos = int(np.ceil(dist_y_pos / quadrat_size))
        num_y_neg = int(np.ceil(dist_y_neg / quadrat_size))
        
        quadrats = []
        for i in range(-num_x_neg, num_x_pos + 1):
            for j in range(-num_y_neg, num_y_pos + 1):
                x_center = centroid_x + i * quadrat_size
                y_center = centroid_y + j * quadrat_size
                x_quad_min = x_center - quadrat_size / 2
                x_quad_max = x_center + quadrat_size / 2
                y_quad_min = y_center - quadrat_size / 2
                y_quad_max = y_center + quadrat_size / 2
                
                if (x_center < x_min or x_center > x_max or
                        y_center < y_min or y_center > y_max):
                    continue
                
                x_quad_min = max(x_quad_min, x_min)
                x_quad_max = min(x_quad_max, x_max)
                y_quad_min = max(y_quad_min, y_min)
                y_quad_max = min(y_quad_max, y_max)
                
                quadrats.append({
                    "x_id": i,
                    "y_id": j,
                    "x_center": x_center,
                    "y_center": y_center,
                    "size": quadrat_size,
                    "x_min": x_quad_min,
                    "x_max": x_quad_max,
                    "y_min": y_quad_min,
                    "y_max": y_quad_max
                })
        
        return quadrats
    
    def _filter_mesh_to_quadrat(self, quadrat):
        import numpy as np
        import pyvista as pv
        
        points = self.mesh.points
        faces = self.mesh.faces
        
        triangular_faces = []
        i = 0
        while i < len(faces):
            n_vertices = faces[i]
            if n_vertices == 3:
                face = faces[i+1:i+1+n_vertices]
                triangular_faces.append(face)
            i += n_vertices + 1
        
        if not triangular_faces:
            return pv.PolyData()
        
        triangular_faces = np.array(triangular_faces)
        face_centroids = np.mean(points[triangular_faces], axis=1)
        
        mask = ((face_centroids[:, 0] >= quadrat["x_min"]) & 
                (face_centroids[:, 0] <= quadrat["x_max"]) &
                (face_centroids[:, 1] >= quadrat["y_min"]) & 
                (face_centroids[:, 1] <= quadrat["y_max"]))
        
        filtered_faces = triangular_faces[mask]
        
        if len(filtered_faces) == 0:
            return pv.PolyData()
        
        pv_faces = []
        for face in filtered_faces:
            pv_faces.extend([3] + face.tolist())
        
        return pv.PolyData(points, pv_faces)
    
    def _calculate_quadrat_metrics(self, mesh_file, filtered_mesh, quadrat, 
                                 shading_metrics, shading_light_dir, shading_sample_size):
        import numpy as np
        
        sf_area = float(filtered_mesh.area) if hasattr(filtered_mesh, 'area') else np.nan
        
        points = filtered_mesh.points
        if len(points) > 0:
            from scipy.spatial import ConvexHull
            try:
                points_2d = points[:, :2]
                hull_2d = ConvexHull(points_2d)
                sf_area_2d = float(hull_2d.volume)
            except:
                sf_area_2d = np.nan
        else:
            sf_area_2d = np.nan
        
        rugosity = float(sf_area / sf_area_2d) if sf_area_2d > 0 else np.nan
        watertight = filtered_mesh.is_manifold if hasattr(filtered_mesh, 'is_manifold') else False
        
        if len(filtered_mesh.points) < 4:
            cvh_vol = np.nan
            vol = np.nan
        else:
            try:
                cvh_vol = float(filtered_mesh.convex_hull().volume) if hasattr(filtered_mesh.convex_hull(), 'volume') else np.nan
                vol = float(filtered_mesh.volume) if watertight and hasattr(filtered_mesh, 'volume') else np.nan
            except:
                cvh_vol = np.nan
                vol = np.nan
        
        asr = cvh_vol - vol if not np.isnan(cvh_vol) and not np.isnan(vol) else np.nan
        prop_occ = vol / cvh_vol if not np.isnan(cvh_vol) and not np.isnan(vol) and cvh_vol > 0 else np.nan
        ssf = asr / sf_area if not np.isnan(asr) and not np.isnan(sf_area) and sf_area > 0 else np.nan
        
        shaded_pct = np.nan
        illuminated_pct = np.nan
        
        if shading_metrics and len(filtered_mesh.points) > 0:
            try:
                shading = Shading(cpu_percentage=25.0)
                shading.mesh = filtered_mesh
                result = shading.calculate(
                    light_dir=shading_light_dir,
                    sample_size=min(50000, shading_sample_size // 10),
                    verbose=False
                )
                shaded_pct = result.get('shaded_percentage', np.nan)
                illuminated_pct = result.get('illuminated_percentage', np.nan)
            except:
                pass
        
        return {
            'mesh_file': mesh_file,
            'quadrat_size': quadrat['size'],
            'quadrat_x_id': quadrat['x_id'],
            'quadrat_y_id': quadrat['y_id'],
            'quadrat_x_min': quadrat['x_min'],
            'quadrat_x_max': quadrat['x_max'],
            'quadrat_y_min': quadrat['y_min'],
            'quadrat_y_max': quadrat['y_max'],
            'quadrat_x_center': quadrat['x_center'],
            'quadrat_y_center': quadrat['y_center'],
            'is_watertight': watertight,
            'num_faces': filtered_mesh.n_cells,
            'num_vertices': filtered_mesh.n_points,
            '3d_surface_area': sf_area,
            '2d_surface_area': sf_area_2d,
            'volume': vol,
            'convex_hull_volume': cvh_vol,
            'absolute_spatial_refuge': asr,
            'proportion_occupied': prop_occ,
            'shelter_size_factor': ssf,
            'surface_rugosity': rugosity,
            'shaded_percentage': shaded_pct,
            'illuminated_percentage': illuminated_pct
        }