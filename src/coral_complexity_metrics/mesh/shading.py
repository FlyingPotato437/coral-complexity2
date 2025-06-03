import os
import numpy as np
import pyvista as pv
from tqdm import tqdm
import multiprocessing as mp
from functools import partial
from typing import Optional, Tuple, Dict, Any, Union
import warnings
from datetime import datetime
import math
from ._internal._shading_utils import AABB, BVHNode


class Shading:
    """
    A class to calculate shading percentage based on the coral structure.
    
    **IMPORTANT: STRUCTURAL-ONLY SCOPE**
    This implementation only considers the geometric structure of the coral mesh
    for shadow calculations. It does NOT account for:
    - Water column effects (depth-dependent light attenuation)
    - Turbidity and suspended particles
    - Underwater light scattering and refraction
    - Spectral changes with depth
    - Bio-optical properties of water
    
    For accurate underwater light modeling, additional physics-based models
    are required. This tool is best suited for comparative structural analysis
    and relative shading assessments.
    """

    def __init__(self, cpu_percentage: float = 80.0):
        """
        Initialize the Shading class with default values.
        
        Parameters:
        cpu_percentage (float): Percentage of available CPU cores to use (1-100).
                               Defaults to 80% for system stability.
        """
        self.mesh_file = None
        self.mesh = None
        self._validate_cpu_percentage(cpu_percentage)
        self.cpu_percentage = cpu_percentage
        self.cpu_limit = max(1, int(mp.cpu_count() * cpu_percentage / 100))

    def _validate_cpu_percentage(self, cpu_percentage: float) -> None:
        """Validate CPU percentage input."""
        if not isinstance(cpu_percentage, (int, float)):
            raise TypeError("cpu_percentage must be a number")
        if not 1 <= cpu_percentage <= 100:
            raise ValueError("cpu_percentage must be between 1 and 100")

    def _validate_sampling_points(self, sample_size: int) -> None:
        """Validate sampling point count."""
        if not isinstance(sample_size, int):
            raise TypeError("sample_size must be an integer")
        if sample_size < 1:
            raise ValueError("sample_size must be positive")
        if sample_size > 10_000_000:
            warnings.warn(
                f"Large sample size ({sample_size:,}) may cause memory issues. "
                "Consider using a smaller value (< 1M points).",
                UserWarning
            )

    def _warn_unsupported_parameters(self, **kwargs) -> None:
        """Warn when users request unsupported environmental factors."""
        unsupported = {
            'depth': 'Water depth effects on light attenuation',
            'turbidity': 'Turbidity and suspended particle scattering',
            'water_properties': 'Bio-optical water properties',
            'spectral_response': 'Spectral changes with depth',
            'refraction': 'Underwater light refraction effects'
        }
        
        for param, description in unsupported.items():
            if param in kwargs and kwargs[param] is not None:
                warnings.warn(
                    f"Parameter '{param}' is not supported in structural-only analysis. "
                    f"{description} are not modeled. Results represent geometric "
                    f"shading only.", 
                    UserWarning
                )

    def calculate_solar_position(self, day_of_year: int, time_of_day: float, 
                               latitude: float = 0.0, longitude: float = 0.0) -> np.ndarray:
        """
        Calculate solar position based on time and location.
        
        Parameters:
        day_of_year (int): Day of year (1-365)
        time_of_day (float): Time in hours (0-24)
        latitude (float): Latitude in degrees (-90 to 90)
        longitude (float): Longitude in degrees (-180 to 180)
        
        Returns:
        np.ndarray: Light direction vector (normalized)
        """
        # Validate inputs
        if not 1 <= day_of_year <= 365:
            raise ValueError("day_of_year must be between 1 and 365")
        if not 0 <= time_of_day <= 24:
            raise ValueError("time_of_day must be between 0 and 24")
        if not -90 <= latitude <= 90:
            raise ValueError("latitude must be between -90 and 90 degrees")
        if not -180 <= longitude <= 180:
            raise ValueError("longitude must be between -180 and 180 degrees")
        
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
        # In mesh coordinates: +Z up, +X east, +Y north
        x = math.sin(azimuth) * math.cos(elevation)  # East component
        y = math.cos(azimuth) * math.cos(elevation)  # North component  
        z = math.sin(elevation)  # Up component
        
        # Return normalized vector pointing FROM sun TO surface (for ray casting)
        return -np.array([x, y, z])

    def adjust_light_for_slope_aspect(self, base_light_dir: np.ndarray, 
                                    slope: float, aspect: float) -> np.ndarray:
        """
        Adjust light direction based on seafloor slope and aspect.
        
        Parameters:
        base_light_dir (np.ndarray): Base light direction vector
        slope (float): Slope angle in degrees (0-90)
        aspect (float): Aspect angle in degrees (0-360), where 0=North
        
        Returns:
        np.ndarray: Adjusted light direction vector
        """
        if not 0 <= slope <= 90:
            raise ValueError("slope must be between 0 and 90 degrees")
        if not 0 <= aspect <= 360:
            raise ValueError("aspect must be between 0 and 360 degrees")
        
        # Convert to radians
        slope_rad = math.radians(slope)
        aspect_rad = math.radians(aspect)
        
        # Create rotation matrix for slope and aspect
        # This is a simplified model - in reality, underwater light fields are complex
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
        
        return rotation_matrix @ base_light_dir

    def load_mesh(self, mesh_file: str, verbose: bool = True) -> None:
        """
        Load a 3D mesh from the specified file.

        Parameters:
        mesh_file (str): Path to the 3D model file.
        verbose (bool): Whether to print loading information.
        """
        self.mesh_file = mesh_file
        if not os.path.exists(mesh_file):
            raise FileNotFoundError(f"3D model file not found: {mesh_file}")

        if verbose:
            print("Loading 3D mesh...")
        try:
            self.mesh = pv.read(mesh_file)
        except Exception as e:
            raise RuntimeError(f"Failed to load 3D model: {e}")

        if verbose:
            print(f"Number of points: {self.mesh.n_points}")
            print(f"Number of faces: {self.mesh.n_cells}")

    def build_bvh(self, triangles: np.ndarray, indices: np.ndarray, 
                  start: int, end: int, depth: int = 0, max_depth: int = 20) -> Optional[BVHNode]:
        """
        Build a Bounding Volume Hierarchy (BVH) for the given triangles.

        Parameters:
        triangles (np.ndarray): Array of triangles.
        indices (np.ndarray): Array of triangle indices.
        start (int): Start index for the BVH node.
        end (int): End index for the BVH node.
        depth (int): Current depth of the BVH node.
        max_depth (int): Maximum depth of the BVH tree.

        Returns:
        Optional[BVHNode]: The root node of the BVH tree.
        """
        if start >= end or depth > max_depth:
            return None

        aabb_min = np.min(triangles[indices[start:end]], axis=(0, 1))
        aabb_max = np.max(triangles[indices[start:end]], axis=(0, 1))
        node = BVHNode(start, end, AABB(aabb_min, aabb_max))

        if end - start <= 4 or depth == max_depth:  # Leaf node
            return node

        # Choose longest axis to split
        axis = np.argmax(aabb_max - aabb_min)
        mid = (start + end) // 2

        # Sort indices based on triangle centroids
        centroids = np.mean(triangles[indices[start:end]], axis=1)
        indices[start:end] = indices[start:end][np.argsort(centroids[:, axis])]

        node.left = self.build_bvh(triangles, indices, start, mid, depth + 1, max_depth)
        node.right = self.build_bvh(triangles, indices, mid, end, depth + 1, max_depth)
        return node

    def ray_triangle_intersect(self, ray_origin: np.ndarray, ray_direction: np.ndarray, 
                             v0: np.ndarray, v1: np.ndarray, v2: np.ndarray) -> bool:
        """
        Check if a ray intersects with a triangle using Möller-Trumbore algorithm.

        Parameters:
        ray_origin (np.ndarray): Origin of the ray.
        ray_direction (np.ndarray): Direction of the ray.
        v0, v1, v2 (np.ndarray): Triangle vertices.

        Returns:
        bool: True if the ray intersects the triangle, False otherwise.
        """
        epsilon = 1e-6
        edge1 = v1 - v0
        edge2 = v2 - v0
        h = np.cross(ray_direction, edge2)
        a = np.dot(edge1, h)
        if abs(a) < epsilon:
            return False
        f = 1.0 / a
        s = ray_origin - v0
        u = f * np.dot(s, h)
        if u < 0.0 or u > 1.0:
            return False
        q = np.cross(s, edge1)
        v = f * np.dot(ray_direction, q)
        if v < 0.0 or u + v > 1.0:
            return False
        t = f * np.dot(edge2, q)
        return t > epsilon

    def intersect_bvh(self, node: Optional[BVHNode], ray_origin: np.ndarray, 
                     ray_direction: np.ndarray, triangles: np.ndarray, 
                     indices: np.ndarray) -> bool:
        """
        Check if a ray intersects with any triangle in the BVH.

        Parameters:
        node (Optional[BVHNode]): The current BVH node.
        ray_origin (np.ndarray): Origin of the ray.
        ray_direction (np.ndarray): Direction of the ray.
        triangles (np.ndarray): Array of triangles.
        indices (np.ndarray): Array of triangle indices.

        Returns:
        bool: True if the ray intersects any triangle, False otherwise.
        """
        if node is None or not node.aabb.intersect(ray_origin, ray_direction):
            return False

        if node.left is None and node.right is None:
            for i in range(node.start, node.end):
                triangle = triangles[indices[i]]
                if self.ray_triangle_intersect(ray_origin, ray_direction, 
                                             triangle[0], triangle[1], triangle[2]):
                    return True
            return False

        return (self.intersect_bvh(node.left, ray_origin, ray_direction, triangles, indices) or 
                self.intersect_bvh(node.right, ray_origin, ray_direction, triangles, indices))

    def process_chunk(self, args: Tuple) -> np.ndarray:
        """
        Process a chunk of points to determine if they are shadowed.

        Parameters:
        args (Tuple): A tuple containing the chunk of points, BVH root, triangles, indices, and light direction.

        Returns:
        np.ndarray: Array indicating which points are shadowed.
        """
        chunk, bvh_root, triangles, indices, light_dir = args
        shadowed = np.zeros(len(chunk), dtype=bool)
        for i, point in enumerate(chunk):
            if self.intersect_bvh(bvh_root, point, light_dir, triangles, indices):
                shadowed[i] = True
        return shadowed

    def point_in_box(self, point: np.ndarray, box_min: np.ndarray, box_max: np.ndarray) -> bool:
        """Check if a point is inside a bounding box."""
        return np.all(point >= box_min) and np.all(point <= box_max)

    def triangle_intersects_box(self, v0: np.ndarray, v1: np.ndarray, v2: np.ndarray, 
                              box_min: np.ndarray, box_max: np.ndarray) -> bool:
        """Check if a triangle intersects with a bounding box."""
        # Check if any vertex is inside the box
        if any(self.point_in_box(v, box_min, box_max) for v in [v0, v1, v2]):
            return True

        # Check if the triangle intersects any of the 12 edges of the box
        edges = [
            (box_min, [box_max[0], box_min[1], box_min[2]]),
            (box_min, [box_min[0], box_max[1], box_min[2]]),
            (box_min, [box_min[0], box_min[1], box_max[2]]),
            (box_max, [box_min[0], box_max[1], box_max[2]]),
            (box_max, [box_max[0], box_min[1], box_max[2]]),
            (box_max, [box_max[0], box_max[1], box_min[2]]),
            ([box_min[0], box_max[1], box_min[2]], [box_max[0], box_max[1], box_min[2]]),
            ([box_min[0], box_max[1], box_min[2]], [box_min[0], box_max[1], box_max[2]]),
            ([box_min[0], box_min[1], box_max[2]], [box_max[0], box_min[1], box_max[2]]),
            ([box_min[0], box_min[1], box_max[2]], [box_min[0], box_max[1], box_max[2]]),
            ([box_max[0], box_min[1], box_min[2]], [box_max[0], box_max[1], box_min[2]]),
            ([box_max[0], box_min[1], box_min[2]], [box_max[0], box_min[1], box_max[2]])
        ]

        for edge_start, edge_end in edges:
            if self.ray_triangle_intersect(edge_start, np.array(edge_end) - np.array(edge_start), 
                                         v0, v1, v2):
                return True

        return False

    def parallel_triangle_filtering(self, triangles: np.ndarray, box_min: np.ndarray, 
                                  box_max: np.ndarray) -> np.ndarray:
        """Filter triangles in parallel to determine which ones intersect with a bounding box."""
        num_processes = self.cpu_limit
        chunk_size = max(1, len(triangles) // (num_processes * 10))

        with mp.Pool(num_processes) as pool:
            triangle_intersects_box_wrapper = partial(
                lambda triangle, box_min, box_max: self.triangle_intersects_box(
                    triangle[0], triangle[1], triangle[2], box_min, box_max
                ),
                box_min=box_min, box_max=box_max
            )
            
            filtered_indices = list(tqdm(
                pool.imap(triangle_intersects_box_wrapper, triangles, chunksize=chunk_size),
                total=len(triangles),
                desc="Filtering triangles",
                mininterval=0.1,
                smoothing=0.1
            ))

        return np.where(filtered_indices)[0]

    def calculate(self, 
                 light_dir: Optional[np.ndarray] = None,
                 point_of_interest: Optional[np.ndarray] = None, 
                 window_size: Optional[np.ndarray] = None, 
                 sample_size: int = 1000000,
                 verbose: bool = True,
                 # Environmental parameters (with warnings)
                 depth: Optional[float] = None,
                 turbidity: Optional[float] = None, 
                 aspect: Optional[float] = None,
                 slope: Optional[float] = None,
                 time_of_day: Optional[float] = None,
                 day_of_year: Optional[int] = None,
                 latitude: float = 0.0,
                 longitude: float = 0.0,
                 **kwargs) -> Dict[str, Any]:
        """
        Calculate the shading percentage based on the coral structure.

        Parameters:
        light_dir (Optional[np.ndarray]): Direction of the light source. If None and time/date provided, calculated from solar position.
        point_of_interest (Optional[np.ndarray]): Point of interest for localized calculation.
        window_size (Optional[np.ndarray]): Size of the window around the point of interest.
        sample_size (int): Number of points to sample for the calculation.
        verbose (bool): Whether to print progress information.
        depth (Optional[float]): Water depth (WARNING: not implemented - structural only)
        turbidity (Optional[float]): Water turbidity (WARNING: not implemented - structural only)
        aspect (Optional[float]): Seafloor aspect in degrees (0-360, 0=North)
        slope (Optional[float]): Seafloor slope in degrees (0-90)
        time_of_day (Optional[float]): Time of day in hours (0-24)
        day_of_year (Optional[int]): Day of year (1-365)
        latitude (float): Latitude in degrees for solar position calculation
        longitude (float): Longitude in degrees for solar position calculation

        Returns:
        Dict[str, Any]: Dictionary containing calculation results.
        """
        if self.mesh is None:
            raise RuntimeError("No 3D model loaded. Please load a 3D model first.")

        # Validate inputs
        self._validate_sampling_points(sample_size)
        self._warn_unsupported_parameters(depth=depth, turbidity=turbidity, **kwargs)

        # Determine light direction
        if light_dir is None:
            if time_of_day is not None and day_of_year is not None:
                light_dir = self.calculate_solar_position(day_of_year, time_of_day, latitude, longitude)
                if verbose:
                    print(f"Calculated solar position for day {day_of_year}, time {time_of_day}h")
            else:
                light_dir = np.array([0, 0, -1])  # Default: straight down
                if verbose:
                    print("Using default downward light direction")
        
        # Adjust for slope and aspect if provided
        if slope is not None and aspect is not None:
            light_dir = self.adjust_light_for_slope_aspect(light_dir, slope, aspect)
            if verbose:
                print(f"Adjusted light direction for slope={slope}°, aspect={aspect}°")

        if verbose:
            print("Calculating shading percentage based on coral structure...")
            print("⚠️  STRUCTURAL-ONLY ANALYSIS: Results do not include water column effects")
            print("Preparing mesh data...")

        self.mesh.compute_normals(inplace=True)
        points = self.mesh.points
        
        # Extract triangular faces from PyVista format
        triangular_faces = []
        faces_array = self.mesh.faces
        i = 0
        while i < len(faces_array):
            n_vertices = faces_array[i]
            if n_vertices == 3:  # Only triangular faces
                face = faces_array[i+1:i+1+n_vertices]
                triangular_faces.append(face)
            i += n_vertices + 1
        
        if len(triangular_faces) == 0:
            raise ValueError("No triangular faces found in mesh")
        
        faces = np.array(triangular_faces)
        triangles = points[faces]

        if point_of_interest is not None and window_size is not None:
            # Bounding box calculation
            box_min = point_of_interest - window_size / 2
            box_max = point_of_interest + window_size / 2

            if verbose:
                print("Filtering points...")
            mask = np.all((points >= box_min) & (points <= box_max), axis=1)
            window_points = points[mask]

            if len(window_points) > sample_size:
                if verbose:
                    print(f"Sampling {sample_size} points from {len(window_points)} points in the window...")
                sampled_indices = np.random.choice(len(window_points), sample_size, replace=False)
                sampled_points = window_points[sampled_indices]
            else:
                sampled_points = window_points

            if verbose:
                print("Filtering triangles...")
            indices = self.parallel_triangle_filtering(triangles, box_min, box_max)
        else:
            # Full model calculation
            if verbose:
                print("Processing full model...")
            if len(points) > sample_size:
                if verbose:
                    print(f"Sampling {sample_size} points from {len(points)} total points...")
                sampled_indices = np.random.choice(len(points), sample_size, replace=False)
                sampled_points = points[sampled_indices]
            else:
                sampled_points = points
            indices = np.arange(len(triangles))

        if verbose:
            print("Building BVH...")
        bvh_root = self.build_bvh(triangles, indices, 0, len(indices))

        chunk_size = max(1, len(sampled_points) // (self.cpu_limit * 2))
        chunks = [sampled_points[i:i + chunk_size] for i in range(0, len(sampled_points), chunk_size)]

        if verbose:
            print(f"Using {self.cpu_limit} CPU cores ({self.cpu_percentage}% of {mp.cpu_count()}) to process {len(chunks)} chunks...")

        with mp.Pool(self.cpu_limit) as pool:
            if verbose:
                results = list(tqdm(
                    pool.imap(self.process_chunk, [
                        (chunk, bvh_root, triangles, indices, light_dir) for chunk in chunks]),
                    total=len(chunks),
                    desc="Processing chunks",
                    mininterval=0.1,
                    smoothing=0.1
                ))
                print("All chunks processed. Calculating final result...")
            else:
                results = pool.map(self.process_chunk, [
                    (chunk, bvh_root, triangles, indices, light_dir) for chunk in chunks])
        
        shadowed = np.concatenate(results)
        shaded_percentage = np.mean(shadowed) * 100

        result = {
            'mesh_file': self.mesh_file,
            'shaded_percentage': shaded_percentage,
            'illuminated_percentage': 100 - shaded_percentage,
            'sample_points': len(sampled_points),
            'cpu_cores_used': self.cpu_limit,
            'parameters': {
                'light_direction': light_dir.tolist() if isinstance(light_dir, np.ndarray) else light_dir,
                'depth': depth,
                'turbidity': turbidity,
                'aspect': aspect,
                'slope': slope,
                'time_of_day': time_of_day,
                'day_of_year': day_of_year,
                'latitude': latitude,
                'longitude': longitude
            }
        }

        if verbose:
            print(f"Shading calculation complete: {shaded_percentage:.2f}% shaded")

        return result

    def process_directory(self, directory: str, csv_file: Optional[str] = None, **kwargs) -> list:
        """
        Process all 3D models in the specified directory.

        Parameters:
        directory (str): Path to the directory containing 3D models.
        csv_file (Optional[str]): Path to save results as CSV.
        **kwargs: Parameters to pass to calculate method.

        Returns:
        list: List of dictionaries containing results for each mesh.
        """
        if not os.path.exists(directory):
            raise FileNotFoundError(f"Directory not found: {directory}")

        mesh_files = [f for f in os.listdir(directory) 
                     if f.lower().endswith(('.obj', '.ply', '.stl', '.vtk'))]

        if not mesh_files:
            raise ValueError(f"No supported mesh files found in {directory}")

        results = []
        for mesh_file in tqdm(mesh_files, desc="Processing 3D models"):
            try:
                self.load_mesh(os.path.join(directory, mesh_file), verbose=False)
                result = self.calculate(verbose=False, **kwargs)
                results.append(result)
            except Exception as e:
                print(f"Error processing {mesh_file}: {e}")
                continue

        if csv_file:
            self._save_results_to_csv(results, csv_file)

        return results

    def _save_results_to_csv(self, results: list, csv_file: str) -> None:
        """Save results to CSV file."""
        import csv
        
        with open(csv_file, 'w', newline='') as f:
            if not results:
                return
                
            fieldnames = ['mesh_file', 'shaded_percentage', 'illuminated_percentage', 
                         'sample_points', 'cpu_cores_used']
            
            # Add parameter fields
            if results[0]['parameters']:
                param_fields = [f"param_{k}" for k in results[0]['parameters'].keys()]
                fieldnames.extend(param_fields)
            
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            
            for result in results:
                row = {
                    'mesh_file': result['mesh_file'],
                    'shaded_percentage': result['shaded_percentage'],
                    'illuminated_percentage': result['illuminated_percentage'],
                    'sample_points': result['sample_points'],
                    'cpu_cores_used': result['cpu_cores_used']
                }
                
                # Add parameters
                for k, v in result['parameters'].items():
                    row[f"param_{k}"] = v
                
                writer.writerow(row)
        
        print(f"Results saved to: {csv_file}")
