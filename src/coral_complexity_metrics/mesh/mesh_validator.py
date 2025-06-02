"""
Mesh validation and repair module for coral complexity analysis.

This module provides comprehensive mesh validation, repair, and quality assessment
tools, including integration with shapefile-based mesh cropping.
"""

import numpy as np
import pyvista as pv
import pymeshlab
import warnings
from typing import Dict, Any, List, Tuple, Optional, Union
from dataclasses import dataclass
from pathlib import Path
import geopandas as gpd
from shapely.geometry import Point, Polygon
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class MeshValidationResult:
    """Result of mesh validation."""
    is_valid: bool
    is_closed: bool
    n_open_edges: int
    n_holes: int
    n_isolated_vertices: int
    n_duplicated_vertices: int
    n_non_manifold_edges: int
    n_non_manifold_vertices: int
    genus: int
    euler_characteristic: int
    volume: float
    surface_area: float
    bbox_volume: float
    validation_errors: List[str]
    repair_suggestions: List[str]


class MeshValidator:
    """Comprehensive mesh validation and repair toolkit."""
    
    def __init__(self, verbose: bool = True):
        """
        Initialize the mesh validator.
        
        Parameters:
        verbose: Whether to print validation progress
        """
        self.verbose = verbose
        self.ms = None  # MeshLab mesh set
        
    def validate_mesh(self, mesh: Union[pv.PolyData, str, Path], 
                     repair_if_needed: bool = True) -> MeshValidationResult:
        """
        Comprehensive mesh validation with optional automatic repair.
        
        Parameters:
        mesh: PyVista mesh or path to mesh file
        repair_if_needed: Whether to attempt automatic repairs
        
        Returns:
        MeshValidationResult: Detailed validation results
        """
        if isinstance(mesh, (str, Path)):
            mesh = pv.read(str(mesh))
        
        if self.verbose:
            logger.info("Starting mesh validation...")
        
        # Initialize MeshLab for advanced operations
        self.ms = pymeshlab.MeshSet()
        self._pyvista_to_meshlab(mesh)
        
        # Perform validation checks
        errors = []
        suggestions = []
        
        # Basic topology checks
        n_points = mesh.n_points
        n_cells = mesh.n_cells
        
        # Check for empty mesh
        if n_points == 0 or n_cells == 0:
            errors.append("Mesh is empty")
            return MeshValidationResult(
                is_valid=False, is_closed=False, n_open_edges=-1, n_holes=-1,
                n_isolated_vertices=-1, n_duplicated_vertices=-1,
                n_non_manifold_edges=-1, n_non_manifold_vertices=-1,
                genus=-1, euler_characteristic=-1, volume=0, surface_area=0,
                bbox_volume=0, validation_errors=errors, repair_suggestions=suggestions
            )
        
        # Advanced topology analysis using MeshLab
        topology_info = self._analyze_topology()
        
        # Check mesh closure
        is_closed = topology_info['n_open_edges'] == 0
        
        # Volume calculation (only meaningful for closed meshes)
        volume = 0
        try:
            if is_closed:
                volume = mesh.volume
        except:
            errors.append("Volume calculation failed")
        
        # Surface area
        surface_area = mesh.area
        
        # Bounding box volume
        bounds = mesh.bounds
        bbox_volume = ((bounds[1] - bounds[0]) * 
                      (bounds[3] - bounds[2]) * 
                      (bounds[5] - bounds[4]))
        
        # Validation assessment
        is_valid = (len(errors) == 0 and 
                   topology_info['n_non_manifold_edges'] == 0 and
                   topology_info['n_non_manifold_vertices'] == 0 and
                   topology_info['n_duplicated_vertices'] == 0)
        
        # Generate repair suggestions
        if topology_info['n_open_edges'] > 0:
            suggestions.append(f"Close {topology_info['n_holes']} holes")
        if topology_info['n_duplicated_vertices'] > 0:
            suggestions.append("Remove duplicated vertices")
        if topology_info['n_non_manifold_edges'] > 0:
            suggestions.append("Fix non-manifold edges")
        if topology_info['n_isolated_vertices'] > 0:
            suggestions.append("Remove isolated vertices")
        
        result = MeshValidationResult(
            is_valid=is_valid,
            is_closed=is_closed,
            n_open_edges=topology_info['n_open_edges'],
            n_holes=topology_info['n_holes'],
            n_isolated_vertices=topology_info['n_isolated_vertices'],
            n_duplicated_vertices=topology_info['n_duplicated_vertices'],
            n_non_manifold_edges=topology_info['n_non_manifold_edges'],
            n_non_manifold_vertices=topology_info['n_non_manifold_vertices'],
            genus=topology_info['genus'],
            euler_characteristic=topology_info['euler_characteristic'],
            volume=volume,
            surface_area=surface_area,
            bbox_volume=bbox_volume,
            validation_errors=errors,
            repair_suggestions=suggestions
        )
        
        # Attempt repairs if requested
        if repair_if_needed and not is_valid:
            repaired_mesh = self.repair_mesh(mesh, result)
            if repaired_mesh is not None:
                if self.verbose:
                    logger.info("Mesh repair completed, re-validating...")
                # Re-validate repaired mesh
                return self.validate_mesh(repaired_mesh, repair_if_needed=False)
        
        if self.verbose:
            self._print_validation_summary(result)
        
        return result
    
    def repair_mesh(self, mesh: pv.PolyData, 
                   validation_result: Optional[MeshValidationResult] = None) -> Optional[pv.PolyData]:
        """
        Attempt to repair mesh issues automatically.
        
        Parameters:
        mesh: Input mesh to repair
        validation_result: Previous validation result (optional)
        
        Returns:
        Repaired mesh or None if repair failed
        """
        try:
            if self.verbose:
                logger.info("Starting mesh repair...")
            
            # Ensure we have a MeshLab mesh set
            if self.ms is None:
                self.ms = pymeshlab.MeshSet()
                self._pyvista_to_meshlab(mesh)
            
            # Step 1: Remove duplicated vertices
            if self.verbose:
                logger.info("Removing duplicated vertices...")
            self.ms.meshing_remove_duplicate_vertices()
            
            # Step 2: Remove isolated vertices
            if self.verbose:
                logger.info("Removing isolated vertices...")
            self.ms.meshing_remove_unreferenced_vertices()
            
            # Step 3: Fix non-manifold edges
            if self.verbose:
                logger.info("Fixing non-manifold edges...")
            self.ms.meshing_repair_non_manifold_edges()
            
            # Step 4: Remove small isolated components
            if self.verbose:
                logger.info("Removing small components...")
            self.ms.meshing_remove_connected_component_by_diameter(
                mincomponentdiag=pymeshlab.PercentageValue(1.0),
                removeunref=True
            )
            
            # Step 5: Close holes
            if self.verbose:
                logger.info("Closing holes...")
            try:
                self.ms.meshing_close_holes(
                    maxholesize=1000,
                    selected=False,
                    newfaceselected=True,
                    selfintersection=False
                )
            except Exception as e:
                if self.verbose:
                    logger.warning(f"Hole closing failed: {e}")
            
            # Step 6: Final cleanup
            if self.verbose:
                logger.info("Final cleanup...")
            self.ms.meshing_remove_duplicate_vertices()
            self.ms.meshing_remove_unreferenced_vertices()
            
            # Convert back to PyVista
            repaired_mesh = self._meshlab_to_pyvista()
            
            if self.verbose:
                logger.info("Mesh repair completed successfully")
            
            return repaired_mesh
            
        except Exception as e:
            if self.verbose:
                logger.error(f"Mesh repair failed: {e}")
            return None
    
    def crop_mesh_with_shapefile(self, mesh: pv.PolyData, 
                                shapefile_path: str,
                                polygon_id_field: str = 'ID',
                                expansion_percentage: float = 0.0) -> Dict[str, pv.PolyData]:
        """
        Crop mesh using polygons from a shapefile.
        
        Parameters:
        mesh: Input mesh to crop
        shapefile_path: Path to shapefile containing crop polygons
        polygon_id_field: Field name for polygon IDs
        expansion_percentage: Percentage to expand bounding boxes
        
        Returns:
        Dictionary mapping polygon IDs to cropped meshes
        """
        if self.verbose:
            logger.info(f"Loading shapefile: {shapefile_path}")
        
        # Load shapefile
        gdf = gpd.read_file(shapefile_path)
        
        # Project mesh points to 2D for spatial queries
        mesh_points_2d = mesh.points[:, :2]  # Use X,Y coordinates
        
        cropped_meshes = {}
        
        for idx, row in gdf.iterrows():
            polygon_id = row[polygon_id_field]
            geometry = row['geometry']
            
            if not isinstance(geometry, Polygon):
                continue
            
            if self.verbose:
                logger.info(f"Processing polygon {polygon_id}")
            
            # Find points inside the polygon
            points_inside = []
            for i, point_2d in enumerate(mesh_points_2d):
                if Point(point_2d).within(geometry):
                    points_inside.append(i)
            
            if len(points_inside) == 0:
                if self.verbose:
                    logger.warning(f"No points found inside polygon {polygon_id}")
                continue
            
            # Create bounding box for cropping
            bounds_2d = geometry.bounds
            
            if expansion_percentage > 0:
                # Expand bounding box
                width = bounds_2d[2] - bounds_2d[0]
                height = bounds_2d[3] - bounds_2d[1]
                expansion_x = width * expansion_percentage / 100
                expansion_y = height * expansion_percentage / 100
                
                bounds_2d = (
                    bounds_2d[0] - expansion_x,
                    bounds_2d[1] - expansion_y,
                    bounds_2d[2] + expansion_x,
                    bounds_2d[3] + expansion_y
                )
            
            # Get Z bounds from the mesh
            z_min = mesh.bounds[4]
            z_max = mesh.bounds[5]
            
            # Create 3D bounds
            bounds_3d = [
                bounds_2d[0], bounds_2d[2],  # X min, max
                bounds_2d[1], bounds_2d[3],  # Y min, max
                z_min, z_max                 # Z min, max
            ]
            
            # Crop mesh using bounding box
            try:
                cropped_mesh = mesh.clip_box(bounds_3d)
                
                if cropped_mesh.n_points > 0:
                    # Validate and optionally repair the cropped mesh
                    validation_result = self.validate_mesh(cropped_mesh, repair_if_needed=True)
                    
                    if validation_result.is_valid or validation_result.is_closed:
                        cropped_meshes[str(polygon_id)] = cropped_mesh
                    else:
                        if self.verbose:
                            logger.warning(f"Cropped mesh for polygon {polygon_id} failed validation")
                else:
                    if self.verbose:
                        logger.warning(f"Empty cropped mesh for polygon {polygon_id}")
                        
            except Exception as e:
                if self.verbose:
                    logger.error(f"Error cropping mesh for polygon {polygon_id}: {e}")
        
        if self.verbose:
            logger.info(f"Successfully cropped {len(cropped_meshes)} mesh regions")
        
        return cropped_meshes
    
    def _analyze_topology(self) -> Dict[str, Any]:
        """Analyze mesh topology using MeshLab filters."""
        # self.ms is the pymeshlab.MeshSet
        if self.ms is None or self.ms.number_meshes() == 0:
            logger.warning("No mesh in MeshSet to analyze topology.")
            return {
                'n_open_edges': -1, 'n_holes': -1, 'n_isolated_vertices': -1,
                'n_duplicated_vertices': -1, 'n_non_manifold_edges': -1,
                'n_non_manifold_vertices': -1, 'genus': -1, 'euler_characteristic': -1,
                'mesh_volume': 0.0, 'surface_area': 0.0, 'bbox_diagonal': 0.0
            }

        m = self.ms.current_mesh() # Get the current pymeshlab.Mesh object

        results = {
            'n_open_edges': -1, 'n_holes': -1, 'n_isolated_vertices': -1,
            'n_duplicated_vertices': -1, 'n_non_manifold_edges': -1,
            'n_non_manifold_vertices': -1, 'genus': -1, 'euler_characteristic': -1,
            'mesh_volume': 0.0, 'surface_area': 0.0, 'bbox_diagonal': 0.0
        }

        try:
            # --- Basic Geometric Measures ---
            # This filter computes various geometric and topological measures
            # The results are typically stored as custom properties on the mesh or returned by the filter
            # PyMeshLab's API for retrieving these can be tricky and version-dependent.
            # We will attempt to get some common ones.
            
            # Try to get volume and area directly if available (often after some computations)
            # For PyMeshLab, you often compute measures and they might be stored or directly accessible.
            # Let's ensure we have some values.
            # PyVista mesh object is 'mesh' in validate_mesh, let's use that for direct area/volume.
            # This part of _analyze_topology is primarily for MeshLab specific details.

            # --- Topological Properties using specific filters ---
            
            # Non-manifold edges
            try:
                # Some versions of pymeshlab might use slightly different filter names or parameters
                # This is a common way to count non-manifold edges
                self.ms.apply_filter('select_non_manifold_edges_by_number_of_faces', onselected=False)
                results['n_non_manifold_edges'] = m.selected_edges_number()
                self.ms.apply_filter('set_selection_none', applytoface=False, applytoedge=True, applytovertex=False) # Clear selection
            except Exception as e_nm_edge:
                logger.debug(f"Could not get non-manifold edge count via select_non_manifold_edges_by_number_of_faces: {e_nm_edge}")
                # Fallback or older method might be different
                # results['n_non_manifold_edges'] will remain -1 if this fails

            # Non-manifold vertices
            try:
                self.ms.apply_filter('select_non_manifold_vertices', onselected=False)
                results['n_non_manifold_vertices'] = m.selected_vertices_number()
                self.ms.apply_filter('set_selection_none', applytoface=False, applytoedge=False, applytovertex=True)
            except Exception as e_nm_vert:
                logger.debug(f"Could not get non-manifold vertex count: {e_nm_vert}")

            # Boundary Edges (Open Edges)
            try:
                self.ms.apply_filter('select_boundary_edges', onselected=False) # Name can vary, e.g., 'select_border_edges'
                results['n_open_edges'] = m.selected_edges_number()
                self.ms.apply_filter('set_selection_none', applytoface=False, applytoedge=True, applytovertex=False)
            except Exception as e_bound:
                logger.debug(f"Could not get boundary edge count: {e_bound}")
            
            # Number of connected components (can indicate isolated pieces, but not directly isolated vertices)
            # Number of holes (can be estimated)
            # Pymeshlab's hole counting can be part of meshing_close_holes or specific diagnostics
            # For now, we'll rely on n_open_edges. If > 0, likely has holes.
            if results['n_open_edges'] > 0:
                results['n_holes'] = 1 # Crude: if open edges exist, assume at least one hole.
                                       # More accurate hole counting is complex.
            elif results['n_open_edges'] == 0:
                results['n_holes'] = 0


            # Duplicated Vertices - this is usually done as a cleaning step.
            # `remove_duplicate_vertices` can report, or one can check before/after.
            # For now, this is harder to get as a direct count without altering the mesh.
            # The repair step handles this.

            # Isolated (unreferenced) vertices
            try:
                # This counts vertices that are not part of any face.
                self.ms.apply_filter('select_unreferenced_vertices')
                results['n_isolated_vertices'] = m.selected_vertices_number()
                self.ms.apply_filter('set_selection_none', applytoface=False, applytoedge=False, applytovertex=True)
            except Exception as e_iso_v:
                logger.debug(f"Could not get isolated vertex count: {e_iso_v}")
                
            # Genus and Euler Characteristic are more complex topological invariants.
            # PyMeshLab can compute these, often via "Topology Computations" or similar filters.
            # Example: ms.apply_filter('compute_topological_measures')
            # And then retrieve them. This might be version dependent.
            # For now, we'll leave them as -1 if not easily available or if it requires complex filter chains.

            logger.info(f"Pymeshlab topology analysis (partial): {results}")
            return results

        except Exception as e:
            logger.error(f"Core topology analysis with PyMeshLab failed: {e}", exc_info=True)
            # Fallback to default error values
            return {
                'n_open_edges': -1, 'n_holes': -1, 'n_isolated_vertices': -1,
                'n_duplicated_vertices': -1, 'n_non_manifold_edges': -1,
                'n_non_manifold_vertices': -1, 'genus': -1, 'euler_characteristic': -1,
                'mesh_volume': 0.0, 'surface_area': 0.0, 'bbox_diagonal': 0.0
            }
    
    def _pyvista_to_meshlab(self, mesh: pv.PolyData) -> None:
        """Convert PyVista mesh to MeshLab format."""
        vertices = mesh.points
        
        # Extract faces (assuming triangular faces)
        faces = []
        for i in range(mesh.n_cells):
            cell = mesh.get_cell(i)
            if cell.type == pv.CellType.TRIANGLE:
                faces.append(cell.point_ids)
        
        faces = np.array(faces)
        
        # Clear existing meshes and add new one
        self.ms.clear()
        self.ms.add_mesh(pymeshlab.Mesh(vertices, faces))
    
    def _meshlab_to_pyvista(self) -> pv.PolyData:
        """Convert MeshLab mesh back to PyVista format."""
        mesh = self.ms.current_mesh()
        vertices = mesh.vertex_matrix()
        faces = mesh.face_matrix()
        
        # Create PyVista mesh
        pv_faces = []
        for face in faces:
            pv_faces.extend([3, face[0], face[1], face[2]])
        
        return pv.PolyData(vertices, pv_faces)
    
    def _print_validation_summary(self, result: MeshValidationResult) -> None:
        """Print validation summary."""
        print("\n" + "="*60)
        print("MESH VALIDATION SUMMARY")
        print("="*60)
        print(f"Overall Status: {'✓ VALID' if result.is_valid else '✗ INVALID'}")
        print(f"Mesh Closure: {'✓ CLOSED' if result.is_closed else '✗ OPEN'}")
        
        if result.volume > 0:
            print(f"Volume: {result.volume:.6f}")
        print(f"Surface Area: {result.surface_area:.6f}")
        
        print("\nTopology:")
        print(f"  Open Edges: {result.n_open_edges}")
        print(f"  Holes: {result.n_holes}")
        print(f"  Non-manifold Edges: {result.n_non_manifold_edges}")
        print(f"  Non-manifold Vertices: {result.n_non_manifold_vertices}")
        print(f"  Isolated Vertices: {result.n_isolated_vertices}")
        print(f"  Duplicated Vertices: {result.n_duplicated_vertices}")
        
        if result.validation_errors:
            print("\nErrors:")
            for error in result.validation_errors:
                print(f"  ✗ {error}")
        
        if result.repair_suggestions:
            print("\nRepair Suggestions:")
            for suggestion in result.repair_suggestions:
                print(f"  → {suggestion}")
        
        print("="*60)


def validate_and_repair_mesh(mesh_path: str, 
                           output_path: Optional[str] = None,
                           repair: bool = True) -> MeshValidationResult:
    """
    Convenience function to validate and optionally repair a mesh.
    
    Parameters:
    mesh_path: Path to input mesh file
    output_path: Path to save repaired mesh (optional)
    repair: Whether to attempt repairs
    
    Returns:
    MeshValidationResult: Validation results
    """
    validator = MeshValidator(verbose=True)
    mesh = pv.read(mesh_path)
    
    result = validator.validate_mesh(mesh, repair_if_needed=repair)
    
    if repair and output_path and result.is_valid:
        # Save repaired mesh if validation passed
        repaired_mesh = validator._meshlab_to_pyvista()
        repaired_mesh.save(output_path)
        logger.info(f"Repaired mesh saved to: {output_path}")
    
    return result


def batch_validate_meshes(mesh_directory: str, 
                         shapefile_path: Optional[str] = None,
                         output_directory: Optional[str] = None) -> Dict[str, MeshValidationResult]:
    """
    Batch validate multiple mesh files.
    
    Parameters:
    mesh_directory: Directory containing mesh files
    shapefile_path: Optional shapefile for cropping
    output_directory: Directory to save results
    
    Returns:
    Dictionary mapping filenames to validation results
    """
    mesh_dir = Path(mesh_directory)
    results = {}
    
    validator = MeshValidator(verbose=True)
    
    # Find all mesh files
    mesh_extensions = ['.obj', '.ply', '.stl', '.vtk', '.vtp']
    mesh_files = []
    for ext in mesh_extensions:
        mesh_files.extend(mesh_dir.glob(f"*{ext}"))
    
    logger.info(f"Found {len(mesh_files)} mesh files to validate")
    
    for mesh_file in mesh_files:
        logger.info(f"Validating: {mesh_file.name}")
        
        try:
            mesh = pv.read(str(mesh_file))
            result = validator.validate_mesh(mesh, repair_if_needed=True)
            results[mesh_file.name] = result
            
            # Optionally crop with shapefile
            if shapefile_path and result.is_valid:
                cropped_meshes = validator.crop_mesh_with_shapefile(
                    mesh, shapefile_path
                )
                
                if output_directory and cropped_meshes:
                    output_dir = Path(output_directory)
                    output_dir.mkdir(exist_ok=True)
                    
                    for polygon_id, cropped_mesh in cropped_meshes.items():
                        output_file = output_dir / f"{mesh_file.stem}_{polygon_id}.ply"
                        cropped_mesh.save(str(output_file))
                        logger.info(f"Saved cropped mesh: {output_file}")
        
        except Exception as e:
            logger.error(f"Error processing {mesh_file.name}: {e}")
            results[mesh_file.name] = MeshValidationResult(
                is_valid=False, is_closed=False, n_open_edges=-1, n_holes=-1,
                n_isolated_vertices=-1, n_duplicated_vertices=-1,
                n_non_manifold_edges=-1, n_non_manifold_vertices=-1,
                genus=-1, euler_characteristic=-1, volume=0, surface_area=0,
                bbox_volume=0, validation_errors=[str(e)], repair_suggestions=[]
            )
    
    return results 