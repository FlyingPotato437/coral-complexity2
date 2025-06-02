#!/usr/bin/env python3
"""
Temporary test script for coral-complexity-metrics using a real PLY mesh.
Evaluates enhanced shading, metric calculation, mesh validation, and visualization components.
This script is intended to be run from the parent directory of the 'src' and 'coral-complexity-metrics' dirs.
"""

import numpy as np
import pyvista as pv
import os
import tempfile
import sys
from pathlib import Path
import traceback

# Adjust path to find the 'src' directory, assuming this script is inside 'coral-complexity-metrics'
# and 'src' is a sibling to 'coral-complexity-metrics' directory, or within it.
# Path(__file__).resolve() gives the path to this script.
# .parent gives the directory of this script (e.g., .../coral-complexity-metrics)
# .parent again gives the parent of that (e.g., .../AIMS -- the workspace root)
# Then append 'src'

# If src is a subdirectory of where this script is (e.g. coral-complexity-metrics/src)
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))


try:
    import coral_complexity_metrics as ccm
except ImportError as e:
    print(f"❌ CRITICAL: Could not import coral_complexity_metrics. Error: {e}")
    print(f"Python sys.path: {sys.path}")
    print(f"Current working directory: {os.getcwd()}")
    print("Ensure the package is installed or the src path is correctly pointing to the package location.")
    sys.exit(1)

# Helper function to extract triangular faces from a PyVista mesh
def extract_triangular_faces_from_pv(mesh_pv):
    """Extracts only the triangular faces from a PyVista mesh object."""
    raw_faces = mesh_pv.faces
    tri_faces_list = []
    i = 0
    while i < len(raw_faces):
        n_vertices_in_face = raw_faces[i]
        if n_vertices_in_face == 3: # Check if the face is a triangle
            tri_faces_list.append(raw_faces[i+1 : i+1+n_vertices_in_face])
        i += (n_vertices_in_face + 1)
    
    if not tri_faces_list:
        return np.array([]).reshape(0, 3) # Return empty array if no triangles
    return np.array(tri_faces_list)

def main():
    # This path needs to be accessible from where the script runs.
    # It's an absolute path, so it should be fine.
    mesh_path = "/Users/srikanthsamy1/Desktop/AIMS/3D_Models/Flat/TSMA_BA1D_P3_202203.ply"
    print(f"🧪 Testing new features with real mesh: {mesh_path}")
    print("=" * 70)

    if not os.path.exists(mesh_path):
        print(f"❌ ERROR: Mesh file not found at {mesh_path}")
        return

    # --- 1. Load Mesh (once for all tests) ---
    print("\nSTEP 1: Loading mesh...")
    pv_mesh = None
    try:
        pv_mesh = pv.read(mesh_path)
        print(f"  ✓ Mesh loaded successfully: {pv_mesh.n_points} points, {pv_mesh.n_cells} cells")
    except Exception as e:
        print(f"  ❌ FAILED to load mesh: {e}")
        traceback.print_exc()
        return # Stop if mesh cannot be loaded
    print("-" * 70)

    # --- 2. Enhanced Shading Module ---
    print("\nSTEP 2: Testing Enhanced Shading Module...")
    if hasattr(ccm, 'Shading'):
        try:
            shading_calculator = ccm.Shading(cpu_percentage=50) 
            print(f"  Initializing Shading with {shading_calculator.cpu_limit} CPU cores (50% of available).")
            shading_calculator.load_mesh(mesh_path, verbose=True)
            
            shading_params = {
                'sample_size': 10000,  # Reduced sample size for this large mesh test
                'time_of_day': 10.0, 'day_of_year': 150,
                'latitude': -18.0, 'longitude': 147.0,
                'slope': 5.0, 'aspect': 180.0, 'verbose': True
            }
            print(f"  Calculating shading with params: {shading_params}")
            result = shading_calculator.calculate(**shading_params)
            
            if 'shaded_percentage' in result:
                print(f"  ✓ Shading calculation result: {result['shaded_percentage']:.2f}% shaded")
                print(f"  Processed {result.get('sample_points', 'N/A')} sample points using {result.get('cpu_cores_used', 'N/A')} cores.")
                print("  ✅ Enhanced Shading Module test: PASSED (basic execution)")
            else:
                print(f"  ❌ Shading result dictionary missing 'shaded_percentage'. Keys: {list(result.keys())}")
                print("  ❌ Enhanced Shading Module test: FAILED (output format error)")

        except Exception as e:
            print(f"  ❌ Enhanced Shading Module test: FAILED - {e}")
            traceback.print_exc()
    else:
        print("  ⚠️ Shading module (ccm.Shading) not found.")
    print("-" * 70)

    # --- 3. Metric Calculation (All Metrics) ---
    print("\nSTEP 3: Testing Metric Calculation System (calculate_all_metrics)...")
    if hasattr(ccm, 'calculate_all_metrics') and pv_mesh is not None:
        try:
            points = pv_mesh.points
            tri_faces_np = extract_triangular_faces_from_pv(pv_mesh) 
            
            if tri_faces_np.ndim != 2 or tri_faces_np.shape[1] != 3:
                if tri_faces_np.size == 0 :
                    print("  ℹ️ Mesh has no triangular faces. Some metrics may yield NaN or default values.")
                else: 
                    print(f"  ⚠️ Extracted triangular faces have unexpected shape: {tri_faces_np.shape}. Metrics might fail.")
            
            min_x, max_x = points[:,0].min(), points[:,0].max()
            min_y, max_y = points[:,1].min(), points[:,1].max()
            projected_area_bbox = (max_x - min_x) * (max_y - min_y)

            mesh_data = {
                'points': points,
                'faces': tri_faces_np, 
                'mesh': pv_mesh,
                'surface_area': pv_mesh.area,
                'projected_area': projected_area_bbox,
                'volume': pv_mesh.volume if pv_mesh.is_closed else 0.0
            }
            
            print(f"  Mesh data prepared: {mesh_data['points'].shape[0]} points, {mesh_data['faces'].shape[0]} triangular faces.")
            print(f"  Surface Area (PyVista): {mesh_data['surface_area']:.2f}, Projected BBox Area: {mesh_data['projected_area']:.2f}, Volume (PyVista if closed): {mesh_data['volume']:.2f}")
            print("  Calculating all metrics (this may take a while)...")
            
            all_metrics_results = ccm.calculate_all_metrics(mesh_data, check_mesh_closure=True)
            print(f"  ✓ Calculated {len(all_metrics_results)} metric categories:")
            for name, res_dict in all_metrics_results.items():
                if isinstance(res_dict, dict):
                    values_str = ", ".join([f"{k}: {v:.3f}" if isinstance(v, float) else f"{k}: {v}" for k,v in res_dict.items()])
                    print(f"    - {name}: {values_str}")
                else:
                     print(f"    - {name}: Unexpected_Format({type(res_dict)}) - {res_dict}")
            print("  ✅ Metric Calculation System test: PASSED (basic execution)")
        except Exception as e:
            print(f"  ❌ Metric Calculation System test: FAILED - {e}")
            traceback.print_exc()
    elif pv_mesh is None:
        print("  ⚠️ Metric Calculation skipped: Mesh failed to load.")
    else:
        print("  ⚠️ calculate_all_metrics not found in ccm.")
    print("-" * 70)

    # --- 4. Mesh Validation ---
    print("\nSTEP 4: Testing Mesh Validation...")
    if hasattr(ccm, 'HAS_MESH_VALIDATION') and ccm.HAS_MESH_VALIDATION and hasattr(ccm, 'MeshValidator') and pv_mesh is not None:
        try:
            validator = ccm.MeshValidator(verbose=True)
            print("  Validating mesh (repair_if_needed=False)...")
            validation_result = validator.validate_mesh(pv_mesh, repair_if_needed=False)
            
            print(f"  ✓ Validation complete.")
            print(f"    Is Valid: {validation_result.is_valid}")
            print(f"    Is Closed: {validation_result.is_closed}")
            print(f"    Open Edges: {validation_result.n_open_edges}")
            print(f"    Holes: {validation_result.n_holes}")
            print(f"    Non-manifold Edges: {validation_result.n_non_manifold_edges}")
            print(f"    Duplicated Vertices: {validation_result.n_duplicated_vertices}")
            print(f"    Volume (from validator): {validation_result.volume:.2f}")
            print(f"    Surface Area (from validator): {validation_result.surface_area:.2f}")
            if validation_result.validation_errors:
                print("    Validation Errors reported:")
                for err in validation_result.validation_errors:
                    print(f"      - {err}")
            if validation_result.repair_suggestions:
                print("    Repair Suggestions:")
                for sug in validation_result.repair_suggestions:
                    print(f"      - {sug}")
            print("  ✅ Mesh Validation test: PASSED (basic execution)")
        except Exception as e:
            print(f"  ❌ Mesh Validation test: FAILED - {e}")
            traceback.print_exc()
    elif pv_mesh is None:
        print("  ⚠️ Mesh Validation skipped: Mesh failed to load.")
    else:
        print("  ⚠️ Mesh Validation module or MeshValidator not available/found.")
    print("-" * 70)

    # --- 5. Mesh Visualization (Preview availability) ---
    print("\nSTEP 5: Testing Mesh Visualization (Preview components)...")
    if hasattr(ccm, 'HAS_VISUALIZATION') and ccm.HAS_VISUALIZATION and hasattr(ccm, 'MeshVisualizer'):
        try:
            visualizer = ccm.MeshVisualizer()
            print("  ✓ MeshVisualizer initialized successfully.")
            print("  Note: Full preview generation for large meshes is resource-intensive.")
            print("  ✅ Mesh Visualization components test: PASSED (availability checked)")
        except Exception as e:
            print(f"  ❌ Mesh Visualization components test: FAILED - {e}")
            traceback.print_exc()
    else:
        print("  ⚠️ Visualization module or MeshVisualizer not available/found.")
    print("-" * 70)
    
    print("\n🏁 Real mesh testing script finished.")

if __name__ == "__main__":
    main() 