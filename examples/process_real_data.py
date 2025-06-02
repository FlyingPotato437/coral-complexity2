from pathlib import Path
import pyvista as pv
import geopandas as gpd
import numpy as np
import logging

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

try:
    from coral_complexity_metrics.mesh.mesh_utils import clip_mesh_by_polygon
    from coral_complexity_metrics.mesh._metric import Rugosity
    logger.info("Successfully imported library components.")
except ImportError as e:
    logger.error(f"Error importing library components: {e}")
    logger.error("Please ensure 'pip install -e .' has been run in the 'coral-complexity-metrics' directory and a an __init__.py file exists in the mesh folder")
    exit()

def main():
    # --- Configuration ---
    # IMPORTANT: Update these paths to your actual files!
    # Assumes this script is run from the root of the coral-complexity-metrics repository.
    # Adjust relative paths if running from elsewhere.
    project_root = Path(__file__).resolve().parent.parent # Assuming examples/ is one level down from repo root
    
    # Path to your large PLY file
    # ply_file_path = project_root.parent / "testply" / "TSMA_BA1S_P4_202203.ply" # Example for running from examples dir
    ply_file_path = Path("/Users/srikanthsamy1/Desktop/AIMS/testply/TSMA_BA1S_P4_202203.ply")

    # Path to your shapefile
    # shapefile_path = project_root.parent / "testshp" / "TSMA_BA1S_P4_202203.shp"
    shapefile_path = Path("/Users/srikanthsamy1/Desktop/AIMS/testshp/TSMA_BA1S_P4_202203.shp")

    # Field in the shapefile that contains the unique ID for each polygon
    polygon_id_field = "TL_id" # Corrected from "ID"
    
    # Optional: Expansion percentage for the bounding box of each polygon before clipping
    expansion_percentage = 10.0 # e.g., 10% expansion
    # --- End Configuration ---

    logger.info(f"Processing PLY file: {ply_file_path}")
    logger.info(f"Using shapefile: {shapefile_path}")

    if not ply_file_path.exists():
        logger.error(f"PLY file not found: {ply_file_path}")
        return
    if not shapefile_path.exists():
        logger.error(f"Shapefile not found: {shapefile_path}")
        return

    # 1. Load the main mesh
    try:
        logger.info("Loading main mesh...")
        main_mesh = pv.read(ply_file_path)
        logger.info(f"Main mesh loaded: {main_mesh.n_points} points, {main_mesh.n_cells} cells")
    except Exception as e:
        logger.error(f"Failed to load PLY file: {e}")
        return

    # 2. Load the shapefile
    try:
        logger.info("Loading shapefile...")
        gdf = gpd.read_file(shapefile_path)
        logger.info(f"Shapefile loaded with {len(gdf)} polygons.")
        if polygon_id_field not in gdf.columns:
            logger.error(f"Polygon ID field '{polygon_id_field}' not found in shapefile. Available fields: {gdf.columns.tolist()}")
            return
    except Exception as e:
        logger.error(f"Failed to load shapefile: {e}")
        return

    # Instantiate the Rugosity metric
    rugosity_metric = Rugosity()
    all_results = []

    # 3. Iterate through each polygon in the shapefile
    logger.info("Starting polygon processing...")
    for index, row in gdf.iterrows():
        polygon_geom = row['geometry']
        poly_id = row[polygon_id_field]
        logger.info(f"--- Processing Polygon ID: {poly_id} ---")

        # 4. Clip the mesh by the current polygon
        clip_data = clip_mesh_by_polygon(
            main_mesh=main_mesh,
            polygon_geom=polygon_geom,
            expansion_percentage=expansion_percentage
        )

        if clip_data is None:
            logger.warning(f"Clipping failed or polygon was invalid for Polygon ID: {poly_id}. Skipping.")
            continue
        
        if clip_data['clipped_mesh_pv'].n_points == 0:
            logger.info(f"Polygon ID: {poly_id} - No mesh data found within the clipping bounds.")
            # Still calculate rugosity to get NaN and 100% missing data
        
        # 5. Prepare mesh_data for the Rugosity metric
        mesh_data_for_rugosity = {
            'surface_area_3d': clip_data['surface_area_3d'],
            'mesh_clipped_points': clip_data['mesh_clipped_points'],
            'clipped_region_size_2d': clip_data['clipped_region_size_2d']
        }

        # 6. Calculate Rugosity and coverage metrics
        logger.info(f"Calculating Rugosity for Polygon ID: {poly_id}...")
        rugosity_results = rugosity_metric.calculate(mesh_data_for_rugosity)
        
        logger.info(f"Polygon ID: {poly_id} Results: {rugosity_results}")
        
        current_poly_results = {
            'polygon_id': poly_id,
            **rugosity_results # Unpack all results from the metric
        }
        all_results.append(current_poly_results)

    # 7. Output results (e.g., print or save to CSV)
    logger.info("\n--- All Polygon Processing Finished ---")
    if all_results:
        logger.info("Summary of Results:")
        for res in all_results:
            logger.info(f"  Polygon ID {res['polygon_id']}: Rugosity={res.get('rugosity', 'N/A'):.4f}, MissingData={res.get('missing_data_percentage', 'N/A'):.2f}%")
        
        # Example: Save to CSV
        try:
            output_csv_path = project_root / "examples" / "real_data_rugosity_results.csv"
            output_csv_path.parent.mkdir(parents=True, exist_ok=True) # Ensure examples dir exists
            
            fieldnames = ['polygon_id', 'rugosity', 'actual_mesh_2d_area', 'defined_2d_area', 'missing_data_percentage', 'mesh_coverage_percentage', 'error']
            import csv
            with open(output_csv_path, 'w', newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames, extrasaction='ignore')
                writer.writeheader()
                writer.writerows(all_results)
            logger.info(f"Results saved to: {output_csv_path}")
        except Exception as e:
            logger.error(f"Failed to save results to CSV: {e}")
    else:
        logger.info("No results to summarize.")

if __name__ == "__main__":
    main() 