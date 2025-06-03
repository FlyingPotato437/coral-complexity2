"""
Enhanced mesh-by-shapefile cropping and processing system.

This module provides comprehensive functionality for cropping 3D meshes using
shapefile polygons, calculating metrics on cropped sections, and exporting
results with data quality assessments.
"""

import numpy as np
import pandas as pd
import pyvista as pv
import geopandas as gpd
from pathlib import Path
from typing import Dict, Any, List, Optional, Union, Tuple
import warnings
import logging
from tqdm import tqdm
import csv

from .mesh_utils import (
    clip_mesh_by_polygon, 
    prepare_mesh_data_for_metrics,
    validate_mesh_for_metrics,
    is_mesh_watertight
)
from ._metric import MetricRegistry

logger = logging.getLogger(__name__)


class ShapefileMeshProcessor:
    """
    Enhanced processor for mesh-by-shapefile cropping with metric calculation.
    
    This class provides comprehensive functionality for:
    - Batch processing of polygons from shapefiles
    - Cropping meshes with quality assessment
    - Calculating specified metrics on cropped sections
    - Exporting results with issue flagging
    """
    
    def __init__(self, 
                 expansion_percentage: float = 5.0,
                 center_on_centroid: bool = True,
                 default_metrics: Optional[List[str]] = None,
                 verbose: bool = True):
        """
        Initialize the shapefile mesh processor.
        
        Parameters:
        expansion_percentage: Default percentage to expand polygon bounds
        center_on_centroid: Whether to center crops on polygon centroids
        default_metrics: List of default metrics to calculate
        verbose: Whether to print processing information
        """
        self.expansion_percentage = expansion_percentage
        self.center_on_centroid = center_on_centroid
        self.verbose = verbose
        
        # Initialize metric registry
        self.metric_registry = MetricRegistry()
        
        # Set default metrics if not provided
        if default_metrics is None:
            self.default_metrics = [
                'rugosity', 'fractal_dimension', 'height_range',
                'slope', 'plane_of_best_fit'
            ]
        else:
            self.default_metrics = default_metrics
    
    def process_mesh_with_shapefile(self,
                                   mesh_path: Union[str, Path],
                                   shapefile_path: Union[str, Path],
                                   output_csv: Optional[Union[str, Path]] = None,
                                   polygon_id_field: str = 'ID',
                                   metrics_to_calculate: Optional[List[str]] = None,
                                   save_cropped_meshes: bool = False,
                                   output_mesh_dir: Optional[Union[str, Path]] = None) -> pd.DataFrame:
        """
        Process a mesh with a shapefile, cropping and calculating metrics.
        
        Parameters:
        mesh_path: Path to the input mesh file
        shapefile_path: Path to the shapefile containing crop polygons
        output_csv: Optional path to save results CSV
        polygon_id_field: Field name for polygon identifiers
        metrics_to_calculate: List of metrics to calculate (uses defaults if None)
        save_cropped_meshes: Whether to save cropped mesh files
        output_mesh_dir: Directory to save cropped meshes
        
        Returns:
        DataFrame with results for all processed polygons
        """
        if self.verbose:
            logger.info(f"Processing mesh: {mesh_path}")
            logger.info(f"Using shapefile: {shapefile_path}")
        
        # Load mesh
        try:
            main_mesh = pv.read(str(mesh_path))
            if self.verbose:
                logger.info(f"Loaded mesh: {main_mesh.n_points} points, {main_mesh.n_cells} faces")
        except Exception as e:
            raise RuntimeError(f"Failed to load mesh from {mesh_path}: {e}")
        
        # Load shapefile
        try:
            gdf = gpd.read_file(str(shapefile_path))
            if self.verbose:
                logger.info(f"Loaded shapefile: {len(gdf)} polygons")
        except Exception as e:
            raise RuntimeError(f"Failed to load shapefile from {shapefile_path}: {e}")
        
        # Validate polygon ID field
        if polygon_id_field not in gdf.columns:
            raise ValueError(f"Polygon ID field '{polygon_id_field}' not found in shapefile. "
                           f"Available fields: {list(gdf.columns)}")
        
        # Use default metrics if none specified
        if metrics_to_calculate is None:
            metrics_to_calculate = self.default_metrics
        
        # Validate requested metrics
        available_metrics = self.metric_registry.list_metrics()
        invalid_metrics = [m for m in metrics_to_calculate if m not in available_metrics]
        if invalid_metrics:
            warnings.warn(f"Unknown metrics: {invalid_metrics}. Available: {available_metrics}")
            metrics_to_calculate = [m for m in metrics_to_calculate if m in available_metrics]
        
        # Process each polygon
        results = []
        for idx, row in tqdm(gdf.iterrows(), total=len(gdf), 
                           desc="Processing polygons", disable=not self.verbose):
            
            polygon_id = row[polygon_id_field]
            polygon_geom = row['geometry']
            
            if self.verbose:
                logger.debug(f"Processing polygon {polygon_id}")
            
            result = self._process_single_polygon(
                main_mesh=main_mesh,
                polygon_id=polygon_id,
                polygon_geom=polygon_geom,
                metrics_to_calculate=metrics_to_calculate,
                mesh_basename=Path(mesh_path).stem,
                save_cropped_mesh=save_cropped_meshes,
                output_mesh_dir=output_mesh_dir
            )
            
            results.append(result)
        
        # Convert to DataFrame
        results_df = pd.DataFrame(results)
        
        # Save to CSV if requested
        if output_csv is not None:
            self._save_results_to_csv(results_df, output_csv)
        
        if self.verbose:
            successful = len(results_df[results_df['processing_status'] == 'success'])
            logger.info(f"Processing complete: {successful}/{len(results_df)} polygons successful")
        
        return results_df
    
    def _process_single_polygon(self,
                               main_mesh: pv.PolyData,
                               polygon_id: Any,
                               polygon_geom,
                               metrics_to_calculate: List[str],
                               mesh_basename: str,
                               save_cropped_mesh: bool = False,
                               output_mesh_dir: Optional[Path] = None) -> Dict[str, Any]:
        """
        Process a single polygon: crop mesh and calculate metrics.
        
        Parameters:
        main_mesh: Input mesh
        polygon_id: Identifier for the polygon
        polygon_geom: Shapely polygon geometry
        metrics_to_calculate: List of metrics to calculate
        mesh_basename: Base name of the mesh file
        save_cropped_mesh: Whether to save the cropped mesh
        output_mesh_dir: Directory to save cropped meshes
        
        Returns:
        Dictionary with processing results
        """
        result = {
            'polygon_id': polygon_id,
            'mesh_file': mesh_basename,
            'processing_status': 'failed',
            'error_message': None,
            'n_points_cropped': 0,
            'n_faces_cropped': 0,
            'is_watertight': False,
        }
        
        try:
            # Crop the mesh
            clip_data = clip_mesh_by_polygon(
                main_mesh=main_mesh,
                polygon_geom=polygon_geom,
                expansion_percentage=self.expansion_percentage,
                center_on_centroid=self.center_on_centroid
            )
            
            if clip_data is None:
                result['error_message'] = "Clipping failed"
                result['processing_status'] = 'clip_failed'
                return result
            
            cropped_mesh = clip_data['clipped_mesh_pv']
            
            # Update basic info
            result.update({
                'n_points_cropped': cropped_mesh.n_points,
                'n_faces_cropped': cropped_mesh.n_cells,
                'is_watertight': clip_data['is_watertight'],
            })
            
            # Add data quality metrics
            result.update({
                'mesh_coverage_percentage': clip_data['mesh_coverage_percentage'],
                'missing_data_percentage': clip_data['missing_data_percentage'],
                'data_quality_score': clip_data['data_quality_score'],
                'point_density': clip_data['point_density'],
            })
            
            # Check if we have any mesh data
            if cropped_mesh.n_points == 0:
                result['processing_status'] = 'no_mesh_data'
                result['error_message'] = "No mesh data found within polygon bounds"
                # Still continue to calculate NaN metrics
            
            # Prepare mesh data for metric calculations
            mesh_data = prepare_mesh_data_for_metrics(cropped_mesh, clip_data)
            
            # Calculate requested metrics
            metric_results = self._calculate_metrics_safely(
                mesh_data, metrics_to_calculate
            )
            
            # Add metric results to main result
            result.update(metric_results)
            
            # Save cropped mesh if requested
            if save_cropped_mesh and output_mesh_dir and cropped_mesh.n_points > 0:
                self._save_cropped_mesh(
                    cropped_mesh, polygon_id, mesh_basename, output_mesh_dir
                )
            
            # Update status
            if result['processing_status'] == 'failed':  # Only update if not already set
                result['processing_status'] = 'success'
                
        except Exception as e:
            result['error_message'] = str(e)
            result['processing_status'] = 'exception'
            if self.verbose:
                logger.error(f"Error processing polygon {polygon_id}: {e}")
        
        return result
    
    def _calculate_metrics_safely(self, 
                                 mesh_data: Dict[str, Any], 
                                 metrics_to_calculate: List[str]) -> Dict[str, Any]:
        """
        Calculate metrics with proper error handling and watertight checking.
        
        Parameters:
        mesh_data: Prepared mesh data dictionary
        metrics_to_calculate: List of metric names to calculate
        
        Returns:
        Dictionary with metric results (NaN for failed calculations)
        """
        metric_results = {}
        
        # Check mesh closure for volume-dependent metrics
        is_watertight = mesh_data.get('is_watertight', False)
        
        for metric_name in metrics_to_calculate:
            try:
                # Get the metric instance
                metric = self.metric_registry.get_metric(metric_name)
                
                # Check if metric requires watertight mesh
                if metric.requires_closed_mesh and not is_watertight:
                    # Return NaN for volume-dependent metrics on non-closed meshes
                    if metric_name in ['volume', 'convex_hull_volume', 'proportion_occupied']:
                        metric_results[metric_name] = float('nan')
                        continue
                
                # Calculate the metric
                result = metric.calculate(mesh_data)
                
                # Handle different result formats
                if isinstance(result, dict):
                    # Flatten nested results with prefixes
                    for key, value in result.items():
                        result_key = f"{metric_name}_{key}" if key != metric_name else key
                        metric_results[result_key] = value
                else:
                    metric_results[metric_name] = result
                    
            except Exception as e:
                # Log error and set NaN
                if self.verbose:
                    logger.warning(f"Failed to calculate {metric_name}: {e}")
                metric_results[metric_name] = float('nan')
                metric_results[f"{metric_name}_error"] = str(e)
        
        return metric_results
    
    def _save_cropped_mesh(self, 
                          cropped_mesh: pv.PolyData,
                          polygon_id: Any,
                          mesh_basename: str,
                          output_mesh_dir: Path) -> None:
        """Save a cropped mesh to file."""
        try:
            output_mesh_dir = Path(output_mesh_dir)
            output_mesh_dir.mkdir(parents=True, exist_ok=True)
            
            output_filename = f"{mesh_basename}_polygon_{polygon_id}.ply"
            output_path = output_mesh_dir / output_filename
            
            cropped_mesh.save(str(output_path))
            
            if self.verbose:
                logger.debug(f"Saved cropped mesh: {output_path}")
                
        except Exception as e:
            if self.verbose:
                logger.warning(f"Failed to save cropped mesh for polygon {polygon_id}: {e}")
    
    def _save_results_to_csv(self, results_df: pd.DataFrame, output_csv: Union[str, Path]) -> None:
        """Save results DataFrame to CSV with proper formatting."""
        try:
            output_csv = Path(output_csv)
            output_csv.parent.mkdir(parents=True, exist_ok=True)
            
            # Format the DataFrame for better CSV output
            formatted_df = results_df.copy()
            
            # Round numeric columns to reasonable precision
            numeric_columns = formatted_df.select_dtypes(include=[np.number]).columns
            for col in numeric_columns:
                if col in formatted_df.columns:
                    formatted_df[col] = formatted_df[col].round(6)
            
            # Save to CSV
            formatted_df.to_csv(output_csv, index=False, na_rep='NaN')
            
            if self.verbose:
                logger.info(f"Results saved to: {output_csv}")
                
        except Exception as e:
            logger.error(f"Failed to save results to CSV: {e}")
            raise
    
    def batch_process_directory(self,
                               mesh_directory: Union[str, Path],
                               shapefile_path: Union[str, Path],
                               output_directory: Union[str, Path],
                               polygon_id_field: str = 'ID',
                               metrics_to_calculate: Optional[List[str]] = None,
                               save_cropped_meshes: bool = False,
                               mesh_extensions: List[str] = None) -> pd.DataFrame:
        """
        Batch process all mesh files in a directory with a single shapefile.
        
        Parameters:
        mesh_directory: Directory containing mesh files
        shapefile_path: Path to shapefile for cropping
        output_directory: Directory to save results
        polygon_id_field: Field name for polygon identifiers
        metrics_to_calculate: List of metrics to calculate
        save_cropped_meshes: Whether to save cropped mesh files
        mesh_extensions: List of file extensions to process
        
        Returns:
        Combined DataFrame with results for all meshes
        """
        if mesh_extensions is None:
            mesh_extensions = ['.ply', '.obj', '.stl', '.vtk']
        
        mesh_dir = Path(mesh_directory)
        output_dir = Path(output_directory)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Find all mesh files
        mesh_files = []
        for ext in mesh_extensions:
            mesh_files.extend(mesh_dir.glob(f"*{ext}"))
        
        if not mesh_files:
            raise ValueError(f"No mesh files found in {mesh_directory} with extensions {mesh_extensions}")
        
        if self.verbose:
            logger.info(f"Found {len(mesh_files)} mesh files to process")
        
        # Process each mesh file
        all_results = []
        for mesh_file in tqdm(mesh_files, desc="Processing mesh files", disable=not self.verbose):
            try:
                # Set up output paths
                csv_output = output_dir / f"{mesh_file.stem}_results.csv"
                mesh_output_dir = output_dir / f"{mesh_file.stem}_cropped" if save_cropped_meshes else None
                
                # Process the mesh
                results_df = self.process_mesh_with_shapefile(
                    mesh_path=mesh_file,
                    shapefile_path=shapefile_path,
                    output_csv=csv_output,
                    polygon_id_field=polygon_id_field,
                    metrics_to_calculate=metrics_to_calculate,
                    save_cropped_meshes=save_cropped_meshes,
                    output_mesh_dir=mesh_output_dir
                )
                
                # Add mesh file identifier
                results_df['source_mesh_file'] = mesh_file.name
                all_results.append(results_df)
                
            except Exception as e:
                if self.verbose:
                    logger.error(f"Failed to process {mesh_file.name}: {e}")
                continue
        
        # Combine all results
        if all_results:
            combined_results = pd.concat(all_results, ignore_index=True)
            
            # Save combined results
            combined_csv = output_dir / "combined_results.csv"
            self._save_results_to_csv(combined_results, combined_csv)
            
            return combined_results
        else:
            return pd.DataFrame()
    
    def get_processing_summary(self, results_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate a summary of processing results.
        
        Parameters:
        results_df: DataFrame with processing results
        
        Returns:
        Dictionary with summary statistics
        """
        summary = {
            'total_polygons': len(results_df),
            'successful_crops': len(results_df[results_df['processing_status'] == 'success']),
            'failed_crops': len(results_df[results_df['processing_status'] != 'success']),
            'watertight_meshes': len(results_df[results_df['is_watertight'] == True]),
            'mean_coverage_percentage': results_df['mesh_coverage_percentage'].mean(),
            'mean_data_quality_score': results_df['data_quality_score'].mean(),
        }
        
        # Add failure reasons
        failure_counts = results_df[results_df['processing_status'] != 'success']['processing_status'].value_counts()
        summary['failure_reasons'] = failure_counts.to_dict()
        
        # Add metric statistics
        metric_columns = [col for col in results_df.columns 
                         if col not in ['polygon_id', 'mesh_file', 'processing_status', 'error_message']]
        
        for col in metric_columns:
            if results_df[col].dtype in [np.float64, np.int64]:
                summary[f'{col}_mean'] = results_df[col].mean()
                summary[f'{col}_valid_count'] = results_df[col].notna().sum()
        
        return summary


def process_mesh_shapefile_batch(mesh_files: List[Union[str, Path]],
                                shapefile_path: Union[str, Path],
                                output_csv: Union[str, Path],
                                polygon_id_field: str = 'ID',
                                metrics: Optional[List[str]] = None,
                                expansion_percentage: float = 5.0,
                                save_cropped_meshes: bool = False,
                                output_mesh_dir: Optional[Union[str, Path]] = None) -> pd.DataFrame:
    """
    Convenience function for batch processing multiple mesh files.
    
    Parameters:
    mesh_files: List of mesh file paths
    shapefile_path: Path to shapefile
    output_csv: Path to save combined results
    polygon_id_field: Field name for polygon IDs
    metrics: List of metrics to calculate
    expansion_percentage: Polygon expansion percentage
    save_cropped_meshes: Whether to save cropped meshes
    output_mesh_dir: Directory for cropped meshes
    
    Returns:
    Combined DataFrame with all results
    """
    processor = ShapefileMeshProcessor(
        expansion_percentage=expansion_percentage,
        default_metrics=metrics
    )
    
    all_results = []
    for mesh_file in tqdm(mesh_files, desc="Processing meshes"):
        try:
            results_df = processor.process_mesh_with_shapefile(
                mesh_path=mesh_file,
                shapefile_path=shapefile_path,
                polygon_id_field=polygon_id_field,
                metrics_to_calculate=metrics,
                save_cropped_meshes=save_cropped_meshes,
                output_mesh_dir=output_mesh_dir
            )
            results_df['source_mesh_file'] = Path(mesh_file).name
            all_results.append(results_df)
        except Exception as e:
            logger.error(f"Failed to process {mesh_file}: {e}")
            continue
    
    # Combine and save results
    if all_results:
        combined_df = pd.concat(all_results, ignore_index=True)
        processor._save_results_to_csv(combined_df, output_csv)
        return combined_df
    else:
        return pd.DataFrame()