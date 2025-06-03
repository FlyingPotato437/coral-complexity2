#!/usr/bin/env python3
"""
Real data processing example for coral complexity metrics.

This script demonstrates processing real coral mesh data with the enhanced
EcoRRAP package, including shapefile-based analysis and quality assessment.

Usage:
    python examples/process_real_data.py --mesh mesh.ply --shapefile regions.shp
"""

import argparse
import sys
import warnings
from pathlib import Path
import pandas as pd

# Import coral complexity metrics
import coral_complexity_metrics as ccm

def validate_inputs(mesh_path, shapefile_path):
    """Validate input files exist and are accessible."""
    mesh_path = Path(mesh_path)
    shapefile_path = Path(shapefile_path)
    
    if not mesh_path.exists():
        raise FileNotFoundError(f"Mesh file not found: {mesh_path}")
    
    if not shapefile_path.exists():
        raise FileNotFoundError(f"Shapefile not found: {shapefile_path}")
    
    # Check file extensions
    if mesh_path.suffix.lower() not in ['.ply', '.obj', '.stl']:
        warnings.warn(f"Unexpected mesh file extension: {mesh_path.suffix}")
    
    if shapefile_path.suffix.lower() != '.shp':
        warnings.warn(f"Expected .shp file, got: {shapefile_path.suffix}")
    
    return mesh_path, shapefile_path

def check_dependencies():
    """Check that all required dependencies are available."""
    info = ccm.get_info()
    required_features = ['mesh_processing', 'shapefile_processing']
    
    missing_features = []
    for feature in required_features:
        if not info['features'].get(feature, False):
            missing_features.append(feature)
    
    if missing_features:
        print("ERROR: Missing required features:")
        for feature in missing_features:
            print(f"  - {feature}")
        print("\nInstall full dependencies with: pip install coral-complexity-metrics[full]")
        return False
    
    return True

def process_mesh_data(mesh_path, shapefile_path, output_dir):
    """Process mesh data using shapefile regions."""
    print(f"Processing mesh: {mesh_path}")
    print(f"Using regions from: {shapefile_path}")
    print(f"Output directory: {output_dir}")
    
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Initialize processor with comprehensive metrics
    metrics = [
        'surface_area',
        'surface_rugosity', 
        'height_range',
        'diameter',
        'volume',
        'proportion_occupied',
        'absolute_spatial_refuge'
    ]
    
    processor = ccm.mesh.ShapefileMeshProcessor(
        expansion_percentage=5.0,
        center_on_centroid=True,
        default_metrics=metrics,
        verbose=True
    )
    
    # Process the mesh
    output_csv = output_dir / 'complexity_metrics.csv'
    mesh_output_dir = output_dir / 'cropped_meshes'
    
    print("\nProcessing regions...")
    results_df = processor.process_mesh_with_shapefile(
        mesh_path=str(mesh_path),
        shapefile_path=str(shapefile_path),
        output_csv=str(output_csv),
        save_cropped_meshes=True,
        output_mesh_dir=str(mesh_output_dir)
    )
    
    return results_df, output_csv

def analyze_results(results_df, output_csv):
    """Analyze and summarize processing results."""
    print(f"\nResults saved to: {output_csv}")
    print(f"Processed {len(results_df)} regions")
    
    # Summarize processing status
    status_counts = results_df['processing_status'].value_counts()
    print(f"\nProcessing status:")
    for status, count in status_counts.items():
        print(f"  {status}: {count}")
    
    # Analyze data quality
    if 'data_quality_score' in results_df.columns:
        quality_stats = results_df['data_quality_score'].describe()
        print(f"\nData quality statistics:")
        print(f"  Mean: {quality_stats['mean']:.3f}")
        print(f"  Std:  {quality_stats['std']:.3f}")
        print(f"  Min:  {quality_stats['min']:.3f}")
        print(f"  Max:  {quality_stats['max']:.3f}")
        
        # Flag low quality regions
        low_quality = results_df[results_df['data_quality_score'] < 0.5]
        if len(low_quality) > 0:
            print(f"\nWARNING: {len(low_quality)} regions have low data quality (<0.5):")
            for idx, row in low_quality.iterrows():
                print(f"  - {row['polygon_id']}: {row['data_quality_score']:.3f}")
    
    # Coverage analysis
    if 'mesh_coverage_percentage' in results_df.columns:
        coverage_stats = results_df['mesh_coverage_percentage'].describe()
        print(f"\nMesh coverage statistics:")
        print(f"  Mean: {coverage_stats['mean']:.1f}%")
        print(f"  Min:  {coverage_stats['min']:.1f}%")
        print(f"  Max:  {coverage_stats['max']:.1f}%")
        
        # Flag low coverage regions
        low_coverage = results_df[results_df['mesh_coverage_percentage'] < 50]
        if len(low_coverage) > 0:
            print(f"\nWARNING: {len(low_coverage)} regions have low mesh coverage (<50%):")
            for idx, row in low_coverage.iterrows():
                print(f"  - {row['polygon_id']}: {row['mesh_coverage_percentage']:.1f}%")

def calculate_additional_shading(mesh_path, output_dir):
    """Calculate shading metrics for the full mesh."""
    print(f"\nCalculating shading metrics for full mesh...")
    
    try:
        # Initialize shading calculator
        shading = ccm.Shading(cpu_percentage=50.0)
        shading.load_mesh(str(mesh_path), verbose=False)
        
        # Calculate shading for typical conditions
        scenarios = [
            {
                'name': 'Winter_Morning',
                'day_of_year': 355,  # Winter solstice
                'time_of_day': 9.0,
                'latitude': -16.3,
                'longitude': 145.8
            },
            {
                'name': 'Summer_Noon', 
                'day_of_year': 172,  # Summer solstice
                'time_of_day': 12.0,
                'latitude': -16.3,
                'longitude': 145.8
            },
            {
                'name': 'Equinox_Afternoon',
                'day_of_year': 80,   # Spring equinox
                'time_of_day': 15.0,
                'latitude': -16.3,
                'longitude': 145.8
            }
        ]
        
        shading_results = []
        for scenario in scenarios:
            result = shading.calculate(
                day_of_year=scenario['day_of_year'],
                time_of_day=scenario['time_of_day'],
                latitude=scenario['latitude'],
                longitude=scenario['longitude'],
                verbose=False
            )
            
            shading_results.append({
                'scenario': scenario['name'],
                'shaded_percentage': result['shaded_percentage'],
                'illuminated_percentage': result['illuminated_percentage'],
                'sample_points': result['sample_points']
            })
            
            print(f"  {scenario['name']}: {result['shaded_percentage']:.1f}% shaded")
        
        # Save shading results
        shading_df = pd.DataFrame(shading_results)
        shading_csv = Path(output_dir) / 'shading_analysis.csv'
        shading_df.to_csv(shading_csv, index=False)
        print(f"Shading results saved to: {shading_csv}")
        
        return shading_results
        
    except Exception as e:
        print(f"Shading calculation failed: {e}")
        print("This may be due to missing dependencies or mesh loading issues")
        return None

def main():
    """Main processing function."""
    parser = argparse.ArgumentParser(
        description='Process coral mesh data with shapefile regions'
    )
    parser.add_argument(
        '--mesh', 
        required=True,
        help='Path to mesh file (.ply, .obj, .stl)'
    )
    parser.add_argument(
        '--shapefile',
        required=True, 
        help='Path to shapefile (.shp) with analysis regions'
    )
    parser.add_argument(
        '--output',
        default='./output',
        help='Output directory for results (default: ./output)'
    )
    parser.add_argument(
        '--no-shading',
        action='store_true',
        help='Skip shading analysis'
    )
    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Suppress warnings'
    )
    
    args = parser.parse_args()
    
    if args.quiet:
        warnings.filterwarnings('ignore')
    
    print("EcoRRAP Real Data Processing")
    print("=" * 40)
    
    try:
        # Check dependencies
        if not check_dependencies():
            return 1
        
        # Validate inputs
        mesh_path, shapefile_path = validate_inputs(args.mesh, args.shapefile)
        
        # Process mesh data
        results_df, output_csv = process_mesh_data(
            mesh_path, shapefile_path, args.output
        )
        
        # Analyze results
        analyze_results(results_df, output_csv)
        
        # Calculate shading if requested
        if not args.no_shading:
            shading_results = calculate_additional_shading(mesh_path, args.output)
        
        print("\nProcessing completed successfully!")
        return 0
        
    except Exception as e:
        print(f"ERROR: {e}")
        if not args.quiet:
            import traceback
            traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main()) 