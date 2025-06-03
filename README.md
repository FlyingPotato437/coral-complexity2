# EcoRRAP: Enhanced Coral Reef Complexity Metrics

A comprehensive Python package for analyzing 3D coral reef structural complexity using photogrammetry and LiDAR-derived mesh data.

## Overview

EcoRRAP provides advanced tools for quantifying coral reef structural complexity through various geometric, optical, and spatial metrics. The package supports both traditional complexity measures and cutting-edge analysis techniques including mesh-by-shapefile processing, data quality assessment, and modular shading analysis.

## Key Features

- **Comprehensive Metrics**: 17+ standardized complexity metrics including surface rugosity, fractal dimension, volume calculations, and spatial refuge measurements
- **Shading Analysis**: Ray-traced shading calculations with environmental adjustments for slope, aspect, and solar positioning
- **Shapefile Integration**: Batch processing of mesh data using GIS polygon boundaries with automatic data quality assessment
- **Data Quality Control**: Coverage analysis, missing data detection, and mesh validation for reliable results
- **Modular Architecture**: Extensible framework for adding new metrics and environmental factors
- **Robust Mesh Handling**: Proper handling of non-watertight meshes with appropriate NaN returns for volume-dependent metrics

## Installation

### Basic Installation
```bash
pip install coral-complexity-metrics
```

### Full Installation (Recommended)
```bash
pip install coral-complexity-metrics[full]
```

The full installation includes all optional dependencies for complete functionality:
- PyVista (3D mesh processing)
- SciPy (spatial algorithms)
- Scikit-learn (complexity metrics)
- GeoPandas/Shapely (shapefile processing)
- Matplotlib (visualization)

## Quick Start

### Basic Mesh Analysis
```python
import coral_complexity_metrics as ccm

# Load and analyze a mesh
shading = ccm.Shading()
shading.load_mesh('reef_mesh.ply')

# Calculate shading metrics
result = shading.calculate(
    time_of_day=12.0,
    day_of_year=180,
    latitude=-16.3,
    longitude=145.8
)

print(f"Shaded area: {result['shaded_percentage']:.1f}%")
```

### Shapefile-Based Processing
```python
# Process multiple regions defined by shapefile
processor = ccm.mesh.ShapefileMeshProcessor()

results = processor.process_mesh_with_shapefile(
    mesh_path='reef_mesh.ply',
    shapefile_path='analysis_regions.shp',
    output_csv='results.csv',
    metrics=['surface_area', 'rugosity', 'height_range']
)
```

### Individual Metrics
```python
# Calculate specific metrics
from coral_complexity_metrics.mesh.unified_metrics import (
    SurfaceRugosity, HeightRange, FractalDimension
)

# Load mesh data
mesh_data = ccm.mesh.prepare_mesh_data_for_metrics(mesh)

# Calculate metrics
rugosity = SurfaceRugosity().calculate(mesh_data)
height_stats = HeightRange().calculate(mesh_data)
fractal_dim = FractalDimension().calculate(mesh_data)
```

## Available Metrics

### Surface Metrics
- Surface Area (3D)
- Surface Rugosity
- Projected Area (2D)
- Mesh Counts (faces/vertices)

### Volume Metrics (Watertight Meshes Only)
- Volume
- Convex Hull Volume
- Proportion Occupied
- Absolute Spatial Refuge
- Shelter Size Factor

### Complexity Metrics
- Fractal Dimension (box-counting)
- Slope Statistics
- Height Range and Distribution
- Plane of Best Fit

### Spatial Metrics
- Diameter (max XY extent)
- Height (Z range)
- Quadrat Positioning
- Coverage Quality

## Data Quality Assessment

Every analysis includes comprehensive quality metrics:

- **Coverage Percentage**: Portion of analysis region covered by mesh data
- **Missing Data Percentage**: Areas with no mesh coverage
- **Data Quality Score**: Overall quality rating (0-1 scale)
- **Point Density**: Mesh resolution within analysis areas

## Architecture

### Modular Design
- **Base Classes**: Abstract interfaces for extensible metric development
- **Metric Registry**: Automatic discovery and categorization of available metrics
- **Optional Dependencies**: Graceful fallbacks when specialized libraries unavailable
- **Type Safety**: Full type hints and validation throughout

### Supported Data Formats
- PLY (Stanford Polygon format)
- STL (STereoLithography)
- OBJ (Wavefront)
- VTK/VTP (Visualization Toolkit)

## Scientific Applications

EcoRRAP supports research in:
- Coral reef biodiversity assessment
- Habitat complexity quantification
- Climate change impact studies
- Restoration monitoring
- Fisheries habitat evaluation

## Contributing

We welcome contributions! Please see our development guidelines:

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## Citation

When using EcoRRAP in scientific publications, please cite:

```
EcoRRAP: Enhanced Coral Reef Complexity Metrics Package
Version 2.0.0
https://github.com/your-org/coral-complexity-metrics
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Australian Institute of Marine Science (AIMS)
- University of Sydney Geosciences
- Coral reef research community

## Support

For questions, bug reports, or feature requests:
- GitHub Issues: https://github.com/your-org/coral-complexity-metrics/issues
- Documentation: https://coral-complexity-metrics.readthedocs.io
- Email: support@ecorap.org