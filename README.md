# Coral Complexity Metrics

A Python package for calculating structural complexity metrics from 3D coral mesh files. This tool provides quantitative measures of coral structure using 3D models in OBJ or PLY format, supporting both plot-level and quadrat-level metrics, and can process individual files or entire directories.

## Features

- Load 3D mesh files (`.obj` or `.ply`)
- Calculates the following complexity metrics:
  - `is_watertight`: Boolean indicating whether the mesh is closed/watertight. If False, volume-based metrics will be NaN.
  - `num_faces`: Number of faces (polygons) in the mesh.
  - `num_vertices`: Number of vertices (points) in the mesh.
  - `3d_surface_area`: Total 3D surface area of the mesh (not the convex hull).
  - `2d_surface_area`: Projected 2D area of the mesh (ignoring the Z component).
  - `volume`: Volume enclosed by the mesh (if watertight; otherwise NaN).
  - `convex_hull_volume`: Volume of the minimum bounding convex hull enclosing the mesh.
  - `absolute_spatial_refuge`: Volumetric measure of shelter capacity (interstitial space) of the object. Calculated as `convex_hull_volume - volume` (ASR).
  - `proportion_occupied`: Proportion of the convex hull volume occupied by the mesh. Measures compactness. Calculated as `volume / convex_hull_volume` (PrOcc).
  - `shelter_size_factor`: Ratio of absolute spatial refuge to 3D surface area. Measures the size structure of refuges. Calculated as `ASR / 3d_surface_area` (SSF).
  - `surface_rugosity`: Ratio of 3D surface area to 2D surface area. Indicates surface complexity.
  - `shaded_percentage`: Percentage of the mesh surface that is shaded, based on simulated light direction. (Not returned if `shading_metrics` argument set to `False`.)
  - `illuminated_percentage`: Percentage of the mesh surface that is illuminated (1 - shaded_percentage). (Not returned if `shading_metrics` argument set to `False`.)
- Quadrat-based metrics 
  - The mesh is divided into quadrats of the specified size(s), starting at the centroid of the bounding box around the mesh.
  - Quadrat sizes must be specified in the same coordinate units as the mesh (e.g., if mesh coordinates are in meters, use [1.0, 0.5] for 1m and 0.5m quadrats).
  - For each quadrat, all the above complexity metrics are calculated, plus:
    - `quadrat_size`: The size (length of side) of the quadrat.
    - `quadrat_x_id`, `quadrat_y_id`: Indices of the quadrat in the X and Y directions.
    - `quadrat_x_min`, `quadrat_x_max`: Minimum and maximum X coordinates of the quadrat.
    - `quadrat_y_min`, `quadrat_y_max`: Minimum and maximum Y coordinates of the quadrat.
    - `quadrat_x_center`, `quadrat_y_center`: Center coordinates of the quadrat.
- Batch processing of directories with automatic CSV export

## Installation

```bash
git clone https://github.com/open-AIMS/coral-complexity-metrics.git
cd coral-complexity-metrics/
pip install .
```

## Usage

### Calculate Metrics for a Single Mesh

```python
from coral_complexity_metrics import ComplexityMetrics
import numpy as np

cm = ComplexityMetrics()

results = cm.calculate(
    mesh_file="path/to/mesh.obj",          # Path to the 3D model file (.obj or .ply)
    shading_metrics=True,                  # Whether to apply shading calculations (default: True)
    shading_light_dir=np.array([0, 0, -1]),# Direction of the light source for shading (default: np.array([0, 0, -1]))
    shading_sample_size=25000,             # Number of samples for shading calculation (default: 25,000)
    quadrat_metrics=True,                  # Whether to calculate quadrat metrics (default: False)
    quadrat_sizes=[1, 0.5],                # List of quadrat sizes in mesh units (default: [1])
    verbose=True                           # Print progress messages (default: True)
)
print(results)
```

Arguments:

- `mesh_file` (str): Path to the 3D model file (.obj or .ply).
- `shading_metrics` (bool): Whether to apply shading calculations to the mesh.
- `shading_light_dir` (np.array): Direction of the light source for shading.
- `shading_sample_size` (int): Number of samples for shading calculation.
- `quadrat_metrics` (bool): Whether to calculate quadrat-based metrics.
- `quadrat_sizes` (list): Quadrat sizes (in mesh coordinate units) for quadrat-based metrics.
- `verbose` (bool): Print progress and status messages.

### Process an Entire Directory

```python
results = cm.process_directory(
    directory="path/to/mesh_directory",    # Directory containing .obj or .ply files
    shading_metrics=True,                  # Whether to apply shading calculations (default: True)
    shading_light_dir=np.array([0, 0, -1]),# Direction of the light source for shading (default: np.array([0, 0, -1]))
    shading_sample_size=25000,             # Number of samples for shading calculation (default: 25,000)
    quadrat_metrics=True,                  # Whether to calculate quadrat metrics (default: False)
    quadrat_sizes=[1, 0.5],                # List of quadrat sizes in mesh units (default: [1])
    verbose=True,                          # Print progress messages (default: False)
    save_results=True,                     # Save results to CSV files (default: True)
    save_dir="results"                     # Directory to save CSV files (default: current directory)
)
print(results)
```

Arguments:

- `directory` (str): Path to the directory containing 3D model files.
- `shading_metrics` (bool): Whether to apply shading calculations to each mesh.
- `shading_light_dir` (np.array): Direction of the light source for shading.
- `shading_sample_size` (int): Number of samples for shading calculation.
- `quadrat_metrics` (bool): Whether to calculate quadrat-based metrics.
- `quadrat_sizes` (list): Quadrat sizes (in mesh coordinate units) for quadrat-based metrics.
- `verbose` (bool): Print progress and status messages.
- `save_results` (bool): Whether to save the results as CSV files.
- `save_dir` (str): Directory to save the CSV files.

Output:

Returns a list of dictionaries with plot-level and quadrat-level metrics for each mesh.
If save_results=True, two CSV files are created:
- plot_complexity_metrics.csv
- quadrat_complexity_metrics.csv