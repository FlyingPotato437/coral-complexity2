# Coral Complexity Measures

**coral-complexity-measures** is a Python package for extracting 3D complexity metrics from models (in `.ply` or `.obj` format) of coral reefs.

## Installation

```
pip install coral_complexity_metrics
```

## Usage

### Shading

The `Shading` class takes in a mesh `.ply` or `.obj` file as input and returns shaded percentage and illuminated percentage. It uses code written by Srikanth Samy and published in [this repository.](https://github.com/FlyingPotato437/srikanth_coral_shading_script)

```python
>>> from coral_complexity_metrics import Shading

>>> sh = Shading()
>>> sh.load_mesh("path/to/mesh/file")

>>> # calculate shading metric
>>> sh.calculate()

{
    'mesh_file': "mesh.ply",
    'shaded_percentage': "20%",
    'illuminated_percentage': "80%"
}
```

### Colony Geometric Measures

The `GeometricMeasures` class computes geometric measures of a given mesh. It has been validated for individual coral colonies, but can also be used on entire plots if the mesh can be closed to compute volume. By default the scipt only closes holes below a size of 1000, but this can be changed by specifying the `max_hole_size` argument. This function takes a mesh `.ply` or `.obj` as input and returns the following geometric calculations:

* `File_Path` : File path to the original input mesh. Identifies each coral in the file
* `Vol`: Volume of first mesh (the coral)
* `CVH_Vol`: Volume of minimum bounding convex hull enclosing original mesh
* `ASR`: Absolute spatial refuge. Volumetric measure of shelter capacity (interstitial space) of the object. Calculation : CVH_Vol - Vol = ASR
* `PrOcc`: Proportion Occupied. Proportion of the convex hull occupied by the coral lying inside it. Measures compactness. Calculation: Vol / CVH_Vol = PrOcc
* `Surface_Area`: 3D surface area of input colony (not the convex hull)
* `SSF`: Shelter size factor. Ratio of ASR to 3D surface area. Measure of size structure of refuges. Calculation: ASR / Surface_area = SSF
* `Diameter`: Maximum colony diameter (length along x axis which is by default the longest horizontal axis in meshlab)
* `Height`: Colony height (length along Z-axis of bounding box) 

Please also note:
* *Transformations must be carried out by the user to get to square and cubic cm*
* *Models must have been scaled in the software you used to create them for this code to work*

This class uses code originally written by Eoghan Aston, repository [here](https://github.com/E-Aston/CoralGeometry) and paper [here](https://www.frontiersin.org/journals/marine-science/articles/10.3389/fmars.2022.854395/full).

```python
>>> from coral_complexity_metrics import GeometricMeasures

>>> gm = GeometricMeasures()
>>> gm.load_mesh("path/to/mesh/file")

>>> # calculate geometric measures
>>> gm.calculate(
>>>     "max_hole_size"=1500 # OPTIONAL: if not provided, defaults to 1000
>>>     )

{
    'File_Path': 'mesh.ply', 
    'Vol': 0.025045684469949477, 
    'CVH_Vol': 0.06598048853525199, 
    'ASR': 0.040934804065302505, 
    'PrOcc': 0.379592285931137, 
    'Surface_Area': 0.6946476101875305, 
    'SSF': 0.05892887769994853, 
    'Diameter': 0.9493150040507317, 
    'Height': 0.43454796075820923
}
```

### Quadrat Metrics 

The `QuadratMetrics` class takes an `.obj` or `.ply` mesh file, dimensions of the input file, and size of quadrats as input. It outputs a dictionary per quadrat containing the following metrics:
* `quadrat_size_m`: The size of the fitted quadrats
* `quadrat_rel_x`: The relative x coordinates of the quadrat
* `quadrat_rel_y`: The relative y coordinates of the quadrat
* `quadrat_rel_z_mean`: The average of the relative z coordinates of the quadrat
* `quadrat_rel_z_sd`: The standard deviation of the relative z coordinates of the quadrat
* `quadrat_abs_x`: The absolute x coordinates of the quadrat
* `quadrat_abs_y`: The absolute x coordinates of the quadrat
* `quadrat_abs_z`: The absolute x coordinates of the quadrat
* `num_faces`: The number of faces in the quadrat
* `num_vertices`: The number of vertices in the quadrat
* `3d_surface_area`: The area of the faces in the quadrat
* `2d_surface_area`: The area of the faces in the quadrat without the Z component
* `surface_rugosity`: Surface rugosity is calculated as 3d_surface_area/2d_surface_area

This class uses code forked from [this repository.](https://github.com/shawes/mesh3d-python)

**NOTE:** *If a `.ply` file is used it is first converted to an `.obj` file and saved in the same directory.*

```python
>>> from coral_complexity_metrics import QuadratMetrics

>>> qm = QuadratMetrics(
    dim="XYZ", # the dimensions of the input files WLH (width-length-height)
    size=1 # the size of a quadrat (standard is metres, but depends on the mesh units)
    )

>>> qm.load_mesh("path/to/mesh/file")

>>> # calculate quadrat metrics
>>> qm.calculate()

[
    {
        'mesh_name': 'mesh.obj', 
        'quadrat_size_m': 1, 
        'quadrat_rel_x': 0, 
        'quadrat_rel_y': 0, 
        'quadrat_rel_z_mean': 0.3477004543933338, 
        'quadrat_rel_z_sd': 0.11053319793189083, 
        'quadrat_abs_x': 0.3118795, 
        'quadrat_abs_y': 0.24110750000000003, 
        'quadrat_abs_z': 0.0, 
        'num_faces': 56172, 
        'num_vertices': 28088, 
        '3d_surface_area': 0.9226908132507138, 
        '2d_surface_area': 0.48581063044899947, 
        'surface_rugosity': 1.899280821413763
    }
]

```

## Contact

Hannah White (Data Scientist) - ha.white@aims.gov.au

Renata Ferrari Legorreta (Ecological Risk Modeller) - r.ferrarilegorreta@aims.gov.au 

## Acknowledgements

* Srikanth Samy: https://github.com/FlyingPotato437/srikanth_coral_shading_script
* Eoghan Aston: https://github.com/E-Aston/CoralGeometry
* Steven Hawes: https://github.com/shawes/mesh3d-python

## License

[MIT](https://choosealicense.com/licenses/mit/)