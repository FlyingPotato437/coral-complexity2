# coral-complexity-metrics

[!CAUTION]
This package is currently under development.

## Installation

```
pip install .
```

## Usage

### Shading

The `Shading` class takes in a mesh `.ply` file as input and returns shaded percentage and illuminated percentage. It uses code written by Srikanth Samy and published in [this repository.](https://aus01.safelinks.protection.outlook.com/?url=https%3A%2F%2Fgithub.com%2FFlyingPotato437%2Fsrikanth_coral_shading_script&data=05%7C02%7Cha.white%40aims.gov.au%7C5f6c78dc6a764492c49108dd51770c90%7Ce054a73b40dc4ae39fce60c537aa6fac%7C0%7C0%7C638756293945567609%7CUnknown%7CTWFpbGZsb3d8eyJFbXB0eU1hcGkiOnRydWUsIlYiOiIwLjAuMDAwMCIsIlAiOiJXaW4zMiIsIkFOIjoiTWFpbCIsIldUIjoyfQ%3D%3D%7C0%7C%7C%7C&sdata=w%2Bqu7GIhWA3qaDQK63UYCVqsK0Vu6mpwbgqOwgTigH4%3D&reserved=0)

```python
>>> from coral_complexity_metrics import Shading

>>> sh = Shading()
>>> sh.load_mesh("path/to/.ply/file")

>>> # calculate shading metric
>>> sh.calculate()

{
    'mesh_file': "mesh.ply",
    'shaded_percentage': "20%",
    'illuminated_percentage': "80%"
}
```

### Geometric Measures

The `GeometricMeasures` class takes a mesh `.ply` as input and returns the following geometric calculations:
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
* Transformations must be carried out by the user to get to square and cubic cm
* Models must have been scaled in the software you used to create them for this code to work

This class uses code originally written by Eoghan Aston, which was published [here.](https://github.com/E-Aston/CoralGeometry)

```python
>>> from coral_complexity_metrics import GeometricMeasures

>>> gm = GeometricMeasures()
>>> gm.load_mesh("path/to/.ply/file")

>>> # calculate geometric measures
>>> gm.calculate()

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

The `QuadratMetrics` class takes an `.obj` mesh file, dimensions of the input file, and size of quadrats as input. It outputs a dictionary per quadrat containing the following metrics:
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

```python
>>> from coral_complexity_metrics import QuadratMetrics

>>> qm = QuadratMetrics(
    dim="XYZ", # the dimensions of the input files WLH (width-length-height)
    size=1 # the size of a quadrat (standard is metres, but depends on the mesh units)
    )

>>> qm.load_mesh("path/to/.obj/file")

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