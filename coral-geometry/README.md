## Coral Geometry 

This code has been taken and slightly modified from https://github.com/E-Aston/CoralGeometry.

It takes an directory of meshes as input (in either `.ply` or `.obj` format) and outputs a CSV file with the following metrics per input mesh:
* **Vol**: Volume of first mesh (the coral)
* **CVH_Vol**: Volume of minimum bounding convex hull enclosing original mesh
* **ASR**: Absolute spatial refuge. Volumetric measure of shelter capacity (interstitial space) of the object. Calculation : CVH_Vol - Vol = ASR
* **PrOcc**: Proportion Occupied. Proportion of the convex hull occupied by the coral lying inside it. Measures compactness. Calculation: Vol / CVH_Vol = PrOcc
* **Surface_Area**: 3D surface area of input colony (not the convex hull)
* **SSF**: Shelter size factor. Ratio of ASR to 3D surface area. Measure of size structure of refuges. Calculation: ASR / Surface_area = SSF
* **Diameter**: Maximum colony diameter (length along x axis which is by default the longest horizontal axis in meshlab)
* **Height**: Colony height (length along Z-axis of bounding box)

### Installation 

`pip install -r requirments.txt`

### Usage 

`python geometric_measures.py --input_dir path/to/folder/containing/meshes`