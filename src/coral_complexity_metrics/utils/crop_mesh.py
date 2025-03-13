import geopandas as gpd
import numpy as np
import open3d
import os


def crop_polygon(MaskCoordinate, minZ, maxZ, mesh):
    """
    Crop a mesh to a polygon 
    :param MaskCoordinate: MaskCoordinate is a shapely polygon object
    :param minZ: minZ is the minimum Z value of the mesh
    :param maxZ: maxZ is the maximum Z value of the mesh
    :param mesh: mesh is the open3d mesh object
    :return: open3d mesh object
    """

    # create empty array and populate with polygon x y values
    polygon = np.dstack(MaskCoordinate.boundary.xy).tolist()[0][:-1]
    arr = np.array([np.array(i) for i in polygon])

    # create min and max of xyz
    min_bound = [(min(arr[:, 0])-0.05), (min(arr[:, 1])-0.05), minZ]
    max_bound = [(max(arr[:, 0])+0.05), (max(arr[:, 1])+0.05), maxZ]

    # create new bounding box that match x, y extent of ortho and Z of original mesh
    box = np.asarray([min_bound, max_bound])

    # create points vector from min & max and create new bounding box
    pts = open3d.utility.Vector3dVector(box)
    bpt = open3d.geometry.AxisAlignedBoundingBox.create_from_points(pts)

    # return the cropped mesh
    return open3d.geometry.TriangleMesh.crop(mesh, bpt)


def crop_mesh_to_segments(mesh_file, shp_file, output_dir):
    """
    Crop a mesh to a shapefile
    :param mesh_file: mesh_file is the path to the mesh file
    :param shp_file: shp_file is the path to the shapefile
    :param output_dir: output_dir is the path to the output directory
    """
    # load ply file
    print("Loading mesh file...")
    mesh = open3d.io.read_triangle_mesh(mesh_file)

    # load shapefile
    print("Loading shapefile...")
    shp = gpd.read_file(shp_file)

    print("Cropping mesh...")
    # get min and max Z values
    vert = np.asarray(mesh.vertices)
    minZ = min(vert[:, 2])
    maxZ = max(vert[:, 2])

    count = 0
    for i, row in shp.iterrows():
        cropped_mesh = crop_polygon(row['geometry'], minZ, maxZ, mesh)
        if np.asarray(cropped_mesh.vertices).size > 0:
            output = os.path.join(output_dir, os.path.basename(
                mesh_file).split(".")[0] + f'_{i}.obj')
            open3d.io.write_triangle_mesh(output, cropped_mesh)
            count += 1

    print(f"Number of meshes saved to ouput folder: {count}")
