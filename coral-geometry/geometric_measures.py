import csv
import os
import pymeshlab
import argparse
import glob


def parse_args():
    parser = argparse.ArgumentParser(
        description='Get geometric measures for a directory of .ply or .obj files.')
    parser.add_argument(
        '--input_dir', help='Input directory containing input .ply or .obj files.')
    args = parser.parse_args()
    return parser, args


def geometric_measures(file):

    ms = pymeshlab.MeshSet()

    ms.load_new_mesh(file)  # Loads a mesh from input folder

    ms.set_current_mesh(0)  # Makes the current mesh the original
    # Compute measures of original mesh
    dict = (ms.get_geometric_measures())
    mesh_sa = dict['surface_area']  # Assigns variable name

    # Loads the filter script (this one cleans mesh and closes holes)
    ms.meshing_repair_non_manifold_edges()
    ms.load_filter_script('Clean_Close.mlx')
    ms.apply_filter_script()  # Applies script

    dict = (ms.get_geometric_measures())  # Compute measures of closed mesh
    # Assigns mesh volume to variable mesh_volume
    mesh_volume = dict['mesh_volume']

    boundingbox = ms.current_mesh().bounding_box()
    width = boundingbox.dim_x()
    height = boundingbox.dim_z()

    # Loads the filter script (this one makes a convex hull)
    ms.load_filter_script('Filter_script.mlx')
    ms.apply_filter_script()  # Applies script

    ms.set_current_mesh(1)  # Sets closed model as current mesh
    dict = (ms.compute_geometric_measures())  # Compute measures of convex hull
    cvh_volume = dict['mesh_volume']  # Assigns variable name

    ASR = (cvh_volume - mesh_volume)  # Basic calculation for ASR
    PrOcc = (mesh_volume / cvh_volume)
    SSF = (ASR / mesh_sa)

    value_list = [str(file), mesh_sa, mesh_volume, cvh_volume,
                  ASR, PrOcc, SSF, width, height]

    with open("geometric_results.csv", "a", newline='') as f:
        write = csv.writer(f)
        write.writerow(value_list)


def main():

    parser, args = parse_args()

    Variable_names = ['File_Path', "Vol", "CVH_Vol", "ASR", "PrOcc",
                      "Surface_Area", "SSF", "Diameter", "Height"]  # Sets up a CSV with variable

    # names in current dir.
    with open("geometric_results.csv", "w", newline='') as f:
        write = csv.writer(f)
        write.writerow(Variable_names)

    for filename in glob.glob(args.input_dir + "/*"):
        if filename.endswith(".obj") or filename.endswith(".ply"):
            print(f"Calculating metrics for file: {filename}")
            geometric_measures(filename)
        else:
            continue


if __name__ == "__main__":
    main()
