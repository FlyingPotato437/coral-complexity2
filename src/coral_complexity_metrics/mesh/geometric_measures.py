import os
import pymeshlab
from tqdm import tqdm


class GeometricMeasures:
    """
    Class to calculate geometric measures of a coral colony 3D mesh.
    """

    def __init__(self):
        """Initialize the GeometricMeasures class with default values."""
        self.mesh_file = None
        self.mesh = None

    def load_mesh(self, file, verbose=True):
        """
        Load a 3D mesh from the specified file.

        Parameters:
        file (str): Path to the 3D model file.
        """
        self.mesh_file = file  # Assigns file to class variable
        if not os.path.exists(file):
            print(f"3D model file not found: {file}")
            return

        if verbose:
            print("Loading 3D mesh...")
        ms = pymeshlab.MeshSet()
        ms.load_new_mesh(file)

        ms.set_current_mesh(0)  # Makes the current mesh the original
        self.mesh = ms  # Assigns the mesh to the class variable
        if verbose:
            print("Mesh loaded")

    def calculate(self, max_hole_size=1000, verbose=True):
        """
        Calculate geometric measures of the mesh.

        Parameters:
        max_hole_size (int): Maximum hole size to close in the mesh. The size is expressed as number of edges composing the hole boundary

        Returns:
        dict: Dictionary containing various geometric measures of the mesh.
        """
        # Compute measures of original mesh
        if verbose:
            print("Calculating geometric measures...")
        dict = (self.mesh.get_geometric_measures())
        mesh_sa = dict['surface_area']  # Assigns variable name

        # Loads the filter script (this one cleans mesh and closes holes)
        self.mesh.meshing_repair_non_manifold_edges()

        # remove isolated pieces
        self.mesh.meshing_remove_connected_component_by_diameter(
            mincomponentdiag=pymeshlab.PercentageValue(1), removeunref=True)

        # close holes
        self.mesh.meshing_close_holes(
            maxholesize=max_hole_size, selected=False, newfaceselected=True, selfintersection=False)

        # Compute measures of closed mesh
        dict = (self.mesh.get_geometric_measures())

        # Assigns mesh volume to variable mesh_volume
        try:
            mesh_volume = dict['mesh_volume']
        except KeyError:
            print("Error: Mesh volume not calculated. Consider increasing the maximum hole size for closing the mesh by setting the max_hole_size parameter to a higher value.")
            return

        boundingbox = self.mesh.current_mesh().bounding_box()
        width = boundingbox.dim_x()
        height = boundingbox.dim_z()

        # Compute convex hull
        self.mesh.generate_convex_hull()

        self.mesh.set_current_mesh(1)  # Sets closed model as current mesh

        # Compute measures of convex hull
        dict = (self.mesh.compute_geometric_measures())
        cvh_volume = dict['mesh_volume']  # Assigns variable name

        ASR = (cvh_volume - mesh_volume)  # Basic calculation for ASR
        PrOcc = (mesh_volume / cvh_volume)
        SSF = (ASR / mesh_sa)

        return {
            "mesh_file": str(self.mesh_file),
            "volume": mesh_volume,
            "CVH_volume": cvh_volume,
            "ASR": ASR,
            "proportion_occupied": PrOcc,
            "surface_area": mesh_sa,
            "SSF": SSF,
            "diameter": width,
            "height": height
        }

    def process_directory(self, directory, csv_file=None, max_hole_size=1000):
        """
        Process all mesh files in the specified directory.

        Parameters:
        directory (str): Path to the directory containing 3D model files.
        csv_file (str): Path to the CSV file to save the results.
        max_hole_size (int): Maximum hole size to close in the mesh.

        Returns:
        list: List of dictionaries containing geometric measures of each mesh.
        """

        if not os.path.exists(directory):
            print(f"Directory not found: {directory}")
            return

        mesh_files = [file for file in os.listdir(
            directory) if file.endswith(".obj") or file.endswith(".ply")]

        results = []
        for mesh_file in tqdm(mesh_files, desc="Processing 3D models"):
            self.load_mesh(os.path.join(directory, mesh_file), verbose=False)
            results.append(self.calculate(max_hole_size, verbose=False))

        if csv_file:
            with open(csv_file, "w") as f:
                f.write(
                    "mesh_file,volume,CVH_volume,ASR,proportion_occupied,surface_area,SSF,diameter,height\n")
                for result in results:
                    f.write(f"{result['mesh_file']},{result['volume']},{result['CVH_volume']},{result['ASR']},{result['proportion_occupied']},{result['surface_area']},{result['SSF']},{result['diameter']},{result['height']}\n")
        return results
