import os
import pymeshlab


class GeometricMeasures:
    """
    Class to calculate geometric measures of a coral colony 3D mesh.
    """

    def __init__(self):
        """Initialize the GeometricMeasures class with default values."""
        self.mesh_file = None
        self.mesh = None

    def load_mesh(self, file):
        """
        Load a 3D mesh from the specified file.

        Parameters:
        file (str): Path to the 3D model file.
        """
        self.mesh_file = file  # Assigns file to class variable
        if not os.path.exists(file):
            print(f"3D model file not found: {file}")
            return

        ms = pymeshlab.MeshSet()
        ms.load_new_mesh(file)

        ms.set_current_mesh(0)  # Makes the current mesh the original
        self.mesh = ms  # Assigns the mesh to the class variable
        print("Mesh loaded")

    def calculate(self):
        """
        Calculate geometric measures of the mesh.

        Returns:
        dict: Dictionary containing various geometric measures of the mesh.
        """
        # Compute measures of original mesh
        dict = (self.mesh.get_geometric_measures())
        mesh_sa = dict['surface_area']  # Assigns variable name

        # Loads the filter script (this one cleans mesh and closes holes)
        self.mesh.meshing_repair_non_manifold_edges()

        # remove isolated pieces
        self.mesh.meshing_remove_connected_component_by_diameter(
            mincomponentdiag=pymeshlab.Percentage(1), removeunref=True)

        # close holes
        self.mesh.meshing_close_holes(
            maxholesize=1000, selected=False, newfaceselected=True, selfintersection=False)

        # Compute measures of closed mesh
        dict = (self.mesh.get_geometric_measures())

        # Assigns mesh volume to variable mesh_volume
        try:
            mesh_volume = dict['mesh_volume']
        except KeyError:
            print("Error: Mesh volume not found. Please make sure you are using a valid mesh file containing a single coral colony.")
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
