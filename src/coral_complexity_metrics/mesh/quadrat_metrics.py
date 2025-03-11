from ._dimension_order import DimensionOrder
from ._quadrat_builder import QuadratBuilder
from ._quadrilateral import Quadrilateral
from ._vertex import Vertex
from ._mesh_io import read_obj
from tqdm import tqdm
import sys
import os
import meshio


class QuadratMetrics:
    """
    Class for calculating metrics of quadrats within a mesh.
    """

    def __init__(self, dim, size):
        """
        Initialize the QuadratMetrics class with dimension and size.

        Parameters:
        dim (int): Dimension order for the mesh.
        size (float): Size of the quadrats.
        """
        self.dim = dim
        self.size = size
        self.mesh_file = None
        self.mesh = None

    def ply_to_obj(self, ply_file, verbose=True):
        """
        Converts a .ply file to a .obj file and saves it in the same directory.

        Parameters:
        ply_file (str): Path to the .ply file to be converted.

        Returns:
        str: Path to the saved .obj file.
        """
        if verbose:
            print("Converting .ply file to .obj file...")
        mesh = meshio.read(ply_file)
        obj_file = ply_file.replace(".ply", ".obj")
        mesh.write(obj_file)
        return obj_file

    def load_mesh(self, file, verbose=True):
        """
        Reads in a mesh file in .obj format and stores it in a corresponding mesh object.

        Parameters:
        file (str): Path to the 3D model file.

        Returns:
        None
        """
        self.mesh_file = file
        if not os.path.exists(file):
            print(f"3D model file not found: {file}")
            return

        if file.endswith(".ply"):
            self.mesh_file = self.ply_to_obj(file, verbose)
        self.mesh = read_obj(self.mesh_file, verbose, DimensionOrder(self.dim))
        if verbose:
            print("Mesh loaded")

    def _calculate_bounding_box(self, verbose=True):
        """
        Calculates the bounding box from the extreme vertices in the mesh.

        Returns:
        None
        """
        if verbose:
            print("Calculating the bounding box...")

        max_x = sys.maxsize * -1
        max_y = sys.maxsize * -1
        min_x = sys.maxsize
        min_y = sys.maxsize

        for face in self.mesh.faces:
            for vertex in face.vertices:
                if vertex.x < min_x:
                    min_x = vertex.x
                if vertex.x > max_x:
                    max_x = vertex.x
                if vertex.y < min_y:
                    min_y = vertex.y
                if vertex.y > max_y:
                    max_y = vertex.y

        self.bounding_box = Quadrilateral(Vertex(min_x, min_y, 0), Vertex(max_x, min_y, 0),
                                          Vertex(max_x, max_y, 0), Vertex(min_x, max_y, 0))

    def _fit_quadrats_to_meshes(self, verbose=True):
        """
        Generates the quadrats inside the bounding box.

        Returns:
        None
        """
        if verbose:
            print("Generating the quadrats inside the bounding box...")

        self.quadrats = QuadratBuilder().build(self.bounding_box, self.size)

        if verbose:
            print("There are this many quadrats: " + str(len(self.quadrats)))

    def _calculate_metrics_of_quadrats(self, verbose=True):
        """
        Calculates metrics for each quadrat.

        Returns:
        None
        """
        if verbose:
            print("Calculating metrics...")
            self.metrics = self.mesh.calculate_metrics(self.quadrats)

        else:
            self.metrics = self.mesh.calculate_metrics(
                self.quadrats, disable_tqdm=True)

    def calculate(self, verbose=True):
        """
        Calculates the bounding box, fits quadrats to meshes, and calculates metrics for each quadrat.

        Returns:
        list: List of dictionaries containing metrics for each quadrat.
        """
        self._calculate_bounding_box(verbose)
        self._fit_quadrats_to_meshes(verbose)
        self._calculate_metrics_of_quadrats(verbose)
        results = []
        for metric in self.metrics:
            results.append({
                "mesh_name": self.mesh_file,
                "quadrat_size_m": self.size,
                "quadrat_rel_x": metric.quadrat_id[0],
                "quadrat_rel_y": metric.quadrat_id[1],
                "quadrat_rel_z_mean": metric.relative_z_mean,
                "quadrat_rel_z_sd": metric.relative_z_sd,
                "quadrat_abs_x": metric.quadrat_midpoint.x,
                "quadrat_abs_y": metric.quadrat_midpoint.y,
                "quadrat_abs_z": metric.quadrat_midpoint.z,
                "num_faces": metric.face_count,
                "num_vertices": metric.vertices_count,
                "3d_surface_area": metric.area3d,
                "2d_surface_area": metric.area2d,
                "surface_rugosity": metric.surface_rugosity()
            })
        return results

    def process_directory(self, directory, csv_file=None):
        """
        Process all mesh files in the specified directory.

        Parameters:
        directory (str): Path to the directory containing 3D model files.
        csv_file (str): Path to the CSV file to save the results.

        Returns:
        list: List of dictionaries containing metrics for each quadrat.
        """

        if not os.path.exists(directory):
            print(f"Directory not found: {directory}")
            return

        mesh_files = [file for file in os.listdir(
            directory) if file.endswith(".obj") or file.endswith(".ply")]

        results = []
        for mesh_file in tqdm(mesh_files, desc="Processing 3D models"):
            self.load_mesh(os.path.join(directory, mesh_file), verbose=False)
            results.extend(self.calculate(verbose=False))

        if csv_file:
            with open(csv_file, "w") as f:
                f.write(
                    "mesh_name,quadrat_size_m,quadrat_rel_x,quadrat_rel_y,quadrat_rel_z_mean,quadrat_rel_z_sd,quadrat_abs_x,quadrat_abs_y,quadrat_abs_z,num_faces,num_vertices,3d_surface_area,2d_surface_area,surface_rugosity\n")
                for result in results:
                    f.write(",".join([str(result[key])
                            for key in result.keys()]) + "\n")

        return results
