from ._dimension_order import DimensionOrder
from ._quadrat_builder import QuadratBuilder
from ._quadrilateral import Quadrilateral
from ._vertex import Vertex
from ._mesh_io import read_obj
import sys
import meshio


class QuadratMetrics:
    def __init__(self, dim, size):
        self.dim = dim
        self.size = size
        self.mesh_file = None
        self.mesh = None

    def ply_to_obj(self, ply_file):
        """
        Converts a .ply file to a .obj file and saves it in the same directory

        Args:
            ply_file: Path to the .ply file to be converted

        Returns:
            Returns the path to the saved .obj file
        """
        print("Converting .ply file to .obj file...")
        mesh = meshio.read(ply_file)
        obj_file = ply_file.replace(".ply", ".obj")
        mesh.write(obj_file)
        return obj_file

    def load_mesh(self, file):
        """
        Reads in mesh file in .obj format and stores them in a
        corresponding mesh objects

        Args:
            file: List of command line arguments

        Returns:
            A mesh object, essentially a collection of faces
        """
        if file.endswith(".ply"):
            file = self.ply_to_obj(file)
        self.mesh_file = file  # Assigns file to class variable
        self.mesh = read_obj(file, True, DimensionOrder(self.dim))
        print("Mesh loaded")

    def _calculate_bounding_box(self):
        """Calculates the bounding box from the extreme vertices in
        each mesh"""
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

    def _fit_quadrats_to_meshes(self):
        print("Generating the quadrats inside the bounding box...")

        self.quadrats = QuadratBuilder().build(self.bounding_box, self.size)

        print("There are this many quadrats: " + str(len(self.quadrats)))

    def _calculate_metrics_of_quadrats(self):
        print("Calculating metrics...")

        self.metrics = self.mesh.calculate_metrics(self.quadrats)

    def calculate(self):
        self._calculate_bounding_box()
        self._fit_quadrats_to_meshes()
        self._calculate_metrics_of_quadrats()
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
