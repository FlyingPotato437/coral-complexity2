import math
import pdb
import pytest
from mesh3d.face import Face
from mesh3d.vertex import Vertex


class TestFace:

    def test_face_contructor_not_none(self):
        my_face = Face(Vertex(0, 0, 0), Vertex(1, 1, 0), Vertex(1, 3, 0))
        assert my_face is not None

    def test_face_vertex1(self):
        my_face = Face(Vertex(1, 1, 1), Vertex(0, 0, 0), Vertex(0, 0, 0))
        assert my_face.vertices[0] is not None
        assert my_face.vertices[0] == Vertex(1, 1, 1)

    def test_face_vertex2(self):
        my_face = Face(Vertex(0, 0, 0), Vertex(1, 1, 1), Vertex(0, 0, 0))
        assert my_face.vertices[1] is not None
        assert my_face.vertices[1] == Vertex(1, 1, 1)

    def test_face_vertex3(self):
        my_face = Face(vertex1=Vertex(0, 0, 0), vertex2=Vertex(0, 0, 0), vertex3=Vertex(1, 1, 1))
        assert my_face.vertices[2] is not None
        assert my_face.vertices[2] == Vertex(1, 1, 1)

    def test_face_centroid_returns_vertex(self):
        my_face = Face(Vertex(1, 1, 1), Vertex(2, 5, 7), Vertex(3, 0, 1))
        my_centroid = my_face.centroid
        assert isinstance(my_centroid, Vertex)

    def test_face_centroid_returns_correct_values(self):
        my_face = Face(Vertex(1, 1, 1), Vertex(2, 5, 7), Vertex(3, 0, 1))
        my_centroid = my_face.centroid
        assert my_centroid.x == 2
        assert my_centroid.y == 2
        assert my_centroid.z == 3

    def test_face_area(self):
        my_face = Face(vertex1=Vertex(0, 0, 0), vertex2=Vertex(0, 0, 0), vertex3=Vertex(1, 1, 1))
        assert my_face.area3d == 0

    def test_face_area2d(self):
        my_face = Face(Vertex(-2, 2, 5), Vertex(1, 6, 4), Vertex(6, 1, 3))
        #(-2,2), (1,5), and (6,1).
        result = my_face.area2d
        assert result == 17.5

    def test_face_area3d(self):
        my_face = Face(Vertex(-2, 2, 5), Vertex(1, 5, 4), Vertex(6, 1, 3))
        result = my_face.area3d
        assert result == 13.982131454109568
