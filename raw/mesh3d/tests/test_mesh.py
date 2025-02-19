import pytest


class TestMesh:
    pass

#     def test_mesh_contructor(mocker):
#         """Constructor test."""
#         vertices = list()
#         faces = list()
#         shapes_patch = mocker.patch('mesh3d.shapes')
#         vertex_path = shapes_patch.object(vertex,'__init__')
#
#
#         # faces_stub = mocker.stub(faces)
#         # mesh = Mesh(vertices_stub, faces_stub)
#         assert mesh is not None
#
#
# def test_stub(mocker):
#     def foo(on_something):
#         on_something('foo', 'bar')
#
#     stub = mocker.stub(name='on_something_stub')
#
#     foo(stub)
#     stub.assert_called_once_with('foo', 'bar')
