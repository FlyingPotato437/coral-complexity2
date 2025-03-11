from ._mesh import Mesh
from ._face import Face


def read_obj(file, verbose, order):
    vertices = []
    faces = []
    is_zero_vn = False
    contains_normal_vertex = False
    contains_texture_vertex = False
    if verbose:
        print("Reading mesh file: " + file)
    file = open(file, "r")
    for line in file:
        instructions = line.rstrip().split()
        if len(instructions) > 0:
            if instructions[0] == "v" and is_zero_vn is False:
                vertices.append(_create_vertex(instructions, order))
            elif instructions[0] == "f":
                faces.append(_create_face(instructions,
                                          vertices,
                                          contains_normal_vertex,
                                          contains_texture_vertex))
            elif instructions[0] == "vn":
                contains_normal_vertex = True
                if (float(instructions[1]) == 0.0):
                    is_zero_vn = True
                else:
                    is_zero_vn = False
            elif instructions[0] == "vt":
                contains_texture_vertex = True
            else:
                pass

    mesh = Mesh(faces, file.name)
    if verbose is True:
        print("Vertices: " + str(len(vertices)) +
              ", Faces: " + str(len(faces)))

    return mesh


def _create_vertex(instructions, order):
    return order.get_vertex(float(instructions[1]),
                            float(instructions[2]),
                            float(instructions[3]))


def _create_face(instructions, vertices, contains_normal_vertex,
                 contains_texture_vertex):
    """Create a face using the instructions (removes 1 for the index)"""
    vertex1_index = 0
    vertex2_index = 0
    vertex3_index = 0
    if contains_normal_vertex is True and contains_texture_vertex is True:
        vertex1_index = instructions[1].split("/")[0]
        vertex2_index = instructions[2].split("/")[0]
        vertex3_index = instructions[3].split("/")[0]
    elif contains_normal_vertex is True and contains_texture_vertex is False:
        vertex1_index = instructions[1].split("//")[0]
        vertex2_index = instructions[2].split("//")[0]
        vertex3_index = instructions[3].split("//")[0]
    elif contains_normal_vertex is False and contains_texture_vertex is True:
        vertex1_index = instructions[1].split("/")[0]
        vertex2_index = instructions[2].split("/")[0]
        vertex3_index = instructions[3].split("/")[0]
    else:
        vertex1_index = instructions[1]
        vertex2_index = instructions[2]
        vertex3_index = instructions[3]
    face_recipe = (int(vertex1_index), int(vertex2_index), int(vertex3_index))
    face = Face(vertices[face_recipe[0] - 1], vertices[face_recipe[1] - 1],
                vertices[face_recipe[2] - 1])
    return face
