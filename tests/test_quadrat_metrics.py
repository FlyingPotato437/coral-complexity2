import pytest
import os
from coral_complexity_metrics import QuadratMetrics


@pytest.fixture
def quadrat_metrics():
    return QuadratMetrics(dim="XYZ", size=1.0)


def test_load_quad_obj(quadrat_metrics):
    """
    Test the load_mesh method of the QuadratMetrics class with an `obj file.
    """
    sample_obj_file = 'tests/sample_data/colony_plot.obj'
    quadrat_metrics.load_mesh(sample_obj_file)
    assert quadrat_metrics.mesh_file.endswith('.obj')
    assert quadrat_metrics.mesh is not None


def test_load_quad_ply(quadrat_metrics):
    """
    Test the load_mesh method of the QuadratMetrics class with a .ply file.
    """
    sample_ply_file = 'tests/sample_data/colony_plot.ply'
    quadrat_metrics.load_mesh(sample_ply_file)
    assert quadrat_metrics.mesh_file.endswith('.obj')
    assert quadrat_metrics.mesh is not None


def test_load_quad_file_not_found(quadrat_metrics):
    """
    Test the load_mesh method of the QuadratMetrics class when the file does not exist.
    """
    sample_mesh = 'tests/sample_data/non_existent_mesh.obj'
    quadrat_metrics.load_mesh(sample_mesh)
    assert quadrat_metrics.mesh_file == sample_mesh
    assert quadrat_metrics.mesh == None


def test_quad_calculate_ply(quadrat_metrics):
    """
    Test the calculate method of the QuadratMetrics class using a .ply file.
    """
    sample_ply_file = 'tests/sample_data/colony_plot.ply'
    quadrat_metrics.load_mesh(sample_ply_file)
    results = quadrat_metrics.calculate()
    assert isinstance(results, list)
    assert len(results) == 1
    assert results[0]['mesh_name'] == sample_ply_file.replace('.ply', '.obj')
    assert results[0]['quadrat_size_m'] == 1
    assert results[0]['quadrat_rel_x'] == 0
    assert results[0]['quadrat_rel_y'] == 0
    assert results[0]['quadrat_rel_z_mean'] == -7.43032050037764
    assert results[0]['quadrat_rel_z_sd'] == 0.0555879143480548
    assert results[0]['quadrat_abs_x'] == -1.3303690552711487
    assert results[0]['quadrat_abs_y'] == -0.035120585933327675
    assert results[0]['quadrat_abs_z'] == 0.0
    assert results[0]['num_faces'] == 11117
    assert results[0]['num_vertices'] == 5773
    assert results[0]['3d_surface_area'] == 0.06191374971805021
    assert results[0]['2d_surface_area'] == 0.03885501641085011
    assert results[0]['surface_rugosity'] == 1.5934557603419526


def test_quad_calculate_obj(quadrat_metrics):
    """
    Test the calculate method of the QuadratMetrics class using an .obj file.
    """
    sample_obj_file = 'tests/sample_data/colony_plot.obj'
    quadrat_metrics.load_mesh(sample_obj_file)
    results = quadrat_metrics.calculate()
    assert isinstance(results, list)
    assert len(results) == 1
    assert results[0]['mesh_name'] == sample_obj_file
    assert results[0]['quadrat_size_m'] == 1
    assert results[0]['quadrat_rel_x'] == 0
    assert results[0]['quadrat_rel_y'] == 0
    assert results[0]['quadrat_rel_z_mean'] == -7.43032050037764
    assert results[0]['quadrat_rel_z_sd'] == 0.0555879143480548
    assert results[0]['quadrat_abs_x'] == -1.3303690552711487
    assert results[0]['quadrat_abs_y'] == -0.035120585933327675
    assert results[0]['quadrat_abs_z'] == 0.0
    assert results[0]['num_faces'] == 11117
    assert results[0]['num_vertices'] == 5773
    assert results[0]['3d_surface_area'] == 0.06191374971805021
    assert results[0]['2d_surface_area'] == 0.03885501641085011
    assert results[0]['surface_rugosity'] == 1.5934557603419526


def test_quad_process_directory(quadrat_metrics):
    """
    Test the process_directory method of the QuadratMetrics class.
    """
    directory = 'tests/sample_data'
    results = quadrat_metrics.process_directory(directory)
    assert len(results) == 2
    assert results[0]['quadrat_size_m'] == 1
    assert results[0]['quadrat_rel_x'] == 0
    assert results[0]['quadrat_rel_y'] == 0
    assert results[0]['quadrat_rel_z_mean'] == -7.43032050037764
    assert results[0]['quadrat_rel_z_sd'] == 0.0555879143480548
    assert results[0]['quadrat_abs_x'] == -1.3303690552711487
    assert results[0]['quadrat_abs_y'] == -0.035120585933327675
    assert results[0]['quadrat_abs_z'] == 0.0
    assert results[0]['num_faces'] == 11117
    assert results[0]['num_vertices'] == 5773
    assert results[0]['3d_surface_area'] == 0.06191374971805021
    assert results[0]['2d_surface_area'] == 0.03885501641085011
    assert results[0]['surface_rugosity'] == 1.5934557603419526
    assert results[1]['quadrat_size_m'] == 1
    assert results[1]['quadrat_rel_x'] == 0
    assert results[1]['quadrat_rel_y'] == 0
    assert results[1]['quadrat_rel_z_mean'] == -7.43032050037764
    assert results[1]['quadrat_rel_z_sd'] == 0.0555879143480548
    assert results[1]['quadrat_abs_x'] == -1.3303690552711487
    assert results[1]['quadrat_abs_y'] == -0.035120585933327675
    assert results[1]['quadrat_abs_z'] == 0.0
