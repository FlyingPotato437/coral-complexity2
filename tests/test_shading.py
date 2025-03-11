import pytest
from coral_complexity_metrics import Shading


@pytest.fixture
def shading():
    return Shading()


def test_load_shading_obj(shading):
    """ 
    Test the load_mesh method of the Shading class.
    """
    sample_plot = 'tests/sample_data/colony_plot.obj'

    shading.load_mesh(sample_plot)
    assert shading.mesh_file == sample_plot
    assert shading.mesh.n_points == 5773
    assert shading.mesh.n_cells == 11117


def test_load_shading_ply(shading):
    """ 
    Test the load_mesh method of the Shading class.
    """
    sample_plot = 'tests/sample_data/colony_plot.ply'

    shading.load_mesh(sample_plot)
    assert shading.mesh_file == sample_plot
    assert shading.mesh.n_points == 5773
    assert shading.mesh.n_cells == 11117


def test_load_shading_file_not_found(shading):
    """
    Test the load_mesh method of the Shading class when the file does not exist.
    """
    sample_plot = 'tests/sample_data/non_existent_plot.obj'
    shading.load_mesh(sample_plot)
    assert shading.mesh_file == sample_plot
    assert shading.mesh is None


def test_calculate_shading_obj(shading):
    """
    Test the calculate method of the Shading class using an .obj file.
    """
    sample_plot = 'tests/sample_data/colony_plot.obj'
    shading.load_mesh(sample_plot)
    result = shading.calculate()
    assert result['mesh_file'] == sample_plot
    assert result['shaded_percentage'] == '29.48%'
    assert result['illuminated_percentage'] == '70.52%'


def test_calculate_shading_ply(shading):
    """
    Test the calculate method of the Shading class using a .ply file.
    """
    sample_plot = 'tests/sample_data/colony_plot.ply'
    shading.load_mesh(sample_plot)
    result = shading.calculate()
    assert result['mesh_file'] == sample_plot
    assert result['shaded_percentage'] == '29.48%'
    assert result['illuminated_percentage'] == '70.52%'


def test_shading_process_directory(shading):
    """
    Test the process_directory method of the Shading class.
    """
    directory = 'tests/sample_data'
    results = shading.process_directory(directory)
    assert len(results) == 2
    assert results[0]['shaded_percentage'] == '29.48%'
    assert results[0]['illuminated_percentage'] == '70.52%'
    assert results[1]['shaded_percentage'] == '29.48%'
    assert results[1]['illuminated_percentage'] == '70.52%'
