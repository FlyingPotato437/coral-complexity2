import pytest
from coral_complexity_metrics import GeometricMeasures


@pytest.fixture
def geometric_measures():
    return GeometricMeasures()


def test_load_geo_obj(geometric_measures):
    """ 
    Test the load_mesh method of the GeometricMeasures class with an `obj file.
    """
    sample_mesh = 'tests/sample_data/colony_plot.obj'

    geometric_measures.load_mesh(sample_mesh)
    assert geometric_measures.mesh_file == sample_mesh
    assert geometric_measures.mesh is not None


def test_load_geo_ply(geometric_measures):
    """
    Test the load_mesh method of the GeometricMeasures class with a .ply file.
    """
    sample_mesh = 'tests/sample_data/colony_plot.ply'

    geometric_measures.load_mesh(sample_mesh)
    assert geometric_measures.mesh_file == sample_mesh
    assert geometric_measures.mesh is not None


def test_load_geo_file_not_found(geometric_measures):
    """
    Test the load_mesh method of the GeometricMeasures class when the file does not exist.
    """
    sample_mesh = 'tests/sample_data/non_existent_mesh.obj'
    geometric_measures.load_mesh(sample_mesh)
    assert geometric_measures.mesh_file == sample_mesh
    assert geometric_measures.mesh == None


def test_calculate_geo_obj(geometric_measures):
    """
    Test the calculate method of the GeometricMeasures class using an .obj file.
    """
    sample_mesh = 'tests/sample_data/colony_plot.obj'
    geometric_measures.load_mesh(sample_mesh)
    result = geometric_measures.calculate()
    assert result['mesh_file'] == sample_mesh
    assert result['volume'] == -4.905071485827925e-05
    assert result['CVH_volume'] == 0.0036832554685674347
    assert result['ASR'] == 0.0037323061834257138
    assert result['proportion_occupied'] == -0.013317217683343869
    assert result['surface_area'] == 0.06191375106573105
    assert result['SSF'] == 0.060282346315332955
    assert result['diameter'] == 0.1614149808883667
    assert result['height'] == 0.2713489532470703


def test_calculate_geo_ply(geometric_measures):
    """
    Test the calculate method of the GeometricMeasures class using a .ply file.
    """
    sample_mesh = 'tests/sample_data/colony_plot.ply'
    geometric_measures.load_mesh(sample_mesh)
    result = geometric_measures.calculate()
    assert result['mesh_file'] == sample_mesh
    assert result['volume'] == -4.905071485827925e-05
    assert result['CVH_volume'] == 0.0036832554685674347
    assert result['ASR'] == 0.0037323061834257138
    assert result['proportion_occupied'] == -0.013317217683343869
    assert result['surface_area'] == 0.06191375106573105
    assert result['SSF'] == 0.060282346315332955
    assert result['diameter'] == 0.1614149808883667
    assert result['height'] == 0.2713489532470703


def test_geo_process_directory(geometric_measures):
    """
    Test the process_directory method of the GeometricMeasures class.
    """
    sample_directory = 'tests/sample_data'
    results = geometric_measures.process_directory(sample_directory)
    assert isinstance(results, list)
    assert len(results) == 2
    assert results[0]['volume'] == -4.905071485827925e-05
    assert results[0]['CVH_volume'] == 0.0036832554685674347
    assert results[0]['ASR'] == 0.0037323061834257138
    assert results[0]['proportion_occupied'] == -0.013317217683343869
    assert results[0]['surface_area'] == 0.06191375106573105
    assert results[0]['SSF'] == 0.060282346315332955
    assert results[0]['diameter'] == 0.1614149808883667
    assert results[0]['height'] == 0.2713489532470703
    assert results[1]['volume'] == -4.905071485827925e-05
    assert results[1]['CVH_volume'] == 0.0036832554685674347
    assert results[1]['ASR'] == 0.0037323061834257138
    assert results[1]['proportion_occupied'] == -0.013317217683343869
    assert results[1]['surface_area'] == 0.06191375106573105
    assert results[1]['SSF'] == 0.060282346315332955
    assert results[1]['diameter'] == 0.1614149808883667
    assert results[1]['height'] == 0.2713489532470703
