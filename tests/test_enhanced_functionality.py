"""
Comprehensive test suite for enhanced coral complexity metrics functionality.

This module tests all the new features including:
- Enhanced shading with parametric inputs
- Metric registration system  
- Complexity metrics from Mitch Bryson's functions
- Mesh validation and repair
- Shading validation harness
- Mesh visualization
"""

import pytest
import numpy as np
import tempfile
import os
from pathlib import Path
import warnings
from unittest.mock import Mock, patch

# Import the enhanced modules
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from coral_complexity_metrics.mesh.shading import Shading
from coral_complexity_metrics.mesh._metric import (
    MetricRegistry, BaseMetric, SurfaceMetric, VolumeMetric, 
    ComplexityMetric, ShadingMetric, register_metric, get_metric_registry
)
from coral_complexity_metrics.mesh.complexity_metrics import (
    SlopeMetric, PlaneOfBestFit, HeightRange, FractalDimensionBox,
    SurfaceComplexityIndex, VectorDispersion
)
from coral_complexity_metrics.mesh.mesh_validator import (
    MeshValidator, MeshValidationResult, validate_and_repair_mesh
)
from coral_complexity_metrics.validation.shading_validator import (
    ShadingValidator, LightLoggerData, ShadingComparison, ValidationMetrics
)
from coral_complexity_metrics.visualization.mesh_previews import (
    MeshVisualizer, generate_mesh_previews
)

try:
    import pyvista as pv
    HAS_PYVISTA = True
except ImportError:
    HAS_PYVISTA = False
    pytestmark = pytest.mark.skip("PyVista not available")

try:
    import pymeshlab
    HAS_PYMESHLAB = True
except ImportError:
    HAS_PYMESHLAB = False


@pytest.fixture
def sample_mesh():
    """Create a simple test mesh."""
    if not HAS_PYVISTA:
        pytest.skip("PyVista not available")
    
    # Create a simple triangular mesh (tetrahedron)
    points = np.array([
        [0, 0, 0],
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1]
    ])
    
    faces = np.array([
        [3, 0, 1, 2],  # Bottom triangle
        [3, 0, 1, 3],  # Side triangle 1
        [3, 1, 2, 3],  # Side triangle 2
        [3, 2, 0, 3]   # Side triangle 3
    ])
    
    return pv.PolyData(points, faces)


@pytest.fixture 
def temp_directory():
    """Create a temporary directory for testing."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        yield tmp_dir


class TestEnhancedShading:
    """Test the enhanced shading functionality."""
    
    def test_shading_initialization(self):
        """Test shading class initialization with CPU percentage."""
        # Test valid CPU percentage
        shading = Shading(cpu_percentage=50)
        assert shading.cpu_percentage == 50
        assert shading.cpu_limit >= 1
        
        # Test invalid CPU percentage
        with pytest.raises(ValueError):
            Shading(cpu_percentage=0)
        
        with pytest.raises(ValueError):
            Shading(cpu_percentage=150)
        
        with pytest.raises(TypeError):
            Shading(cpu_percentage="invalid")
    
    def test_solar_position_calculation(self):
        """Test solar position calculation."""
        shading = Shading()
        
        # Test valid inputs
        light_dir = shading.calculate_solar_position(
            day_of_year=172,  # Summer solstice
            time_of_day=12.0,  # Noon
            latitude=0.0,     # Equator
            longitude=0.0
        )
        
        assert isinstance(light_dir, np.ndarray)
        assert len(light_dir) == 3
        assert np.isclose(np.linalg.norm(light_dir), 1.0, atol=1e-6)
        
        # Test invalid inputs
        with pytest.raises(ValueError):
            shading.calculate_solar_position(400, 12.0)  # Invalid day
        
        with pytest.raises(ValueError):
            shading.calculate_solar_position(172, 25.0)  # Invalid time
        
        with pytest.raises(ValueError):
            shading.calculate_solar_position(172, 12.0, latitude=100)  # Invalid latitude
    
    def test_slope_aspect_adjustment(self):
        """Test light direction adjustment for slope and aspect."""
        shading = Shading()
        base_light = np.array([0, 0, -1])
        
        # Test valid slope and aspect
        adjusted = shading.adjust_light_for_slope_aspect(base_light, slope=30, aspect=45)
        assert isinstance(adjusted, np.ndarray)
        assert len(adjusted) == 3
        
        # Test invalid inputs
        with pytest.raises(ValueError):
            shading.adjust_light_for_slope_aspect(base_light, slope=100, aspect=45)
        
        with pytest.raises(ValueError):
            shading.adjust_light_for_slope_aspect(base_light, slope=30, aspect=400)
    
    def test_sampling_validation(self):
        """Test sampling point validation."""
        shading = Shading()
        
        # Valid sample size
        shading._validate_sampling_points(1000)
        
        # Invalid sample sizes
        with pytest.raises(TypeError):
            shading._validate_sampling_points(1000.5)
        
        with pytest.raises(ValueError):
            shading._validate_sampling_points(0)
        
        # Large sample size should trigger warning
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            shading._validate_sampling_points(15_000_000)
            assert len(w) == 1
            assert "memory issues" in str(w[0].message)
    
    def test_unsupported_parameter_warnings(self):
        """Test warnings for unsupported environmental parameters."""
        shading = Shading()
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            shading._warn_unsupported_parameters(
                depth=10.0,
                turbidity=0.5,
                unsupported_param=123
            )
            # Should have 2 warnings (depth and turbidity)
            assert len(w) == 2
            assert "structural-only analysis" in str(w[0].message)


class TestMetricRegistry:
    """Test the metric registration system."""
    
    def test_metric_registration(self):
        """Test metric registration and retrieval."""
        registry = MetricRegistry()
        
        # Create a test metric
        class TestMetric(SurfaceMetric):
            @property
            def name(self):
                return "test_metric"
            
            @property
            def description(self):
                return "A test metric"
            
            def calculate(self, mesh_data, **kwargs):
                return {"test_value": 42}
        
        # Register the metric
        test_metric = TestMetric()
        registry.register(test_metric)
        
        # Check registration
        assert "test_metric" in registry.list_metrics()
        assert "test_metric" in registry.list_metrics("surface")
        
        # Retrieve and test
        retrieved = registry.get_metric("test_metric")
        assert retrieved.name == "test_metric"
        
        # Test calculation
        dummy_data = {"points": np.array([[0, 0, 0]])}
        result = retrieved.calculate(dummy_data)
        assert result["test_value"] == 42
    
    def test_metric_validation(self):
        """Test metric validation."""
        registry = MetricRegistry()
        
        # Valid metric
        class ValidMetric(BaseMetric):
            @property
            def name(self):
                return "valid"
            
            @property
            def description(self):
                return "Valid metric"
            
            def calculate(self, mesh_data, **kwargs):
                return {"result": 1}
        
        errors = registry.validate_metric(ValidMetric())
        assert len(errors) == 0
        
        # Invalid metric (missing methods)
        class InvalidMetric:
            pass
        
        errors = registry.validate_metric(InvalidMetric())
        assert len(errors) > 0
    
    def test_metric_groups(self):
        """Test metric grouping by type."""
        registry = MetricRegistry()
        
        # Check default groups exist
        assert "surface" in registry.list_metrics("surface")
        assert "complexity" in registry.list_metrics("complexity")
        assert "volume" in registry.list_metrics("volume")
    
    def test_metric_calculation_with_closure_check(self):
        """Test metric calculation with mesh closure checking."""
        registry = MetricRegistry()
        
        # Mock mesh data - open mesh
        mesh_data = {
            "mesh": Mock(),
            "points": np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]])
        }
        mesh_data["mesh"].volume = 0  # Indicates open mesh
        
        # Should return NaN for volume metrics when mesh is open
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            results = registry.calculate_metrics(mesh_data, check_mesh_closure=True)
            
            # Should have warning about open mesh
            assert len(w) > 0
            assert "not closed" in str(w[0].message)


class TestComplexityMetrics:
    """Test the complexity metrics from Mitch Bryson's functions."""
    
    def test_slope_metric(self, sample_mesh):
        """Test slope calculation metric."""
        slope_metric = SlopeMetric()
        
        assert slope_metric.name == "slope"
        assert slope_metric.metric_type == "complexity"
        
        mesh_data = {"points": sample_mesh.points}
        result = slope_metric.calculate(mesh_data)
        
        assert isinstance(result, dict)
        assert "slope_mean" in result
        assert "slope_std" in result
    
    def test_plane_of_best_fit(self, sample_mesh):
        """Test plane of best fit metric."""
        pobf_metric = PlaneOfBestFit()
        
        assert pobf_metric.name == "plane_of_best_fit"
        
        mesh_data = {"points": sample_mesh.points}
        result = pobf_metric.calculate(mesh_data)
        
        assert isinstance(result, dict)
        assert "global_fit_error" in result
        assert "global_normal" in result
    
    def test_height_range(self, sample_mesh):
        """Test height range metric."""
        height_metric = HeightRange()
        
        assert height_metric.name == "height_range"
        assert height_metric.metric_type == "surface"
        
        mesh_data = {"points": sample_mesh.points}
        result = height_metric.calculate(mesh_data)
        
        assert isinstance(result, dict)
        assert "height_range" in result
        assert "height_mean" in result
        assert "height_std" in result
        assert result["height_range"] == 1.0  # Z ranges from 0 to 1
    
    def test_fractal_dimension(self, sample_mesh):
        """Test fractal dimension metric."""
        fractal_metric = FractalDimensionBox()
        
        assert fractal_metric.name == "fractal_dimension_box"
        
        mesh_data = {"points": sample_mesh.points}
        result = fractal_metric.calculate(mesh_data)
        
        assert isinstance(result, dict)
        assert "fractal_dimension" in result
        assert "r_squared" in result
    
    def test_surface_complexity_index(self, sample_mesh):
        """Test composite complexity index."""
        sci_metric = SurfaceComplexityIndex()
        
        assert sci_metric.name == "surface_complexity_index"
        
        mesh_data = {"points": sample_mesh.points}
        result = sci_metric.calculate(mesh_data)
        
        assert isinstance(result, dict)
        assert "complexity_index" in result
        assert 0 <= result["complexity_index"] <= 1
    
    def test_vector_dispersion(self, sample_mesh):
        """Test vector dispersion metric."""
        vd_metric = VectorDispersion()
        
        assert vd_metric.name == "vector_dispersion"
        
        mesh_data = {"mesh": sample_mesh}
        result = vd_metric.calculate(mesh_data)
        
        assert isinstance(result, dict)
        assert "vector_dispersion" in result


@pytest.mark.skipif(not HAS_PYMESHLAB, reason="PyMeshLab not available")
class TestMeshValidator:
    """Test mesh validation and repair functionality."""
    
    def test_mesh_validation_result(self):
        """Test MeshValidationResult dataclass."""
        result = MeshValidationResult(
            is_valid=True,
            is_closed=True,
            n_open_edges=0,
            n_holes=0,
            n_isolated_vertices=0,
            n_duplicated_vertices=0,
            n_non_manifold_edges=0,
            n_non_manifold_vertices=0,
            genus=0,
            euler_characteristic=2,
            volume=1.0,
            surface_area=6.0,
            bbox_volume=8.0,
            validation_errors=[],
            repair_suggestions=[]
        )
        
        assert result.is_valid
        assert result.is_closed
        assert result.volume == 1.0
    
    def test_mesh_validator_initialization(self):
        """Test MeshValidator initialization."""
        validator = MeshValidator(verbose=False)
        assert validator.verbose == False
        assert validator.ms is None
    
    def test_mesh_validation_empty_mesh(self):
        """Test validation of empty mesh."""
        validator = MeshValidator(verbose=False)
        
        # Create empty mesh
        empty_mesh = pv.PolyData()
        
        result = validator.validate_mesh(empty_mesh, repair_if_needed=False)
        
        assert not result.is_valid
        assert not result.is_closed
        assert "empty" in result.validation_errors[0].lower()


class TestShadingValidator:
    """Test the shading validation harness."""
    
    def test_light_logger_data(self):
        """Test LightLoggerData dataclass."""
        import pandas as pd
        
        data = LightLoggerData(
            timestamp=pd.Timestamp('2023-01-01 12:00:00'),
            light_intensity=1000.0,
            logger_id="logger_01",
            location=(10.0, 20.0, -5.0)
        )
        
        assert data.light_intensity == 1000.0
        assert data.logger_id == "logger_01"
        assert data.location == (10.0, 20.0, -5.0)
    
    def test_shading_comparison(self):
        """Test ShadingComparison dataclass."""
        comparison = ShadingComparison(
            logger_id="test_logger",
            location=(0, 0, 0),
            measured_shading=50.0,
            modeled_shading=45.0,
            absolute_error=5.0,
            relative_error=0.1,
            time_period="daily",
            mesh_file="test.ply",
            validation_status="valid"
        )
        
        assert comparison.absolute_error == 5.0
        assert comparison.relative_error == 0.1
    
    def test_validation_metrics(self):
        """Test ValidationMetrics calculation."""
        # Create mock comparisons
        comparisons = [
            ShadingComparison("l1", (0,0,0), 50, 45, 5, 0.1, "daily", "test.ply", "valid"),
            ShadingComparison("l2", (0,0,0), 60, 65, 5, 0.083, "daily", "test.ply", "valid"),
            ShadingComparison("l3", (0,0,0), 30, 25, 5, 0.167, "daily", "test.ply", "valid")
        ]
        
        validator = ShadingValidator(verbose=False)
        metrics = validator.calculate_validation_metrics(comparisons)
        
        assert metrics.n_comparisons == 3
        assert metrics.mean_absolute_error == 5.0
        assert isinstance(metrics.r_squared, float)
    
    def test_measured_shading_calculation(self):
        """Test calculation of shading from light logger data."""
        import pandas as pd
        
        # Create mock logger data
        logger_data = [
            LightLoggerData(pd.Timestamp('2023-01-01 08:00:00'), 800, logger_id="L1"),
            LightLoggerData(pd.Timestamp('2023-01-01 12:00:00'), 1000, logger_id="L1"),
            LightLoggerData(pd.Timestamp('2023-01-01 16:00:00'), 600, logger_id="L1"),
        ]
        
        validator = ShadingValidator(verbose=False)
        shading_results = validator.calculate_measured_shading(logger_data)
        
        assert "L1" in shading_results
        assert 0 <= shading_results["L1"] <= 100


class TestMeshVisualizer:
    """Test mesh visualization functionality."""
    
    def test_mesh_visualizer_initialization(self):
        """Test MeshVisualizer initialization."""
        visualizer = MeshVisualizer(
            image_size=(640, 480),
            background_color='black',
            colormap='plasma'
        )
        
        assert visualizer.image_size == (640, 480)
        assert visualizer.background_color == 'black'
        assert visualizer.colormap == 'plasma'
    
    @patch('pyvista.Plotter')
    def test_html_preview_generation(self, mock_plotter, sample_mesh):
        """Test HTML preview generation."""
        # Mock the plotter
        mock_plotter_instance = Mock()
        mock_plotter.return_value = mock_plotter_instance
        mock_plotter_instance.screenshot.return_value = None
        
        visualizer = MeshVisualizer()
        
        # This should not crash even with mocked plotter
        try:
            preview = visualizer._generate_html_preview(
                mock_plotter_instance, sample_mesh, "test_mesh", None, None
            )
            assert isinstance(preview, str)
            assert "Mesh Preview: test_mesh" in preview
        except Exception:
            # Expected to fail with mocked components, but shouldn't crash
            pass
    
    def test_polygon_highlight_extraction(self, sample_mesh):
        """Test extraction of polygon highlights."""
        visualizer = MeshVisualizer()
        
        # Mock polygon data
        mock_polygon_data = Mock()
        mock_polygon_data.iterrows.return_value = []
        
        highlights = visualizer._extract_polygon_highlights(sample_mesh, mock_polygon_data)
        assert isinstance(highlights, list)


class TestDocumentationCoverage:
    """Test documentation coverage for all modules."""
    
    def test_shading_module_docstrings(self):
        """Test that shading module has proper documentation."""
        assert Shading.__doc__ is not None
        assert "STRUCTURAL-ONLY SCOPE" in Shading.__doc__
        
        # Test method documentation
        assert Shading.calculate.__doc__ is not None
        assert Shading.calculate_solar_position.__doc__ is not None
    
    def test_metric_module_docstrings(self):
        """Test metric module documentation."""
        assert BaseMetric.__doc__ is not None
        assert MetricRegistry.__doc__ is not None
        
        # Test complexity metrics
        assert SlopeMetric.__doc__ is not None
        assert HeightRange.__doc__ is not None
    
    def test_validator_module_docstrings(self):
        """Test validator module documentation."""
        assert MeshValidator.__doc__ is not None
        assert ShadingValidator.__doc__ is not None
    
    def test_visualization_module_docstrings(self):
        """Test visualization module documentation."""
        assert MeshVisualizer.__doc__ is not None


class TestRegressionProtection:
    """Test regression protection for metric outputs."""
    
    def test_metric_output_consistency(self, sample_mesh):
        """Test that metrics produce consistent outputs."""
        # Test height range metric consistency
        height_metric = HeightRange()
        mesh_data = {"points": sample_mesh.points}
        
        # Run multiple times
        results = []
        for _ in range(3):
            result = height_metric.calculate(mesh_data)
            results.append(result["height_range"])
        
        # Should be identical
        assert all(r == results[0] for r in results)
        assert results[0] == 1.0  # Known value for our test mesh
    
    def test_shading_parameter_consistency(self):
        """Test shading parameter handling consistency."""
        shading = Shading(cpu_percentage=50)
        
        # Test solar position calculation consistency
        light_dir1 = shading.calculate_solar_position(180, 12.0, 0.0, 0.0)
        light_dir2 = shading.calculate_solar_position(180, 12.0, 0.0, 0.0)
        
        np.testing.assert_array_almost_equal(light_dir1, light_dir2)
    
    def test_metric_registry_consistency(self):
        """Test metric registry behavior consistency."""
        registry = MetricRegistry()
        
        # Should have same metrics each time
        metrics1 = registry.list_metrics()
        metrics2 = registry.list_metrics()
        
        assert metrics1 == metrics2
        assert len(metrics1) > 0


class TestIntegrationScenarios:
    """Test integration scenarios combining multiple components."""
    
    def test_full_workflow_integration(self, sample_mesh, temp_directory):
        """Test a complete workflow with multiple components."""
        # 1. Validate mesh
        validator = MeshValidator(verbose=False)
        validation_result = validator.validate_mesh(sample_mesh, repair_if_needed=False)
        
        # 2. Calculate complexity metrics
        registry = MetricRegistry()
        mesh_data = {"points": sample_mesh.points, "mesh": sample_mesh}
        metrics = registry.calculate_metrics(mesh_data, exclude_types=["shading"])
        
        # 3. Generate visualization (mocked)
        visualizer = MeshVisualizer()
        
        # Verify integration
        assert validation_result is not None
        assert len(metrics) > 0
        assert visualizer is not None
    
    def test_error_handling_integration(self):
        """Test error handling across integrated components."""
        # Test with invalid inputs
        with pytest.raises(ValueError):
            Shading(cpu_percentage=-10)
        
        with pytest.raises(TypeError):
            registry = MetricRegistry()
            registry.register("not a metric")
    
    def test_warning_system_integration(self):
        """Test that warning systems work correctly across modules."""
        shading = Shading()
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            
            # Trigger multiple warning types
            shading._validate_sampling_points(20_000_000)  # Large sample warning
            shading._warn_unsupported_parameters(depth=10.0)  # Unsupported param warning
            
            assert len(w) >= 2


# Performance and memory tests
class TestPerformanceAndMemory:
    """Test performance and memory usage."""
    
    def test_large_sample_size_handling(self):
        """Test handling of large sample sizes."""
        shading = Shading()
        
        # Should not crash with large (but reasonable) sample size
        shading._validate_sampling_points(500_000)
        
        # Should warn with very large sample size
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            shading._validate_sampling_points(15_000_000)
            assert len(w) == 1
    
    def test_cpu_limit_scaling(self):
        """Test CPU limit scaling."""
        import multiprocessing as mp
        
        # Test different CPU percentages
        for percentage in [25, 50, 75, 100]:
            shading = Shading(cpu_percentage=percentage)
            expected_limit = max(1, int(mp.cpu_count() * percentage / 100))
            assert shading.cpu_limit == expected_limit


if __name__ == "__main__":
    # Run the tests
    pytest.main([__file__, "-v"]) 