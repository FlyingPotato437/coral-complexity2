"""Validation tools for coral complexity metrics."""

# Optional imports that require additional dependencies
try:
    from .shading_validator import (
        ShadingValidator,
        LightLoggerData,
        ShadingComparison,
        ValidationMetrics,
        run_validation_study
    )
    _VALIDATION_AVAILABLE = True
except ImportError:
    # Create placeholders for missing functionality
    def _validation_not_available(*args, **kwargs):
        raise ImportError("Validation functionality requires additional dependencies. Install with: pip install coral-complexity-metrics[full]")
    
    ShadingValidator = _validation_not_available
    LightLoggerData = _validation_not_available
    ShadingComparison = _validation_not_available
    ValidationMetrics = _validation_not_available
    run_validation_study = _validation_not_available
    _VALIDATION_AVAILABLE = False

__all__ = [
    'ShadingValidator',
    'LightLoggerData', 
    'ShadingComparison',
    'ValidationMetrics',
    'run_validation_study'
] 