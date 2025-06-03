# Changelog

All notable changes to the EcoRRAP (Enhanced Coral Reef Complexity Metrics) package are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [2.0.0] - 2024-12-19

### Major Release - Complete Package Overhaul

This version represents a complete rewrite and enhancement of the coral complexity metrics package with significant architectural improvements and new capabilities.

### Added

#### Enhanced Shading Analysis
- **CPU Percentage Control**: Configure CPU usage from 1-100% for computational resource management
- **Environmental Parameters**: Support for slope, aspect, depth, and turbidity adjustments
- **Solar Position Calculation**: Accurate astronomical calculations for realistic lighting conditions
- **Modular Architecture**: Extensible framework with abstract base classes for lighting models, environmental factors, and shading metrics
- **Quality Control**: Comprehensive validation and warning systems for parameter ranges

#### Unified Metrics System
- **17+ Standardized Metrics**: Complete set of surface, volume, complexity, and shading metrics
- **Automatic Registration**: Dynamic metric discovery and categorization system
- **USYD Translations**: Python implementations of slope, plane of best fit, height range, and fractal dimension algorithms
- **Proper NaN Handling**: Volume-dependent metrics correctly return NaN for non-watertight meshes
- **Context Awareness**: Metrics work across whole meshes, quadrats, and cropped regions

#### Mesh-by-Shapefile Processing
- **Complete Pipeline**: End-to-end processing from shapefile polygons to CSV results
- **Data Quality Assessment**: Coverage percentage, missing data detection, and quality scoring (0-1 scale)
- **Batch Processing**: Handle multiple regions with comprehensive error handling
- **Mesh Output**: Option to save cropped mesh files for detailed analysis
- **Polygon Expansion**: Configurable boundary expansion for edge case handling

#### Robust Architecture
- **Optional Dependencies**: Graceful fallbacks when specialized libraries unavailable
- **Type Safety**: Full type hints throughout codebase
- **Error Handling**: Comprehensive validation with informative error messages
- **Memory Management**: Efficient processing of large mesh files
- **Cross-Platform**: Compatible with Windows, macOS, and Linux

#### Data Quality Features
- **Mesh Validation**: Comprehensive mesh integrity checking
- **Coverage Analysis**: Quantify mesh data coverage within analysis regions
- **Quality Scoring**: Automated quality assessment with flagging of problematic regions
- **Missing Data Tracking**: Precise quantification of data gaps

### Changed

#### Breaking Changes
- **Package Structure**: Complete reorganization into logical modules
- **API Interface**: Simplified and standardized function signatures
- **Metric Names**: Consistent naming convention across all metrics
- **Dependencies**: Moved from required to optional dependencies for core functionality
- **Configuration**: New parameter validation and defaults

#### Improvements
- **Performance**: Significant speed improvements through optimized algorithms
- **Memory Usage**: Reduced memory footprint for large mesh processing
- **Documentation**: Complete rewrite with professional examples and API documentation
- **Testing**: Comprehensive test suite with real mesh data validation
- **Code Quality**: Professional code style without emojis or unnecessary comments

### Removed

#### Deprecated Features
- **Legacy Interfaces**: Removed outdated function signatures
- **Hardcoded Parameters**: Replaced with configurable options
- **Redundant Metrics**: Consolidated overlapping functionality
- **Unnecessary Files**: Cleaned up temporary and test files

#### Dependencies
- **PyMeshLab**: No longer required for basic functionality
- **Fixed Versions**: Relaxed dependency version constraints

### Technical Details

#### Architecture Improvements
- **Modular Design**: Clear separation of concerns across modules
- **Abstract Base Classes**: Extensible framework for new metrics and environmental factors
- **Registry System**: Automatic metric discovery and metadata management
- **Optional Imports**: Graceful handling of missing dependencies

#### Data Processing
- **Mesh Utilities**: Enhanced spatial calculations and coverage analysis
- **Quality Metrics**: Comprehensive assessment of data completeness and reliability
- **Error Recovery**: Robust handling of edge cases and malformed data
- **Scalability**: Efficient processing of large datasets

#### Scientific Accuracy
- **Solar Calculations**: Precise astronomical algorithms for realistic lighting
- **Environmental Modeling**: Support for slope, aspect, and underwater conditions
- **Metric Validation**: Extensive validation against known test cases
- **Quality Control**: Automated detection of processing issues

### Migration Guide

#### For Existing Users
1. **Update Installation**: Use `pip install coral-complexity-metrics[full]` for complete functionality
2. **API Changes**: Review updated function signatures in documentation
3. **Configuration**: Update parameter names to new standardized format
4. **Dependencies**: Install optional dependencies as needed for specific features

#### New Features to Explore
- **Shapefile Processing**: Automated batch analysis of multiple regions
- **Data Quality Assessment**: Built-in validation and quality scoring
- **Enhanced Shading**: Environmental parameter support and CPU control
- **Unified Metrics**: Standardized interface for all complexity measurements

### Validation

This release has been extensively tested on:
- **Real Mesh Data**: Validated on 1.2GB TSMA coral reef mesh
- **Multiple Platforms**: Tested on Windows, macOS, and Linux
- **Dependency Scenarios**: Validated with various dependency combinations
- **Performance**: Benchmarked against previous versions
- **Scientific Accuracy**: Verified against established complexity metrics

### Acknowledgments

- **Australian Institute of Marine Science (AIMS)**: Core development and testing
- **University of Sydney**: Complexity metric algorithms and validation
- **Original Contributors**: Srikanth Samy, Eoghan Aston, Steven Hawes
- **Scientific Community**: Feedback and validation on metric implementations

### Links

- **Documentation**: https://coral-complexity-metrics.readthedocs.io
- **GitHub Repository**: https://github.com/open-AIMS/coral-complexity-metrics
- **Issues**: https://github.com/open-AIMS/coral-complexity-metrics/issues
- **PyPI**: https://pypi.org/project/coral-complexity-metrics/

---

## [1.x.x] - Previous Versions

Previous versions were development releases with limited functionality. Version 2.0.0 represents the first production-ready release with comprehensive features and documentation.