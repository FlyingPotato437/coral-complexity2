"""
Shading validation and comparison harness for coral complexity analysis.

This module provides tools to compare structural shading estimates with in-situ
light logger data and generate quality control metrics.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
from datetime import datetime, timedelta
import warnings
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy import stats
import pyvista as pv

from ..mesh.shading import Shading
from ..mesh.mesh_validator import MeshValidator


@dataclass
class LightLoggerData:
    """Container for light logger measurement data."""
    timestamp: pd.Timestamp
    light_intensity: float  # μmol photons m⁻² s⁻¹
    depth: Optional[float] = None
    temperature: Optional[float] = None
    location: Optional[Tuple[float, float, float]] = None  # x, y, z coordinates
    logger_id: Optional[str] = None
    quality_flag: Optional[str] = None


@dataclass 
class ShadingComparison:
    """Result of comparing model estimates with logger data."""
    logger_id: str
    location: Tuple[float, float, float]
    measured_shading: float  # % shaded from light reduction
    modeled_shading: float   # % shaded from structural analysis
    absolute_error: float
    relative_error: float
    time_period: str
    mesh_file: str
    validation_status: str


@dataclass
class ValidationMetrics:
    """Aggregate validation metrics for a comparison study."""
    n_comparisons: int
    mean_absolute_error: float
    root_mean_square_error: float
    r_squared: float
    correlation_coefficient: float
    bias: float
    precision: float
    accuracy_within_10_percent: float
    accuracy_within_20_percent: float
    outlier_count: int
    outlier_threshold: float


class ShadingValidator:
    """Validation harness for comparing shading estimates with light logger data."""
    
    def __init__(self, verbose: bool = True):
        """
        Initialize the shading validator.
        
        Parameters:
        verbose: Whether to print validation progress
        """
        self.verbose = verbose
        self.shading_calculator = Shading(cpu_percentage=80.0)
        self.mesh_validator = MeshValidator(verbose=False)
        
    def load_light_logger_data(self, data_path: Union[str, Path], 
                              file_format: str = 'csv') -> List[LightLoggerData]:
        """
        Load light logger data from various formats.
        
        Parameters:
        data_path: Path to data file
        file_format: Format of data file ('csv', 'xlsx', 'netcdf')
        
        Returns:
        List of LightLoggerData objects
        """
        if self.verbose:
            print(f"Loading light logger data from: {data_path}")
        
        data_path = Path(data_path)
        logger_data = []
        
        if file_format.lower() == 'csv':
            df = pd.read_csv(data_path)
        elif file_format.lower() == 'xlsx':
            df = pd.read_excel(data_path)
        else:
            raise ValueError(f"Unsupported file format: {file_format}")
        
        # Expected columns: timestamp, light_intensity, logger_id, x, y, z
        required_cols = ['timestamp', 'light_intensity', 'logger_id']
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"Required column '{col}' not found in data")
        
        for _, row in df.iterrows():
            # Parse timestamp
            if isinstance(row['timestamp'], str):
                timestamp = pd.to_datetime(row['timestamp'])
            else:
                timestamp = row['timestamp']
            
            # Parse location if available
            location = None
            if all(col in df.columns for col in ['x', 'y', 'z']):
                location = (float(row['x']), float(row['y']), float(row['z']))
            
            logger_data.append(LightLoggerData(
                timestamp=timestamp,
                light_intensity=float(row['light_intensity']),
                depth=row.get('depth'),
                temperature=row.get('temperature'),
                location=location,
                logger_id=str(row['logger_id']),
                quality_flag=row.get('quality_flag')
            ))
        
        if self.verbose:
            print(f"Loaded {len(logger_data)} light measurements")
        
        return logger_data
    
    def calculate_measured_shading(self, logger_data: List[LightLoggerData],
                                 reference_depth: Optional[float] = None,
                                 time_window: str = 'daily') -> Dict[str, float]:
        """
        Calculate shading percentages from light logger measurements.
        
        Parameters:
        logger_data: List of light measurements
        reference_depth: Reference depth for normalization
        time_window: Time aggregation window ('hourly', 'daily', 'weekly')
        
        Returns:
        Dictionary mapping logger_id to shading percentage
        """
        if self.verbose:
            print("Calculating measured shading from light data...")
        
        # Convert to DataFrame for easier manipulation
        df = pd.DataFrame([
            {
                'timestamp': data.timestamp,
                'light_intensity': data.light_intensity,
                'logger_id': data.logger_id,
                'depth': data.depth,
                'location': data.location
            }
            for data in logger_data
        ])
        
        # Group by logger and time window
        df['date'] = df['timestamp'].dt.date
        
        if time_window == 'hourly':
            df['time_group'] = df['timestamp'].dt.floor('H')
        elif time_window == 'daily':
            df['time_group'] = df['timestamp'].dt.date
        elif time_window == 'weekly':
            df['time_group'] = df['timestamp'].dt.to_period('W')
        
        # Calculate shading for each logger
        shading_results = {}
        
        for logger_id in df['logger_id'].unique():
            logger_df = df[df['logger_id'] == logger_id].copy()
            
            # Find reference (unshaded) light intensity
            # Use maximum observed intensity as reference
            max_intensity = logger_df['light_intensity'].max()
            
            # Calculate average intensity for the time period
            avg_intensity = logger_df['light_intensity'].mean()
            
            # Calculate shading percentage
            if max_intensity > 0:
                shading_pct = ((max_intensity - avg_intensity) / max_intensity) * 100
                shading_results[logger_id] = max(0, min(100, shading_pct))
            else:
                shading_results[logger_id] = 0
        
        if self.verbose:
            print(f"Calculated shading for {len(shading_results)} loggers")
        
        return shading_results
    
    def run_shading_comparison(self, 
                             mesh_file: str,
                             logger_data: List[LightLoggerData],
                             comparison_scales: List[str] = ['plot', 'local'],
                             local_window_size: float = 2.0,
                             **shading_kwargs) -> List[ShadingComparison]:
        """
        Compare structural shading estimates with light logger measurements.
        
        Parameters:
        mesh_file: Path to 3D mesh file
        logger_data: List of light measurements
        comparison_scales: Scales for comparison ('plot', 'local')
        local_window_size: Size of local analysis window (m)
        **shading_kwargs: Additional arguments for shading calculation
        
        Returns:
        List of ShadingComparison objects
        """
        if self.verbose:
            print(f"Running shading comparison for: {mesh_file}")
        
        # Load and validate mesh
        mesh = pv.read(mesh_file)
        validation_result = self.mesh_validator.validate_mesh(mesh, repair_if_needed=True)
        
        if not (validation_result.is_valid or validation_result.is_closed):
            warnings.warn(f"Mesh validation failed for {mesh_file}")
        
        # Load mesh into shading calculator
        self.shading_calculator.load_mesh(mesh_file, verbose=False)
        
        # Calculate measured shading from logger data
        measured_shading = self.calculate_measured_shading(logger_data)
        
        comparisons = []
        
        for scale in comparison_scales:
            if self.verbose:
                print(f"Calculating shading at {scale} scale...")
            
            if scale == 'plot':
                # Full plot analysis
                result = self.shading_calculator.calculate(
                    verbose=False,
                    **shading_kwargs
                )
                plot_shading = result['shaded_percentage']
                
                # Compare with all loggers
                for logger_id, measured_pct in measured_shading.items():
                    comparison = ShadingComparison(
                        logger_id=logger_id,
                        location=(0, 0, 0),  # Plot center
                        measured_shading=measured_pct,
                        modeled_shading=plot_shading,
                        absolute_error=abs(plot_shading - measured_pct),
                        relative_error=abs(plot_shading - measured_pct) / max(measured_pct, 1e-6),
                        time_period='aggregate',
                        mesh_file=mesh_file,
                        validation_status='valid' if validation_result.is_valid else 'warning'
                    )
                    comparisons.append(comparison)
            
            elif scale == 'local':
                # Local analysis around each logger position
                for data in logger_data:
                    if data.location is None:
                        continue
                    
                    logger_id = data.logger_id
                    if logger_id not in measured_shading:
                        continue
                    
                    # Create local window around logger position
                    center = np.array(data.location)
                    window = np.array([local_window_size, local_window_size, local_window_size])
                    
                    try:
                        result = self.shading_calculator.calculate(
                            point_of_interest=center,
                            window_size=window,
                            verbose=False,
                            **shading_kwargs
                        )
                        local_shading = result['shaded_percentage']
                        
                        comparison = ShadingComparison(
                            logger_id=logger_id,
                            location=data.location,
                            measured_shading=measured_shading[logger_id],
                            modeled_shading=local_shading,
                            absolute_error=abs(local_shading - measured_shading[logger_id]),
                            relative_error=abs(local_shading - measured_shading[logger_id]) / max(measured_shading[logger_id], 1e-6),
                            time_period='local',
                            mesh_file=mesh_file,
                            validation_status='valid' if validation_result.is_valid else 'warning'
                        )
                        comparisons.append(comparison)
                        
                    except Exception as e:
                        if self.verbose:
                            print(f"Error calculating local shading for logger {logger_id}: {e}")
                        continue
        
        if self.verbose:
            print(f"Generated {len(comparisons)} shading comparisons")
        
        return comparisons
    
    def calculate_validation_metrics(self, comparisons: List[ShadingComparison],
                                   outlier_threshold: float = 2.0) -> ValidationMetrics:
        """
        Calculate aggregate validation metrics from comparisons.
        
        Parameters:
        comparisons: List of shading comparisons
        outlier_threshold: Z-score threshold for outlier detection
        
        Returns:
        ValidationMetrics object
        """
        if len(comparisons) == 0:
            raise ValueError("No comparisons provided")
        
        measured = np.array([c.measured_shading for c in comparisons])
        modeled = np.array([c.modeled_shading for c in comparisons])
        errors = np.array([c.absolute_error for c in comparisons])
        
        # Basic metrics
        mae = mean_absolute_error(measured, modeled)
        rmse = np.sqrt(mean_squared_error(measured, modeled))
        r2 = r2_score(measured, modeled)
        correlation = stats.pearsonr(measured, modeled)[0]
        
        # Bias and precision
        bias = np.mean(modeled - measured)
        precision = np.std(modeled - measured)
        
        # Accuracy within thresholds
        accuracy_10 = np.mean(errors <= 10) * 100
        accuracy_20 = np.mean(errors <= 20) * 100
        
        # Outlier detection
        z_scores = np.abs(stats.zscore(errors))
        outliers = np.sum(z_scores > outlier_threshold)
        
        return ValidationMetrics(
            n_comparisons=len(comparisons),
            mean_absolute_error=mae,
            root_mean_square_error=rmse,
            r_squared=r2,
            correlation_coefficient=correlation,
            bias=bias,
            precision=precision,
            accuracy_within_10_percent=accuracy_10,
            accuracy_within_20_percent=accuracy_20,
            outlier_count=outliers,
            outlier_threshold=outlier_threshold
        )
    
    def generate_validation_report(self, comparisons: List[ShadingComparison],
                                 metrics: ValidationMetrics,
                                 output_dir: Optional[str] = None,
                                 create_plots: bool = True) -> str:
        """
        Generate a comprehensive validation report.
        
        Parameters:
        comparisons: List of shading comparisons
        metrics: Validation metrics
        output_dir: Directory to save report and plots
        create_plots: Whether to create visualization plots
        
        Returns:
        Report text
        """
        report_lines = [
            "SHADING VALIDATION REPORT",
            "=" * 50,
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "SUMMARY STATISTICS",
            "-" * 20,
            f"Number of comparisons: {metrics.n_comparisons}",
            f"Mean Absolute Error: {metrics.mean_absolute_error:.2f}%",
            f"Root Mean Square Error: {metrics.root_mean_square_error:.2f}%",
            f"R-squared: {metrics.r_squared:.3f}",
            f"Correlation Coefficient: {metrics.correlation_coefficient:.3f}",
            "",
            "ACCURACY ASSESSMENT",
            "-" * 20,
            f"Bias: {metrics.bias:.2f}%",
            f"Precision: {metrics.precision:.2f}%",
            f"Accuracy within 10%: {metrics.accuracy_within_10_percent:.1f}%",
            f"Accuracy within 20%: {metrics.accuracy_within_20_percent:.1f}%",
            f"Outliers (Z > {metrics.outlier_threshold}): {metrics.outlier_count}",
            "",
        ]
        
        # Add detailed comparison results
        report_lines.extend([
            "DETAILED COMPARISONS",
            "-" * 20,
            "Logger ID | Measured | Modeled | Error | Status"
        ])
        
        for comp in comparisons[:20]:  # Show first 20
            report_lines.append(
                f"{comp.logger_id:>9} | {comp.measured_shading:>8.1f} | "
                f"{comp.modeled_shading:>7.1f} | {comp.absolute_error:>5.1f} | "
                f"{comp.validation_status}"
            )
        
        if len(comparisons) > 20:
            report_lines.append(f"... and {len(comparisons) - 20} more")
        
        report_text = "\n".join(report_lines)
        
        # Save report and plots if output directory specified
        if output_dir:
            output_path = Path(output_dir)
            output_path.mkdir(exist_ok=True)
            
            # Save report text
            report_file = output_path / "shading_validation_report.txt"
            with open(report_file, 'w') as f:
                f.write(report_text)
            
            # Create validation plots
            if create_plots:
                self._create_validation_plots(comparisons, metrics, output_path)
        
        return report_text
    
    def _create_validation_plots(self, comparisons: List[ShadingComparison],
                               metrics: ValidationMetrics,
                               output_dir: Path) -> None:
        """Create validation visualization plots."""
        measured = [c.measured_shading for c in comparisons]
        modeled = [c.modeled_shading for c in comparisons]
        errors = [c.absolute_error for c in comparisons]
        
        # Set up the plotting style
        plt.style.use('seaborn-v0_8')
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # 1. Scatter plot: Measured vs Modeled
        ax1 = axes[0, 0]
        ax1.scatter(measured, modeled, alpha=0.6)
        ax1.plot([0, 100], [0, 100], 'r--', label='1:1 line')
        ax1.set_xlabel('Measured Shading (%)')
        ax1.set_ylabel('Modeled Shading (%)')
        ax1.set_title(f'Measured vs Modeled (R² = {metrics.r_squared:.3f})')
        ax1.legend()
        ax1.grid(True)
        
        # 2. Residual plot
        ax2 = axes[0, 1]
        residuals = np.array(modeled) - np.array(measured)
        ax2.scatter(measured, residuals, alpha=0.6)
        ax2.axhline(y=0, color='r', linestyle='--')
        ax2.set_xlabel('Measured Shading (%)')
        ax2.set_ylabel('Residuals (%)')
        ax2.set_title(f'Residual Plot (Bias = {metrics.bias:.2f}%)')
        ax2.grid(True)
        
        # 3. Error distribution
        ax3 = axes[1, 0]
        ax3.hist(errors, bins=20, alpha=0.7, edgecolor='black')
        ax3.axvline(x=metrics.mean_absolute_error, color='r', linestyle='--', 
                   label=f'MAE = {metrics.mean_absolute_error:.1f}%')
        ax3.set_xlabel('Absolute Error (%)')
        ax3.set_ylabel('Frequency')
        ax3.set_title('Error Distribution')
        ax3.legend()
        ax3.grid(True)
        
        # 4. Summary statistics text
        ax4 = axes[1, 1]
        ax4.axis('off')
        stats_text = f"""
        Validation Summary
        
        n = {metrics.n_comparisons}
        MAE = {metrics.mean_absolute_error:.2f}%
        RMSE = {metrics.root_mean_square_error:.2f}%
        R² = {metrics.r_squared:.3f}
        r = {metrics.correlation_coefficient:.3f}
        
        Accuracy:
        Within 10%: {metrics.accuracy_within_10_percent:.1f}%
        Within 20%: {metrics.accuracy_within_20_percent:.1f}%
        
        Outliers: {metrics.outlier_count}
        """
        ax4.text(0.1, 0.9, stats_text, transform=ax4.transAxes, 
                fontsize=12, verticalalignment='top', fontfamily='monospace')
        
        plt.tight_layout()
        plt.savefig(output_dir / 'shading_validation_plots.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Additional plot: Time series if temporal data available
        if any(hasattr(c, 'timestamp') for c in comparisons):
            fig, ax = plt.subplots(figsize=(12, 6))
            
            # This would require timestamps in the comparison data
            # For now, just create a simple comparison by logger ID
            logger_ids = [c.logger_id for c in comparisons]
            ax.plot(logger_ids, measured, 'o-', label='Measured', alpha=0.7)
            ax.plot(logger_ids, modeled, 's-', label='Modeled', alpha=0.7)
            ax.set_xlabel('Logger ID')
            ax.set_ylabel('Shading (%)')
            ax.set_title('Measured vs Modeled Shading by Logger')
            ax.legend()
            ax.grid(True)
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(output_dir / 'shading_comparison_by_logger.png', dpi=300, bbox_inches='tight')
            plt.close()


def run_validation_study(mesh_files: List[str],
                        logger_data_path: str,
                        output_dir: str,
                        comparison_scales: List[str] = ['plot', 'local'],
                        **kwargs) -> Dict[str, ValidationMetrics]:
    """
    Run a complete validation study comparing multiple meshes with logger data.
    
    Parameters:
    mesh_files: List of paths to mesh files
    logger_data_path: Path to light logger data file
    output_dir: Output directory for reports
    comparison_scales: Scales for comparison
    **kwargs: Additional arguments for shading calculation
    
    Returns:
    Dictionary mapping mesh files to validation metrics
    """
    validator = ShadingValidator(verbose=True)
    
    # Load logger data
    logger_data = validator.load_light_logger_data(logger_data_path)
    
    all_results = {}
    
    for mesh_file in mesh_files:
        print(f"\nProcessing: {mesh_file}")
        
        try:
            # Run comparison
            comparisons = validator.run_shading_comparison(
                mesh_file, logger_data, comparison_scales, **kwargs
            )
            
            if comparisons:
                # Calculate metrics
                metrics = validator.calculate_validation_metrics(comparisons)
                
                # Generate report
                mesh_name = Path(mesh_file).stem
                mesh_output_dir = Path(output_dir) / mesh_name
                mesh_output_dir.mkdir(parents=True, exist_ok=True)
                
                report = validator.generate_validation_report(
                    comparisons, metrics, str(mesh_output_dir)
                )
                
                all_results[mesh_file] = metrics
                
                print(f"Validation complete. MAE: {metrics.mean_absolute_error:.2f}%")
            else:
                print("No valid comparisons generated")
                
        except Exception as e:
            print(f"Error processing {mesh_file}: {e}")
            continue
    
    return all_results 