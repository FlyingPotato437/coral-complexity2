"""
Mesh visualization and preview generation for coral complexity analysis.

This module provides tools to generate HTML and PNG previews of mesh data,
highlighting focal colonies and validation status.
"""

import numpy as np
import pyvista as pv
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple
import warnings
import base64
from io import BytesIO
import json
from datetime import datetime

from ..mesh.mesh_validator import MeshValidator, MeshValidationResult


class MeshVisualizer:
    """Generate visualization previews for coral mesh data."""
    
    def __init__(self, 
                 image_size: Tuple[int, int] = (800, 600),
                 background_color: str = 'white',
                 colormap: str = 'viridis'):
        """
        Initialize the mesh visualizer.
        
        Parameters:
        image_size: Width and height for generated images
        background_color: Background color for plots
        colormap: Colormap for scalar data visualization
        """
        self.image_size = image_size
        self.background_color = background_color
        self.colormap = colormap
        self.validator = MeshValidator(verbose=False)
        
    def generate_mesh_preview(self, 
                            mesh: pv.PolyData,
                            mesh_id: str,
                            highlight_regions: Optional[List[Dict]] = None,
                            validation_result: Optional[MeshValidationResult] = None,
                            output_format: str = 'html',
                            camera_position: Optional[str] = 'isometric') -> str:
        """
        Generate a preview visualization of a mesh.
        
        Parameters:
        mesh: PyVista mesh to visualize
        mesh_id: Identifier for the mesh
        highlight_regions: List of regions to highlight with colors
        validation_result: Mesh validation results to display
        output_format: Output format ('html', 'png', 'both')
        camera_position: Camera angle ('isometric', 'top', 'side', 'front')
        
        Returns:
        Path to generated preview file(s)
        """
        # Create plotter
        plotter = pv.Plotter(off_screen=True, window_size=self.image_size)
        plotter.background_color = self.background_color
        
        # Determine mesh color based on validation status
        if validation_result:
            if validation_result.is_valid:
                mesh_color = 'lightblue'
                mesh_opacity = 0.8
            elif validation_result.is_closed:
                mesh_color = 'orange'
                mesh_opacity = 0.7
            else:
                mesh_color = 'red'
                mesh_opacity = 0.6
        else:
            mesh_color = 'lightgray'
            mesh_opacity = 0.8
        
        # Add main mesh
        plotter.add_mesh(
            mesh,
            color=mesh_color,
            opacity=mesh_opacity,
            show_edges=True,
            edge_color='darkgray',
            line_width=1
        )
        
        # Add highlight regions if provided
        if highlight_regions:
            for i, region in enumerate(highlight_regions):
                if 'points' in region and 'color' in region:
                    highlight_mesh = pv.PolyData(region['points'])
                    plotter.add_mesh(
                        highlight_mesh,
                        color=region['color'],
                        point_size=8,
                        render_points_as_spheres=True,
                        label=region.get('label', f'Region {i+1}')
                    )
        
        # Set camera position
        if camera_position == 'isometric':
            plotter.camera_position = 'iso'
        elif camera_position == 'top':
            plotter.view_xy()
        elif camera_position == 'side':
            plotter.view_xz()
        elif camera_position == 'front':
            plotter.view_yz()
        
        # Add title and legend
        title = f"Mesh Preview: {mesh_id}"
        if validation_result:
            status = "✓ Valid" if validation_result.is_valid else "⚠ Invalid"
            title += f" ({status})"
        
        plotter.add_title(title, font_size=16)
        
        # Generate appropriate output
        if output_format in ['html', 'both']:
            html_content = self._generate_html_preview(
                plotter, mesh, mesh_id, validation_result, highlight_regions
            )
            
        if output_format in ['png', 'both']:
            png_path = self._generate_png_preview(plotter, mesh_id)
            
        plotter.close()
        
        if output_format == 'html':
            return html_content
        elif output_format == 'png':
            return png_path
        else:
            return {'html': html_content, 'png': png_path}
    
    def generate_batch_previews(self,
                               mesh_files: List[str],
                               output_dir: str,
                               shapefile_path: Optional[str] = None,
                               validate_meshes: bool = True,
                               output_format: str = 'html') -> Dict[str, str]:
        """
        Generate previews for multiple mesh files.
        
        Parameters:
        mesh_files: List of paths to mesh files
        output_dir: Directory to save preview files
        shapefile_path: Optional shapefile for polygon highlighting
        validate_meshes: Whether to validate meshes before visualization
        output_format: Output format for previews
        
        Returns:
        Dictionary mapping mesh files to preview file paths
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        results = {}
        
        # Load shapefile data if provided
        polygon_data = None
        if shapefile_path:
            try:
                import geopandas as gpd
                gdf = gpd.read_file(shapefile_path)
                polygon_data = gdf
            except Exception as e:
                warnings.warn(f"Failed to load shapefile: {e}")
        
        for mesh_file in mesh_files:
            try:
                print(f"Generating preview for: {Path(mesh_file).name}")
                
                # Load mesh
                mesh = pv.read(mesh_file)
                mesh_id = Path(mesh_file).stem
                
                # Validate mesh if requested
                validation_result = None
                if validate_meshes:
                    validation_result = self.validator.validate_mesh(
                        mesh, repair_if_needed=True
                    )
                
                # Prepare highlight regions from shapefile
                highlight_regions = []
                if polygon_data is not None:
                    highlight_regions = self._extract_polygon_highlights(
                        mesh, polygon_data
                    )
                
                # Generate preview
                preview = self.generate_mesh_preview(
                    mesh=mesh,
                    mesh_id=mesh_id,
                    highlight_regions=highlight_regions,
                    validation_result=validation_result,
                    output_format=output_format
                )
                
                # Save preview
                if output_format == 'html':
                    preview_file = output_path / f"{mesh_id}_preview.html"
                    with open(preview_file, 'w') as f:
                        f.write(preview)
                    results[mesh_file] = str(preview_file)
                
                elif output_format == 'png':
                    preview_file = output_path / f"{mesh_id}_preview.png"
                    # preview is already the file path
                    import shutil
                    shutil.move(preview, preview_file)
                    results[mesh_file] = str(preview_file)
                
                else:  # both
                    html_file = output_path / f"{mesh_id}_preview.html"
                    with open(html_file, 'w') as f:
                        f.write(preview['html'])
                    
                    png_file = output_path / f"{mesh_id}_preview.png"
                    shutil.move(preview['png'], png_file)
                    
                    results[mesh_file] = {
                        'html': str(html_file),
                        'png': str(png_file)
                    }
                
            except Exception as e:
                print(f"Error generating preview for {mesh_file}: {e}")
                continue
        
        # Generate index file
        if len(results) > 1:
            index_file = self._generate_index_file(results, output_path, output_format)
            print(f"Generated index file: {index_file}")
        
        return results
    
    def _generate_html_preview(self,
                             plotter: pv.Plotter,
                             mesh: pv.PolyData,
                             mesh_id: str,
                             validation_result: Optional[MeshValidationResult],
                             highlight_regions: Optional[List[Dict]]) -> str:
        """Generate HTML preview with embedded plot and metadata."""
        
        # Capture plot as base64 image
        img_buffer = BytesIO()
        plotter.screenshot(img_buffer, transparent_background=False)
        img_buffer.seek(0)
        img_base64 = base64.b64encode(img_buffer.read()).decode()
        
        # Prepare mesh statistics
        stats = {
            'n_points': mesh.n_points,
            'n_cells': mesh.n_cells,
            'surface_area': f"{mesh.area:.2f}",
            'bounds': [f"{b:.2f}" for b in mesh.bounds]
        }
        
        # Prepare validation info
        validation_html = ""
        if validation_result:
            status_color = "green" if validation_result.is_valid else "orange" if validation_result.is_closed else "red"
            validation_html = f"""
            <div class="validation-section">
                <h3>Validation Results</h3>
                <div class="status-badge" style="background-color: {status_color};">
                    {'✓ Valid' if validation_result.is_valid else '⚠ Warning' if validation_result.is_closed else '✗ Invalid'}
                </div>
                <table class="validation-table">
                    <tr><td>Closed Mesh:</td><td>{'Yes' if validation_result.is_closed else 'No'}</td></tr>
                    <tr><td>Open Edges:</td><td>{validation_result.n_open_edges}</td></tr>
                    <tr><td>Holes:</td><td>{validation_result.n_holes}</td></tr>
                    <tr><td>Volume:</td><td>{validation_result.volume:.6f}</td></tr>
                    <tr><td>Surface Area:</td><td>{validation_result.surface_area:.6f}</td></tr>
                </table>
            """
            
            if validation_result.validation_errors:
                validation_html += "<h4>Errors:</h4><ul>"
                for error in validation_result.validation_errors:
                    validation_html += f"<li>{error}</li>"
                validation_html += "</ul>"
            
            if validation_result.repair_suggestions:
                validation_html += "<h4>Repair Suggestions:</h4><ul>"
                for suggestion in validation_result.repair_suggestions:
                    validation_html += f"<li>{suggestion}</li>"
                validation_html += "</ul>"
            
            validation_html += "</div>"
        
        # Generate HTML
        html_template = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Mesh Preview: {mesh_id}</title>
            <style>
                body {{
                    font-family: Arial, sans-serif;
                    margin: 20px;
                    background-color: #f5f5f5;
                }}
                .container {{
                    max-width: 1200px;
                    margin: 0 auto;
                    background-color: white;
                    padding: 20px;
                    border-radius: 8px;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                }}
                .header {{
                    text-align: center;
                    margin-bottom: 30px;
                }}
                .preview-image {{
                    text-align: center;
                    margin-bottom: 30px;
                }}
                .preview-image img {{
                    max-width: 100%;
                    border: 1px solid #ddd;
                    border-radius: 4px;
                }}
                .info-grid {{
                    display: grid;
                    grid-template-columns: 1fr 1fr;
                    gap: 30px;
                }}
                .stats-section, .validation-section {{
                    background-color: #f9f9f9;
                    padding: 15px;
                    border-radius: 4px;
                }}
                .stats-table, .validation-table {{
                    width: 100%;
                    border-collapse: collapse;
                }}
                .stats-table td, .validation-table td {{
                    padding: 5px 10px;
                    border-bottom: 1px solid #eee;
                }}
                .stats-table td:first-child, .validation-table td:first-child {{
                    font-weight: bold;
                    width: 40%;
                }}
                .status-badge {{
                    display: inline-block;
                    padding: 5px 10px;
                    border-radius: 15px;
                    color: white;
                    font-weight: bold;
                    margin-bottom: 10px;
                }}
                .timestamp {{
                    text-align: center;
                    color: #666;
                    font-size: 0.9em;
                    margin-top: 30px;
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>Mesh Preview: {mesh_id}</h1>
                </div>
                
                <div class="preview-image">
                    <img src="data:image/png;base64,{img_base64}" alt="Mesh Preview">
                </div>
                
                <div class="info-grid">
                    <div class="stats-section">
                        <h3>Mesh Statistics</h3>
                        <table class="stats-table">
                            <tr><td>Points:</td><td>{stats['n_points']:,}</td></tr>
                            <tr><td>Cells:</td><td>{stats['n_cells']:,}</td></tr>
                            <tr><td>Surface Area:</td><td>{stats['surface_area']}</td></tr>
                            <tr><td>Bounds (X):</td><td>{stats['bounds'][0]} - {stats['bounds'][1]}</td></tr>
                            <tr><td>Bounds (Y):</td><td>{stats['bounds'][2]} - {stats['bounds'][3]}</td></tr>
                            <tr><td>Bounds (Z):</td><td>{stats['bounds'][4]} - {stats['bounds'][5]}</td></tr>
                        </table>
                    </div>
                    
                    {validation_html}
                </div>
                
                <div class="timestamp">
                    Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
                </div>
            </div>
        </body>
        </html>
        """
        
        return html_template
    
    def _generate_png_preview(self, plotter: pv.Plotter, mesh_id: str) -> str:
        """Generate PNG preview and return the file path."""
        import tempfile
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            tmp_path = tmp.name
        
        # Save screenshot
        plotter.screenshot(tmp_path, transparent_background=False)
        
        return tmp_path
    
    def _extract_polygon_highlights(self, mesh: pv.PolyData, 
                                  polygon_data) -> List[Dict]:
        """Extract highlight regions from shapefile polygons."""
        highlights = []
        
        try:
            mesh_points_2d = mesh.points[:, :2]  # X, Y coordinates
            
            for idx, row in polygon_data.iterrows():
                geometry = row['geometry']
                polygon_id = row.get('ID', f'polygon_{idx}')
                
                # Find points inside polygon
                from shapely.geometry import Point
                points_inside = []
                for i, point_2d in enumerate(mesh_points_2d):
                    if Point(point_2d).within(geometry):
                        points_inside.append(mesh.points[i])
                
                if points_inside:
                    highlights.append({
                        'points': np.array(points_inside),
                        'color': 'red',
                        'label': f'Polygon {polygon_id}'
                    })
        
        except Exception as e:
            warnings.warn(f"Failed to extract polygon highlights: {e}")
        
        return highlights
    
    def _generate_index_file(self, results: Dict[str, str], 
                           output_path: Path, output_format: str) -> str:
        """Generate an index file linking to all previews."""
        
        index_html = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Mesh Preview Index</title>
            <style>
                body {
                    font-family: Arial, sans-serif;
                    margin: 20px;
                    background-color: #f5f5f5;
                }
                .container {
                    max-width: 1000px;
                    margin: 0 auto;
                    background-color: white;
                    padding: 20px;
                    border-radius: 8px;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                }
                .preview-grid {
                    display: grid;
                    grid-template-columns: repeat(auto-fill, minmax(250px, 1fr));
                    gap: 20px;
                    margin-top: 20px;
                }
                .preview-card {
                    border: 1px solid #ddd;
                    border-radius: 4px;
                    padding: 10px;
                    text-align: center;
                    background-color: #f9f9f9;
                }
                .preview-card h3 {
                    margin: 0 0 10px 0;
                    font-size: 1.1em;
                }
                .preview-links a {
                    display: inline-block;
                    margin: 5px;
                    padding: 5px 10px;
                    background-color: #007bff;
                    color: white;
                    text-decoration: none;
                    border-radius: 3px;
                    font-size: 0.9em;
                }
                .preview-links a:hover {
                    background-color: #0056b3;
                }
            </style>
        </head>
        <body>
            <div class="container">
                <h1>Mesh Preview Index</h1>
                <p>Generated on """ + datetime.now().strftime('%Y-%m-%d %H:%M:%S') + f"""</p>
                <p>Total meshes: {len(results)}</p>
                
                <div class="preview-grid">
        """
        
        for mesh_file, preview_path in results.items():
            mesh_name = Path(mesh_file).stem
            
            index_html += f"""
                    <div class="preview-card">
                        <h3>{mesh_name}</h3>
                        <div class="preview-links">
            """
            
            if output_format == 'html':
                rel_path = Path(preview_path).relative_to(output_path)
                index_html += f'<a href="{rel_path}">View HTML</a>'
            elif output_format == 'png':
                rel_path = Path(preview_path).relative_to(output_path)
                index_html += f'<a href="{rel_path}">View PNG</a>'
            else:  # both
                html_rel = Path(preview_path['html']).relative_to(output_path)
                png_rel = Path(preview_path['png']).relative_to(output_path)
                index_html += f'<a href="{html_rel}">HTML</a>'
                index_html += f'<a href="{png_rel}">PNG</a>'
            
            index_html += """
                        </div>
                    </div>
            """
        
        index_html += """
                </div>
            </div>
        </body>
        </html>
        """
        
        index_file = output_path / "index.html"
        with open(index_file, 'w') as f:
            f.write(index_html)
        
        return str(index_file)


def generate_mesh_previews(mesh_files: Union[str, List[str]],
                         output_dir: str,
                         shapefile_path: Optional[str] = None,
                         output_format: str = 'html',
                         validate_meshes: bool = True) -> Dict[str, str]:
    """
    Convenience function to generate mesh previews.
    
    Parameters:
    mesh_files: Single mesh file or list of mesh files
    output_dir: Directory to save preview files
    shapefile_path: Optional shapefile for polygon highlighting
    output_format: Output format ('html', 'png', 'both')
    validate_meshes: Whether to validate meshes
    
    Returns:
    Dictionary mapping mesh files to preview file paths
    """
    if isinstance(mesh_files, str):
        mesh_files = [mesh_files]
    
    visualizer = MeshVisualizer()
    
    return visualizer.generate_batch_previews(
        mesh_files=mesh_files,
        output_dir=output_dir,
        shapefile_path=shapefile_path,
        validate_meshes=validate_meshes,
        output_format=output_format
    ) 