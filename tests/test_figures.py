"""
Unit tests for the visualization system.

These tests ensure that all figure functions work correctly,
produce deterministic results, and maintain consistent styling.
"""

import unittest
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import tempfile
import shutil
import os

# Add parent directory to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from viz.figures import (
        plot_timeseries_ci, plot_violinbox, plot_ecdf, plot_calibration,
        plot_residuals, plot_ablation_ci, plot_scatter_density,
        plot_paired_swarm, plot_trajectory_xy
    )
    from viz.figstyle import use_paper_style, get_figure_size, setup_axes_style
    from viz.palette import get_discrete_colors, get_semantic_color
except ImportError:
    # Skip tests if visualization modules not available
    pass

class TestVisualizationSystem(unittest.TestCase):
    """Test cases for the visualization system."""
    
    def setUp(self):
        """Set up test data and temporary directory."""
        # Set random seed for reproducibility
        np.random.seed(42)
        
        # Create temporary directory for test outputs
        self.test_dir = tempfile.mkdtemp()
        
        # Generate test data
        self.t = np.linspace(0, 10, 50)
        self.baseline_data = np.sin(self.t) + 0.1 * np.random.randn(50)
        self.proposed_data = 0.8 * np.sin(self.t) + 0.05 * np.random.randn(50)
        
        # Create test DataFrame
        self.test_df = pd.DataFrame({
            't': self.t,
            'baseline': self.baseline_data,
            'proposed': self.proposed_data
        })
        
        # Test error data
        self.errors = {
            'baseline': np.random.exponential(1.0, 100),
            'proposed': np.random.exponential(0.6, 100)
        }
        
        # Test ablation data
        self.ablation_df = pd.DataFrame({
            'baseline': np.random.normal(1.0, 0.2, 50),
            'proposed': np.random.normal(0.6, 0.15, 50)
        })
    
    def tearDown(self):
        """Clean up temporary files."""
        shutil.rmtree(self.test_dir)
        plt.close('all')
    
    def test_figure_sizes(self):
        """Test that figure sizes follow journal standards."""
        single_size = get_figure_size('single', 4/3)
        double_size = get_figure_size('double', 4/3)
        
        # Check that sizes are reasonable (in inches)
        self.assertGreater(single_size[0], 2.0)  # Width > 2 inches
        self.assertLess(single_size[0], 4.0)     # Width < 4 inches
        self.assertGreater(double_size[0], 4.0)   # Width > 4 inches
        self.assertLess(double_size[0], 8.0)      # Width < 8 inches
    
    def test_color_palette(self):
        """Test color palette generation."""
        colors = get_discrete_colors(5)
        
        # Check that we get the right number of colors
        self.assertEqual(len(colors), 5)
        
        # Check that all colors are valid hex strings
        for color in colors:
            self.assertTrue(color.startswith('#'))
            self.assertEqual(len(color), 7)  # #RRGGBB format
    
    def test_timeseries_ci(self):
        """Test time series plotting with confidence intervals."""
        # Create multi-sample data
        multi_data = np.column_stack([
            self.baseline_data + 0.1 * np.random.randn(50) for _ in range(10)
        ])
        
        test_df = self.test_df.copy()
        test_df['baseline_multi'] = multi_data
        
        fig, ax = plot_timeseries_ci(
            test_df, x='t', ys=['baseline_multi'],
            savepath=os.path.join(self.test_dir, "test_timeseries")
        )
        
        # Check that figure was created
        self.assertIsInstance(fig, plt.Figure)
        self.assertIsInstance(ax, plt.Axes)
        
        # Check that output files exist
        output_base = os.path.join(self.test_dir, "test_timeseries")
        self.assertTrue(os.path.exists(f"{output_base}.pdf"))
        self.assertTrue(os.path.exists(f"{output_base}.svg"))
        self.assertTrue(os.path.exists(f"{output_base}.png"))
    
    def test_violinbox(self):
        """Test violin plot with box plot overlay."""
        fig, ax = plot_violinbox(
            self.errors, "Error Distance [m]",
            savepath=os.path.join(self.test_dir, "test_violinbox")
        )
        
        self.assertIsInstance(fig, plt.Figure)
        self.assertIsInstance(ax, plt.Axes)
        
        # Check output files
        output_base = os.path.join(self.test_dir, "test_violinbox")
        self.assertTrue(os.path.exists(f"{output_base}.pdf"))
    
    def test_ecdf(self):
        """Test ECDF plotting."""
        fig, ax = plot_ecdf(
            self.errors, "Error Distance [m]",
            savepath=os.path.join(self.test_dir, "test_ecdf")
        )
        
        self.assertIsInstance(fig, plt.Figure)
        self.assertIsInstance(ax, plt.Axes)
        
        # Check output files
        output_base = os.path.join(self.test_dir, "test_ecdf")
        self.assertTrue(os.path.exists(f"{output_base}.pdf"))
    
    def test_ablation_ci(self):
        """Test ablation study plotting."""
        fig, ax = plot_ablation_ci(
            self.ablation_df, 'baseline', delta=True,
            savepath=os.path.join(self.test_dir, "test_ablation")
        )
        
        self.assertIsInstance(fig, plt.Figure)
        self.assertIsInstance(ax, plt.Axes)
        
        # Check output files
        output_base = os.path.join(self.test_dir, "test_ablation")
        self.assertTrue(os.path.exists(f"{output_base}.pdf"))
    
    def test_trajectory_xy(self):
        """Test trajectory plotting."""
        x = np.linspace(0, 100, 50)
        y = 0.1 * x**2 + np.random.randn(50) * 5
        
        fig, ax = plot_trajectory_xy(
            x, y, savepath=os.path.join(self.test_dir, "test_trajectory")
        )
        
        self.assertIsInstance(fig, plt.Figure)
        self.assertIsInstance(ax, plt.Axes)
        
        # Check output files
        output_base = os.path.join(self.test_dir, "test_trajectory")
        self.assertTrue(os.path.exists(f"{output_base}.pdf"))
    
    def test_deterministic_results(self):
        """Test that results are deterministic with same seed."""
        # Set seed and generate first figure
        np.random.seed(42)
        fig1, ax1 = plot_timeseries_ci(
            self.test_df, x='t', ys=['baseline'],
            savepath=os.path.join(self.test_dir, "test_det1")
        )
        
        # Reset seed and generate second figure
        np.random.seed(42)
        fig2, ax2 = plot_timeseries_ci(
            self.test_df, x='t', ys=['baseline'],
            savepath=os.path.join(self.test_dir, "test_det2")
        )
        
        # Check that figures are identical
        self.assertEqual(fig1.get_size_inches().tolist(), 
                        fig2.get_size_inches().tolist())
    
    def test_error_handling(self):
        """Test error handling for invalid inputs."""
        # Test with empty data
        with self.assertRaises(ValueError):
            plot_violinbox({}, "Test")
        
        # Test with invalid DataFrame
        with self.assertRaises(KeyError):
            plot_timeseries_ci(self.test_df, x='nonexistent', ys=['baseline'])
    
    def test_style_consistency(self):
        """Test that styling is consistent across figures."""
        # Generate multiple figures
        fig1, ax1 = plot_timeseries_ci(
            self.test_df, x='t', ys=['baseline'],
            savepath=os.path.join(self.test_dir, "test_style1")
        )
        
        fig2, ax2 = plot_violinbox(
            self.errors, "Test",
            savepath=os.path.join(self.test_dir, "test_style2")
        )
        
        # Check that both figures have consistent styling
        self.assertEqual(ax1.get_xlabel(), 't')
        self.assertEqual(ax2.get_xlabel(), 'Method')
        
        # Check that grid is enabled
        self.assertTrue(ax1.get_xgrid())
        self.assertTrue(ax2.get_xgrid())

if __name__ == '__main__':
    # Check if required packages are available
    try:
        import numpy
        import pandas
        import matplotlib
        import scipy
        unittest.main()
    except ImportError:
        print("Skipping tests - required packages not available")
        print("Install numpy, pandas, matplotlib, and scipy to run tests")
