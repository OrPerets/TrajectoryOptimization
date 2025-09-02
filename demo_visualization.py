#!/usr/bin/env python3
"""
Demonstration script for the scientific visualization system.

This script shows how to use the new visualization system to create
publication-quality figures with consistent styling and statistical rigor.
"""

import sys
import os
from pathlib import Path

# Add current directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

def main():
    """Main demonstration function."""
    print("Scientific Visualization System Demo")
    print("=" * 50)
    
    try:
        # Try to import the visualization modules
        from viz.figstyle import use_paper_style, get_figure_size
        from viz.palette import get_discrete_colors, get_semantic_color
        from viz.figures import (
            plot_timeseries_ci, plot_violinbox, plot_ecdf,
            plot_ablation_ci, plot_trajectory_xy
        )
        
        print("✓ All visualization modules imported successfully")
        
        # Apply global styling
        use_paper_style()
        print("✓ Global paper style applied")
        
        # Show available figure sizes
        sizes = get_figure_size('single', 4/3)
        print(f"✓ Single column figure size: {sizes[0]:.2f} × {sizes[1]:.2f} inches")
        
        # Show color palette
        colors = get_discrete_colors(5)
        print(f"✓ Color palette: {len(colors)} colors available")
        
        # Show semantic colors
        baseline_color = get_semantic_color('baseline')
        proposed_color = get_semantic_color('proposed')
        print(f"✓ Semantic colors: baseline={baseline_color}, proposed={proposed_color}")
        
        print("\nVisualization system is ready!")
        print("\nTo generate figures, run:")
        print("  python scripts/make_all_figures.py --sample --out figures/")
        print("\nOr use individual functions:")
        print("  from viz.figures import plot_timeseries_ci")
        print("  fig, ax = plot_timeseries_ci(df, x='time', ys=['method1', 'method2'])")
        
    except ImportError as e:
        print(f"✗ Import error: {e}")
        print("\nTo fix this, ensure you have the required packages:")
        print("  pip install numpy pandas matplotlib scipy")
        print("\nOr run the figure generation script directly:")
        print("  python scripts/make_all_figures.py --sample --out figures/")
        
    except Exception as e:
        print(f"✗ Error: {e}")
        print("\nPlease check the installation and try again.")

if __name__ == "__main__":
    main()