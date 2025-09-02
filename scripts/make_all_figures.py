#!/usr/bin/env python3
"""
Command-line interface to regenerate all figures for publication.

This script provides deterministic figure generation with proper error handling,
metadata tracking, and support for different output formats.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Any
import warnings

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from scipy import stats
except ImportError as e:
    print(f"Error: Required packages not available: {e}")
    print("Please install required packages: numpy, pandas, matplotlib, scipy")
    sys.exit(1)

try:
    from viz.figures import (
        plot_timeseries_ci, plot_violinbox, plot_ecdf, plot_calibration,
        plot_residuals, plot_ablation_ci, plot_scatter_density,
        plot_paired_swarm, plot_trajectory_xy, create_summary_panel
    )
    from viz.figstyle import use_paper_style, save_figure
    from viz.palette import get_discrete_colors, get_semantic_color
except ImportError as e:
    print(f"Error: Visualization modules not available: {e}")
    print("Please ensure viz/ package is properly installed")
    sys.exit(1)

# Set random seed for reproducibility
np.random.seed(42)

def load_data(input_path: str) -> Dict[str, Any]:
    """Load data from various input formats.
    
    Parameters
    ----------
    input_path : str
        Path to input data file
        
    Returns
    -------
    Dict[str, Any]
        Dictionary containing loaded data
    """
    input_path = Path(input_path)
    
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    
    data = {}
    
    if input_path.suffix == '.csv':
        # Load CSV data
        df = pd.read_csv(input_path)
        data['dataframe'] = df
        
        # Try to identify time series columns
        time_cols = [col for col in df.columns if 'time' in col.lower() or 't' in col.lower()]
        if time_cols:
            data['time_column'] = time_cols[0]
        
        # Try to identify metric columns
        metric_cols = [col for col in df.columns if col not in time_cols]
        data['metric_columns'] = metric_cols
        
    elif input_path.suffix == '.parquet':
        # Load Parquet data
        df = pd.read_parquet(input_path)
        data['dataframe'] = df
        
    elif input_path.suffix == '.json':
        # Load JSON data
        with open(input_path, 'r') as f:
            data = json.load(f)
    
    else:
        raise ValueError(f"Unsupported file format: {input_path.suffix}")
    
    return data

def generate_sample_data() -> Dict[str, Any]:
    """Generate sample data for testing figure generation.
    
    Returns
    -------
    Dict[str, Any]
        Dictionary containing sample data
    """
    print("Generating sample data for demonstration...")
    
    # Time series data
    t = np.linspace(0, 10, 100)
    np.random.seed(42)  # Ensure reproducibility
    
    # Generate multiple samples for confidence intervals
    n_samples = 20
    baseline_data = np.column_stack([
        np.sin(t) + 0.1 * np.random.randn(len(t)) for _ in range(n_samples)
    ])
    
    proposed_data = np.column_stack([
        0.8 * np.sin(t) + 0.05 * np.random.randn(len(t)) for _ in range(n_samples)
    ])
    
    ablation_data = np.column_stack([
        0.9 * np.sin(t) + 0.08 * np.random.randn(len(t)) for _ in range(n_samples)
    ])
    
    # Create DataFrame
    df = pd.DataFrame({
        't': t,
        'baseline': np.mean(baseline_data, axis=1),
        'proposed': np.mean(proposed_data, axis=1),
        'ablation': np.mean(ablation_data, axis=1)
    })
    
    # Error data for distribution plots
    errors_baseline = np.random.exponential(1.0, 100)
    errors_proposed = np.random.exponential(0.6, 100)
    errors_ablation = np.random.exponential(0.8, 100)
    
    # Ablation results
    ablation_results = pd.DataFrame({
        'baseline': np.random.normal(1.0, 0.2, 50),
        'proposed': np.random.normal(0.6, 0.15, 50),
        'ablation': np.random.normal(0.8, 0.18, 50)
    })
    
    # Trajectory data
    x = np.linspace(0, 100, 50)
    y = 0.1 * x**2 + np.random.randn(50) * 5
    
    return {
        'timeseries_df': df,
        'errors_by_method': {
            'baseline': errors_baseline,
            'proposed': errors_proposed,
            'ablation': errors_ablation
        },
        'ablation_results': ablation_results,
        'trajectory_xy': {'x': x, 'y': y},
        'sample_size': 100
    }

def generate_all_figures(data: Dict[str, Any], 
                        output_dir: str,
                        formats: List[str] = None) -> Dict[str, str]:
    """Generate all figures using the new visualization system.
    
    Parameters
    ----------
    data : Dict[str, Any]
        Data dictionary containing all required datasets
    output_dir : str
        Output directory for figures
    formats : List[str]
        List of output formats
        
    Returns
    -------
    Dict[str, str]
        Dictionary mapping figure names to file paths
    """
    if formats is None:
        formats = ['pdf', 'svg', 'png']
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Ensure paper style is applied
    use_paper_style()
    
    generated_figures = {}
    
    try:
        # Figure 1: Time series with confidence intervals
        print("Generating Figure 1: Time series with confidence intervals...")
        if 'timeseries_df' in data:
            df = data['timeseries_df']
            fig, ax = plot_timeseries_ci(
                df, x='t', ys=['baseline', 'proposed', 'ablation'],
                ci=0.95, savepath=str(output_path / "Fig_01_Timeseries_CI")
            )
            generated_figures['timeseries_ci'] = str(output_path / "Fig_01_Timeseries_CI")
            plt.close(fig)
        
        # Figure 2: Distribution comparison (violin + box)
        print("Generating Figure 2: Distribution comparison...")
        if 'errors_by_method' in data:
            fig, ax = plot_violinbox(
                data['errors_by_method'], "Error Distance [m]",
                savepath=str(output_path / "Fig_02_Distribution_Comparison")
            )
            generated_figures['distribution_comparison'] = str(output_path / "Fig_02_Distribution_Comparison")
            plt.close(fig)
        
        # Figure 3: ECDF plot
        print("Generating Figure 3: ECDF plot...")
        if 'errors_by_method' in data:
            fig, ax = plot_ecdf(
                data['errors_by_method'], "Error Distance [m]",
                savepath=str(output_path / "Fig_03_ECDF_Analysis")
            )
            generated_figures['ecdf_analysis'] = str(output_path / "Fig_03_ECDF_Analysis")
            plt.close(fig)
        
        # Figure 4: Ablation study with confidence intervals
        print("Generating Figure 4: Ablation study...")
        if 'ablation_results' in data:
            fig, ax = plot_ablation_ci(
                data['ablation_results'], 'baseline', delta=True,
                savepath=str(output_path / "Fig_04_Ablation_Study")
            )
            generated_figures['ablation_study'] = str(output_path / "Fig_04_Ablation_Study")
            plt.close(fig)
        
        # Figure 5: Trajectory visualization
        print("Generating Figure 5: Trajectory visualization...")
        if 'trajectory_xy' in data:
            traj_data = data['trajectory_xy']
            fig, ax = plot_trajectory_xy(
                traj_data['x'], traj_data['y'],
                savepath=str(output_path / "Fig_05_Trajectory_XY")
            )
            generated_figures['trajectory_xy'] = str(output_path / "Fig_05_Trajectory_XY")
            plt.close(fig)
        
        # Figure 6: Scatter density plot
        print("Generating Figure 6: Scatter density plot...")
        if 'errors_by_method' in data:
            baseline_errors = data['errors_by_method']['baseline']
            proposed_errors = data['errors_by_method']['proposed']
            
            # Ensure same length for correlation
            min_len = min(len(baseline_errors), len(proposed_errors))
            fig, ax = plot_scatter_density(
                baseline_errors[:min_len], proposed_errors[:min_len],
                savepath=str(output_path / "Fig_06_Scatter_Density")
            )
            generated_figures['scatter_density'] = str(output_path / "Fig_06_Scatter_Density")
            plt.close(fig)
        
        # Figure 7: Paired comparison
        print("Generating Figure 7: Paired comparison...")
        if 'errors_by_method' in data:
            baseline_errors = data['errors_by_method']['baseline']
            proposed_errors = data['errors_by_method']['proposed']
            
            min_len = min(len(baseline_errors), len(proposed_errors))
            fig, ax = plot_paired_swarm(
                baseline_errors[:min_len], proposed_errors[:min_len],
                savepath=str(output_path / "Fig_07_Paired_Comparison")
            )
            generated_figures['paired_comparison'] = str(output_path / "Fig_07_Paired_Comparison")
            plt.close(fig)
        
        # Figure 8: Summary panel (2x2 grid)
        print("Generating Figure 8: Summary panel...")
        summary_data = {
            'timeseries_data': data.get('timeseries_df'),
            'distribution_data': data.get('errors_by_method'),
            'ablation_data': data.get('ablation_results'),
            'calibration_data': None  # Would need actual pred/truth data
        }
        
        fig, axes = create_summary_panel(
            summary_data,
            savepath=str(output_path / "Fig_08_Summary_Panel")
        )
        generated_figures['summary_panel'] = str(output_path / "Fig_08_Summary_Panel")
        plt.close(fig)
        
    except Exception as e:
        print(f"Error generating figures: {e}")
        import traceback
        traceback.print_exc()
        return {}
    
    return generated_figures

def create_manifest(figures: Dict[str, str], 
                   output_dir: str,
                   metadata: Dict[str, Any] = None) -> str:
    """Create a manifest file documenting all generated figures.
    
    Parameters
    ----------
    figures : Dict[str, str]
        Dictionary mapping figure names to file paths
    output_dir : str
        Output directory path
    metadata : Dict[str, Any]
        Additional metadata to include
        
    Returns
    -------
    str
        Path to manifest file
    """
    manifest = {
        'generated_at': time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime()),
        'total_figures': len(figures),
        'output_directory': output_dir,
        'formats': ['pdf', 'svg', 'png'],
        'figures': {},
        'metadata': metadata or {}
    }
    
    # Add figure information
    for fig_name, base_path in figures.items():
        manifest['figures'][fig_name] = {
            'base_path': base_path,
            'files': {
                'pdf': f"{base_path}.pdf",
                'svg': f"{base_path}.svg", 
                'png': f"{base_path}.png"
            },
            'caption': f"Figure: {fig_name.replace('_', ' ').title()}",
            'paper_section': "Results",  # Would be customized per paper
            'function_used': fig_name,
            'sample_size': metadata.get('sample_size', 'N/A') if metadata else 'N/A'
        }
    
    # Write manifest
    manifest_path = Path(output_dir) / "manifest.json"
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)
    
    return str(manifest_path)

def main():
    """Main CLI function."""
    parser = argparse.ArgumentParser(
        description="Generate all figures for publication using new visualization system",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate figures from data file
  python scripts/make_all_figures.py --input data/metrics.csv --out figures/
  
  # Generate figures with sample data
  python scripts/make_all_figures.py --sample --out figures/
  
  # Generate figures in specific formats
  python scripts/make_all_figures.py --input data.csv --out figures/ --formats pdf svg
        """
    )
    
    parser.add_argument(
        '--input', '-i',
        type=str,
        help='Input data file path (CSV, Parquet, or JSON)'
    )
    
    parser.add_argument(
        '--out', '-o',
        type=str,
        default='figures/',
        help='Output directory for figures (default: figures/)'
    )
    
    parser.add_argument(
        '--formats',
        nargs='+',
        default=['pdf', 'svg', 'png'],
        choices=['pdf', 'svg', 'png'],
        help='Output formats (default: pdf svg png)'
    )
    
    parser.add_argument(
        '--sample',
        action='store_true',
        help='Generate sample data for demonstration'
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility (default: 42)'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Verbose output'
    )
    
    args = parser.parse_args()
    
    # Set random seed
    np.random.seed(args.seed)
    
    # Validate arguments
    if not args.input and not args.sample:
        parser.error("Must specify either --input or --sample")
    
    # Load or generate data
    try:
        if args.sample:
            print("Using sample data...")
            data = generate_sample_data()
        else:
            print(f"Loading data from {args.input}...")
            data = load_data(args.input)
        
        if args.verbose:
            print(f"Data loaded: {list(data.keys())}")
        
    except Exception as e:
        print(f"Error loading data: {e}")
        sys.exit(1)
    
    # Generate figures
    print(f"\nGenerating figures in {args.out}...")
    print(f"Formats: {', '.join(args.formats)}")
    
    try:
        generated_figures = generate_all_figures(
            data, args.out, args.formats
        )
        
        if generated_figures:
            print(f"\nSuccessfully generated {len(generated_figures)} figures:")
            for fig_name, base_path in generated_figures.items():
                print(f"  - {fig_name}: {base_path}")
            
            # Create manifest
            metadata = {
                'sample_size': data.get('sample_size', 'N/A'),
                'random_seed': args.seed,
                'input_file': args.input or 'sample_data',
                'formats': args.formats
            }
            
            manifest_path = create_manifest(generated_figures, args.out, metadata)
            print(f"\nManifest created: {manifest_path}")
            
            # Print summary table
            print("\n" + "="*80)
            print("FIGURE SUMMARY TABLE")
            print("="*80)
            print(f"{'Figure':<20} {'Paper Section':<15} {'Function':<20} {'Files':<20}")
            print("-"*80)
            
            for fig_name, fig_info in generated_figures.items():
                paper_section = "Results"  # Would be customized
                function_used = fig_name
                files = f"{len(args.formats)} formats"
                print(f"{fig_name:<20} {paper_section:<15} {function_used:<20} {files:<20}")
            
            print("="*80)
            print(f"\nAll figures generated successfully in: {args.out}")
            print("Use the manifest.json file for figure references and captions.")
            
        else:
            print("No figures were generated.")
            sys.exit(1)
            
    except Exception as e:
        print(f"Error generating figures: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()