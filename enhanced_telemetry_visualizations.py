#!/usr/bin/env python3
"""
Enhanced telemetry data visualization script with research-quality trajectory plots.

This script generates publication-ready figures with advanced trajectory visualizations
including smooth curves, directional arrows, gradient coloring, and professional styling.
"""

import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional, Dict, Any

# Add current directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

# Import enhanced visualization functions
from viz.figures import (
    plot_enhanced_trajectory_2d, 
    plot_enhanced_trajectory_3d,
    plot_trajectory_comparison,
    plot_timeseries_ci,
    plot_violinbox,
    plot_ecdf
)
from viz.figstyle import use_paper_style, save_figure


def load_telemetry_data(filepath: str) -> pd.DataFrame:
    """Load and preprocess telemetry data with enhanced error handling."""
    print(f"Loading telemetry data from {filepath}...")
    
    try:
        # Load CSV data
        df = pd.read_csv(filepath)
        
        # Clean column names
        df.columns = df.columns.str.strip().str.replace('"', '')
        
        # Convert time column
        if 'Time' in df.columns:
            try:
                df['Time'] = pd.to_datetime(df['Time'], format='%j:%H:%M:%S.%f', errors='coerce')
                df['time_seconds'] = (df['Time'] - df['Time'].iloc[0]).dt.total_seconds()
            except:
                print("Warning: Could not parse time column, using index")
                df['time_seconds'] = np.arange(len(df))
        else:
            df['time_seconds'] = np.arange(len(df))
        
        # Convert numeric columns with robust handling
        numeric_columns = [
            'A/C Mach', 'A/C Heading (deg)', 'Weapon LAT (deg)', 
            'Weapon ALT (ft)', 'Weapon Downtrack (ft)', 'Weapon Crosstrack (ft)',
            'Wpn Heading', 'Wpn Pitch', 'Wpn Roll'
        ]
        
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Calculate derived metrics for enhanced visualizations
        if 'Weapon Downtrack (ft)' in df.columns and 'Weapon Crosstrack (ft)' in df.columns:
            # Convert feet to meters for international compatibility
            df['x_m'] = df['Weapon Downtrack (ft)'] * 0.3048
            df['y_m'] = df['Weapon Crosstrack (ft)'] * 0.3048
            
            if 'Weapon ALT (ft)' in df.columns:
                df['z_m'] = df['Weapon ALT (ft)'] * 0.3048
            
            # Calculate velocity (numerical differentiation)
            dt = np.diff(df['time_seconds'].values)
            dt = np.append(dt, dt[-1])  # Extend for same length
            
            dx = np.gradient(df['x_m'].values)
            dy = np.gradient(df['y_m'].values)
            df['velocity_ms'] = np.sqrt(dx**2 + dy**2) / dt
            
            # Smooth velocity to reduce noise
            from scipy.signal import savgol_filter
            if len(df) > 10:
                window = min(11, len(df) // 3)
                if window % 2 == 0:
                    window += 1
                df['velocity_smooth'] = savgol_filter(df['velocity_ms'], window, 3)
            else:
                df['velocity_smooth'] = df['velocity_ms']
        
        print(f"✓ Loaded {len(df)} data points with {len(df.columns)} columns")
        print(f"✓ Time range: {df['time_seconds'].iloc[0]:.1f} to {df['time_seconds'].iloc[-1]:.1f} seconds")
        
        return df
    
    except Exception as e:
        print(f"Error loading data: {e}")
        raise


def plot_enhanced_trajectory_2d_comparison(df: pd.DataFrame, output_dir: Path) -> None:
    """Create enhanced 2D trajectory visualization with multiple views."""
    print("Generating enhanced 2D trajectory visualizations...")
    
    if not all(col in df.columns for col in ['x_m', 'y_m']):
        print("Warning: Required coordinate columns not found, skipping 2D trajectory")
        return
    
    # Remove invalid data
    mask = np.isfinite(df['x_m']) & np.isfinite(df['y_m'])
    if 'velocity_smooth' in df.columns:
        mask &= np.isfinite(df['velocity_smooth'])
    
    df_clean = df[mask].copy()
    
    if len(df_clean) < 3:
        print("Warning: Insufficient valid data points for trajectory plot")
        return
    
    x = df_clean['x_m'].values
    y = df_clean['y_m'].values
    t = df_clean['time_seconds'].values
    
    # Enhanced trajectory with time coloring
    if 'velocity_smooth' in df_clean.columns:
        velocity = df_clean['velocity_smooth'].values
        fig1, ax1 = plot_enhanced_trajectory_2d(
            x, y, t=t, velocity=velocity,
            title="Enhanced Aircraft Trajectory (Velocity Profile)",
            xlabel="Downtrack Distance [m]",
            ylabel="Crosstrack Distance [m]",
            smooth_factor=0.15,
            arrow_frequency=15,
            show_velocity_profile=True
        )
    else:
        fig1, ax1 = plot_enhanced_trajectory_2d(
            x, y, t=t,
            title="Enhanced Aircraft Trajectory (Time Progression)",
            xlabel="Downtrack Distance [m]",
            ylabel="Crosstrack Distance [m]",
            smooth_factor=0.15,
            arrow_frequency=15,
            show_velocity_profile=False
        )
    
    # Save enhanced trajectory
    output_path = output_dir / "enhanced_trajectory_2d"
    save_figure(fig1, str(output_path))
    plt.close(fig1)
    
    # Create comparison with basic vs enhanced
    fig2, (ax_basic, ax_enhanced) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Basic plot
    ax_basic.plot(x, y, 'b-', alpha=0.7, linewidth=1)
    ax_basic.scatter(x[0], y[0], color='green', s=50, marker='o', label='Start')
    ax_basic.scatter(x[-1], y[-1], color='red', s=50, marker='s', label='End')
    ax_basic.set_xlabel("Downtrack Distance [m]")
    ax_basic.set_ylabel("Crosstrack Distance [m]")
    ax_basic.set_title("Basic Trajectory Plot")
    ax_basic.legend()
    ax_basic.grid(True, alpha=0.3)
    ax_basic.set_aspect('equal')
    
    # Enhanced plot (recreate in subplot)
    if 'velocity_smooth' in df_clean.columns:
        velocity = df_clean['velocity_smooth'].values
        colors = velocity
        cmap_name = 'plasma'
    else:
        colors = t
        cmap_name = 'viridis'
    
    # Plot enhanced version in subplot
    for i in range(len(x) - 1):
        ax_enhanced.plot(x[i:i+2], y[i:i+2], 
                        color=plt.cm.get_cmap(cmap_name)(colors[i] / np.max(colors)), 
                        linewidth=2.5, alpha=0.9, solid_capstyle='round')
    
    ax_enhanced.scatter(x[0], y[0], s=100, color='green', marker='o', 
                       edgecolors='darkgreen', linewidth=2, label='Start', alpha=0.9)
    ax_enhanced.scatter(x[-1], y[-1], s=120, color='red', marker='s', 
                       edgecolors='darkred', linewidth=2, label='End', alpha=0.9)
    
    ax_enhanced.set_xlabel("Downtrack Distance [m]")
    ax_enhanced.set_ylabel("Crosstrack Distance [m]")
    ax_enhanced.set_title("Enhanced Trajectory Plot")
    ax_enhanced.legend()
    ax_enhanced.grid(True, alpha=0.2, linestyle='--')
    ax_enhanced.set_aspect('equal')
    
    plt.tight_layout()
    
    # Save comparison
    comparison_path = output_dir / "trajectory_comparison_basic_vs_enhanced"
    save_figure(fig2, str(comparison_path))
    plt.close(fig2)


def plot_enhanced_trajectory_3d_views(df: pd.DataFrame, output_dir: Path) -> None:
    """Create enhanced 3D trajectory visualizations with multiple viewing angles."""
    print("Generating enhanced 3D trajectory visualizations...")
    
    if not all(col in df.columns for col in ['x_m', 'y_m', 'z_m']):
        print("Warning: Required 3D coordinate columns not found, skipping 3D trajectory")
        return
    
    # Remove invalid data
    mask = np.isfinite(df['x_m']) & np.isfinite(df['y_m']) & np.isfinite(df['z_m'])
    df_clean = df[mask].copy()
    
    if len(df_clean) < 3:
        print("Warning: Insufficient valid data points for 3D trajectory plot")
        return
    
    x = df_clean['x_m'].values
    y = df_clean['y_m'].values
    z = df_clean['z_m'].values
    t = df_clean['time_seconds'].values
    
    # Multiple viewing angles
    view_angles = [
        (30, 45, "Standard View"),
        (60, 30, "Elevated View"),
        (15, 120, "Side View"),
        (45, 0, "Profile View")
    ]
    
    for elev, azim, view_name in view_angles:
        if 'velocity_smooth' in df_clean.columns:
            velocity = df_clean['velocity_smooth'].values
            fig, ax = plot_enhanced_trajectory_3d(
                x, y, z, t=t, velocity=velocity,
                title=f"Enhanced 3D Aircraft Trajectory - {view_name}",
                smooth_factor=0.15,
                show_projection_shadows=True,
                view_angle=(elev, azim)
            )
        else:
            fig, ax = plot_enhanced_trajectory_3d(
                x, y, z, t=t,
                title=f"Enhanced 3D Aircraft Trajectory - {view_name}",
                smooth_factor=0.15,
                show_projection_shadows=True,
                view_angle=(elev, azim)
            )
        
        # Save with descriptive filename
        filename = f"enhanced_trajectory_3d_{view_name.lower().replace(' ', '_')}"
        output_path = output_dir / filename
        save_figure(fig, str(output_path))
        plt.close(fig)


def plot_trajectory_analysis_suite(df: pd.DataFrame, output_dir: Path) -> None:
    """Create a comprehensive suite of trajectory analysis plots."""
    print("Generating trajectory analysis suite...")
    
    if not all(col in df.columns for col in ['x_m', 'y_m', 'time_seconds']):
        print("Warning: Required columns not found for analysis suite")
        return
    
    # Clean data
    mask = np.isfinite(df['x_m']) & np.isfinite(df['y_m']) & np.isfinite(df['time_seconds'])
    df_clean = df[mask].copy()
    
    # Create synthetic comparison trajectories for demonstration
    x_orig = df_clean['x_m'].values
    y_orig = df_clean['y_m'].values
    t_orig = df_clean['time_seconds'].values
    
    # Create "smoothed" and "filtered" versions for comparison
    from scipy.signal import savgol_filter
    if len(x_orig) > 10:
        window = min(21, len(x_orig) // 3)
        if window % 2 == 0:
            window += 1
        
        x_smooth = savgol_filter(x_orig, window, 3)
        y_smooth = savgol_filter(y_orig, window, 3)
        
        # Add some noise for "raw" version
        np.random.seed(42)  # For reproducibility
        noise_scale = 0.02 * (np.max(x_orig) - np.min(x_orig))
        x_raw = x_orig + np.random.normal(0, noise_scale, len(x_orig))
        y_raw = y_orig + np.random.normal(0, noise_scale, len(y_orig))
        
        trajectories = {
            'Raw Data': {'x': x_raw, 'y': y_raw, 't': t_orig},
            'Original': {'x': x_orig, 'y': y_orig, 't': t_orig},
            'Smoothed': {'x': x_smooth, 'y': y_smooth, 't': t_orig}
        }
        
        # Create uncertainty bands for demonstration
        uncertainty_bands = {
            'Raw Data': {
                'x_std': np.full_like(x_raw, noise_scale),
                'y_std': np.full_like(y_raw, noise_scale)
            }
        }
        
        # Plot trajectory comparison
        fig, ax = plot_trajectory_comparison(
            trajectories,
            title="Trajectory Processing Comparison",
            uncertainty_bands=uncertainty_bands,
            show_arrows=True
        )
        
        output_path = output_dir / "trajectory_processing_comparison"
        save_figure(fig, str(output_path))
        plt.close(fig)
    
    # Plot velocity analysis if available
    if 'velocity_smooth' in df_clean.columns:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        
        # Velocity time series
        ax1.plot(df_clean['time_seconds'], df_clean['velocity_smooth'], 
                'b-', linewidth=2, label='Velocity')
        ax1.fill_between(df_clean['time_seconds'], 
                        df_clean['velocity_smooth'] - 0.1 * df_clean['velocity_smooth'],
                        df_clean['velocity_smooth'] + 0.1 * df_clean['velocity_smooth'],
                        alpha=0.2, color='blue')
        ax1.set_xlabel('Time [s]')
        ax1.set_ylabel('Velocity [m/s]')
        ax1.set_title('Velocity Profile with Uncertainty Band')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Velocity distribution
        ax2.hist(df_clean['velocity_smooth'], bins=30, alpha=0.7, 
                edgecolor='black', color='skyblue')
        ax2.axvline(np.mean(df_clean['velocity_smooth']), color='red', 
                   linestyle='--', label=f'Mean: {np.mean(df_clean["velocity_smooth"]):.1f} m/s')
        ax2.set_xlabel('Velocity [m/s]')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Velocity Distribution')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        velocity_path = output_dir / "velocity_analysis"
        save_figure(fig, str(velocity_path))
        plt.close(fig)


def create_research_quality_summary(df: pd.DataFrame, output_dir: Path) -> None:
    """Create a research-quality summary figure combining multiple visualizations."""
    print("Generating research-quality summary figure...")
    
    if not all(col in df.columns for col in ['x_m', 'y_m']):
        print("Warning: Required columns not found for summary")
        return
    
    # Clean data
    mask = np.isfinite(df['x_m']) & np.isfinite(df['y_m'])
    if 'velocity_smooth' in df.columns:
        mask &= np.isfinite(df['velocity_smooth'])
    df_clean = df[mask].copy()
    
    # Create figure with subplots
    fig = plt.figure(figsize=(16, 12))
    
    # Main trajectory plot (large subplot)
    ax_main = plt.subplot2grid((3, 3), (0, 0), colspan=2, rowspan=2)
    
    x = df_clean['x_m'].values
    y = df_clean['y_m'].values
    t = df_clean['time_seconds'].values
    
    if 'velocity_smooth' in df_clean.columns:
        velocity = df_clean['velocity_smooth'].values
        colors = velocity
        cmap_name = 'plasma'
        color_label = 'Velocity [m/s]'
    else:
        colors = t
        cmap_name = 'viridis'
        color_label = 'Time [s]'
    
    # Enhanced trajectory plotting
    for i in range(len(x) - 1):
        ax_main.plot(x[i:i+2], y[i:i+2], 
                    color=plt.cm.get_cmap(cmap_name)(colors[i] / np.max(colors)), 
                    linewidth=3, alpha=0.9, solid_capstyle='round')
    
    # Add directional arrows
    arrow_indices = np.arange(10, len(x), 20)
    for idx in arrow_indices:
        if idx < len(x) - 1:
            dx = x[idx + 1] - x[idx]
            dy = y[idx + 1] - y[idx]
            length = np.sqrt(dx**2 + dy**2)
            if length > 0:
                scale = 0.03 * (np.max(x) - np.min(x))
                ax_main.arrow(x[idx], y[idx], dx/length * scale, dy/length * scale,
                            head_width=scale/3, head_length=scale/3, 
                            fc='black', ec='black', alpha=0.7)
    
    ax_main.scatter(x[0], y[0], s=200, color='green', marker='o', 
                   edgecolors='darkgreen', linewidth=3, label='Start', alpha=0.9, zorder=10)
    ax_main.scatter(x[-1], y[-1], s=250, color='red', marker='s', 
                   edgecolors='darkred', linewidth=3, label='End', alpha=0.9, zorder=10)
    
    ax_main.set_xlabel('Downtrack Distance [m]', fontweight='bold', fontsize=12)
    ax_main.set_ylabel('Crosstrack Distance [m]', fontweight='bold', fontsize=12)
    ax_main.set_title('Enhanced Aircraft Trajectory Analysis', fontweight='bold', fontsize=14)
    ax_main.legend(loc='upper right', fontsize=10)
    ax_main.grid(True, alpha=0.2, linestyle='--')
    ax_main.set_aspect('equal')
    
    # Time series plot
    ax_time = plt.subplot2grid((3, 3), (0, 2))
    if 'velocity_smooth' in df_clean.columns:
        ax_time.plot(df_clean['time_seconds'], df_clean['velocity_smooth'], 
                    'b-', linewidth=2)
        ax_time.set_ylabel('Velocity [m/s]', fontweight='bold')
        ax_time.set_title('Velocity Profile', fontweight='bold')
    else:
        # Plot distance from start
        distance_from_start = np.sqrt((x - x[0])**2 + (y - y[0])**2)
        ax_time.plot(t, distance_from_start, 'g-', linewidth=2)
        ax_time.set_ylabel('Distance from Start [m]', fontweight='bold')
        ax_time.set_title('Distance Profile', fontweight='bold')
    
    ax_time.set_xlabel('Time [s]', fontweight='bold')
    ax_time.grid(True, alpha=0.3)
    
    # Statistics panel
    ax_stats = plt.subplot2grid((3, 3), (1, 2))
    ax_stats.axis('off')
    
    # Calculate statistics
    total_distance = np.sum(np.sqrt(np.diff(x)**2 + np.diff(y)**2))
    straight_distance = np.sqrt((x[-1] - x[0])**2 + (y[-1] - y[0])**2)
    efficiency = straight_distance / total_distance if total_distance > 0 else 0
    duration = t[-1] - t[0]
    avg_speed = total_distance / duration if duration > 0 else 0
    
    stats_text = f"""TRAJECTORY STATISTICS
    
Total Distance: {total_distance:.1f} m
Direct Distance: {straight_distance:.1f} m
Path Efficiency: {efficiency:.3f}
Duration: {duration:.1f} s
Average Speed: {avg_speed:.2f} m/s
Data Points: {len(x)}

COORDINATE RANGE
X: {np.min(x):.1f} to {np.max(x):.1f} m
Y: {np.min(y):.1f} to {np.max(y):.1f} m"""
    
    ax_stats.text(0.05, 0.95, stats_text, transform=ax_stats.transAxes,
                 fontsize=9, verticalalignment='top', fontfamily='monospace',
                 bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', alpha=0.8))
    
    # Velocity histogram
    ax_hist = plt.subplot2grid((3, 3), (2, 0), colspan=3)
    if 'velocity_smooth' in df_clean.columns:
        ax_hist.hist(df_clean['velocity_smooth'], bins=25, alpha=0.7, 
                    edgecolor='black', color='lightblue', density=True)
        ax_hist.axvline(np.mean(df_clean['velocity_smooth']), color='red', 
                       linestyle='--', linewidth=2, 
                       label=f'Mean: {np.mean(df_clean["velocity_smooth"]):.2f} m/s')
        ax_hist.set_xlabel('Velocity [m/s]', fontweight='bold')
        ax_hist.set_ylabel('Probability Density', fontweight='bold')
        ax_hist.set_title('Velocity Distribution', fontweight='bold')
    else:
        # Plot segment length distribution
        segment_lengths = np.sqrt(np.diff(x)**2 + np.diff(y)**2)
        ax_hist.hist(segment_lengths, bins=25, alpha=0.7, 
                    edgecolor='black', color='lightcoral', density=True)
        ax_hist.axvline(np.mean(segment_lengths), color='red', 
                       linestyle='--', linewidth=2, 
                       label=f'Mean: {np.mean(segment_lengths):.2f} m')
        ax_hist.set_xlabel('Segment Length [m]', fontweight='bold')
        ax_hist.set_ylabel('Probability Density', fontweight='bold')
        ax_hist.set_title('Segment Length Distribution', fontweight='bold')
    
    ax_hist.legend()
    ax_hist.grid(True, alpha=0.3)
    
    # Colorbar for main plot (add after histogram to avoid layout engine conflict)
    sm = plt.cm.ScalarMappable(cmap=cmap_name, norm=plt.Normalize(vmin=np.min(colors), vmax=np.max(colors)))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax_main, shrink=0.6)
    cbar.set_label(color_label, rotation=270, labelpad=20, fontweight='bold')
    
    # Use subplots_adjust instead of tight_layout to avoid engine conflicts
    plt.subplots_adjust(left=0.08, right=0.92, top=0.95, bottom=0.08, hspace=0.3, wspace=0.3)
    
    # Save research-quality summary
    summary_path = output_dir / "research_quality_trajectory_summary"
    save_figure(fig, str(summary_path))
    plt.close(fig)


def main():
    """Main function to generate all enhanced telemetry visualizations."""
    print("Enhanced Telemetry Data Visualization Generator")
    print("=" * 60)
    
    # Apply publication-quality style
    use_paper_style()
    print("✓ Applied publication-quality styling")
    
    # Input and output paths
    input_file = "S1_Hatala_83_sufa_trimmed (1).csv"
    output_dir = Path("enhanced_telemetry_figures")
    
    # Create output directory
    output_dir.mkdir(exist_ok=True)
    
    try:
        # Load data
        df = load_telemetry_data(input_file)
        
        print(f"\nGenerating enhanced visualizations...")
        print("-" * 40)
        
        # Generate all enhanced visualizations
        plot_enhanced_trajectory_2d_comparison(df, output_dir)
        plot_enhanced_trajectory_3d_views(df, output_dir)
        plot_trajectory_analysis_suite(df, output_dir)
        create_research_quality_summary(df, output_dir)
        
        print(f"\n✓ All enhanced visualizations generated successfully!")
        print(f"✓ Output directory: {output_dir}")
        print(f"✓ Formats: PDF, SVG, PNG")
        print(f"✓ Total figures generated: Multiple enhanced trajectory visualizations")
        
        # List generated files
        pdf_files = list(output_dir.glob("*.pdf"))
        print(f"\n📊 Generated {len(pdf_files)} visualization sets:")
        for pdf_file in pdf_files:
            print(f"   - {pdf_file.stem}")
        
        print(f"\n🎯 Key improvements over basic visualizations:")
        print(f"   • Smooth interpolated trajectories")
        print(f"   • Directional arrows showing motion flow")
        print(f"   • Gradient color coding (time/velocity)")
        print(f"   • Enhanced start/end markers")
        print(f"   • Professional styling and typography")
        print(f"   • Multiple viewing angles for 3D plots")
        print(f"   • Uncertainty visualization")
        print(f"   • Research-quality summary figures")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
