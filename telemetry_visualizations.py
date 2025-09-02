#!/usr/bin/env python3
"""
Custom telemetry data visualization script.

This script generates publication-quality figures specifically for telemetry data
analysis, including time series, trajectory plots, and performance metrics.
"""

import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Add current directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

def load_telemetry_data(filepath):
    """Load and preprocess telemetry data."""
    print(f"Loading telemetry data from {filepath}...")
    
    # Load CSV data
    df = pd.read_csv(filepath)
    
    # Clean column names (remove quotes and extra spaces)
    df.columns = df.columns.str.strip().str.replace('"', '')
    
    # Convert time column to datetime if possible
    if 'Time' in df.columns:
        try:
            # Handle the specific time format in the data
            df['Time'] = pd.to_datetime(df['Time'], format='%j:%H:%M:%S.%f', errors='coerce')
        except:
            print("Warning: Could not parse time column, using as string")
    
    # Convert numeric columns
    numeric_columns = ['A/C Mach', 'A/C Heading (deg)', 'Weapon LAT (deg)', 
                      'Weapon ALT (ft)', 'Weapon Downtrack (ft)', 'Weapon Crosstrack (ft)',
                      'Wpn Heading', 'Wpn Pitch', 'Wpn Roll']
    
    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    print(f"Loaded {len(df)} data points with {len(df.columns)} columns")
    print(f"Columns: {list(df.columns)}")
    
    return df

def plot_aircraft_trajectory(df, output_dir):
    """Plot aircraft trajectory in 3D space."""
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Extract trajectory data
    x = df['Weapon Downtrack (ft)'].values
    y = df['Weapon Crosstrack (ft)'].values
    z = df['Weapon ALT (ft)'].values
    
    # Remove invalid data
    mask = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    x, y, z = x[mask], y[mask], z[mask]
    
    # Create color gradient based on time
    colors = plt.cm.viridis(np.linspace(0, 1, len(x)))
    
    # Plot trajectory
    scatter = ax.scatter(x, y, z, c=colors, s=20, alpha=0.7)
    
    # Add trajectory line
    ax.plot(x, y, z, 'k-', alpha=0.3, linewidth=1)
    
    # Labels and title
    ax.set_xlabel('Downtrack (ft)')
    ax.set_ylabel('Crosstrack (ft)')
    ax.set_zlabel('Altitude (ft)')
    ax.set_title('Aircraft Trajectory in 3D Space')
    
    # Save figure
    output_path = output_dir / "telemetry_trajectory_3d"
    fig.savefig(f"{output_path}.pdf", dpi=300, bbox_inches='tight')
    fig.savefig(f"{output_path}.png", dpi=300, bbox_inches='tight')
    fig.savefig(f"{output_path}.svg", bbox_inches='tight')
    print(f"Saved: {output_path}")
    
    plt.close(fig)

def plot_aircraft_performance(df, output_dir):
    """Plot aircraft performance metrics over time."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Aircraft Performance Metrics', fontsize=16)
    
    # Time index for x-axis
    time_idx = np.arange(len(df))
    
    # 1. Mach number over time
    if 'A/C Mach' in df.columns:
        axes[0, 0].plot(time_idx, df['A/C Mach'], 'b-', linewidth=2)
        axes[0, 0].set_ylabel('Mach Number')
        axes[0, 0].set_title('Aircraft Mach Number')
        axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Heading over time
    if 'A/C Heading (deg)' in df.columns:
        axes[0, 1].plot(time_idx, df['A/C Heading (deg)'], 'g-', linewidth=2)
        axes[0, 1].set_ylabel('Heading (degrees)')
        axes[0, 1].set_title('Aircraft Heading')
        axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Altitude over time
    if 'Weapon ALT (ft)' in df.columns:
        axes[1, 0].plot(time_idx, df['Weapon ALT (ft)'], 'r-', linewidth=2)
        axes[1, 0].set_ylabel('Altitude (ft)')
        axes[1, 0].set_xlabel('Time Index')
        axes[1, 0].set_title('Aircraft Altitude')
        axes[1, 0].grid(True, alpha=0.3)
    
    # 4. Weapon orientation
    if 'Wpn Pitch' in df.columns and 'Wpn Roll' in df.columns:
        axes[1, 1].plot(time_idx, df['Wpn Pitch'], 'm-', linewidth=2, label='Pitch')
        axes[1, 1].plot(time_idx, df['Wpn Roll'], 'c-', linewidth=2, label='Roll')
        axes[1, 1].set_ylabel('Angle (degrees)')
        axes[1, 1].set_xlabel('Time Index')
        axes[1, 1].set_title('Weapon Orientation')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save figure
    output_path = output_dir / "telemetry_performance_metrics"
    fig.savefig(f"{output_path}.pdf", dpi=300, bbox_inches='tight')
    fig.savefig(f"{output_path}.png", dpi=300, bbox_inches='tight')
    fig.savefig(f"{output_path}.svg", bbox_inches='tight')
    print(f"Saved: {output_path}")
    
    plt.close(fig)

def plot_trajectory_2d(df, output_dir):
    """Plot 2D trajectory projections."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('Aircraft Trajectory Projections', fontsize=16)
    
    # 1. Top-down view (downtrack vs crosstrack)
    if 'Weapon Downtrack (ft)' in df.columns and 'Weapon Crosstrack (ft)' in df.columns:
        x = df['Weapon Downtrack (ft)'].values
        y = df['Weapon Crosstrack (ft)'].values
        mask = np.isfinite(x) & np.isfinite(y)
        
        axes[0].scatter(x[mask], y[mask], c=range(sum(mask)), cmap='viridis', s=20, alpha=0.7)
        axes[0].plot(x[mask], y[mask], 'k-', alpha=0.3, linewidth=1)
        axes[0].set_xlabel('Downtrack (ft)')
        axes[0].set_ylabel('Crosstrack (ft)')
        axes[0].set_title('Top-Down View')
        axes[0].grid(True, alpha=0.3)
        axes[0].set_aspect('equal')
    
    # 2. Side view (downtrack vs altitude)
    if 'Weapon Downtrack (ft)' in df.columns and 'Weapon ALT (ft)' in df.columns:
        x = df['Weapon Downtrack (ft)'].values
        z = df['Weapon ALT (ft)'].values
        mask = np.isfinite(x) & np.isfinite(z)
        
        axes[1].scatter(x[mask], z[mask], c=range(sum(mask)), cmap='plasma', s=20, alpha=0.7)
        axes[1].plot(x[mask], z[mask], 'k-', alpha=0.3, linewidth=1)
        axes[1].set_xlabel('Downtrack (ft)')
        axes[1].set_ylabel('Altitude (ft)')
        axes[1].set_title('Side View')
        axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save figure
    output_path = output_dir / "telemetry_trajectory_2d"
    fig.savefig(f"{output_path}.pdf", dpi=300, bbox_inches='tight')
    fig.savefig(f"{output_path}.png", dpi=300, bbox_inches='tight')
    fig.savefig(f"{output_path}.svg", bbox_inches='tight')
    print(f"Saved: {output_path}")
    
    plt.close(fig)

def plot_statistical_summary(df, output_dir):
    """Plot statistical summary of key metrics."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Statistical Summary of Telemetry Data', fontsize=16)
    
    # 1. Mach number distribution
    if 'A/C Mach' in df.columns:
        mach_data = df['A/C Mach'].dropna()
        axes[0, 0].hist(mach_data, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        axes[0, 0].set_xlabel('Mach Number')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].set_title('Mach Number Distribution')
        axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Altitude distribution
    if 'Weapon ALT (ft)' in df.columns:
        alt_data = df['Weapon ALT (ft)'].dropna()
        axes[0, 1].hist(alt_data, bins=30, alpha=0.7, color='lightgreen', edgecolor='black')
        axes[0, 1].set_xlabel('Altitude (ft)')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].set_title('Altitude Distribution')
        axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Heading distribution
    if 'A/C Heading (deg)' in df.columns:
        heading_data = df['A/C Heading (deg)'].dropna()
        axes[1, 0].hist(heading_data, bins=30, alpha=0.7, color='lightcoral', edgecolor='black')
        axes[1, 0].set_xlabel('Heading (degrees)')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].set_title('Heading Distribution')
        axes[1, 0].grid(True, alpha=0.3)
    
    # 4. Box plot of key metrics
    if 'A/C Mach' in df.columns and 'Weapon ALT (ft)' in df.columns:
        data_to_plot = []
        labels = []
        
        if 'A/C Mach' in df.columns:
            data_to_plot.append(df['A/C Mach'].dropna())
            labels.append('Mach')
        
        if 'Weapon ALT (ft)' in df.columns:
            # Normalize altitude to similar scale
            alt_normalized = (df['Weapon ALT (ft)'].dropna() - df['Weapon ALT (ft)'].mean()) / df['Weapon ALT (ft)'].std()
            data_to_plot.append(alt_normalized)
            labels.append('Alt (norm)')
        
        if data_to_plot:
            axes[1, 1].boxplot(data_to_plot, labels=labels)
            axes[1, 1].set_ylabel('Normalized Values')
            axes[1, 1].set_title('Distribution Comparison')
            axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save figure
    output_path = output_dir / "telemetry_statistical_summary"
    fig.savefig(f"{output_path}.pdf", dpi=300, bbox_inches='tight')
    fig.savefig(f"{output_path}.png", dpi=300, bbox_inches='tight')
    fig.savefig(f"{output_path}.svg", bbox_inches='tight')
    print(f"Saved: {output_path}")
    
    plt.close(fig)

def main():
    """Main function to generate all telemetry visualizations."""
    print("Telemetry Data Visualization Generator")
    print("=" * 50)
    
    # Input and output paths
    input_file = "S1_Hatala_83_sufa_trimmed (1).csv"
    output_dir = Path("telemetry_figures")
    
    # Create output directory
    output_dir.mkdir(exist_ok=True)
    
    try:
        # Load data
        df = load_telemetry_data(input_file)
        
        # Apply matplotlib style
        try:
            from viz.figstyle import use_paper_style
            use_paper_style()
            print("✓ Applied publication style")
        except ImportError:
            print("Warning: Could not import viz.figstyle, using default style")
        
        # Generate visualizations
        print("\nGenerating telemetry visualizations...")
        
        plot_aircraft_trajectory(df, output_dir)
        plot_aircraft_performance(df, output_dir)
        plot_trajectory_2d(df, output_dir)
        plot_statistical_summary(df, output_dir)
        
        print(f"\n✓ All visualizations generated successfully in: {output_dir}")
        print("Formats: PDF, PNG, SVG")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())