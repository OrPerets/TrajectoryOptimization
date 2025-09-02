"""
Comprehensive figure utilities for publication-quality scientific visualization.

This module provides reusable plotting functions with consistent styling,
statistical rigor, and accessibility features.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Any
from scipy import stats
from scipy.signal import savgol_filter
import warnings

from .figstyle import (
    get_figure_size, get_common_sizes, setup_axes_style, 
    save_figure, use_paper_style
)
from .palette import (
    get_discrete_colors, get_semantic_color, get_method_colors,
    get_sequential_colormap, get_diverging_colormap
)

# Ensure paper style is applied
use_paper_style()

def plot_timeseries_ci(df: pd.DataFrame, 
                      x: str = "t", 
                      ys: List[str] = None,
                      ci: float = 0.95,
                      smooth: Optional[Dict[str, Any]] = None,
                      highlight: Optional[Dict[str, Any]] = None,
                      savepath: Optional[str] = None,
                      **kwargs) -> Tuple[plt.Figure, plt.Axes]:
    """Plot time series with confidence intervals and optional smoothing.
    
    Parameters
    ----------
    df : pd.DataFrame
        Data containing time series
    x : str
        Column name for x-axis (time)
    ys : List[str]
        Column names for y-values to plot
    ci : float
        Confidence interval level (default: 0.95)
    smooth : Optional[Dict[str, Any]]
        Smoothing parameters {'window': int, 'polyorder': int}
    highlight : Optional[Dict[str, Any]]
        Highlight parameters {'events': List[float], 'labels': List[str]}
    savepath : Optional[str]
        Path to save figure (without extension)
    **kwargs
        Additional plotting arguments
        
    Returns
    -------
    Tuple[plt.Figure, plt.Axes]
        Figure and axes objects
    """
    if ys is None:
        ys = [col for col in df.columns if col != x]
    
    fig, ax = plt.subplots(figsize=get_figure_size('single', 4/3))
    setup_axes_style(ax)
    
    colors = get_discrete_colors(len(ys))
    
    for i, y_col in enumerate(ys):
        if y_col not in df.columns:
            continue
            
        x_data = df[x].values
        y_data = df[y_col].to_numpy()
        if y_data.dtype == object:
            try:
                y_data = np.vstack(y_data)
            except ValueError:
                y_data = np.asarray(y_data)
        
        # Apply smoothing if requested
        if smooth and len(y_data) > smooth.get('window', 5):
            window = smooth.get('window', 9)
            polyorder = smooth.get('polyorder', 3)
            y_smooth = savgol_filter(y_data, window, polyorder)
        else:
            y_smooth = y_data
        
        # Calculate confidence intervals if multiple samples
        if len(y_data.shape) > 1 and y_data.shape[1] > 1:
            # Multiple samples - calculate CI
            y_mean = np.mean(y_data, axis=1)
            y_std = np.std(y_data, axis=1)
            ci_factor = stats.norm.ppf((1 + ci) / 2)
            y_low = y_mean - ci_factor * y_std
            y_high = y_mean + ci_factor * y_std
            
            # Plot ribbon
            ax.fill_between(x_data, y_low, y_high, 
                           alpha=0.2, color=colors[i])
            ax.plot(x_data, y_smooth, color=colors[i], 
                   label=f"{y_col} (N={y_data.shape[1]})", **kwargs)
        else:
            # Single sample
            ax.plot(x_data, y_smooth, color=colors[i], 
                   label=y_col, **kwargs)
    
    # Add event markers if specified
    if highlight and 'events' in highlight:
        events = highlight['events']
        labels = highlight.get('labels', [f"Event {i+1}" for i in range(len(events))])
        
        for event, label in zip(events, labels):
            ax.axvline(x=event, color='red', linestyle='--', alpha=0.7)
            ax.text(event, ax.get_ylim()[1], label, 
                   rotation=90, ha='right', va='top')
    
    # Add terminal value annotations
    for i, y_col in enumerate(ys):
        if y_col in df.columns:
            y_data = df[y_col].to_numpy()
            if y_data.dtype == object:
                try:
                    y_data = np.vstack(y_data)
                except ValueError:
                    y_data = np.asarray(y_data)
            if len(y_data.shape) > 1:
                y_final = np.mean(y_data[-1, :])
            else:
                y_final = y_data[-1]

            ax.annotate(
                f"{y_col}: {y_final:.2f}",
                xy=(x_data[-1], y_final),
                xytext=(10, 10),
                textcoords="offset points",
                bbox=dict(boxstyle="round,pad=0.3", facecolor=colors[i], alpha=0.7),
                fontsize=8,
            )
    
    ax.set_xlabel(x)
    ax.set_ylabel("Value")
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)
    
    if savepath:
        save_figure(fig, savepath)
    
    return fig, ax

def plot_violinbox(groups: Dict[str, np.ndarray], 
                   metric_name: str,
                   order: Optional[List[str]] = None,
                   savepath: Optional[str] = None) -> Tuple[plt.Figure, plt.Axes]:
    """Plot violin plot with box plot overlay for distribution comparison.
    
    Parameters
    ----------
    groups : Dict[str, np.ndarray]
        Dictionary mapping group names to data arrays
    metric_name : str
        Name of the metric being plotted
    order : Optional[List[str]]
        Order to display groups
    savepath : Optional[str]
        Path to save figure (without extension)
        
    Returns
    -------
    Tuple[plt.Figure, plt.Axes]
        Figure and axes objects
    """
    if order is None:
        order = list(groups.keys())
    
    fig, ax = plt.subplots(figsize=get_figure_size('single', 4/3))
    setup_axes_style(ax)
    
    colors = get_discrete_colors(len(order))
    
    # Prepare data
    data = [groups[name] for name in order if name in groups]
    labels = [name for name in order if name in groups]
    
    if not data:
        raise ValueError("No valid groups found")
    
    # Create violin plot
    violin_parts = ax.violinplot(data, positions=range(len(data)), 
                                showmeans=True, showmedians=True)
    
    # Style violin plot
    for i, pc in enumerate(violin_parts['bodies']):
        pc.set_facecolor(colors[i])
        pc.set_alpha(0.7)
    
    # Add box plot overlay
    box_parts = ax.boxplot(data, positions=range(len(data)), 
                           widths=0.3, patch_artist=True)
    
    for i, box in enumerate(box_parts['boxes']):
        box.set_facecolor(colors[i])
        box.set_alpha(0.5)
    
    # Add individual points with jitter
    for i, (name, group_data) in enumerate(zip(labels, data)):
        if len(group_data) > 0:
            jitter = np.random.normal(0, 0.1, len(group_data))
            ax.scatter(i + jitter, group_data, 
                      color=colors[i], alpha=0.6, s=20, zorder=10)
    
    # Add statistics annotations
    for i, (name, group_data) in enumerate(zip(labels, data)):
        if len(group_data) > 0:
            median = np.median(group_data)
            q25, q75 = np.percentile(group_data, [25, 75])
            n = len(group_data)
            
            ax.text(i, q75 + 0.1 * (ax.get_ylim()[1] - ax.get_ylim()[0]),
                   f"N={n}\nQ1={q25:.2f}\nQ3={q75:.2f}",
                   ha='center', va='bottom', fontsize=8,
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_ylabel(metric_name)
    ax.set_xlabel("Method")
    ax.set_title(f"Distribution Comparison: {metric_name}")
    ax.grid(True, alpha=0.3)
    
    if savepath:
        save_figure(fig, savepath)
    
    return fig, ax

def plot_ecdf(samples_by_group: Dict[str, np.ndarray], 
               metric_name: str,
               savepath: Optional[str] = None) -> Tuple[plt.Figure, plt.Axes]:
    """Plot empirical cumulative distribution function (ECDF).
    
    Parameters
    ----------
    samples_by_group : Dict[str, np.ndarray]
        Dictionary mapping group names to sample arrays
    metric_name : str
        Name of the metric being plotted
    savepath : Optional[str]
        Path to save figure (without extension)
        
    Returns
    -------
    Tuple[plt.Figure, plt.Axes]
        Figure and axes objects
    """
    fig, ax = plt.subplots(figsize=get_figure_size('single', 4/3))
    setup_axes_style(ax)
    
    colors = get_discrete_colors(len(samples_by_group))
    
    for i, (group_name, samples) in enumerate(samples_by_group.items()):
        if len(samples) == 0:
            continue
            
        # Sort samples for ECDF
        sorted_samples = np.sort(samples)
        ecdf_values = np.arange(1, len(sorted_samples) + 1) / len(sorted_samples)
        
        # Plot ECDF
        ax.step(sorted_samples, ecdf_values, where='post', 
                color=colors[i], label=f"{group_name} (N={len(samples)})", 
                linewidth=2)
        
        # Add key percentile annotations
        percentiles = [0.5, 0.9, 0.95]
        for p in percentiles:
            if p <= 1.0:
                idx = int(p * len(sorted_samples))
                if idx < len(sorted_samples):
                    value = sorted_samples[idx]
                    ax.axhline(y=p, color=colors[i], linestyle='--', alpha=0.5)
                    ax.axvline(x=value, color=colors[i], linestyle='--', alpha=0.5)
                    ax.annotate(f"{p*100:.0f}%: {value:.2f}", 
                               xy=(value, p), xytext=(10, 10),
                               textcoords='offset points',
                               bbox=dict(boxstyle='round,pad=0.3', 
                                       facecolor=colors[i], alpha=0.7),
                               fontsize=8)
    
    ax.set_xlabel(metric_name)
    ax.set_ylabel("Cumulative Probability")
    ax.set_title(f"ECDF: {metric_name}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    if savepath:
        save_figure(fig, savepath)
    
    return fig, ax

def plot_calibration(pred: np.ndarray, 
                     truth: np.ndarray,
                     savepath: Optional[str] = None) -> Tuple[plt.Figure, plt.Axes]:
    """Plot calibration plot for model predictions.
    
    Parameters
    ----------
    pred : np.ndarray
        Predicted values
    truth : np.ndarray
        True values
    savepath : Optional[str]
        Path to save figure (without extension)
        
    Returns
    -------
    Tuple[plt.Figure, plt.Axes]
        Figure and axes objects
    """
    fig, ax = plt.subplots(figsize=get_figure_size('single', 4/3))
    setup_axes_style(ax)
    
    # Calculate residuals
    residuals = pred - truth
    
    # Create binned reliability plot
    n_bins = min(10, len(pred) // 10)
    if n_bins < 2:
        n_bins = 2
    
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    # Calculate calibration statistics
    mse = np.mean(residuals**2)
    mae = np.mean(np.abs(residuals))
    r2 = 1 - np.sum(residuals**2) / np.sum((truth - np.mean(truth))**2)
    
    # Plot predictions vs truth
    ax.scatter(truth, pred, alpha=0.6, s=30)
    
    # Add identity line
    min_val = min(np.min(truth), np.min(pred))
    max_val = max(np.max(truth), np.max(pred))
    ax.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.7, label='Perfect Calibration')
    
    # Add statistics text
    stats_text = f"MSE: {mse:.3f}\nMAE: {mae:.3f}\nR²: {r2:.3f}\nN: {len(pred)}"
    ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, 
            verticalalignment='top', bbox=dict(boxstyle='round,pad=0.5', 
                                            facecolor='white', alpha=0.8))
    
    ax.set_xlabel("True Values")
    ax.set_ylabel("Predicted Values")
    ax.set_title("Calibration Plot")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    if savepath:
        save_figure(fig, savepath)
    
    return fig, ax

def plot_residuals(residuals: np.ndarray,
                   savepath: Optional[str] = None) -> Tuple[plt.Figure, plt.Axes]:
    """Plot residual analysis for model validation.
    
    Parameters
    ----------
    residuals : np.ndarray
        Residual values (predicted - true)
    savepath : Optional[str]
        Path to save figure (without extension)
        
    Returns
    -------
    Tuple[plt.Figure, plt.Axes]
        Figure and axes objects
    """
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, 
                                                   figsize=get_figure_size('double', 4/3))
    
    # Residuals vs fitted values
    ax1.scatter(range(len(residuals)), residuals, alpha=0.6, s=20)
    ax1.axhline(y=0, color='red', linestyle='--', alpha=0.7)
    ax1.set_xlabel("Observation Index")
    ax1.set_ylabel("Residuals")
    ax1.set_title("Residuals vs Index")
    ax1.grid(True, alpha=0.3)
    
    # Residuals histogram
    ax2.hist(residuals, bins=min(20, len(residuals)//5), alpha=0.7, edgecolor='black')
    ax2.axvline(x=0, color='red', linestyle='--', alpha=0.7)
    ax2.set_xlabel("Residual Value")
    ax2.set_ylabel("Frequency")
    ax2.set_title("Residual Distribution")
    ax2.grid(True, alpha=0.3)
    
    # Q-Q plot
    stats.probplot(residuals, dist="norm", plot=ax3)
    ax3.set_title("Q-Q Plot (Normal)")
    ax3.grid(True, alpha=0.3)
    
    # Residuals vs predicted
    ax4.scatter(residuals[:-1], residuals[1:], alpha=0.6, s=20)
    ax4.axhline(y=0, color='red', linestyle='--', alpha=0.7)
    ax4.axvline(x=0, color='red', linestyle='--', alpha=0.7)
    ax4.set_xlabel("Residual t")
    ax4.set_ylabel("Residual t+1")
    ax4.set_title("Residual Autocorrelation")
    ax4.grid(True, alpha=0.3)
    
    # Add overall statistics
    mse = np.mean(residuals**2)
    mae = np.mean(np.abs(residuals))
    std_res = np.std(residuals)
    
    stats_text = f"Residual Statistics:\nMSE: {mse:.3f}\nMAE: {mae:.3f}\nStd: {std_res:.3f}\nN: {len(residuals)}"
    fig.text(0.02, 0.02, stats_text, fontsize=10, 
             bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    
    if savepath:
        save_figure(fig, savepath)
    
    return fig, (ax1, ax2, ax3, ax4)

def plot_ablation_ci(results_df: pd.DataFrame, 
                     baseline_col: str,
                     delta: bool = True,
                     savepath: Optional[str] = None) -> Tuple[plt.Figure, plt.Axes]:
    """Plot ablation study results with confidence intervals.
    
    Parameters
    ----------
    results_df : pd.DataFrame
        DataFrame with ablation results
    baseline_col : str
        Column name for baseline method
    delta : bool
        Whether to plot deltas from baseline
    savepath : Optional[str]
        Path to save figure (without extension)
        
    Returns
    -------
    Tuple[plt.Figure, plt.Axes]
        Figure and axes objects
    """
    fig, ax = plt.subplots(figsize=get_figure_size('single', 16/10))
    setup_axes_style(ax)
    
    # Get method columns (excluding baseline)
    method_cols = [col for col in results_df.columns if col != baseline_col]
    
    if delta:
        # Calculate deltas from baseline
        baseline_values = results_df[baseline_col].values
        deltas = {}
        for col in method_cols:
            deltas[col] = results_df[col].values - baseline_values
        
        # Sort by median delta
        median_deltas = {col: np.median(deltas[col]) for col in method_cols}
        sorted_methods = sorted(method_cols, key=lambda x: median_deltas[x])
        
        # Plot deltas
        colors = get_discrete_colors(len(sorted_methods))
        for i, method in enumerate(sorted_methods):
            delta_data = deltas[method]
            
            # Calculate confidence interval
            mean_delta = np.mean(delta_data)
            std_delta = np.std(delta_data)
            ci_95 = 1.96 * std_delta / np.sqrt(len(delta_data))
            
            # Plot bar with error
            ax.bar(i, mean_delta, yerr=ci_95, 
                   color=colors[i], alpha=0.7, capsize=5, label=method)
            
            # Add value label
            ax.text(i, mean_delta + ci_95 + 0.01 * (ax.get_ylim()[1] - ax.get_ylim()[0]),
                   f"{mean_delta:.3f}\n±{ci_95:.3f}", ha='center', va='bottom', fontsize=8)
        
        ax.set_ylabel(f"Δ from {baseline_col}")
        ax.set_title(f"Ablation Study: Performance Deltas from {baseline_col}")
    else:
        # Plot absolute values
        colors = get_discrete_colors(len(method_cols) + 1)
        
        # Plot baseline
        baseline_data = results_df[baseline_col].values
        mean_baseline = np.mean(baseline_data)
        std_baseline = np.std(baseline_data)
        ci_95_baseline = 1.96 * std_baseline / np.sqrt(len(baseline_data))
        
        ax.bar(0, mean_baseline, yerr=ci_95_baseline,
               color=colors[0], alpha=0.7, capsize=5, label=baseline_col)
        
        # Plot other methods
        for i, method in enumerate(method_cols):
            method_data = results_df[method].values
            mean_method = np.mean(method_data)
            std_method = np.std(method_data)
            ci_95_method = 1.96 * std_method / np.sqrt(len(method_data))
            
            ax.bar(i + 1, mean_method, yerr=ci_95_method,
                   color=colors[i + 1], alpha=0.7, capsize=5, label=method)
        
        ax.set_ylabel("Metric Value")
        ax.set_title("Ablation Study: Absolute Performance")
    
    ax.set_xlabel("Method")
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    if savepath:
        save_figure(fig, savepath)
    
    return fig, ax

def plot_scatter_density(x: np.ndarray, 
                        y: np.ndarray,
                        bins: int = 40,
                        savepath: Optional[str] = None) -> Tuple[plt.Figure, plt.Axes]:
    """Plot scatter plot with density information using hexbin.
    
    Parameters
    ----------
    x : np.ndarray
        X-axis data
    y : np.ndarray
        Y-axis data
    bins : int
        Number of bins for hexbin
    savepath : Optional[str]
        Path to save figure (without extension)
        
    Returns
    -------
    Tuple[plt.Figure, plt.Axes]
        Figure and axes objects
    """
    fig, ax = plt.subplots(figsize=get_figure_size('single', 4/3))
    setup_axes_style(ax)
    
    # Create hexbin plot
    hb = ax.hexbin(x, y, gridsize=bins, cmap='viridis', alpha=0.8)
    
    # Add colorbar
    cbar = plt.colorbar(hb, ax=ax)
    cbar.set_label('Count')
    
    # Calculate correlation statistics
    pearson_r, pearson_p = stats.pearsonr(x, y)
    spearman_r, spearman_p = stats.spearmanr(x, y)
    
    # Add correlation text
    corr_text = f"Pearson: r={pearson_r:.3f} (p={pearson_p:.3e})\nSpearman: ρ={spearman_r:.3f} (p={spearman_p:.3e})\nN: {len(x)}"
    ax.text(0.05, 0.95, corr_text, transform=ax.transAxes,
            verticalalignment='top', bbox=dict(boxstyle='round,pad=0.5',
                                            facecolor='white', alpha=0.8))
    
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title("Scatter Density Plot")
    ax.grid(True, alpha=0.3)
    
    if savepath:
        save_figure(fig, savepath)
    
    return fig, ax

def plot_paired_swarm(before: np.ndarray, 
                      after: np.ndarray,
                      link_lines: bool = True,
                      savepath: Optional[str] = None) -> Tuple[plt.Figure, plt.Axes]:
    """Plot paired before/after comparison with swarm plot.
    
    Parameters
    ----------
    before : np.ndarray
        Before values
    after : np.ndarray
        After values
    link_lines : bool
        Whether to connect paired points
    savepath : Optional[str]
        Path to save figure (without extension)
        
    Returns
    -------
    Tuple[plt.Figure, plt.Axes]
        Figure and axes objects
    """
    fig, ax = plt.subplots(figsize=get_figure_size('single', 4/3))
    setup_axes_style(ax)
    
    colors = get_discrete_colors(2)
    
    # Add jitter to x-positions
    jitter = np.random.normal(0, 0.1, len(before))
    
    # Plot before values
    ax.scatter(0 + jitter, before, color=colors[0], alpha=0.7, s=40, label='Before')
    
    # Plot after values
    ax.scatter(1 + jitter, after, color=colors[1], alpha=0.7, s=40, label='After')
    
    # Connect paired points if requested
    if link_lines:
        for i in range(len(before)):
            ax.plot([0 + jitter[i], 1 + jitter[i]], [before[i], after[i]], 
                   color='gray', alpha=0.3, linewidth=0.8)
    
    # Add statistics
    mean_before = np.mean(before)
    mean_after = np.mean(after)
    std_before = np.std(before)
    std_after = np.std(after)
    
    # Perform paired t-test
    t_stat, p_value = stats.ttest_rel(before, after)
    
    # Add mean lines
    ax.axhline(y=mean_before, color=colors[0], linestyle='--', alpha=0.7)
    ax.axhline(y=mean_after, color=colors[1], linestyle='--', alpha=0.7)
    
    # Add statistics text
    stats_text = f"Before: {mean_before:.3f}±{std_before:.3f}\nAfter: {mean_after:.3f}±{std_after:.3f}\nPaired t-test: p={p_value:.3e}\nN: {len(before)}"
    ax.text(0.05, 0.95, stats_text, transform=ax.transAxes,
            verticalalignment='top', bbox=dict(boxstyle='round,pad=0.5',
                                            facecolor='white', alpha=0.8))
    
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['Before', 'After'])
    ax.set_ylabel("Metric Value")
    ax.set_title("Paired Before/After Comparison")
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    if savepath:
        save_figure(fig, savepath)
    
    return fig, ax

def plot_trajectory_xy(x: np.ndarray, 
                       y: np.ndarray,
                       colorby: Optional[np.ndarray] = None,
                       savepath: Optional[str] = None) -> Tuple[plt.Figure, plt.Axes]:
    """Plot 2D trajectory with optional color coding.
    
    Parameters
    ----------
    x : np.ndarray
        X coordinates
    y : np.ndarray
        Y coordinates
    colorby : Optional[np.ndarray]
        Values to color code the trajectory
    savepath : Optional[str]
        Path to save figure (without extension)
        
    Returns
    -------
    Tuple[plt.Figure, plt.Axes]
        Figure and axes objects
    """
    fig, ax = plt.subplots(figsize=get_figure_size('single', 4/3))
    setup_axes_style(ax)
    
    if colorby is not None:
        # Color-coded trajectory
        scatter = ax.scatter(x, y, c=colorby, cmap='viridis', alpha=0.8, s=20)
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Color Value')
    else:
        # Simple trajectory
        ax.plot(x, y, linewidth=2, alpha=0.8)
    
    # Mark start and end points
    ax.scatter(x[0], y[0], color='green', s=100, marker='o', label='Start', zorder=5)
    ax.scatter(x[-1], y[-1], color='red', s=100, marker='s', label='End', zorder=5)
    
    # Add trajectory statistics
    total_distance = np.sum(np.sqrt(np.diff(x)**2 + np.diff(y)**2))
    straight_distance = np.sqrt((x[-1] - x[0])**2 + (y[-1] - y[0])**2)
    efficiency = straight_distance / total_distance if total_distance > 0 else 0
    
    stats_text = f"Total Distance: {total_distance:.2f}\nStraight Distance: {straight_distance:.2f}\nEfficiency: {efficiency:.3f}\nN: {len(x)}"
    ax.text(0.05, 0.95, stats_text, transform=ax.transAxes,
            verticalalignment='top', bbox=dict(boxstyle='round,pad=0.5',
                                            facecolor='white', alpha=0.8))
    
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.set_title("2D Trajectory")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    
    if savepath:
        save_figure(fig, savepath)
    
    return fig, ax


def plot_enhanced_trajectory_2d(x: np.ndarray, 
                               y: np.ndarray,
                               t: Optional[np.ndarray] = None,
                               velocity: Optional[np.ndarray] = None,
                               title: str = "Enhanced 2D Trajectory",
                               xlabel: str = "X [m]",
                               ylabel: str = "Y [m]",
                               smooth_factor: float = 0.1,
                               arrow_frequency: int = 10,
                               show_velocity_profile: bool = True,
                               savepath: Optional[str] = None) -> Tuple[plt.Figure, plt.Axes]:
    """Plot enhanced 2D trajectory with smooth curves, directional arrows, and advanced styling.
    
    Parameters
    ----------
    x, y : np.ndarray
        Trajectory coordinates
    t : Optional[np.ndarray]
        Time points for color coding
    velocity : Optional[np.ndarray]
        Velocity magnitude for color coding and width variation
    title : str
        Plot title
    xlabel, ylabel : str
        Axis labels
    smooth_factor : float
        Smoothing factor for spline interpolation (0-1)
    arrow_frequency : int
        Frequency of directional arrows
    show_velocity_profile : bool
        Whether to show velocity as line width variation
    savepath : Optional[str]
        Path to save figure
        
    Returns
    -------
    Tuple[plt.Figure, plt.Axes]
        Figure and axes objects
    """
    from scipy.interpolate import make_interp_spline
    from matplotlib.patches import FancyArrowPatch
    from matplotlib.patches import Circle
    
    fig, ax = plt.subplots(figsize=get_figure_size('single', 4/3))
    setup_axes_style(ax)
    
    # Create smooth interpolated trajectory
    if len(x) > 3 and smooth_factor > 0:
        # Parameterize by arc length for better interpolation
        distances = np.cumsum(np.sqrt(np.diff(x)**2 + np.diff(y)**2))
        distances = np.insert(distances, 0, 0)
        
        # Create more points for smoother curves
        n_smooth = max(len(x) * 3, 200)
        t_smooth = np.linspace(0, distances[-1], n_smooth)
        
        try:
            # Spline interpolation
            k = min(3, len(x) - 1)  # Spline degree
            spline_x = make_interp_spline(distances, x, k=k)
            spline_y = make_interp_spline(distances, y, k=k)
            
            x_smooth = spline_x(t_smooth)
            y_smooth = spline_y(t_smooth)
        except:
            # Fallback to original points if spline fails
            x_smooth, y_smooth = x, y
            t_smooth = distances
    else:
        x_smooth, y_smooth = x, y
        t_smooth = np.arange(len(x))
    
    # Prepare color mapping
    if t is not None and len(t) == len(x):
        # Interpolate time values to smooth trajectory
        if len(x_smooth) != len(x):
            t_interp = np.interp(np.linspace(0, len(x)-1, len(x_smooth)), 
                                np.arange(len(x)), t)
        else:
            t_interp = t
        colors = t_interp
        cmap_name = 'plasma'
        color_label = 'Time [s]'
    elif velocity is not None and len(velocity) == len(x):
        if len(x_smooth) != len(x):
            v_interp = np.interp(np.linspace(0, len(x)-1, len(x_smooth)), 
                                np.arange(len(x)), velocity)
        else:
            v_interp = velocity
        colors = v_interp
        cmap_name = 'viridis'
        color_label = 'Velocity [m/s]'
    else:
        colors = np.linspace(0, 1, len(x_smooth))
        cmap_name = 'viridis'
        color_label = 'Progress'
    
    # Main trajectory with gradient coloring
    if show_velocity_profile and velocity is not None:
        # Variable line width based on velocity
        v_norm = (velocity - np.min(velocity)) / (np.max(velocity) - np.min(velocity) + 1e-8)
        linewidths = 1 + 3 * v_norm  # Width between 1 and 4
        
        # Plot segments with varying width
        for i in range(len(x) - 1):
            ax.plot(x[i:i+2], y[i:i+2], 
                   color=plt.cm.get_cmap(cmap_name)(colors[i] if len(colors) == len(x) else colors[min(i, len(colors)-1)]),
                   linewidth=linewidths[i], alpha=0.8, solid_capstyle='round')
    else:
        # Standard colored line
        for i in range(len(x_smooth) - 1):
            ax.plot(x_smooth[i:i+2], y_smooth[i:i+2], 
                   color=plt.cm.get_cmap(cmap_name)(colors[i]), 
                   linewidth=2.5, alpha=0.9, solid_capstyle='round')
    
    # Add directional arrows along trajectory
    if arrow_frequency > 0 and len(x) > arrow_frequency:
        arrow_indices = np.arange(arrow_frequency, len(x), arrow_frequency)
        for idx in arrow_indices:
            if idx < len(x) - 1:
                # Calculate arrow direction
                dx = x[idx + 1] - x[idx - 1] if idx > 0 else x[idx + 1] - x[idx]
                dy = y[idx + 1] - y[idx - 1] if idx > 0 else y[idx + 1] - y[idx]
                
                # Normalize direction vector
                length = np.sqrt(dx**2 + dy**2)
                if length > 0:
                    dx, dy = dx/length, dy/length
                    
                    # Scale arrow size based on local trajectory curvature
                    arrow_scale = 0.02 * (ax.get_xlim()[1] - ax.get_xlim()[0])
                    
                    arrow = FancyArrowPatch(
                        (x[idx] - dx * arrow_scale/2, y[idx] - dy * arrow_scale/2),
                        (x[idx] + dx * arrow_scale/2, y[idx] + dy * arrow_scale/2),
                        arrowstyle='->', mutation_scale=15, 
                        color='black', alpha=0.7, zorder=10
                    )
                    ax.add_patch(arrow)
    
    # Enhanced start and end markers
    # Start marker - larger circle with glow effect
    start_circle = Circle((x[0], y[0]), radius=0.01*(ax.get_xlim()[1] - ax.get_xlim()[0]), 
                         facecolor='green', edgecolor='darkgreen', linewidth=2, alpha=0.9, zorder=15)
    ax.add_patch(start_circle)
    ax.scatter(x[0], y[0], s=200, color='lightgreen', marker='o', 
              edgecolors='darkgreen', linewidth=2, label='Start', zorder=16, alpha=0.8)
    
    # End marker - square with glow effect
    ax.scatter(x[-1], y[-1], s=250, color='red', marker='s', 
              edgecolors='darkred', linewidth=2, label='End', zorder=16, alpha=0.9)
    
    # Add colorbar for the trajectory coloring
    sm = plt.cm.ScalarMappable(cmap=cmap_name, norm=plt.Normalize(vmin=np.min(colors), vmax=np.max(colors)))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, shrink=0.8, aspect=20)
    cbar.set_label(color_label, rotation=270, labelpad=15)
    
    # Enhanced statistics with better formatting
    if len(x) > 1:
        total_distance = np.sum(np.sqrt(np.diff(x)**2 + np.diff(y)**2))
        straight_distance = np.sqrt((x[-1] - x[0])**2 + (y[-1] - y[0])**2)
        efficiency = straight_distance / total_distance if total_distance > 0 else 0
        
        if velocity is not None:
            avg_velocity = np.mean(velocity)
            max_velocity = np.max(velocity)
            stats_text = (f"Distance: {total_distance:.1f} m\n"
                         f"Efficiency: {efficiency:.2f}\n"
                         f"Avg Speed: {avg_velocity:.1f} m/s\n"
                         f"Max Speed: {max_velocity:.1f} m/s\n"
                         f"Points: {len(x)}")
        else:
            stats_text = (f"Distance: {total_distance:.1f} m\n"
                         f"Efficiency: {efficiency:.2f}\n"
                         f"Points: {len(x)}")
        
        # Stylish stats box
        props = dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.9, 
                    edgecolor='gray', linewidth=1)
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
                verticalalignment='top', fontsize=8, bbox=props, family='monospace')
    
    # Professional styling
    ax.set_xlabel(xlabel, fontsize=10, fontweight='bold')
    ax.set_ylabel(ylabel, fontsize=10, fontweight='bold')
    ax.set_title(title, fontsize=12, fontweight='bold', pad=20)
    ax.legend(loc='upper right', framealpha=0.9, shadow=True)
    ax.grid(True, alpha=0.2, linestyle='--')
    ax.set_aspect('equal', adjustable='box')
    
    # Adjust limits with padding
    x_range = np.max(x) - np.min(x)
    y_range = np.max(y) - np.min(y)
    padding = 0.1
    
    ax.set_xlim(np.min(x) - padding * x_range, np.max(x) + padding * x_range)
    ax.set_ylim(np.min(y) - padding * y_range, np.max(y) + padding * y_range)
    
    if savepath:
        save_figure(fig, savepath)
    
    return fig, ax


def plot_enhanced_trajectory_3d(x: np.ndarray, 
                               y: np.ndarray, 
                               z: np.ndarray,
                               t: Optional[np.ndarray] = None,
                               velocity: Optional[np.ndarray] = None,
                               title: str = "Enhanced 3D Trajectory",
                               smooth_factor: float = 0.1,
                               show_projection_shadows: bool = True,
                               view_angle: Tuple[float, float] = (30, 45),
                               savepath: Optional[str] = None) -> Tuple[plt.Figure, plt.Axes]:
    """Plot enhanced 3D trajectory with smooth curves, shadows, and professional styling.
    
    Parameters
    ----------
    x, y, z : np.ndarray
        3D trajectory coordinates
    t : Optional[np.ndarray]
        Time points for color coding
    velocity : Optional[np.ndarray]
        Velocity magnitude for color coding
    title : str
        Plot title
    smooth_factor : float
        Smoothing factor for spline interpolation
    show_projection_shadows : bool
        Whether to show projection shadows on coordinate planes
    view_angle : Tuple[float, float]
        Viewing angle (elevation, azimuth) in degrees
    savepath : Optional[str]
        Path to save figure
        
    Returns
    -------
    Tuple[plt.Figure, plt.Axes]
        Figure and 3D axes objects
    """
    from scipy.interpolate import make_interp_spline
    from mpl_toolkits.mplot3d import Axes3D
    
    fig = plt.figure(figsize=get_figure_size('single', 4/3))
    ax = fig.add_subplot(111, projection='3d')
    
    # Create smooth interpolated trajectory
    if len(x) > 3 and smooth_factor > 0:
        try:
            # Parameterize by arc length
            distances = np.cumsum(np.sqrt(np.diff(x)**2 + np.diff(y)**2 + np.diff(z)**2))
            distances = np.insert(distances, 0, 0)
            
            # Create smoother trajectory
            n_smooth = max(len(x) * 2, 100)
            t_smooth = np.linspace(0, distances[-1], n_smooth)
            
            k = min(3, len(x) - 1)
            spline_x = make_interp_spline(distances, x, k=k)
            spline_y = make_interp_spline(distances, y, k=k)
            spline_z = make_interp_spline(distances, z, k=k)
            
            x_smooth = spline_x(t_smooth)
            y_smooth = spline_y(t_smooth)
            z_smooth = spline_z(t_smooth)
        except:
            x_smooth, y_smooth, z_smooth = x, y, z
    else:
        x_smooth, y_smooth, z_smooth = x, y, z
    
    # Prepare color mapping
    if velocity is not None and len(velocity) == len(x):
        if len(x_smooth) != len(x):
            colors = np.interp(np.linspace(0, len(x)-1, len(x_smooth)), 
                              np.arange(len(x)), velocity)
        else:
            colors = velocity
        cmap_name = 'plasma'
        color_label = 'Velocity [m/s]'
    elif t is not None and len(t) == len(x):
        if len(x_smooth) != len(x):
            colors = np.interp(np.linspace(0, len(x)-1, len(x_smooth)), 
                              np.arange(len(x)), t)
        else:
            colors = t
        cmap_name = 'viridis'
        color_label = 'Time [s]'
    else:
        colors = np.linspace(0, 1, len(x_smooth))
        cmap_name = 'viridis'
        color_label = 'Progress'
    
    # Plot main 3D trajectory with color gradient
    for i in range(len(x_smooth) - 1):
        ax.plot3D(x_smooth[i:i+2], y_smooth[i:i+2], z_smooth[i:i+2], 
                 color=plt.cm.get_cmap(cmap_name)(colors[i]),
                 linewidth=2.5, alpha=0.9)
    
    # Add projection shadows if requested
    if show_projection_shadows:
        # Get axis limits
        x_min, x_max = np.min(x), np.max(x)
        y_min, y_max = np.min(y), np.max(y)
        z_min, z_max = np.min(z), np.max(z)
        
        # XY projection (shadow on bottom)
        ax.plot(x, y, z_min, color='gray', alpha=0.3, linewidth=1, linestyle='--')
        
        # XZ projection (shadow on side)
        ax.plot(x, y_max, z, color='gray', alpha=0.3, linewidth=1, linestyle='--')
        
        # YZ projection (shadow on side)
        ax.plot(x_min, y, z, color='gray', alpha=0.3, linewidth=1, linestyle='--')
    
    # Enhanced start and end markers
    ax.scatter(x[0], y[0], z[0], s=200, c='green', marker='o', 
              edgecolors='darkgreen', linewidth=2, label='Start', alpha=0.9)
    ax.scatter(x[-1], y[-1], z[-1], s=250, c='red', marker='s', 
              edgecolors='darkred', linewidth=2, label='End', alpha=0.9)
    
    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap_name, norm=plt.Normalize(vmin=np.min(colors), vmax=np.max(colors)))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, shrink=0.6, aspect=15)
    cbar.set_label(color_label, rotation=270, labelpad=15)
    
    # Professional 3D styling
    ax.set_xlabel('X [m]', fontweight='bold')
    ax.set_ylabel('Y [m]', fontweight='bold')
    ax.set_zlabel('Z [m]', fontweight='bold')
    ax.set_title(title, fontweight='bold', pad=20)
    
    # Set viewing angle
    ax.view_init(elev=view_angle[0], azim=view_angle[1])
    
    # Improve 3D appearance
    ax.grid(True, alpha=0.2)
    ax.legend(loc='upper left')
    
    # Make axes equal aspect ratio
    max_range = np.array([x.max()-x.min(), y.max()-y.min(), z.max()-z.min()]).max() / 2.0
    mid_x = (x.max()+x.min()) * 0.5
    mid_y = (y.max()+y.min()) * 0.5
    mid_z = (z.max()+z.min()) * 0.5
    
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    if savepath:
        save_figure(fig, savepath)
    
    return fig, ax


def plot_trajectory_comparison(trajectories: Dict[str, Dict[str, np.ndarray]], 
                              title: str = "Trajectory Comparison",
                              uncertainty_bands: Optional[Dict[str, Dict[str, np.ndarray]]] = None,
                              show_arrows: bool = True,
                              savepath: Optional[str] = None) -> Tuple[plt.Figure, plt.Axes]:
    """Plot multiple trajectories with uncertainty bands and professional comparison.
    
    Parameters
    ----------
    trajectories : Dict[str, Dict[str, np.ndarray]]
        Dictionary mapping method names to trajectory data {'x': x_array, 'y': y_array, 't': t_array}
    title : str
        Plot title
    uncertainty_bands : Optional[Dict[str, Dict[str, np.ndarray]]]
        Uncertainty data for each method {'x_std': array, 'y_std': array}
    show_arrows : bool
        Whether to show directional arrows
    savepath : Optional[str]
        Path to save figure
        
    Returns
    -------
    Tuple[plt.Figure, plt.Axes]
        Figure and axes objects
    """
    from matplotlib.patches import FancyArrowPatch
    
    fig, ax = plt.subplots(figsize=get_figure_size('single', 4/3))
    setup_axes_style(ax)
    
    colors = get_discrete_colors(len(trajectories))
    
    for i, (method_name, traj_data) in enumerate(trajectories.items()):
        x, y = traj_data['x'], traj_data['y']
        color = colors[i]
        
        # Plot main trajectory
        ax.plot(x, y, color=color, linewidth=2.5, label=method_name, 
               alpha=0.9, solid_capstyle='round')
        
        # Add uncertainty bands if provided
        if uncertainty_bands and method_name in uncertainty_bands:
            unc = uncertainty_bands[method_name]
            if 'x_std' in unc and 'y_std' in unc:
                x_low = x - unc['x_std']
                x_high = x + unc['x_std']
                y_low = y - unc['y_std']
                y_high = y + unc['y_std']
                
                # Plot uncertainty ellipses at key points
                n_ellipses = min(10, len(x) // 5)
                indices = np.linspace(0, len(x)-1, n_ellipses, dtype=int)
                
                for idx in indices:
                    # Create uncertainty ellipse
                    from matplotlib.patches import Ellipse
                    ellipse = Ellipse((x[idx], y[idx]), 
                                    width=2*unc['x_std'][idx], 
                                    height=2*unc['y_std'][idx],
                                    facecolor=color, alpha=0.15, 
                                    edgecolor=color, linewidth=0.5)
                    ax.add_patch(ellipse)
        
        # Add directional arrows
        if show_arrows and len(x) > 5:
            arrow_indices = np.arange(len(x)//4, len(x), len(x)//4)[:3]
            for idx in arrow_indices:
                if idx < len(x) - 1:
                    dx = x[idx + 1] - x[idx]
                    dy = y[idx + 1] - y[idx]
                    
                    length = np.sqrt(dx**2 + dy**2)
                    if length > 0:
                        scale = 0.03 * (np.max(x) - np.min(x))
                        dx_norm, dy_norm = dx/length * scale, dy/length * scale
                        
                        arrow = FancyArrowPatch(
                            (x[idx], y[idx]),
                            (x[idx] + dx_norm, y[idx] + dy_norm),
                            arrowstyle='->', mutation_scale=12, 
                            color=color, alpha=0.8, zorder=10
                        )
                        ax.add_patch(arrow)
        
        # Mark start and end points
        ax.scatter(x[0], y[0], s=80, color=color, marker='o', 
                  edgecolors='white', linewidth=1.5, zorder=15, alpha=0.9)
        ax.scatter(x[-1], y[-1], s=100, color=color, marker='s', 
                  edgecolors='white', linewidth=1.5, zorder=15, alpha=0.9)
    
    # Add trajectory statistics comparison
    stats_text = "Method Statistics:\n"
    for method_name, traj_data in trajectories.items():
        x, y = traj_data['x'], traj_data['y']
        distance = np.sum(np.sqrt(np.diff(x)**2 + np.diff(y)**2))
        stats_text += f"{method_name}: {distance:.1f}m\n"
    
    props = dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.9, 
                edgecolor='gray', linewidth=1)
    ax.text(0.02, 0.98, stats_text.strip(), transform=ax.transAxes,
            verticalalignment='top', fontsize=8, bbox=props, family='monospace')
    
    # Professional styling
    ax.set_xlabel('X [m]', fontweight='bold')
    ax.set_ylabel('Y [m]', fontweight='bold')
    ax.set_title(title, fontweight='bold', pad=20)
    ax.legend(loc='best', framealpha=0.9, shadow=True)
    ax.grid(True, alpha=0.2, linestyle='--')
    ax.set_aspect('equal', adjustable='box')
    
    if savepath:
        save_figure(fig, savepath)
    
    return fig, ax


def create_summary_panel(figures_data: Dict[str, Any],
                        savepath: Optional[str] = None) -> Tuple[plt.Figure, List[plt.Axes]]:
    """Create a 2x2 summary panel combining key results.
    
    Parameters
    ----------
    figures_data : Dict[str, Any]
        Dictionary containing data for all subplots
    savepath : Optional[str]
        Path to save figure (without extension)
        
    Returns
    -------
    Tuple[plt.Figure, List[plt.Axes]]
        Figure and list of axes objects
    """
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, 
                                                   figsize=get_figure_size('double', 4/3))
    
    axes = [ax1, ax2, ax3, ax4]
    
    # Apply consistent styling to all axes
    for ax in axes:
        setup_axes_style(ax)
    
    # Subplot 1: Time series CEP-90
    if 'timeseries_data' in figures_data:
        # Implementation for time series subplot
        pass
    
    # Subplot 2: Distribution violin
    if 'distribution_data' in figures_data:
        # Implementation for distribution subplot
        pass
    
    # Subplot 3: Ablation deltas
    if 'ablation_data' in figures_data:
        # Implementation for ablation subplot
        pass
    
    # Subplot 4: Calibration
    if 'calibration_data' in figures_data:
        # Implementation for calibration subplot
        pass
    
    # Add unified title and legend
    fig.suptitle("Summary of Key Results", fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    if savepath:
        save_figure(fig, savepath)
    
    return fig, axes