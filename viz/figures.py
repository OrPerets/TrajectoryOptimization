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