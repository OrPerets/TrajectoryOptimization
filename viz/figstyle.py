"""
Global figure style system for publication-quality scientific visualization.

This module provides a single source of truth for all figure styling parameters,
following journal standards for typography, sizing, and accessibility.
"""

import matplotlib as mpl
import matplotlib.pyplot as plt
from typing import Tuple, Dict, Any
import numpy as np

# Journal column widths (mm) - golden ratio based
SINGLE_COLUMN_MM = 85.0
DOUBLE_COLUMN_MM = 174.0
MM_TO_INCHES = 1.0 / 25.4

# Typography specifications
FONT_SIZES = {
    'title': 11.0,      # pt
    'axis_label': 9.5,  # pt  
    'tick': 9.0,        # pt
    'legend': 9.0,      # pt
    'caption': 9.0,     # pt
    'annotation': 8.5,  # pt
}

# Color-blind safe discrete palette (8 colors)
DISCRETE_PALETTE = [
    '#1f77b4',  # Blue
    '#ff7f0e',  # Orange  
    '#2ca02c',  # Green
    '#d62728',  # Red
    '#9467bd',  # Purple
    '#8c564b',  # Brown
    '#e377c2',  # Pink
    '#7f7f7f',  # Gray
]

# Semantic color mapping
SEMANTIC_COLORS = {
    'baseline': '#1f77b4',      # Blue
    'proposed': '#ff7f0e',      # Orange
    'ablation': '#2ca02c',      # Green
    'noise': '#d62728',         # Red
    'confidence': '#9467bd',    # Purple
    'outlier': '#8c564b',       # Brown
    'highlight': '#e377c2',     # Pink
    'reference': '#7f7f7f',     # Gray
}

# Sequential colormaps for continuous data
SEQUENTIAL_MAPS = {
    'viridis': 'viridis',      # Default sequential
    'magma': 'magma',          # Alternative sequential
    'cividis': 'cividis',      # Colorblind-friendly
}

# Diverging colormaps
DIVERGING_MAPS = {
    'RdBu': 'RdBu_r',          # Red-Blue diverging
    'PiYG': 'PiYG',            # Pink-Yellow-Green
    'BrBG': 'BrBG',            # Brown-Blue-Green
}

def use_paper_style() -> None:
    """Apply publication-quality style settings globally."""
    mpl.rcParams.update({
        # Figure settings
        'figure.dpi': 150,
        'savefig.dpi': 600,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.1,
        'figure.constrained_layout.use': True,
        'figure.constrained_layout.h_pad': 0.02,
        'figure.constrained_layout.w_pad': 0.02,
        
        # Font settings
        'font.family': 'serif',
        'font.serif': ['STIX', 'Computer Modern', 'Times New Roman', 'DejaVu Serif'],
        'font.size': FONT_SIZES['axis_label'],
        'mathtext.fontset': 'stix',
        
        # Text sizes
        'axes.titlesize': FONT_SIZES['title'],
        'axes.labelsize': FONT_SIZES['axis_label'],
        'xtick.labelsize': FONT_SIZES['tick'],
        'ytick.labelsize': FONT_SIZES['tick'],
        'legend.fontsize': FONT_SIZES['legend'],
        'figure.titlesize': FONT_SIZES['title'],
        
        # Line and marker settings
        'lines.linewidth': 1.8,
        'lines.markersize': 4.0,
        'lines.antialiased': True,
        'lines.solid_capstyle': 'round',
        'lines.solid_joinstyle': 'round',
        
        # Axes settings
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.spines.left': True,
        'axes.spines.bottom': True,
        'axes.linewidth': 0.8,
        'axes.grid': True,
        'axes.grid.axis': 'y',
        'grid.alpha': 0.25,
        'grid.linewidth': 0.6,
        
        # Tick settings
        'xtick.major.size': 3.0,
        'xtick.major.width': 0.8,
        'xtick.minor.size': 1.5,
        'xtick.minor.width': 0.6,
        'ytick.major.size': 3.0,
        'ytick.major.width': 0.8,
        'ytick.minor.size': 1.5,
        'ytick.minor.width': 0.6,
        'xtick.direction': 'out',
        'ytick.direction': 'out',
        
        # Legend settings
        'legend.frameon': False,
        'legend.fancybox': False,
        'legend.shadow': False,
        'legend.numpoints': 1,
        'legend.scatterpoints': 1,
        
        # Color settings
        'axes.prop_cycle': mpl.cycler('color', DISCRETE_PALETTE),
        
        # Save settings
        'savefig.format': 'pdf',
        'savefig.facecolor': 'white',
        'savefig.edgecolor': 'none',
        'savefig.transparent': False,
    })

def get_figure_size(width_type: str = 'single', aspect_ratio: float = 4/3) -> Tuple[float, float]:
    """Get figure dimensions following journal standards.
    
    Parameters
    ----------
    width_type : str
        Either 'single' or 'double' column
    aspect_ratio : float
        Height/width ratio (default: 4:3)
        
    Returns
    -------
    Tuple[float, float]
        Figure dimensions in inches (width, height)
    """
    if width_type == 'single':
        width_mm = SINGLE_COLUMN_MM
    elif width_type == 'double':
        width_mm = DOUBLE_COLUMN_MM
    else:
        raise ValueError("width_type must be 'single' or 'double'")
    
    width_inches = width_mm * MM_TO_INCHES
    height_inches = width_inches * aspect_ratio
    
    return (width_inches, height_inches)

def get_common_sizes() -> Dict[str, Tuple[float, float]]:
    """Get common figure sizes for different content types."""
    return {
        'single_panel': get_figure_size('single', 4/3),
        'single_tall': get_figure_size('single', 16/10),
        'single_wide': get_figure_size('single', 3/4),
        'double_panel': get_figure_size('double', 4/3),
        'double_tall': get_figure_size('double', 16/10),
        'double_wide': get_figure_size('double', 3/4),
        'square': get_figure_size('single', 1.0),
    }

def get_semantic_color(key: str) -> str:
    """Get semantic color for specific data type."""
    return SEMANTIC_COLORS.get(key, DISCRETE_PALETTE[0])

def get_discrete_colors(n_colors: int) -> list:
    """Get n distinct colors from discrete palette."""
    if n_colors <= len(DISCRETE_PALETTE):
        return DISCRETE_PALETTE[:n_colors]
    else:
        # Generate additional colors if needed
        colors = DISCRETE_PALETTE.copy()
        for i in range(len(colors), n_colors):
            # Generate a new color with good contrast
            hue = (i * 0.618033988749895) % 1.0  # Golden ratio
            saturation = 0.7
            value = 0.8
            colors.append(mpl.colors.hsv_to_rgb([hue, saturation, value]))
        return colors

def setup_axes_style(ax: plt.Axes, 
                     show_grid: bool = True,
                     grid_alpha: float = 0.25,
                     spine_width: float = 0.8) -> None:
    """Apply consistent styling to axes."""
    # Grid
    if show_grid:
        ax.grid(True, alpha=grid_alpha, linewidth=0.6)
        ax.set_axisbelow(True)
    
    # Spines
    for spine in ax.spines.values():
        spine.set_linewidth(spine_width)
    
    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Tick parameters
    ax.tick_params(axis='both', which='major', width=0.8, length=3.0)
    ax.tick_params(axis='both', which='minor', width=0.6, length=1.5)

def add_watermark(fig: plt.Figure, text: str = "DRAFT", 
                  alpha: float = 0.1, fontsize: int = 24) -> None:
    """Add watermark to figure (useful for drafts)."""
    fig.text(0.5, 0.5, text, 
             ha='center', va='center', 
             alpha=alpha, fontsize=fontsize,
             transform=fig.transFigure,
             rotation=45)

def save_figure(fig: plt.Figure, 
                base_path: str,
                formats: list = None,
                metadata: Dict[str, Any] = None,
                tight: bool = True) -> None:
    """Save figure in multiple formats with metadata.
    
    Parameters
    ----------
    fig : plt.Figure
        Figure to save
    base_path : str
        Base path without extension
    formats : list, optional
        List of formats to save (default: ['pdf', 'svg', 'png'])
    metadata : Dict[str, Any], optional
        Metadata to include in saved files
    tight : bool
        Whether to use tight layout
    """
    if formats is None:
        formats = ['pdf', 'svg', 'png']
    
    if metadata is None:
        metadata = {
            'Creator': 'Scientific Visualization Pipeline',
            'Title': 'Publication Ready Figure'
        }
    
    for fmt in formats:
        if fmt == 'png':
            fig.savefig(f"{base_path}.{fmt}", 
                       dpi=600, bbox_inches='tight' if tight else None,
                       metadata=metadata)
        else:
            fig.savefig(f"{base_path}.{fmt}", 
                       bbox_inches='tight' if tight else None,
                       metadata=metadata)
    
    print(f"Saved figure: {base_path} ({', '.join(formats)})")

# Initialize style when module is imported
use_paper_style()