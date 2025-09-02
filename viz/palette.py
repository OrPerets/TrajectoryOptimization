"""
Color palette system for scientific visualization.

This module provides colorblind-safe palettes, semantic color mappings,
and accessibility tools for publication-quality figures.
"""

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Dict, Tuple, Optional, Union
from .figstyle import DISCRETE_PALETTE, SEMANTIC_COLORS

# Colorblind-safe discrete palette (8 colors)
# Tested for deuteranopia, protanopia, and tritanopia
COLORBLIND_SAFE_PALETTE = [
    '#1f77b4',  # Blue
    '#ff7f0e',  # Orange  
    '#2ca02c',  # Green
    '#d62728',  # Red
    '#9467bd',  # Purple
    '#8c564b',  # Brown
    '#e377c2',  # Pink
    '#7f7f7f',  # Gray
]

# Alternative palette with better contrast
HIGH_CONTRAST_PALETTE = [
    '#000000',  # Black
    '#e69f00',  # Orange
    '#56b4e9',  # Blue
    '#009e73',  # Green
    '#f0e442',  # Yellow
    '#0072b2',  # Dark Blue
    '#d55e00',  # Red
    '#cc79a7',  # Pink
]

# Semantic color mapping for different data types
SEMANTIC_MAPPING = {
    # Method comparison
    'baseline': '#1f77b4',      # Blue - reference method
    'proposed': '#ff7f0e',      # Orange - new method
    'ablation': '#2ca02c',      # Green - ablation study
    'improvement': '#d62728',   # Red - improvements
    
    # Performance indicators
    'good': '#2ca02c',          # Green - good performance
    'poor': '#d62728',          # Red - poor performance
    'neutral': '#7f7f7f',       # Gray - neutral
    'warning': '#ff7f0e',       # Orange - warning
    
    # Uncertainty and confidence
    'confidence': '#9467bd',    # Purple - confidence intervals
    'uncertainty': '#e377c2',   # Pink - uncertainty bands
    'outlier': '#8c564b',       # Brown - outliers
    
    # Time series
    'past': '#1f77b4',          # Blue - historical
    'present': '#ff7f0e',       # Orange - current
    'future': '#2ca02c',        # Green - predicted
    
    # Trajectory states
    'start': '#2ca02c',         # Green - start point
    'end': '#d62728',           # Red - end point
    'waypoint': '#9467bd',      # Purple - intermediate
}

def get_semantic_color(key: str, fallback: str = None) -> str:
    """Get semantic color for specific data type.
    
    Parameters
    ----------
    key : str
        Semantic key (e.g., 'baseline', 'proposed')
    fallback : str, optional
        Fallback color if key not found
        
    Returns
    -------
    str
        Hex color code
    """
    return SEMANTIC_MAPPING.get(key, fallback or DISCRETE_PALETTE[0])

def get_discrete_colors(n_colors: int, palette_name: str = 'colorblind') -> List[str]:
    """Get n distinct colors from specified palette.
    
    Parameters
    ----------
    n_colors : int
        Number of colors needed
    palette_name : str
        Palette to use ('colorblind', 'high_contrast', 'default')
        
    Returns
    -------
    List[str]
        List of hex color codes
    """
    if palette_name == 'colorblind':
        base_palette = COLORBLIND_SAFE_PALETTE
    elif palette_name == 'high_contrast':
        base_palette = HIGH_CONTRAST_PALETTE
    else:
        base_palette = DISCRETE_PALETTE
    
    if n_colors <= len(base_palette):
        return base_palette[:n_colors]
    else:
        # Generate additional colors with good contrast
        colors = base_palette.copy()
        for i in range(len(colors), n_colors):
            # Use golden ratio to generate well-distributed hues
            hue = (i * 0.618033988749895) % 1.0
            saturation = 0.7
            value = 0.8
            colors.append(mpl.colors.hsv_to_rgb([hue, saturation, value]))
        return colors

def get_sequential_colormap(name: str = 'viridis', n_colors: int = 256) -> mpl.colors.Colormap:
    """Get sequential colormap for continuous data.
    
    Parameters
    ----------
    name : str
        Colormap name ('viridis', 'magma', 'cividis', 'plasma')
    n_colors : int
        Number of color levels
        
    Returns
    -------
    mpl.colors.Colormap
        Matplotlib colormap object
    """
    valid_maps = ['viridis', 'magma', 'cividis', 'plasma', 'inferno']
    if name not in valid_maps:
        name = 'viridis'
    
    return plt.cm.get_cmap(name, n_colors)

def get_diverging_colormap(name: str = 'RdBu', n_colors: int = 256) -> mpl.colors.Colormap:
    """Get diverging colormap for data with center point.
    
    Parameters
    ----------
    name : str
        Colormap name ('RdBu', 'PiYG', 'BrBG', 'PuOr')
    n_colors : int
        Number of color levels
        
    Returns
    -------
    mpl.colors.Colormap
        Matplotlib colormap object
    """
    valid_maps = ['RdBu', 'PiYG', 'BrBG', 'PuOr', 'RdBu_r', 'PiYG_r', 'BrBG_r', 'PuOr_r']
    if name not in valid_maps:
        name = 'RdBu'
    
    return plt.cm.get_cmap(name, n_colors)

def create_custom_colormap(colors: List[str], n_levels: int = 256) -> mpl.colors.Colormap:
    """Create custom colormap from list of colors.
    
    Parameters
    ----------
    colors : List[str]
        List of hex color codes
    n_levels : int
        Number of color levels in colormap
        
    Returns
    -------
    mpl.colors.Colormap
        Custom matplotlib colormap
    """
    if len(colors) < 2:
        raise ValueError("Need at least 2 colors for colormap")
    
    # Normalize positions
    positions = np.linspace(0, 1, len(colors))
    
    # Create colormap
    cmap = mpl.colors.LinearSegmentedColormap.from_list(
        'custom', list(zip(positions, colors)), N=n_levels
    )
    
    return cmap

def get_color_contrast_ratio(color1: str, color2: str) -> float:
    """Calculate contrast ratio between two colors (WCAG standard).
    
    Parameters
    ----------
    color1 : str
        First color (hex code)
    color2 : str
        Second color (hex code)
        
    Returns
    -------
    float
        Contrast ratio (4.5+ for WCAG AA, 7+ for WCAG AAA)
    """
    def hex_to_rgb(hex_color: str) -> Tuple[int, int, int]:
        hex_color = hex_color.lstrip('#')
        return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    
    def rgb_to_luminance(rgb: Tuple[int, int, int]) -> float:
        r, g, b = [c/255.0 for c in rgb]
        r = r/12.92 if r <= 0.03928 else ((r + 0.055)/1.055)**2.4
        g = g/12.92 if g <= 0.03928 else ((g + 0.055)/1.055)**2.4
        b = b/12.92 if b <= 0.03928 else ((b + 0.055)/1.055)**2.4
        return 0.2126*r + 0.7152*g + 0.0722*b
    
    rgb1 = hex_to_rgb(color1)
    rgb2 = hex_to_rgb(color2)
    
    lum1 = rgb_to_luminance(rgb1)
    lum2 = rgb_to_luminance(rgb2)
    
    lighter = max(lum1, lum2)
    darker = min(lum1, lum2)
    
    return (lighter + 0.05) / (darker + 0.05)

def check_colorblind_accessibility(colors: List[str], 
                                 colorblind_type: str = 'deuteranopia') -> Dict[str, Union[bool, float]]:
    """Check if color palette is accessible for colorblind users.
    
    Parameters
    ----------
    colors : List[str]
        List of hex color codes to check
    colorblind_type : str
        Type of colorblindness to simulate
        
    Returns
    -------
    Dict[str, Union[bool, float]]
        Accessibility metrics
    """
    # Simulate colorblind vision (simplified)
    def simulate_colorblind(color: str, cb_type: str) -> str:
        # This is a simplified simulation - in practice, use specialized tools
        rgb = mpl.colors.to_rgb(color)
        if cb_type == 'deuteranopia':
            # Red-green colorblind
            r, g, b = rgb
            # Approximate simulation
            new_r = 0.625 * r + 0.375 * g
            new_g = 0.7 * r + 0.3 * g
            return mpl.colors.rgb2hex([new_r, new_g, b])
        elif cb_type == 'protanopia':
            # Red-green colorblind (different type)
            r, g, b = rgb
            new_r = 0.567 * r + 0.433 * g
            new_g = 0.558 * r + 0.442 * g
            return mpl.colors.rgb2hex([new_r, new_g, b])
        else:
            return color
    
    # Simulate colorblind vision for all colors
    simulated_colors = [simulate_colorblind(c, colorblind_type) for c in colors]
    
    # Check contrast ratios
    min_contrast = float('inf')
    for i in range(len(colors)):
        for j in range(i+1, len(colors)):
            contrast = get_color_contrast_ratio(colors[i], colors[j])
            min_contrast = min(min_contrast, contrast)
    
    # Check if colors are distinguishable in grayscale
    grayscale_colors = [mpl.colors.rgb_to_hsv(mpl.colors.to_rgb(c))[2] for c in colors]
    grayscale_unique = len(set(round(g, 2) for g in grayscale_colors))
    
    return {
        'accessible': min_contrast >= 4.5,  # WCAG AA standard
        'min_contrast_ratio': min_contrast,
        'grayscale_distinguishable': grayscale_unique == len(colors),
        'colorblind_simulated': simulated_colors
    }

def get_grayscale_equivalent(color: str) -> str:
    """Convert color to grayscale equivalent.
    
    Parameters
    ----------
    color : str
        Hex color code
        
    Returns
    -------
    str
        Grayscale hex color code
    """
    rgb = mpl.colors.to_rgb(color)
    # Convert to grayscale using luminance formula
    gray_value = 0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]
    return mpl.colors.rgb2hex([gray_value, gray_value, gray_value])

def create_accessible_palette(n_colors: int, 
                            min_contrast: float = 4.5,
                            colorblind_safe: bool = True) -> List[str]:
    """Create accessible color palette with specified constraints.
    
    Parameters
    ----------
    n_colors : int
        Number of colors needed
    min_contrast : float
        Minimum contrast ratio between colors
    colorblind_safe : bool
        Whether to ensure colorblind accessibility
        
    Returns
    -------
    List[str]
        List of accessible hex color codes
    """
    if n_colors <= len(COLORBLIND_SAFE_PALETTE):
        return COLORBLIND_SAFE_PALETTE[:n_colors]
    
    # Generate additional colors with accessibility constraints
    colors = COLORBLIND_SAFE_PALETTE.copy()
    
    for i in range(len(colors), n_colors):
        attempts = 0
        max_attempts = 100
        
        while attempts < max_attempts:
            # Generate color using golden ratio
            hue = (i * 0.618033988749895) % 1.0
            saturation = 0.6 + 0.2 * np.random.random()  # 0.6-0.8
            value = 0.7 + 0.2 * np.random.random()        # 0.7-0.9
            
            new_color = mpl.colors.hsv_to_rgb([hue, saturation, value])
            new_color_hex = mpl.colors.rgb2hex(new_color)
            
            # Check accessibility
            accessible = True
            for existing_color in colors:
                contrast = get_color_contrast_ratio(new_color_hex, existing_color)
                if contrast < min_contrast:
                    accessible = False
                    break
            
            if accessible:
                colors.append(new_color_hex)
                break
            
            attempts += 1
        
        if attempts >= max_attempts:
            # Fallback to high contrast palette
            colors.append(HIGH_CONTRAST_PALETTE[i % len(HIGH_CONTRAST_PALETTE)])
    
    return colors

# Convenience functions for common use cases
def get_method_colors(methods: List[str]) -> Dict[str, str]:
    """Get semantic colors for different methods."""
    color_map = {}
    colors = get_discrete_colors(len(methods))
    
    for i, method in enumerate(methods):
        if method.lower() in ['baseline', 'reference', 'ground_truth']:
            color_map[method] = get_semantic_color('baseline')
        elif method.lower() in ['proposed', 'new', 'improved']:
            color_map[method] = get_semantic_color('proposed')
        elif method.lower() in ['ablation', 'variant']:
            color_map[method] = get_semantic_color('ablation')
        else:
            color_map[method] = colors[i]
    
    return color_map

def get_performance_colors(values: np.ndarray, 
                          good_threshold: float = None,
                          poor_threshold: float = None) -> List[str]:
    """Get colors based on performance values."""
    if good_threshold is None:
        good_threshold = np.percentile(values, 75)
    if poor_threshold is None:
        poor_threshold = np.percentile(values, 25)
    
    colors = []
    for val in values:
        if val >= good_threshold:
            colors.append(get_semantic_color('good'))
        elif val <= poor_threshold:
            colors.append(get_semantic_color('poor'))
        else:
            colors.append(get_semantic_color('neutral'))
    
    return colors