# Scientific Visualization System

This document describes the comprehensive visualization system for generating publication-quality figures for scientific papers.

## Overview

The visualization system provides:
- **Global styling**: Consistent typography, sizing, and colors across all figures
- **Statistical rigor**: Built-in confidence intervals, error analysis, and significance testing
- **Accessibility**: Colorblind-safe palettes and WCAG-compliant contrast ratios
- **Reproducibility**: Deterministic generation with proper metadata and versioning
- **Multiple formats**: PDF (vector), SVG (web/editing), and PNG (600 DPI) export

## Quick Start

### Generate All Figures

```bash
# Generate figures from existing data
python scripts/make_all_figures.py --input data/metrics.csv --out figures/

# Generate figures with sample data for testing
python scripts/make_all_figures.py --sample --out figures/

# Generate figures in specific formats
python scripts/make_all_figures.py --input data.csv --out figures/ --formats pdf svg
```

### Use Individual Plot Functions

```python
from viz.figures import plot_timeseries_ci, plot_violinbox
from viz.figstyle import use_paper_style

# Apply global styling
use_paper_style()

# Create time series with confidence intervals
fig, ax = plot_timeseries_ci(
    df, x='time', ys=['baseline', 'proposed'],
    ci=0.95, savepath='figures/timeseries'
)

# Create distribution comparison
fig, ax = plot_violinbox(
    {'baseline': errors_baseline, 'proposed': errors_proposed},
    "Error Distance [m]", savepath='figures/distribution'
)
```

## Figure Types

### 1. Time Series with Confidence Intervals
**Function**: `plot_timeseries_ci()`
**Use Case**: Showing performance over time with uncertainty bands
**Features**: 
- Confidence interval ribbons
- Optional Savitzky-Golay smoothing
- Event markers and annotations
- Terminal value labels

```python
fig, ax = plot_timeseries_ci(
    df, x='t', ys=['method1', 'method2'],
    ci=0.95,  # 95% confidence interval
    smooth={'window': 9, 'polyorder': 3},  # Optional smoothing
    highlight={'events': [5.2, 8.1], 'labels': ['Event A', 'Event B']}
)
```

### 2. Distribution Comparison (Violin + Box)
**Function**: `plot_violinbox()`
**Use Case**: Comparing performance distributions across methods
**Features**:
- Violin plots showing full distribution shape
- Box plot overlay for quartiles
- Individual data points with jitter
- Statistical annotations (N, Q1, Q3)

```python
fig, ax = plot_violinbox(
    {'baseline': errors_baseline, 'proposed': errors_proposed},
    "Error Distance [m]",
    order=['baseline', 'proposed']  # Control display order
)
```

### 3. ECDF Analysis
**Function**: `plot_ecdf()`
**Use Case**: Cumulative performance analysis and percentile comparison
**Features**:
- Step-wise ECDF curves
- Key percentile annotations (50%, 90%, 95%)
- Sample size reporting
- Multiple method comparison

```python
fig, ax = plot_ecdf(
    {'baseline': errors_baseline, 'proposed': errors_proposed},
    "Error Distance [m]"
)
```

### 4. Calibration Plot
**Function**: `plot_calibration()`
**Use Case**: Model validation and prediction accuracy
**Features**:
- Predictions vs. true values scatter
- Identity line for perfect calibration
- Statistical metrics (MSE, MAE, R²)
- Sample size reporting

```python
fig, ax = plot_calibration(
    predicted_values, true_values,
    savepath='figures/calibration'
)
```

### 5. Residual Analysis
**Function**: `plot_residuals()`
**Use Case**: Model diagnostics and validation
**Features**:
- 2×2 diagnostic panel
- Residuals vs. index
- Histogram distribution
- Q-Q plot for normality
- Autocorrelation analysis

```python
fig, axes = plot_residuals(
    residuals, savepath='figures/residuals'
)
```

### 6. Ablation Study
**Function**: `plot_ablation_ci()`
**Use Case**: Systematic comparison of method variants
**Features**:
- Performance deltas from baseline
- Confidence intervals with error bars
- Sorted by effect size
- Value labels on bars

```python
fig, ax = plot_ablation_ci(
    results_df, baseline_col='baseline',
    delta=True,  # Show deltas instead of absolute values
    savepath='figures/ablation'
)
```

### 7. Scatter Density
**Function**: `plot_scatter_density()`
**Use Case**: Correlation analysis and error vs. magnitude relationships
**Features**:
- Hexbin density visualization
- Pearson and Spearman correlation
- Statistical significance testing
- Sample size reporting

```python
fig, ax = plot_scatter_density(
    x_values, y_values, bins=40,
    savepath='figures/scatter_density'
)
```

### 8. Paired Comparison
**Function**: `plot_paired_swarm()`
**Use Case**: Before/after analysis and paired experiments
**Features**:
- Swarm plot with jitter
- Connecting lines between pairs
- Paired t-test statistics
- Mean lines and annotations

```python
fig, ax = plot_paired_swarm(
    before_values, after_values,
    link_lines=True,  # Connect paired points
    savepath='figures/paired_comparison'
)
```

### 9. Trajectory Visualization
**Function**: `plot_trajectory_xy()`
**Use Case**: 2D path visualization and analysis
**Features**:
- 2D trajectory plotting
- Optional color coding
- Start/end point markers
- Distance and efficiency metrics

```python
fig, ax = plot_trajectory_xy(
    x_coords, y_coords,
    colorby=velocity_values,  # Optional color coding
    savepath='figures/trajectory'
)
```

### 10. Summary Panel
**Function**: `create_summary_panel()`
**Use Case**: Combined results overview for paper
**Features**:
- 2×2 grid layout
- Unified styling and legend
- Key result combination
- Single figure for multiple insights

## Global Styling

### Typography Standards
- **Font Family**: Serif (STIX, Computer Modern, Times New Roman)
- **Title**: 11pt
- **Axis Labels**: 9.5pt  
- **Tick Labels**: 9pt
- **Legend**: 9pt
- **Captions**: 9pt

### Figure Sizing
- **Single Column**: 85mm width (golden ratio based)
- **Double Column**: 174mm width
- **Aspect Ratios**: 4:3, 16:10, 3:4, 1:1 (square)

### Color System
- **Discrete Palette**: 8 colorblind-safe colors
- **Semantic Mapping**: baseline, proposed, ablation, etc.
- **Sequential Maps**: viridis, magma, cividis
- **Diverging Maps**: RdBu, PiYG, BrBG

### Grid and Spines
- **Major Grid**: Light gray (alpha=0.25) on y-axis
- **Minor Grid**: Optional, lighter weight
- **Spines**: Top and right spines removed
- **Line Widths**: 1.8pt for data, 0.8pt for axes

## Statistical Features

### Confidence Intervals
- **Default**: 95% confidence level
- **Method**: Standard error-based (normal approximation)
- **Visualization**: Shaded ribbons for time series
- **Error Bars**: For bar charts and point plots

### Significance Testing
- **Paired Tests**: Wilcoxon signed-rank, paired t-test
- **Correlation**: Pearson and Spearman with p-values
- **Multiple Comparisons**: Holm-Bonferroni correction support
- **Effect Sizes**: Cohen's d, correlation coefficients

### Sample Size Reporting
- **Automatic**: N values in legends and annotations
- **Consistent**: Same format across all figures
- **Contextual**: Relevant to each specific analysis

## Output and Export

### File Formats
- **PDF**: Vector format for publication (default)
- **SVG**: Web-friendly vector format
- **PNG**: High-resolution raster (600 DPI)

### Metadata
- **Creator**: Scientific Visualization Pipeline
- **Software**: matplotlib version
- **Format**: Publication Ready
- **Timestamp**: Generation date/time

### File Naming
- **Format**: `Fig_XX_Description_Type`
- **Examples**: 
  - `Fig_01_Timeseries_CI_main.pdf`
  - `Fig_02_Distribution_Comparison_main.pdf`
  - `Fig_03_ECDF_Analysis_main.pdf`

## Figure Index for Paper

| Figure | File | Paper Section | Function | Description |
|--------|------|---------------|----------|-------------|
| 1 | `Fig_01_Timeseries_CI` | Results | `plot_timeseries_ci` | Performance over time with uncertainty |
| 2 | `Fig_02_Distribution_Comparison` | Results | `plot_violinbox` | Error distribution comparison |
| 3 | `Fig_03_ECDF_Analysis` | Results | `plot_ecdf` | Cumulative error probability |
| 4 | `Fig_04_Ablation_Study` | Ablation | `plot_ablation_ci` | Method variant comparison |
| 5 | `Fig_05_Trajectory_XY` | Results | `plot_trajectory_xy` | 2D trajectory visualization |
| 6 | `Fig_06_Scatter_Density` | Analysis | `plot_scatter_density` | Error correlation analysis |
| 7 | `Fig_07_Paired_Comparison` | Results | `plot_paired_swarm` | Before/after comparison |
| 8 | `Fig_08_Summary_Panel` | Results | `create_summary_panel` | Combined results overview |

## Usage Examples

### Basic Time Series
```python
import pandas as pd
from viz.figures import plot_timeseries_ci

# Load data
df = pd.read_csv('data/performance.csv')

# Create figure
fig, ax = plot_timeseries_ci(
    df, x='timestamp', ys=['baseline', 'proposed'],
    ci=0.95, savepath='figures/performance_timeseries'
)
```

### Method Comparison
```python
from viz.figures import plot_violinbox

# Prepare data
errors_by_method = {
    'baseline': baseline_errors,
    'proposed': proposed_errors,
    'ablation': ablation_errors
}

# Create comparison
fig, ax = plot_violinbox(
    errors_by_method, "Error Distance [m]",
    savepath='figures/method_comparison'
)
```

### Ablation Analysis
```python
from viz.figures import plot_ablation_ci

# Results DataFrame
results_df = pd.DataFrame({
    'baseline': baseline_performance,
    'proposed': proposed_performance,
    'ablation': ablation_performance
})

# Create ablation plot
fig, ax = plot_ablation_ci(
    results_df, 'baseline', delta=True,
    savepath='figures/ablation_analysis'
)
```

## Customization

### Style Overrides
```python
from viz.figstyle import use_paper_style, get_figure_size

# Use custom size
custom_size = get_figure_size('single', 16/10)

# Apply custom styling
use_paper_style()
plt.rcParams['axes.grid.alpha'] = 0.4  # Override grid alpha
```

### Color Customization
```python
from viz.palette import get_semantic_color, get_discrete_colors

# Get semantic colors
baseline_color = get_semantic_color('baseline')
proposed_color = get_semantic_color('proposed')

# Get custom palette
colors = get_discrete_colors(5, palette_name='high_contrast')
```

### Export Options
```python
from viz.figstyle import save_figure

# Custom export
save_figure(
    fig, 'figures/custom_figure',
    formats=['pdf', 'svg'],  # Only specific formats
    metadata={'Creator': 'Custom Analysis', 'Version': '1.0'}
)
```

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure `viz/` package is in Python path
2. **Missing Dependencies**: Install numpy, pandas, matplotlib, scipy
3. **Style Not Applied**: Call `use_paper_style()` before plotting
4. **Figure Sizes**: Use `get_figure_size()` for journal-compliant dimensions

### Debug Mode
```python
import matplotlib.pyplot as plt
plt.rcParams['figure.dpi'] = 72  # Lower DPI for faster testing
plt.rcParams['savefig.dpi'] = 72
```

### Testing
```bash
# Run unit tests
python -m pytest tests/test_figures.py -v

# Run specific test
python -m pytest tests/test_figures.py::TestVisualizationSystem::test_timeseries_ci -v
```

## Best Practices

1. **Consistency**: Always use the provided functions for consistent styling
2. **Reproducibility**: Set random seeds for deterministic results
3. **Accessibility**: Use semantic colors and ensure contrast ratios
4. **Documentation**: Include sample sizes and statistical methods in captions
5. **Version Control**: Track figure generation scripts and data sources

## Contributing

To add new figure types:

1. **Function**: Add to `viz/figures.py` with proper docstring
2. **Tests**: Add unit tests to `tests/test_figures.py`
3. **Documentation**: Update this README with usage examples
4. **CLI**: Add to `scripts/make_all_figures.py` if appropriate

## License

This visualization system is part of the scientific publication pipeline.
Please ensure proper attribution and follow journal-specific requirements.