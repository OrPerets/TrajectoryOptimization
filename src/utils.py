from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from scipy.stats import iqr, wilcoxon

# Plotting style

def set_pub_style() -> None:
    mpl.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 10,
            "axes.labelsize": 10,
            "legend.fontsize": 9,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "figure.autolayout": True,
        }
    )


# Robust column mapping

@dataclass
class ColumnMapping:
    required: List[str]
    aliases: Dict[str, List[str]]


def map_columns(df: pd.DataFrame, aliases: Dict[str, List[str]], required: Iterable[str]) -> pd.DataFrame:
    lower_cols = {c.lower(): c for c in df.columns}
    out = {}
    for canonical in required:
        alias_list = aliases.get(canonical, []) + [canonical]
        found = None
        for alias in alias_list:
            key = alias.lower()
            if key in lower_cols:
                found = lower_cols[key]
                break
        if found is None:
            raise ValueError(f"Missing required column for '{canonical}'. Available: {list(df.columns)}")
        out[canonical] = df[found]
    return pd.DataFrame(out)


# Preprocessing helpers

def iqr_outlier_mask_first_diff(x: np.ndarray, factor: float = 1.5) -> np.ndarray:
    dx = np.diff(x)
    if dx.size < 4:
        return np.ones_like(x, dtype=bool)
    q1 = np.quantile(dx, 0.25)
    q3 = np.quantile(dx, 0.75)
    rng = q3 - q1
    lo = q1 - factor * rng
    hi = q3 + factor * rng
    good = np.ones_like(x, dtype=bool)
    good[1:] = (dx >= lo) & (dx <= hi)
    return good


def apply_savgol(x: np.ndarray, window: int, polyorder: int) -> np.ndarray:
    window = max(3, window | 1)  # ensure odd
    polyorder = min(polyorder, window - 1)
    return savgol_filter(x, window_length=window, polyorder=polyorder, mode="interp")


# Alignment (E2)

def radial_velocity_from_velocity(v_xyz: np.ndarray, los_xyz: np.ndarray) -> np.ndarray:
    los_unit = los_xyz / (np.linalg.norm(los_xyz, axis=1, keepdims=True) + 1e-12)
    return np.sum(v_xyz * los_unit, axis=1)


def cross_correlation_lag(x: np.ndarray, y: np.ndarray, dt: float, max_lag_seconds: float) -> float:
    n = min(len(x), len(y))
    x = np.asarray(x[:n]) - np.mean(x[:n])
    y = np.asarray(y[:n]) - np.mean(y[:n])
    max_lag = int(max_lag_seconds / dt)
    lags = np.arange(-max_lag, max_lag + 1)
    corrs = []
    for lag in lags:
        if lag < 0:
            c = np.dot(x[: lag], y[-lag:])
        elif lag > 0:
            c = np.dot(x[lag:], y[:-lag])
        else:
            c = np.dot(x, y)
        corrs.append(c)
    corrs = np.array(corrs)
    k = int(np.argmax(corrs))
    # refine by quadratic fit around peak (lag-1, lag, lag+1)
    if 0 < k < len(lags) - 1:
        y0, y1, y2 = corrs[k - 1], corrs[k], corrs[k + 1]
        denom = (y0 - 2 * y1 + y2)
        if abs(denom) > 1e-12:
            delta = 0.5 * (y0 - y2) / denom
        else:
            delta = 0.0
    else:
        delta = 0.0
    lag_est = lags[k] + delta
    return float(lag_est * dt)


# Metrics (E3)

def trajectory_errors(p_est: np.ndarray, p_ref: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    diff = p_est - p_ref
    d = np.linalg.norm(diff, axis=1)
    horiz = np.linalg.norm(diff[:, :2], axis=1)
    vert = np.abs(diff[:, 2])
    return d, horiz, vert


def cep(values: np.ndarray, q: float) -> float:
    return float(np.quantile(values, q))


def rmse(values: np.ndarray) -> float:
    return float(np.sqrt(np.mean(values ** 2)))


def wilcoxon_signed(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    stat, p = wilcoxon(x, y, zero_method="wilcox", correction=False)
    return float(stat), float(p)


# Hash helpers

def sha256_of_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def save_json(path: str, obj: dict) -> None:
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, sort_keys=True)


def ensure_dir(path: str) -> None:
    import os

    os.makedirs(path, exist_ok=True)


# Color palette

def palette() -> List[str]:
    return [
        "#1b9e77",
        "#d95f02",
        "#7570b3",
        "#e7298a",
        "#66a61e",
        "#e6ab02",
        "#a6761d",
        "#666666",
    ]


def format_p_value(p_value: float) -> str:
    """Format p-value for display with significance indicators."""
    if p_value < 0.001:
        return f"{p_value:.3e} ***"
    elif p_value < 0.01:
        return f"{p_value:.3f} **"
    elif p_value < 0.05:
        return f"{p_value:.3f} *"
    else:
        return f"{p_value:.3f}"


def calculate_impact_angles(velocities: np.ndarray) -> np.ndarray:
    """Calculate impact angles from velocity vectors.
    
    Args:
        velocities: Array of shape (N, 3) with velocity vectors [vx, vy, vz]
        
    Returns:
        Array of shape (N,) with impact angles in degrees
    """
    # Impact angle is angle from horizontal (xy-plane) to velocity vector
    # tan(θ) = |v_horizontal| / |v_vertical|
    v_horizontal = np.linalg.norm(velocities[:, :2], axis=1)  # vx, vy
    v_vertical = np.abs(velocities[:, 2])  # |vz|
    
    # Avoid division by zero
    mask = v_vertical > 1e-12
    angles = np.zeros_like(v_horizontal)
    
    # Calculate angles where vz is significant
    angles[mask] = np.arctan2(v_horizontal[mask], v_vertical[mask])
    
    # Convert to degrees
    angles_deg = np.degrees(angles)
    
    return angles_deg


def aggregate_metrics_with_ci(metrics_list: List[Dict], confidence_level: float = 0.95) -> Dict[str, str]:
    """Aggregate metrics across sorties with confidence intervals.
    
    Args:
        metrics_list: List of metric dictionaries
        confidence_level: Confidence level for intervals (default: 0.95)
        
    Returns:
        Dictionary with aggregated metrics in format "median [Q25, Q75] (CI_lower, CI_upper)"
    """
    if not metrics_list:
        return {}
    
    # Convert to DataFrame for easier manipulation
    df = pd.DataFrame(metrics_list)
    
    aggregated = {}
    
    for col in df.columns:
        if df[col].dtype in ['float64', 'int64']:
            values = df[col].dropna()
            if len(values) > 0:
                # Calculate statistics
                median = np.median(values)
                q25, q75 = np.quantile(values, [0.25, 0.75])
                
                # Bootstrap confidence interval
                n_bootstrap = min(1000, len(values))
                bootstrap_medians = []
                for _ in range(n_bootstrap):
                    sample = np.random.choice(values, size=len(values), replace=True)
                    bootstrap_medians.append(np.median(sample))
                
                bootstrap_medians = np.array(bootstrap_medians)
                alpha = 1 - confidence_level
                ci_lower = np.quantile(bootstrap_medians, alpha / 2)
                ci_upper = np.quantile(bootstrap_medians, 1 - alpha / 2)
                
                aggregated[col] = f"{median:.2f} [{q25:.2f}, {q75:.2f}] ({ci_lower:.2f}, {ci_upper:.2f})"
            else:
                aggregated[col] = "N/A"
        else:
            # For non-numeric columns, just show unique values
            unique_vals = df[col].unique()
            aggregated[col] = ", ".join(str(v) for v in unique_vals[:3])  # Limit to first 3
    
    return aggregated
