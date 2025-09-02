from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple
import logging

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


def normalize_inputs(data: Dict[str, np.ndarray], cfg: Dict) -> Tuple[Dict[str, np.ndarray], Dict[str, Dict[str, float]]]:
    """Center and scale position/velocity arrays to O(1) magnitudes.

    Parameters
    ----------
    data: Dict[str, np.ndarray]
        Dictionary containing at minimum ``"positions"`` with shape (N, 3). An optional
        ``"velocities"`` key may provide corresponding velocity samples.
    cfg: Dict
        Configuration dictionary with optional ``general`` scaling entries:
        ``enable_scaling``, ``scale_position_m``, and ``scale_velocity_mps``.

    Returns
    -------
    Tuple[Dict[str, np.ndarray], Dict[str, Dict[str, float]]]
        A tuple of (normalized_data, summary). ``normalized_data`` mirrors the
        structure of ``data`` with centered/scaled arrays. ``summary`` contains
        min/median/max magnitudes for the raw and normalized values, useful for
        unit sanity checks.
    """

    positions = np.asarray(data["positions"], dtype=float)
    velocities = np.asarray(data.get("velocities")) if "velocities" in data else None

    gcfg = cfg.get("general", {})
    enable = bool(gcfg.get("enable_scaling", True))
    scale_p = float(gcfg.get("scale_position_m", 1.0)) if enable else 1.0
    scale_v = float(gcfg.get("scale_velocity_mps", 1.0)) if enable else 1.0

    pos_center = positions - positions.mean(axis=0, keepdims=True)
    pos_norm = pos_center / scale_p

    vel_norm = None
    if velocities is not None:
        vel_norm = velocities / scale_v

    def _stats(arr: np.ndarray) -> Dict[str, float]:
        mag = np.linalg.norm(arr, axis=1)
        return {
            "min": float(np.min(mag)),
            "median": float(np.median(mag)),
            "max": float(np.max(mag)),
        }

    summary: Dict[str, Dict[str, float]] = {
        "position_raw": _stats(positions),
        "position_norm": _stats(pos_norm),
    }
    if velocities is not None and vel_norm is not None:
        summary["velocity_raw"] = _stats(velocities)
        summary["velocity_norm"] = _stats(vel_norm)

    norm_data: Dict[str, np.ndarray] = {"positions": pos_norm}
    if vel_norm is not None:
        norm_data["velocities"] = vel_norm
    return norm_data, summary


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


def wilcoxon_comparison_matrix(error_data: Dict[str, np.ndarray]) -> pd.DataFrame:
    """Compute pairwise Wilcoxon signed-rank tests between methods.

    Parameters
    ----------
    error_data: Dict[str, np.ndarray]
        Mapping from method name to error distances of equal length.

    Returns
    -------
    pd.DataFrame
        Table with columns ["Group1", "Group2", "Statistic", "P_value"].
    """

    methods = list(error_data.keys())
    rows: List[Dict[str, float]] = []
    for i in range(len(methods)):
        for j in range(i + 1, len(methods)):
            a = np.asarray(error_data[methods[i]])
            b = np.asarray(error_data[methods[j]])
            n = min(len(a), len(b))
            stat, p = wilcoxon_signed(a[:n], b[:n])
            rows.append(
                {
                    "Group1": methods[i],
                    "Group2": methods[j],
                    "Statistic": stat,
                    "P_value": p,
                }
            )
    return pd.DataFrame(rows)


def check_error_sanity(errors: np.ndarray, name: str = "errors", max_m: float = 1e5) -> None:
    """Raise ``ValueError`` if metrics contain NaNs or unrealistic magnitudes.

    Parameters
    ----------
    errors: np.ndarray
        Array of error distances in meters.
    name: str, optional
        Name used in the exception message.
    max_m: float, optional
        Median threshold in meters above which values are considered implausible.
    """

    arr = np.asarray(errors, dtype=float)
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contain non-finite values")
    med = float(np.nanmedian(np.abs(arr)))
    if med > max_m:
        raise ValueError(
            f"{name} median {med:.1f} m exceeds sanity threshold {max_m} m; check unit scaling"
        )


def check_mapping_consistency(
    radar_xyz: np.ndarray, telem_xyz: np.ndarray, threshold: float = 1e4
) -> None:
    """Verify radar and telemetry positions are co-located within a reasonable bound.

    This helps catch frame or unit mismatches early in the pipeline.
    """

    diff = np.linalg.norm(radar_xyz - telem_xyz, axis=1)
    med = float(np.nanmedian(diff))
    if med > threshold:
        raise ValueError(
            f"Radar/telemetry mapping mismatch: median position diff {med:.1f} m exceeds {threshold} m"
        )


# Position scale sanity (pre-optimization)

def check_position_scale(
    positions: np.ndarray,
    name: str = "positions",
    abs_position_median_max_m: float = 1e4,
    median_step_max_m: float = 1e3,
    enforce: str = "error",
    hard_max_m: float = 1e7,
) -> None:
    """Validate that positions are within expected scales.

    Raises ValueError if median absolute position magnitude or median
    inter-sample displacement exceeds thresholds.

    Parameters
    ----------
    positions: np.ndarray
        Array of shape (T, 3) with local-frame positions in meters.
    name: str
        Label used in error messages.
    abs_position_median_max_m: float
        Threshold for median absolute position magnitude (meters).
    median_step_max_m: float
        Threshold for median inter-sample displacement (meters).
    """

    arr = np.asarray(positions, dtype=float)
    if arr.ndim != 2 or arr.shape[1] != 3:
        raise ValueError(f"{name} must be (T,3) array in meters")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contain non-finite values")

    pos_med = float(np.median(np.linalg.norm(arr, axis=1)))
    # Helpful diagnostics for logs
    pos_p95 = float(np.percentile(np.linalg.norm(arr, axis=1), 95))
    if arr.shape[0] >= 2:
        steps = np.linalg.norm(np.diff(arr, axis=0), axis=1)
        step_med_diag = float(np.median(steps))
        step_p95_diag = float(np.percentile(steps, 95))
    else:
        step_med_diag = 0.0
        step_p95_diag = 0.0
    logging.getLogger(__name__).info(
        f"{name} scale: median |p|={pos_med:.1f}m (p95={pos_p95:.1f}m), step_med={step_med_diag:.2f}m, step_p95={step_p95_diag:.2f}m"
    )
    if pos_med > hard_max_m:
        raise ValueError(
            f"{name}: median |p| = {pos_med:.1f} m exceeds hard limit {hard_max_m:.1f} m; check mapping/units"
        )
    if pos_med > abs_position_median_max_m:
        msg = (
            f"{name}: median |p| = {pos_med:.1f} m exceeds {abs_position_median_max_m:.1f} m; check mapping/units"
        )
        if str(enforce).lower() == "warn":
            logging.getLogger(__name__).warning(msg)
        else:
            raise ValueError(msg)

    if arr.shape[0] >= 2:
        steps = np.linalg.norm(np.diff(arr, axis=0), axis=1)
        step_med = float(np.median(steps))
        if step_med > hard_max_m:
            raise ValueError(
                f"{name}: median step = {step_med:.1f} m exceeds hard limit {hard_max_m:.1f} m; check dt/units"
            )
        if step_med > median_step_max_m:
            msg = (
                f"{name}: median inter-sample step = {step_med:.1f} m exceeds {median_step_max_m:.1f} m; check dt/units"
            )
            if str(enforce).lower() == "warn":
                logging.getLogger(__name__).warning(msg)
            else:
                raise ValueError(msg)


# Runtime breakdown cleanup (avoid double counting)

def clean_runtime_breakdown(runtime_data: Dict[str, float]) -> Dict[str, float]:
    """Return a cleaned runtime dict without aggregated totals.

    Removes 'total_runtime' always.
    If granular ablation keys are present, removes 'ablation_studies'.
    """

    cleaned: Dict[str, float] = {}
    has_ablation_parts = any(k in runtime_data for k in ("soc_ablation", "theta_sensitivity"))
    for k, v in runtime_data.items():
        if k == "total_runtime":
            continue
        if has_ablation_parts and k == "ablation_studies":
            continue
        cleaned[k] = float(v)
    return cleaned


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
    if not np.all(np.isfinite(velocities)):
        raise ValueError("Velocity array contains NaN or inf values")

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
