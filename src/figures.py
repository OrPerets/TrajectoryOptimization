from __future__ import annotations

import os
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np

from .utils import palette, set_pub_style, cep


def _save(fig: plt.Figure, out_dir: str, name: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    for ext in ["png", "pdf"]:
        fig.savefig(os.path.join(out_dir, f"{name}.{ext}"), bbox_inches="tight")


def fig_trajectory_overlay(t: np.ndarray, telem_xyz: np.ndarray, raw_xyz: np.ndarray, qp_xyz: np.ndarray, qp_soc_xyz: Optional[np.ndarray], out_dir: str) -> None:
    set_pub_style()
    colors = palette()
    fig = plt.figure(figsize=(6, 4))
    ax = fig.add_subplot(1, 2, 1)
    ax.plot(telem_xyz[:, 0], telem_xyz[:, 1], color=colors[0], label="Telemetry")
    ax.plot(raw_xyz[:, 0], raw_xyz[:, 1], color=colors[1], alpha=0.7, label="Raw")
    ax.plot(qp_xyz[:, 0], qp_xyz[:, 1], color=colors[2], label="QP")
    if qp_soc_xyz is not None:
        ax.plot(qp_soc_xyz[:, 0], qp_soc_xyz[:, 1], color=colors[3], label="QP+SOC")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.legend()
    ax.set_title("XY")

    ax2 = fig.add_subplot(1, 2, 2)
    ax2.plot(telem_xyz[:, 0], telem_xyz[:, 2], color=colors[0], label="Telemetry")
    ax2.plot(raw_xyz[:, 0], raw_xyz[:, 2], color=colors[1], alpha=0.7, label="Raw")
    ax2.plot(qp_xyz[:, 0], qp_xyz[:, 2], color=colors[2], label="QP")
    if qp_soc_xyz is not None:
        ax2.plot(qp_soc_xyz[:, 0], qp_soc_xyz[:, 2], color=colors[3], label="QP+SOC")
    ax2.set_xlabel("x [m]")
    ax2.set_ylabel("z [m]")
    ax2.set_title("XZ")
    _save(fig, out_dir, "fig_F1_trajectory_overlay")


def fig_error_timeseries(t: np.ndarray, d_raw: np.ndarray, d_qp: np.ndarray, d_soc: Optional[np.ndarray], out_dir: str) -> None:
    set_pub_style()
    colors = palette()
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.plot(t, d_raw, color=colors[1], label="Raw")
    ax.plot(t, d_qp, color=colors[2], label="QP")
    if d_soc is not None:
        ax.plot(t, d_soc, color=colors[3], label="QP+SOC")
    ax.set_xlabel("t [s]")
    ax.set_ylabel(r"$\|p - \hat{p}\|$ [m]")
    ax.legend()
    _save(fig, out_dir, "fig_F2_time_series_errors")


def fig_error_ecdf_hist(d_raw: np.ndarray, d_qp: np.ndarray, d_soc: Optional[np.ndarray], out_dir: str) -> None:
    """Plot error ECDF and histogram (F3)."""
    set_pub_style()
    colors = palette()
    fig, (ax_ecdf, ax_hist) = plt.subplots(1, 2, figsize=(8, 3))

    def _ecdf(ax: plt.Axes, data: np.ndarray, color: str, label: str) -> None:
        x = np.sort(data)
        y = np.linspace(0.0, 1.0, len(x), endpoint=False)
        ax.step(x, y, where="post", color=color, label=label)

    _ecdf(ax_ecdf, d_raw, colors[1], "Raw")
    _ecdf(ax_ecdf, d_qp, colors[2], "QP")
    if d_soc is not None:
        _ecdf(ax_ecdf, d_soc, colors[3], "QP+SOC")
    ax_ecdf.set_xlabel(r"$\|p - \hat{p}\|$ [m]")
    ax_ecdf.set_ylabel("ECDF")
    ax_ecdf.set_title("Error ECDF")
    ax_ecdf.legend()

    max_val = max(
        np.max(d_raw), np.max(d_qp), np.max(d_soc) if d_soc is not None else 0.0
    )
    bins = np.linspace(0.0, max_val, 30)
    ax_hist.hist(d_raw, bins=bins, color=colors[1], alpha=0.5, label="Raw")
    ax_hist.hist(d_qp, bins=bins, color=colors[2], alpha=0.5, label="QP")
    if d_soc is not None:
        ax_hist.hist(d_soc, bins=bins, color=colors[3], alpha=0.5, label="QP+SOC")
    ax_hist.set_xlabel(r"$\|p - \hat{p}\|$ [m]")
    ax_hist.set_ylabel("Count")
    ax_hist.set_title("Error Histogram")
    ax_hist.legend()

    _save(fig, out_dir, "fig_F3_error_ecdf_hist")


def fig_outer_landscape(history: List[Dict], out_dir: str) -> None:
    """Plot outer-loop objective landscape (F6)."""
    set_pub_style()
    colors = palette()
    ks = [h["k"] for h in history]
    Js = [h["J"] for h in history]
    feas = [h["feasible"] for h in history]
    ks_feas = [k for k, f in zip(ks, feas) if f]
    Js_feas = [j for j, f in zip(Js, feas) if f]
    ks_infeas = [k for k, f in zip(ks, feas) if not f]
    Js_infeas = [j for j, f in zip(Js, feas) if not f]
    fig, ax = plt.subplots(figsize=(6, 4))
    if ks_infeas:
        ax.scatter(ks_infeas, Js_infeas, color=colors[1], marker="x", label="Infeasible")
    if ks_feas:
        ax.scatter(ks_feas, Js_feas, color=colors[2], marker="o", label="Feasible")
        j_best = min(Js_feas)
        k_best = ks_feas[Js_feas.index(j_best)]
        ax.scatter([k_best], [j_best], color=colors[0], marker="*", s=120, label="Best")
    ref_steps = [(h["k"], h["J"]) for h in history if h.get("stage") == "refine"]
    if len(ref_steps) > 1:
        ax.plot([k for k, _ in ref_steps], [J for _, J in ref_steps], color=colors[3], linestyle="--", label="Refine")
    ax.set_xscale("log")
    ax.set_xlabel("k")
    ax.set_ylabel("J")
    ax.legend()
    _save(fig, out_dir, "fig_F6_outer_landscape")


def fig_cep_curves(d_raw: np.ndarray, d_qp: np.ndarray, d_soc: Optional[np.ndarray], out_dir: str) -> None:
    """Plot CEP curves (F4)."""
    set_pub_style()
    colors = palette()
    
    # Calculate CEP values at different percentiles
    percentiles = np.linspace(0.1, 0.99, 50)
    cep_raw = [cep(d_raw, p) for p in percentiles]
    cep_qp = [cep(d_qp, p) for p in percentiles]
    
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(percentiles * 100, cep_raw, color=colors[1], label="Raw", linewidth=2)
    ax.plot(percentiles * 100, cep_qp, color=colors[2], label="QP", linewidth=2)
    
    if d_soc is not None:
        cep_soc = [cep(d_soc, p) for p in percentiles]
        ax.plot(percentiles * 100, cep_soc, color=colors[3], label="QP+SOC", linewidth=2)
    
    # Add horizontal lines for CEP50 and CEP90
    cep50_raw = cep(d_raw, 0.5)
    cep90_raw = cep(d_raw, 0.9)
    cep50_qp = cep(d_qp, 0.5)
    cep90_qp = cep(d_qp, 0.9)
    
    ax.axhline(y=cep50_raw, color=colors[1], linestyle='--', alpha=0.7, label=f"Raw CEP50: {cep50_raw:.1f}m")
    ax.axhline(y=cep90_raw, color=colors[1], linestyle=':', alpha=0.7, label=f"Raw CEP90: {cep90_raw:.1f}m")
    ax.axhline(y=cep50_qp, color=colors[2], linestyle='--', alpha=0.7, label=f"QP CEP50: {cep50_qp:.1f}m")
    ax.axhline(y=cep90_qp, color=colors[2], linestyle=':', alpha=0.7, label=f"QP CEP90: {cep90_qp:.1f}m")
    
    if d_soc is not None:
        cep50_soc = cep(d_soc, 0.5)
        cep90_soc = cep(d_soc, 0.9)
        ax.axhline(y=cep50_soc, color=colors[3], linestyle='--', alpha=0.7, label=f"QP+SOC CEP50: {cep50_soc:.1f}m")
        ax.axhline(y=cep90_soc, color=colors[3], linestyle=':', alpha=0.7, label=f"QP+SOC CEP90: {cep90_soc:.1f}m")
    
    ax.set_xlabel("Percentile [%]")
    ax.set_ylabel("Error [m]")
    ax.set_title("Cumulative Error Probability (CEP) Curves")
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(10, 99)
    
    _save(fig, out_dir, "fig_F4_cep_curves")


def fig_theta_sensitivity(d_raw: np.ndarray, d_qp: np.ndarray, theta_values: List[float], d_soc_by_theta: List[np.ndarray], out_dir: str) -> None:
    """Plot θ_max sensitivity analysis (F5)."""
    set_pub_style()
    colors = palette()
    
    # Calculate CEP90 for each theta value
    cep90_raw = cep(d_raw, 0.9)
    cep90_qp = cep(d_qp, 0.9)
    cep90_soc_values = [cep(d_soc, 0.9) for d_soc in d_soc_by_theta]
    
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.axhline(y=cep90_raw, color=colors[1], linestyle='-', label="Raw", linewidth=2)
    ax.axhline(y=cep90_qp, color=colors[2], linestyle='-', label="QP (no SOC)", linewidth=2)
    ax.plot(theta_values, cep90_soc_values, color=colors[3], marker='o', label="QP+SOC", linewidth=2)
    
    # Find best theta
    best_idx = np.argmin(cep90_soc_values)
    best_theta = theta_values[best_idx]
    best_cep90 = cep90_soc_values[best_idx]
    ax.scatter([best_theta], [best_cep90], color=colors[0], s=100, marker='*', 
               label=f"Best: θ={best_theta}° (CEP90={best_cep90:.1f}m)")
    
    ax.set_xlabel("θ_max [degrees]")
    ax.set_ylabel("CEP90 [m]")
    ax.set_title("Impact Angle Constraint Sensitivity")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    _save(fig, out_dir, "fig_F5_theta_sensitivity")


def fig_runtime_breakdown(runtime_data: Dict[str, float], out_dir: str) -> None:
    """Plot runtime breakdown (F7)."""
    set_pub_style()
    colors = palette()
    
    # Extract timing data
    stages = list(runtime_data.keys())
    times = list(runtime_data.values())
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    
    # Bar chart
    bars = ax1.bar(stages, times, color=colors[:len(stages)])
    ax1.set_ylabel("Time [s]")
    ax1.set_title("Runtime Breakdown")
    ax1.tick_params(axis='x', rotation=45)
    
    # Add value labels on bars
    for bar, time in zip(bars, times):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{time:.2f}s', ha='center', va='bottom')
    
    # Pie chart
    total_time = sum(times)
    if total_time > 0:
        ax2.pie(times, labels=stages, autopct='%1.1f%%', colors=colors[:len(stages)])
        ax2.set_title(f"Total: {total_time:.2f}s")
    
    plt.tight_layout()
    _save(fig, out_dir, "fig_F7_runtime_breakdown")


def fig_kinematics_sanity(t: np.ndarray, v_qp: np.ndarray, v_soc: Optional[np.ndarray], out_dir: str) -> None:
    """Plot kinematics sanity checks (F8)."""
    set_pub_style()
    colors = palette()
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(10, 8))
    
    # Velocity magnitude
    v_mag_qp = np.linalg.norm(v_qp, axis=1)
    ax1.plot(t, v_mag_qp, color=colors[2], label="QP", linewidth=2)
    if v_soc is not None:
        v_mag_soc = np.linalg.norm(v_soc, axis=1)
        ax1.plot(t, v_mag_soc, color=colors[3], label="QP+SOC", linewidth=2)
    ax1.set_xlabel("Time [s]")
    ax1.set_ylabel("Velocity [m/s]")
    ax1.set_title("Velocity Magnitude")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Vertical velocity
    ax2.plot(t, v_qp[:, 2], color=colors[2], label="QP", linewidth=2)
    if v_soc is not None:
        ax2.plot(t, v_soc[:, 2], color=colors[3], label="QP+SOC", linewidth=2)
    ax2.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    ax2.set_xlabel("Time [s]")
    ax2.set_ylabel("v_z [m/s]")
    ax2.set_title("Vertical Velocity")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Horizontal velocity components
    ax3.plot(t, v_qp[:, 0], color=colors[0], label="QP v_x", linewidth=2)
    ax3.plot(t, v_qp[:, 1], color=colors[1], label="QP v_y", linewidth=2)
    if v_soc is not None:
        ax3.plot(t, v_soc[:, 0], color=colors[0], linestyle='--', label="QP+SOC v_x", alpha=0.7)
        ax3.plot(t, v_soc[:, 1], color=colors[1], linestyle='--', label="QP+SOC v_y", alpha=0.7)
    ax3.set_xlabel("Time [s]")
    ax3.set_ylabel("Velocity [m/s]")
    ax3.set_title("Horizontal Velocity Components")
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Acceleration (finite difference)
    dt = t[1] - t[0] if len(t) > 1 else 1.0
    a_qp = np.diff(v_qp, axis=0) / dt
    a_mag_qp = np.linalg.norm(a_qp, axis=1)
    ax4.plot(t[:-1], a_mag_qp, color=colors[2], label="QP", linewidth=2)
    if v_soc is not None:
        a_soc = np.diff(v_soc, axis=0) / dt
        a_mag_soc = np.linalg.norm(a_soc, axis=1)
        ax4.plot(t[:-1], a_mag_soc, color=colors[3], label="QP+SOC", linewidth=2)
    ax4.set_xlabel("Time [s]")
    ax4.set_ylabel("Acceleration [m/s²]")
    ax4.set_title("Acceleration Magnitude")
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    _save(fig, out_dir, "fig_F8_kinematics_sanity")


def fig_impact_angle_hist(impact_angles_qp: np.ndarray, impact_angles_soc: Optional[np.ndarray], out_dir: str) -> None:
    """Plot impact angle histograms (F9)."""
    set_pub_style()
    colors = palette()
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    
    # QP impact angles
    ax1.hist(impact_angles_qp, bins=20, color=colors[2], alpha=0.7, label="QP")
    ax1.axvline(x=np.median(impact_angles_qp), color=colors[2], linestyle='--', 
                label=f"Median: {np.median(impact_angles_qp):.1f}°")
    ax1.set_xlabel("Impact Angle [degrees]")
    ax1.set_ylabel("Count")
    ax1.set_title("QP Impact Angle Distribution")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # QP+SOC impact angles
    if impact_angles_soc is not None:
        ax2.hist(impact_angles_soc, bins=20, color=colors[3], alpha=0.7, label="QP+SOC")
        ax2.axvline(x=np.median(impact_angles_soc), color=colors[3], linestyle='--',
                    label=f"Median: {np.median(impact_angles_soc):.1f}°")
        # Add constraint line
        ax2.axvline(x=15, color='r', linestyle='-', alpha=0.7, label="θ_max = 15°")
        ax2.set_xlabel("Impact Angle [degrees]")
        ax2.set_ylabel("Count")
        ax2.set_title("QP+SOC Impact Angle Distribution")
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    else:
        ax2.text(0.5, 0.5, "No SOC data", ha='center', va='center', transform=ax2.transAxes)
        ax2.set_title("QP+SOC Impact Angle Distribution")
    
    plt.tight_layout()
    _save(fig, out_dir, "fig_F9_impact_angle_hist")
