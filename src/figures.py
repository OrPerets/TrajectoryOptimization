from __future__ import annotations

import os
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np

from .utils import palette, set_pub_style


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
