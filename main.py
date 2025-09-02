from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from typing import Dict, List

import numpy as np
import pandas as pd
import yaml

from src.data import Processed, load_config, load_streams, resample_and_filter
from src.model import ProblemData, Scaling, SolverOptions, solve_inner
from src.outer import OuterConfig, outer_search
from src.tables import (
    table_aggregate, 
    table_per_sortie, 
    table_ablation_soc,
    table_ablation_theta,
    table_runtime_breakdown,
    table_solver_settings,
    write_results_index
)
from src.figures import (
    fig_trajectory_overlay,
    fig_error_timeseries,
    fig_error_ecdf_hist,
    fig_cep_curves,
    fig_theta_sensitivity,
    fig_outer_landscape,
    fig_runtime_breakdown,
    fig_kinematics_sanity,
    fig_impact_angle_hist,
)
from src.utils import (
    ensure_dir,
    rmse,
    save_json,
    set_pub_style,
    trajectory_errors,
    cep,
    calculate_impact_angles,
    wilcoxon_comparison_matrix,
    format_p_value,
    aggregate_metrics_with_ci,
)


def infer_weights_auto(radar_xyz: np.ndarray, dt: float, eta: float, gamma: float) -> Dict[str, float]:
    # stationary segment heuristic: first 5 seconds
    n0 = int(max(5.0 / dt, 10))
    seg = radar_xyz[:n0, :]
    sigmas = np.std(seg - seg.mean(axis=0, keepdims=True), axis=0)
    sigma_p2 = float(np.mean(sigmas ** 2))
    w_pos = 1.0 / max(sigma_p2, 1e-4)
    w_smooth = eta / (dt ** 2 * max(sigma_p2, 1e-4))
    w_TV = gamma * w_smooth
    w_final = w_pos
    # Clamp weights to numerically stable ranges
    def _clamp(x: float, lo: float, hi: float) -> float:
        return float(min(max(x, lo), hi))
    w_pos = _clamp(w_pos, 1e-3, 10.0)
    w_smooth = _clamp(w_smooth, 1e-5, 10.0)
    w_TV = _clamp(w_TV, 0.0, 1.0)
    w_final = _clamp(w_final, 1e-3, 10.0)
    return dict(w_pos=w_pos, w_smooth=w_smooth, w_TV=w_TV, w_final=w_final)


def run_ablation_studies(
    problem_data: ProblemData, 
    scaling: Scaling, 
    ocfg: OuterConfig, 
    cfg: Dict,
    out_dir: str
) -> Dict:
    """Run ablation studies for Sprint 4."""
    
    # Track timing for each stage
    timing_data = {}
    
    print("Running ablation studies...")
    
    # 1. SOC on/off ablation
    start_time = time.time()
    print("  - SOC on/off ablation...")
    
    sopts_qp = SolverOptions(
        enable_soc=False,
        theta_max_deg=cfg["solver"].get("theta_max_deg", 15),
        terminal_altitude_window_m=tuple(cfg["solver"].get("terminal_altitude_window_m", [-5, 5])),
        terminal_xy_box_m=float(cfg["solver"].get("terminal_xy_box_m", 500.0)),
        osqp_opts=cfg["solver"].get("osqp", {}),
        ecos_opts=cfg["solver"].get("ecos", {}),
    )
    
    sopts_soc = SolverOptions(
        enable_soc=True,
        theta_max_deg=cfg["solver"].get("theta_max_deg", 15),
        terminal_altitude_window_m=tuple(cfg["solver"].get("terminal_altitude_window_m", [-5, 5])),
        terminal_xy_box_m=float(cfg["solver"].get("terminal_xy_box_m", 500.0)),
        osqp_opts=cfg["solver"].get("osqp", {}),
        ecos_opts=cfg["solver"].get("ecos", {}),
    )
    
    res_qp = outer_search(problem_data, scaling, sopts_qp, ocfg)
    res_soc = outer_search(problem_data, scaling, sopts_soc, ocfg)
    
    timing_data["soc_ablation"] = time.time() - start_time
    
    # 2. θ_max sensitivity analysis
    start_time = time.time()
    print("  - θ_max sensitivity analysis...")
    
    theta_values = cfg.get("ablations", {}).get("theta_grid_deg", [10, 15, 20])
    theta_results = []
    
    for theta in theta_values:
        sopts_theta = SolverOptions(
            enable_soc=True,
            theta_max_deg=theta,
            terminal_altitude_window_m=tuple(cfg["solver"].get("terminal_altitude_window_m", [-5, 5])),
            terminal_xy_box_m=float(cfg["solver"].get("terminal_xy_box_m", 500.0)),
            osqp_opts=cfg["solver"].get("osqp", {}),
            ecos_opts=cfg["solver"].get("ecos", {}),
        )
        
        res_theta = outer_search(problem_data, scaling, sopts_theta, ocfg)
        if res_theta.sol_best and res_theta.sol_best.p is not None:
            theta_results.append({
                "theta_max_deg": theta,
                "k_best": res_theta.k_best,
                "J_best": res_theta.J_best,
                "solution": res_theta.sol_best
            })
    
    timing_data["theta_sensitivity"] = time.time() - start_time
    
    return {
        "qp_result": res_qp,
        "soc_result": res_soc,
        "theta_results": theta_results,
        "timing_data": timing_data
    }


def generate_all_figures(
    processed: Processed,
    results: Dict,
    ablation_data: Dict,
    out_dir: str
) -> None:
    """Generate all Sprint 4 figures."""
    
    print("Generating figures...")
    
    figures_dir = os.path.join(out_dir, "figures")
    ensure_dir(figures_dir)
    
    # Extract data
    res_qp = results["qp_result"]
    res_soc = results.get("soc_result")
    theta_results = ablation_data.get("theta_results", [])
    timing_data = ablation_data.get("timing_data", {})
    
    # Calculate errors for all methods
    d_raw, _, _ = trajectory_errors(processed.radar_xyz, processed.telem_xyz)
    d_qp, _, _ = trajectory_errors(res_qp.sol_best.p, processed.telem_xyz)
    d_soc = None
    if res_soc and res_soc.sol_best and res_soc.sol_best.p is not None:
        d_soc, _, _ = trajectory_errors(res_soc.sol_best.p, processed.telem_xyz)
    
    # F1: Trajectory overlay (already exists)
    qp_soc_xyz = res_soc.sol_best.p if res_soc and res_soc.sol_best else None
    fig_trajectory_overlay(
        processed.time, processed.telem_xyz, processed.radar_xyz, 
        res_qp.sol_best.p, qp_soc_xyz, figures_dir
    )
    
    # F2: Error time series (already exists)
    fig_error_timeseries(processed.time, d_raw, d_qp, d_soc, figures_dir)
    
    # F3: ECDF/histogram (already exists)
    fig_error_ecdf_hist(d_raw, d_qp, d_soc, figures_dir)
    
    # F4: CEP curves
    fig_cep_curves(d_raw, d_qp, d_soc, figures_dir)
    
    # F5: θ_max sensitivity
    if theta_results:
        theta_values = [r["theta_max_deg"] for r in theta_results]
        d_soc_by_theta = []
        for r in theta_results:
            if r["solution"] and r["solution"].p is not None:
                d_theta, _, _ = trajectory_errors(r["solution"].p, processed.telem_xyz)
                d_soc_by_theta.append(d_theta)
        
        if d_soc_by_theta:
            fig_theta_sensitivity(d_raw, d_qp, theta_values, d_soc_by_theta, figures_dir)
    
    # F6: Outer landscape (already exists - called if history available)
    if hasattr(res_qp, 'history') and res_qp.history:
        fig_outer_landscape(res_qp.history, figures_dir)
    
    # F7: Runtime breakdown
    if timing_data:
        fig_runtime_breakdown(timing_data, figures_dir)
    
    # F8: Kinematics sanity
    if res_qp.sol_best and res_qp.sol_best.v is not None:
        v_soc = res_soc.sol_best.v if res_soc and res_soc.sol_best and res_soc.sol_best.v is not None else None
        fig_kinematics_sanity(processed.time, res_qp.sol_best.v, v_soc, figures_dir)
    
    # F9: Impact angle histogram
    if res_qp.sol_best and res_qp.sol_best.v is not None:
        impact_angles_qp = calculate_impact_angles(res_qp.sol_best.v)
        impact_angles_soc = None
        if res_soc and res_soc.sol_best and res_soc.sol_best.v is not None:
            impact_angles_soc = calculate_impact_angles(res_soc.sol_best.v)
        
        fig_impact_angle_hist(impact_angles_qp, impact_angles_soc, figures_dir)
    
    print(f"  Generated figures in {figures_dir}")


def generate_all_tables(
    processed: Processed,
    results: Dict,
    ablation_data: Dict,
    cfg: Dict,
    out_dir: str
) -> None:
    """Generate all Sprint 4 tables."""
    
    print("Generating tables...")
    
    tables_dir = os.path.join(out_dir, "tables")
    ensure_dir(tables_dir)
    
    # Extract data
    res_qp = results["qp_result"]
    res_soc = results.get("soc_result")
    theta_results = ablation_data.get("theta_results", [])
    timing_data = ablation_data.get("timing_data", {})
    
    # Calculate metrics
    d_raw, _, _ = trajectory_errors(processed.radar_xyz, processed.telem_xyz)
    d_qp, _, _ = trajectory_errors(res_qp.sol_best.p, processed.telem_xyz)
    
    CEP50_raw = cep(d_raw, 0.5)
    CEP90_raw = cep(d_raw, 0.9)
    CEP50_qp = cep(d_qp, 0.5)
    CEP90_qp = cep(d_qp, 0.9)
    RMSE_qp = rmse(d_qp)
    TerminalMiss_qp = float(np.linalg.norm(res_qp.sol_best.p[-1] - processed.telem_xyz[-1]))
    
    has_soc = res_soc is not None and res_soc.sol_best and res_soc.sol_best.p is not None
    if has_soc:
        d_soc, _, _ = trajectory_errors(res_soc.sol_best.p, processed.telem_xyz)
        CEP50_soc = cep(d_soc, 0.5)
        CEP90_soc = cep(d_soc, 0.9)
        RMSE_soc = rmse(d_soc)
        TerminalMiss_soc = float(np.linalg.norm(res_soc.sol_best.p[-1] - processed.telem_xyz[-1]))
    else:
        CEP50_soc = None
        CEP90_soc = None
        RMSE_soc = None
        TerminalMiss_soc = None
    
    # T1 & T2: Basic metrics (already implemented)
    metrics_row = {
        "CEP50_raw": CEP50_raw,
        "CEP90_raw": CEP90_raw,
        "CEP50_qp": CEP50_qp,
        "CEP90_qp": CEP90_qp,
        "RMSE_qp": RMSE_qp,
        "TerminalMiss_qp": TerminalMiss_qp,
        "CEP50_soc": CEP50_soc if CEP50_soc is not None else np.nan,
        "CEP90_soc": CEP90_soc if CEP90_soc is not None else np.nan,
        "RMSE_soc": RMSE_soc if RMSE_soc is not None else np.nan,
        "TerminalMiss_soc": TerminalMiss_soc if TerminalMiss_soc is not None else np.nan,
    }
    
    table_per_sortie([metrics_row], tables_dir)
    table_aggregate([metrics_row], tables_dir)
    
    # T3: SOC ablation
    if has_soc:
        metrics_soc_on = [metrics_row]
        metrics_soc_off = [{
            "CEP50_qp": CEP50_qp,
            "CEP90_qp": CEP90_qp,
            "RMSE_qp": RMSE_qp,
            "TerminalMiss_qp": TerminalMiss_qp,
        }]
        table_ablation_soc(metrics_soc_on, metrics_soc_off, tables_dir)
    
    # T4: θ_max sensitivity
    if theta_results:
        theta_metrics = []
        for r in theta_results:
            if r["solution"] and r["solution"].p is not None:
                d_theta, _, _ = trajectory_errors(r["solution"].p, processed.telem_xyz)
                theta_metrics.append({
                    "theta_max_deg": r["theta_max_deg"],
                    "CEP90_soc": cep(d_theta, 0.9),
                    "RMSE_soc": rmse(d_theta),
                    "TerminalMiss_soc": float(np.linalg.norm(r["solution"].p[-1] - processed.telem_xyz[-1]))
                })
        
        if theta_metrics:
            table_ablation_theta(theta_metrics, tables_dir)
    
    # T5: Runtime breakdown
    if timing_data:
        table_runtime_breakdown(timing_data, tables_dir)
    
    # T6: Solver settings
    table_solver_settings(cfg["solver"], tables_dir)
    
    # Statistical tests
    if has_soc:
        error_data = {
            "Raw": d_raw,
            "QP": d_qp,
            "QP+SOC": d_soc
        }
        
        wilcoxon_results = wilcoxon_comparison_matrix(error_data)
        if not wilcoxon_results.empty:
            wilcoxon_results["P_value_formatted"] = wilcoxon_results["P_value"].apply(format_p_value)
            wilcoxon_results.to_csv(os.path.join(tables_dir, "wilcoxon_tests.csv"), index=False)
    
    # Update results index
    write_results_index(tables_dir)
    
    print(f"  Generated tables in {tables_dir}")


def run_once(args: argparse.Namespace) -> None:
    cfg = load_config(args.config)
    if args.enable_soc is not None:
        cfg["solver"]["enable_soc"] = bool(args.enable_soc)
    if args.theta_max is not None:
        cfg["solver"]["theta_max_deg"] = float(args.theta_max)
    if args.k_grid_points is not None:
        cfg["outer"]["k_grid_points"] = int(args.k_grid_points)

    out_dir = args.out_dir or cfg["paths"]["out_dir"]
    out_dir = os.path.abspath(out_dir)
    ensure_dir(out_dir)

    # Track overall timing
    start_time = time.time()

    # Log environment
    env = {
        "python": sys.version,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    ensure_dir(os.path.join(out_dir, "logs"))
    save_json(os.path.join(out_dir, "logs", "env.txt"), env)

    # Load and preprocess
    load_start = time.time()
    streams = load_streams(args.radar, args.telemetry, cfg)
    processed: Processed = resample_and_filter(streams, cfg, out_dir)
    load_time = time.time() - load_start

    # Weights (E4)
    if cfg["weights"].get("auto_calibrate", True):
        W = infer_weights_auto(processed.radar_xyz, cfg["preprocessing"]["dt"], cfg["weights"]["eta"], cfg["weights"]["gamma"])
    else:
        W = {
            "w_pos": cfg["weights"]["w_pos"],
            "w_smooth": cfg["weights"]["w_smooth"],
            "w_TV": cfg["weights"]["w_TV"],
            "w_final": cfg["weights"]["w_final"],
        }

    # Build inner problem data (radar positions as observations)
    problem_data = ProblemData(
        dt=float(cfg["preprocessing"]["dt"]),
        positions_obs=processed.radar_xyz,
        weights=W,
        launch_fix=processed.radar_xyz[0, :],
    )

    scaling = Scaling(enable=cfg["solver"].get("enable_scaling", True), alpha_p=cfg["solver"].get("alpha_p", 1e4), alpha_v=cfg["solver"].get("alpha_v", 1e3))
    
    ocfg = OuterConfig(
        k_min=float(cfg["outer"].get("k_min", 0.02)),
        k_max=float(cfg["outer"].get("k_max", 0.5)),
        k_grid_points=int(cfg["outer"].get("k_grid_points", 21)),
        tol_k=float(cfg["outer"].get("tol_k", 1e-3)),
        tol_J=float(cfg["outer"].get("tol_J", 1e-3)),
        max_iters=int(cfg["outer"].get("max_iters", 30)),
    )

    # Run baseline optimization
    baseline_start = time.time()
    sopts_qp = SolverOptions(
        enable_soc=False,
        theta_max_deg=cfg["solver"].get("theta_max_deg", 15),
        terminal_altitude_window_m=tuple(cfg["solver"].get("terminal_altitude_window_m", [-5, 5])),
        terminal_xy_box_m=float(cfg["solver"].get("terminal_xy_box_m", 500.0)),
        osqp_opts=cfg["solver"].get("osqp", {}),
        ecos_opts=cfg["solver"].get("ecos", {}),
    )
    
    res_qp = outer_search(problem_data, scaling, sopts_qp, ocfg)
    
    # Fallback if no feasible solution found
    if res_qp.sol_best is None or res_qp.sol_best.p is None:
        k0 = float(np.exp(0.5 * (np.log(ocfg.k_min) + np.log(ocfg.k_max))))
        sol0 = solve_inner(problem_data, k0, scaling, sopts_qp)
        if sol0.p is not None:
            class _Tmp:
                pass
            tmp = _Tmp()
            tmp.k_best = k0
            tmp.J_best = float(sol0.objective) if sol0.objective is not None else float('inf')
            tmp.sol_best = sol0
            res_qp = tmp  # duck-typed for downstream use
    
    baseline_time = time.time() - baseline_start

    # Run ablation studies
    ablation_start = time.time()
    ablation_data = run_ablation_studies(problem_data, scaling, ocfg, cfg, out_dir)
    ablation_time = time.time() - ablation_start

    # Combine results
    results = {
        "qp_result": res_qp,
        "soc_result": ablation_data.get("soc_result"),
    }
    
    # Add timing data
    ablation_data["timing_data"].update({
        "data_loading": load_time,
        "baseline_optimization": baseline_time,
        "ablation_studies": ablation_time,
    })

    # Generate figures
    figures_start = time.time()
    generate_all_figures(processed, results, ablation_data, out_dir)
    figures_time = time.time() - figures_start
    ablation_data["timing_data"]["figure_generation"] = figures_time

    # Generate tables
    tables_start = time.time()
    generate_all_tables(processed, results, ablation_data, cfg, out_dir)
    tables_time = time.time() - tables_start
    ablation_data["timing_data"]["table_generation"] = tables_time

    # Save artifacts
    res_dir = os.path.join(out_dir, "results")
    ensure_dir(res_dir)
    df_sol = pd.DataFrame({"time": processed.time, "x": res_qp.sol_best.p[:, 0], "y": res_qp.sol_best.p[:, 1], "z": res_qp.sol_best.p[:, 2]})
    try:
        df_sol.to_parquet(os.path.join(out_dir, "solution_traj.parquet"))
    except Exception:
        df_sol.to_csv(os.path.join(out_dir, "solution_traj.csv"), index=False)
    
    res_soc = results.get("soc_result")
    save_json(os.path.join(out_dir, "best_k.json"), {
        "k_star_qp": res_qp.k_best,
        "J_star_qp": res_qp.J_best,
        "k_star_soc": (res_soc.k_best if res_soc is not None else None),
        "J_star_soc": (res_soc.J_best if res_soc is not None else None)
    })

    # Save timing summary
    total_time = time.time() - start_time
    ablation_data["timing_data"]["total_runtime"] = total_time
    save_json(os.path.join(out_dir, "timing_summary.json"), ablation_data["timing_data"])

    # Calculate final metrics for output
    d_raw, _, _ = trajectory_errors(processed.radar_xyz, processed.telem_xyz)
    d_qp, _, _ = trajectory_errors(res_qp.sol_best.p, processed.telem_xyz)
    CEP90_qp = cep(d_qp, 0.9)
    
    CEP90_soc = None
    if res_soc and res_soc.sol_best and res_soc.sol_best.p is not None:
        d_soc, _, _ = trajectory_errors(res_soc.sol_best.p, processed.telem_xyz)
        CEP90_soc = cep(d_soc, 0.9)

    # Done
    print("\n" + "="*60)
    print("SPRINT 4 IMPLEMENTATION COMPLETE")
    print("="*60)
    print(f"Total runtime: {total_time:.2f}s")
    print(f"Output directory: {out_dir}")
    print(f"Figures generated: {len(os.listdir(os.path.join(out_dir, 'figures')))} files")
    print(f"Tables generated: {len(os.listdir(os.path.join(out_dir, 'tables')))} files")
    print("="*60)
    
    print(json.dumps({
        "k_star_qp": res_qp.k_best,
        "k_star_soc": (res_soc.k_best if res_soc is not None else None),
        "CEP90_qp": CEP90_qp,
        "CEP90_soc": CEP90_soc,
        "total_runtime_s": total_time
    }, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--radar", type=str, required=True)
    parser.add_argument("--telemetry", type=str, required=True)
    parser.add_argument("--config", type=str, default="config/default.yaml")
    parser.add_argument("--out_dir", type=str, default=None)
    parser.add_argument("--enable_soc", type=lambda x: str(x).lower() == "true", default=None)
    parser.add_argument("--theta_max", type=float, default=None)
    parser.add_argument("--k_grid_points", type=int, default=None)
    args = parser.parse_args()
    run_once(args)
