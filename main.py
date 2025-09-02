from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from typing import Dict

import numpy as np
import pandas as pd
import yaml

from src.data import Processed, load_config, load_streams, resample_and_filter
from src.model import ProblemData, Scaling, SolverOptions, solve_inner
from src.outer import OuterConfig, outer_search
from src.tables import table_aggregate, table_per_sortie, write_results_index
from src.utils import (
    ensure_dir,
    rmse,
    save_json,
    set_pub_style,
    trajectory_errors,
    cep,
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
    return dict(w_pos=w_pos, w_smooth=w_smooth, w_TV=w_TV, w_final=w_final)


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

    # Log environment
    env = {
        "python": sys.version,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    ensure_dir(os.path.join(out_dir, "logs"))
    save_json(os.path.join(out_dir, "logs", "env.txt"), env)

    # Load and preprocess
    streams = load_streams(args.radar, args.telemetry, cfg)
    processed: Processed = resample_and_filter(streams, cfg, out_dir)

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
    pd = ProblemData(
        dt=float(cfg["preprocessing"]["dt"]),
        positions_obs=processed.radar_xyz,
        weights=W,
        launch_fix=processed.radar_xyz[0, :],
    )

    scaling = Scaling(enable=cfg["solver"].get("enable_scaling", True), alpha_p=cfg["solver"].get("alpha_p", 1e4), alpha_v=cfg["solver"].get("alpha_v", 1e3))
    sopts_qp = SolverOptions(
        enable_soc=False,
        theta_max_deg=cfg["solver"].get("theta_max_deg", 15),
        terminal_altitude_window_m=tuple(cfg["solver"].get("terminal_altitude_window_m", [-5, 5])),
        terminal_xy_box_m=float(cfg["solver"].get("terminal_xy_box_m", 500.0)),
        osqp_opts=cfg["solver"].get("osqp", {}),
        ecos_opts=cfg["solver"].get("ecos", {}),
    )
    sopts_soc = SolverOptions(
        enable_soc=cfg["solver"].get("enable_soc", True),
        theta_max_deg=cfg["solver"].get("theta_max_deg", 15),
        terminal_altitude_window_m=tuple(cfg["solver"].get("terminal_altitude_window_m", [-5, 5])),
        terminal_xy_box_m=float(cfg["solver"].get("terminal_xy_box_m", 500.0)),
        osqp_opts=cfg["solver"].get("osqp", {}),
        ecos_opts=cfg["solver"].get("ecos", {}),
    )

    ocfg = OuterConfig(
        k_min=float(cfg["outer"].get("k_min", 0.02)),
        k_max=float(cfg["outer"].get("k_max", 0.5)),
        k_grid_points=int(cfg["outer"].get("k_grid_points", 21)),
        tol_k=float(cfg["outer"].get("tol_k", 1e-3)),
        tol_J=float(cfg["outer"].get("tol_J", 1e-3)),
        max_iters=int(cfg["outer"].get("max_iters", 30)),
    )

    # Outer search: QP baseline
    res_qp = outer_search(pd, scaling, sopts_qp, ocfg)
    # Fallback if no feasible solution found
    if res_qp.sol_best is None or res_qp.sol_best.p is None:
        k0 = float(np.exp(0.5 * (np.log(ocfg.k_min) + np.log(ocfg.k_max))))
        sol0 = solve_inner(pd, k0, scaling, sopts_qp)
        if sol0.p is not None:
            class _Tmp:
                pass
            tmp = _Tmp()
            tmp.k_best = k0
            tmp.J_best = float(sol0.objective) if sol0.objective is not None else float('inf')
            tmp.sol_best = sol0
            res_qp = tmp  # duck-typed for downstream use
    # Outer search: SOC variant (optional)
    res_soc = outer_search(pd, scaling, sopts_soc, ocfg) if sopts_soc.enable_soc else None

    # Metrics
    d_raw, _, _ = trajectory_errors(processed.radar_xyz, processed.telem_xyz)
    if res_qp is None or res_qp.sol_best is None or res_qp.sol_best.p is None:
        raise RuntimeError("Inner solve failed to produce a trajectory. Check data mapping and constraints.")
    d_qp, _, _ = trajectory_errors(res_qp.sol_best.p, processed.telem_xyz)
    CEP50_raw = cep(d_raw, 0.5)
    CEP90_raw = cep(d_raw, 0.9)
    CEP50_qp = cep(d_qp, 0.5)
    CEP90_qp = cep(d_qp, 0.9)
    RMSE_qp = rmse(d_qp)
    TerminalMiss_qp = float(np.linalg.norm(res_qp.sol_best.p[-1] - processed.telem_xyz[-1]))

    has_soc = res_soc is not None and res_soc.sol_best.p is not None
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

    # Save artifacts
    res_dir = os.path.join(out_dir, "results")
    ensure_dir(res_dir)
    df_sol = pd.DataFrame({"time": processed.time, "x": res_qp.sol_best.p[:, 0], "y": res_qp.sol_best.p[:, 1], "z": res_qp.sol_best.p[:, 2]})
    try:
        df_sol.to_parquet(os.path.join(out_dir, "solution_traj.parquet"))
    except Exception:
        df_sol.to_csv(os.path.join(out_dir, "solution_traj.csv"), index=False)
    save_json(os.path.join(out_dir, "best_k.json"), {
        "k_star_qp": res_qp.k_best,
        "J_star_qp": res_qp.J_best,
        "k_star_soc": (res_soc.k_best if res_soc is not None else None),
        "J_star_soc": (res_soc.J_best if res_soc is not None else None)
    })

    # Tables
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
    table_per_sortie([metrics_row], os.path.join(out_dir, "tables"))
    table_aggregate([metrics_row], os.path.join(out_dir, "tables"))
    write_results_index(os.path.join(out_dir, "tables"))

    # Done
    print(json.dumps({
        "k_star_qp": res_qp.k_best,
        "k_star_soc": (res_soc.k_best if res_soc is not None else None),
        "CEP90_qp": CEP90_qp,
        "CEP90_soc": (CEP90_soc if CEP90_soc is not None else None)
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
