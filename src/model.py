from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import cvxpy as cp
import numpy as np

# Physical constants
G0 = 9.80665  # m/s^2
A_WGS84 = 6378137.0
E2_WGS84 = 6.69437999e-3


@dataclass
class Scaling:
    """Numerical scaling for optimization variables.

    If enabled, positions and velocities are scaled as:
      p = alpha_p * p_s,  v = alpha_v * v_s.
    We co-scale gravity and objective weights accordingly to preserve equivalence.
    """
    enable: bool = True
    alpha_p: float = 1.0e4
    alpha_v: float = 1.0e3


@dataclass
class SolverOptions:
    enable_soc: bool = True
    theta_max_deg: float = 15.0
    terminal_altitude_window_m: Tuple[float, float] = (-5.0, 5.0)
    terminal_xy_box_m: float = 500.0
    osqp_opts: Dict = None
    ecos_opts: Dict = None
    scs_opts: Dict = None


@dataclass
class ProblemData:
    dt: float
    positions_obs: np.ndarray  # shape (T,3) observed radar positions in local frame
    weights: Dict[str, float]
    launch_fix: Optional[np.ndarray] = None  # p0 fixed


@dataclass
class Solution:
    success: bool
    status: str
    p: Optional[np.ndarray]
    v: Optional[np.ndarray]
    objective: Optional[float]
    infeasibility: Optional[float]


# Flat-Earth mapping (M5)

def wgs84_radii(phi: float) -> Tuple[float, float]:
    s = math.sin(phi)
    denom = math.sqrt(1 - E2_WGS84 * s * s)
    N = A_WGS84 / denom
    M = A_WGS84 * (1 - E2_WGS84) / (denom ** 3)
    return M, N


def geodetic_to_local(phis: np.ndarray, lams: np.ndarray, hs: np.ndarray, phi0: float, lam0: float, h0: float) -> np.ndarray:
    M, N = wgs84_radii(phi0)
    x = (N + h0) * math.cos(phi0) * (lams - lam0)
    y = (M + h0) * (phis - phi0)
    z = hs - h0
    return np.stack([x, y, z], axis=1)


# Exact discrete dynamics under linear drag (M1)

def discrete_step_matrices(k: float, dt: float) -> Tuple[np.ndarray, np.ndarray, float]:
    x = k * dt
    if x < 1e-6:
        # series expansions
        e = 1 - x + 0.5 * x**2 - x**3 / 6 + x**4 / 24
        one_minus_e_over_k = (dt - 0.5 * k * dt**2 + (k**2) * dt**3 / 6 - (k**3) * dt**4 / 24)
        dt_minus_term = (dt - one_minus_e_over_k)
    else:
        e = math.exp(-x)
        one_minus_e_over_k = (1 - e) / k
        dt_minus_term = dt - one_minus_e_over_k
    A = e * np.eye(3)
    B = np.eye(3) * one_minus_e_over_k
    # gravity drift magnitude coefficient for position update (before applying e_z)
    C_mag = (G0 / k) * dt_minus_term
    return A, B, C_mag


def build_problem(pd: ProblemData, k: float, scaling: Scaling, opts: SolverOptions) -> Tuple[cp.Problem, Dict[str, cp.Expression]]:
    # Ensure numeric arrays
    positions_obs = np.asarray(pd.positions_obs, dtype=float)
    T = positions_obs.shape[0]
    dt = float(pd.dt)
    w_pos = pd.weights.get("w_pos", 1.0)
    w_smooth = pd.weights.get("w_smooth", 0.0)
    w_tv = pd.weights.get("w_TV", 0.0)
    w_final = pd.weights.get("w_final", 0.0)

    # Scaling
    alpha_p = float(scaling.alpha_p) if scaling.enable else 1.0
    alpha_v = float(scaling.alpha_v) if scaling.enable else 1.0

    # Variables (scaled)
    p_s = cp.Variable((T, 3))
    v_s = cp.Variable((T, 3))

    constraints = []

    # launch fix
    if pd.launch_fix is not None:
        launch_fix = np.asarray(pd.launch_fix, dtype=float)
        constraints += [alpha_p * p_s[0, :] == launch_fix]

    # Dynamics for each interval
    A, B, C_mag = discrete_step_matrices(k, dt)
    g_step_mag = float(G0 * (1 - math.exp(-k * dt)) / k)  # for velocity update
    ez = np.array([0.0, 0.0, 1.0])
    for i in range(T - 1):
        # Scaled velocity update
        constraints += [
            v_s[i + 1, :] == A @ v_s[i, :] - ez * (g_step_mag / alpha_v),
        ]
        # Scaled position update
        constraints += [
            p_s[i + 1, :] == p_s[i, :] + (alpha_v / alpha_p) * (B @ v_s[i, :]) - ez * (C_mag / alpha_p),
        ]

    # Terminal altitude window around observed terminal altitude
    lo, hi = opts.terminal_altitude_window_m
    constraints += [
        p_s[T - 1, 2] >= (positions_obs[T - 1, 2] + lo) / alpha_p,
        p_s[T - 1, 2] <= (positions_obs[T - 1, 2] + hi) / alpha_p,
    ]

    # Terminal xy box centered at observed terminal xy (robust box constraint)
    box = opts.terminal_xy_box_m
    constraints += [
        p_s[T - 1, 0] >= (positions_obs[T - 1, 0] - box) / alpha_p,
        p_s[T - 1, 0] <= (positions_obs[T - 1, 0] + box) / alpha_p,
        p_s[T - 1, 1] >= (positions_obs[T - 1, 1] - box) / alpha_p,
        p_s[T - 1, 1] <= (positions_obs[T - 1, 1] + box) / alpha_p,
    ]

    # SOC impact-angle constraint (M2)
    if opts.enable_soc:
        theta = math.radians(opts.theta_max_deg)
        tan_theta = math.tan(theta)
        constraints += [cp.norm(v_s[T - 1, 0:2], 2) <= tan_theta * (-v_s[T - 1, 2]), v_s[T - 1, 2] <= 0]

    # Objective
    obj_terms = []
    # Sensor fit
    y_s = positions_obs / alpha_p
    obj_terms.append((w_pos * (alpha_p ** 2)) * cp.sum_squares(p_s - y_s))

    # Smoothness (second difference on position)
    if w_smooth > 0:
        D2p_s = p_s[:-2, :] - 2 * p_s[1:-1, :] + p_s[2:, :]
        obj_terms.append((w_smooth * (alpha_p ** 2)) * cp.sum_squares(D2p_s))

    # TV on velocity increments
    if w_tv > 0:
        Dv_s = v_s[1:, :] - v_s[:-1, :]
        if opts.enable_soc:
            obj_terms.append((w_tv * alpha_v) * cp.sum(cp.norm(Dv_s, axis=1)))
        else:
            obj_terms.append((w_tv * (alpha_v ** 2)) * cp.sum_squares(Dv_s))

    # Terminal fit
    if w_final > 0:
        obj_terms.append((w_final * (alpha_p ** 2)) * cp.sum_squares(p_s[T - 1, :] - y_s[T - 1, :]))

    # Numerical regularization for stability
    obj_terms.append(1e-8 * cp.sum_squares(v_s))
    obj_terms.append(1e-12 * cp.sum_squares(p_s))

    objective = cp.Minimize(cp.sum(obj_terms))

    prob = cp.Problem(objective, constraints)

    aux = {"p_s": p_s, "v_s": v_s}
    return prob, aux


def solve_inner(
    pd: ProblemData,
    k: float,
    scaling: Scaling,
    opts: SolverOptions,
    warm_start: bool = True,
    init: Optional[Dict[str, np.ndarray]] = None,
) -> Solution:
    """Solve the inner convex problem for a fixed drag k.

    Parameters
    - warm_start: if True, reuse previous variable values as initial point
    - init: optional dict with keys 'p' and 'v' in physical units to warm-start
    """
    prob, aux = build_problem(pd, k, scaling, opts)
    # initialize variables if provided
    if init is not None:
        if "p" in init and init["p"] is not None:
            aux["p_s"].value = init["p"] / (scaling.alpha_p if scaling.enable else 1.0)
        if "v" in init and init["v"] is not None:
            aux["v_s"].value = init["v"] / (scaling.alpha_v if scaling.enable else 1.0)
    def _extract() -> Solution:
        p_s_val = aux["p_s"].value
        v_s_val = aux["v_s"].value
        p_val = None if p_s_val is None else p_s_val * (scaling.alpha_p if scaling.enable else 1.0)
        v_val = None if v_s_val is None else v_s_val * (scaling.alpha_v if scaling.enable else 1.0)
        success = prob.status in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE)
        infeas = prob.status in (cp.INFEASIBLE, cp.INFEASIBLE_INACCURATE)
        return Solution(success=success, status=prob.status, p=p_val, v=v_val, objective=prob.value, infeasibility=float(infeas))

    # Primary solve: prefer ECOS (handles QP and SOCP)
    try:
        prob.solve(solver=cp.ECOS, warm_start=warm_start, **(opts.ecos_opts or {}))
        sol = _extract()
        if sol.success and sol.p is not None:
            return sol
    except Exception:
        pass
    # Secondary: OSQP for QP
    try:
        prob.solve(solver=cp.OSQP, warm_start=warm_start, **(opts.osqp_opts or {}))
        sol = _extract()
        if sol.success and sol.p is not None:
            return sol
    except Exception:
        pass
    # Fallback to SCS
    try:
        prob.solve(solver=cp.SCS, warm_start=warm_start, **(opts.scs_opts or {}))
        return _extract()
    except Exception as e:
        return Solution(success=False, status=str(e), p=None, v=None, objective=None, infeasibility=None)
