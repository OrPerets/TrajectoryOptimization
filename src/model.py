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
    When ``alpha_p`` or ``alpha_v`` is ``None`` the scaling is inferred from
    the data so that variable magnitudes remain close to :math:`O(1)`.
    """

    enable: bool = True
    alpha_p: Optional[float] = None
    alpha_v: Optional[float] = None

    def auto_from_data(self, positions: np.ndarray, dt: float) -> None:
        """Auto-suggest scaling factors based on data magnitudes."""
        if not self.enable:
            return
        if self.alpha_p is None:
            max_p = float(np.max(np.linalg.norm(positions, axis=1)))
            self.alpha_p = max(1.0, 10 ** math.floor(math.log10(max_p + 1e-9)))
        if self.alpha_v is None:
            v_est = np.diff(positions, axis=0) / dt
            max_v = float(np.max(np.linalg.norm(v_est, axis=1)))
            self.alpha_v = max(1.0, 10 ** math.floor(math.log10(max_v + 1e-9)))


@dataclass
class SolverOptions:
    enable_soc: bool = True
    theta_max_deg: float = 15.0
    terminal_altitude_window_m: Tuple[float, float] = (-5.0, 5.0)
    terminal_xy_box_m: float = 500.0
    quadratic_tv: bool = False
    osqp_opts: Dict = None
    ecos_opts: Dict = None
    scs_opts: Dict = None
    tighten_tol: bool = True


@dataclass
class ProblemData:
    dt: float
    positions_obs: np.ndarray  # shape (T,3) observed radar positions in local frame
    weights: Dict[str, float]
    launch_fix: Optional[np.ndarray] = None  # p0 fixed
    terminal_anchor: Optional[np.ndarray] = None  # optional terminal position anchor in local frame


@dataclass
class Solution:
    success: bool
    status: str
    p: Optional[np.ndarray]
    v: Optional[np.ndarray]
    objective: Optional[float]
    infeasibility: Optional[float]
    diagnostics: Optional[Dict[str, float]] = None


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

_STEP_CACHE: Dict[Tuple[float, float], Tuple[np.ndarray, np.ndarray, float, float]] = {}


def discrete_step_matrices(k: float, dt: float) -> Tuple[np.ndarray, np.ndarray, float, float]:
    key = (k, dt)
    if key in _STEP_CACHE:
        return _STEP_CACHE[key]
    x = k * dt
    if x < 1e-6:
        # series expansions
        e = 1 - x + 0.5 * x**2 - x**3 / 6 + x**4 / 24
        one_minus_e_over_k = (dt - 0.5 * k * dt**2 + (k**2) * dt**3 / 6 - (k**3) * dt**4 / 24)
        dt_minus_term = dt - one_minus_e_over_k
    else:
        e = math.exp(-x)
        one_minus_e_over_k = (1 - e) / k
        dt_minus_term = dt - one_minus_e_over_k
    A = e * np.eye(3)
    B = np.eye(3) * one_minus_e_over_k
    C_mag = (G0 / k) * dt_minus_term
    g_step_mag = G0 * (1 - e) / k
    _STEP_CACHE[key] = (A, B, C_mag, g_step_mag)
    return _STEP_CACHE[key]


def build_problem(pd: ProblemData, k: float, scaling: Scaling, opts: SolverOptions) -> Tuple[cp.Problem, Dict[str, cp.Expression]]:
    # Ensure numeric arrays
    positions_obs = np.asarray(pd.positions_obs, dtype=float)
    T = positions_obs.shape[0]
    dt = float(pd.dt)
    w_pos = pd.weights.get("w_pos", 1.0)
    w_smooth = pd.weights.get("w_smooth", 0.0)
    w_tv = pd.weights.get("w_TV", 0.0)
    w_final = pd.weights.get("w_final", 0.0)

    # Scaling (auto-infer if needed)
    scaling.auto_from_data(positions_obs, dt)
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

    # Dynamics for each interval (vectorized)
    A, B, C_mag, g_step_mag = discrete_step_matrices(k, dt)
    ez = np.array([0.0, 0.0, 1.0])
    g_vec_v = np.tile(ez, (T - 1, 1)) * (g_step_mag / alpha_v)
    g_vec_p = np.tile(ez, (T - 1, 1)) * (C_mag / alpha_p)
    constraints += [v_s[1:, :] == v_s[:-1, :] @ A.T - g_vec_v]
    constraints += [
        p_s[1:, :] == p_s[:-1, :] + (alpha_v / alpha_p) * (v_s[:-1, :] @ B.T) - g_vec_p
    ]

    # Terminal altitude window around anchor (telemetry) if provided, else observed terminal
    lo, hi = opts.terminal_altitude_window_m
    term_ref = pd.terminal_anchor if pd.terminal_anchor is not None else positions_obs[T - 1, :]
    constraints += [
        p_s[T - 1, 2] >= (term_ref[2] + lo) / alpha_p,
        p_s[T - 1, 2] <= (term_ref[2] + hi) / alpha_p,
    ]

    # Terminal xy box centered at anchor (telemetry) if provided, else observed terminal (robust box constraint)
    box = opts.terminal_xy_box_m
    constraints += [
        p_s[T - 1, 0] >= (term_ref[0] - box) / alpha_p,
        p_s[T - 1, 0] <= (term_ref[0] + box) / alpha_p,
        p_s[T - 1, 1] >= (term_ref[1] - box) / alpha_p,
        p_s[T - 1, 1] <= (term_ref[1] + box) / alpha_p,
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
        D2p_s = cp.diff(p_s, k=2, axis=0)
        obj_terms.append((w_smooth * (alpha_p ** 2)) * cp.sum_squares(D2p_s))

    # TV on velocity increments
    if w_tv > 0:
        Dv_s = cp.diff(v_s, axis=0)
        if opts.quadratic_tv:
            obj_terms.append((w_tv * (alpha_v ** 2)) * cp.sum_squares(Dv_s))
        else:
            obj_terms.append((w_tv * alpha_v) * cp.sum(cp.norm(Dv_s, axis=1)))

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
        diagnostics = None
        if p_val is not None and v_val is not None:
            A, B, C_mag, g_step_mag = discrete_step_matrices(k, pd.dt)
            ez = np.array([0.0, 0.0, 1.0])
            g_v = np.outer(np.ones(p_val.shape[0] - 1), ez) * g_step_mag
            g_p = np.outer(np.ones(p_val.shape[0] - 1), ez) * C_mag
            v_res = v_val[1:, :] - (v_val[:-1, :] @ A.T - g_v)
            p_res = p_val[1:, :] - (p_val[:-1, :] + (v_val[:-1, :] @ B.T) - g_p)

            # Terminal constraints residuals
            lo, hi = opts.terminal_altitude_window_m
            term_ref = pd.terminal_anchor if pd.terminal_anchor is not None else pd.positions_obs[-1, :]
            alt_low = term_ref[2] + lo
            alt_high = term_ref[2] + hi
            alt_res = max(0.0, alt_low - p_val[-1, 2], p_val[-1, 2] - alt_high)

            box = opts.terminal_xy_box_m
            x_lo = term_ref[0] - box
            x_hi = term_ref[0] + box
            y_lo = term_ref[1] - box
            y_hi = term_ref[1] + box
            xy_res = max(
                0.0,
                x_lo - p_val[-1, 0],
                p_val[-1, 0] - x_hi,
                y_lo - p_val[-1, 1],
                p_val[-1, 1] - y_hi,
            )

            soc_res = 0.0
            if opts.enable_soc:
                theta = math.radians(opts.theta_max_deg)
                tan_theta = math.tan(theta)
                soc_res = max(0.0, np.linalg.norm(v_val[-1, 0:2]) - tan_theta * (-v_val[-1, 2]))

            vz_pos = max(0.0, v_val[-1, 2])

            diagnostics = {
                "dyn_v_max": float(np.max(np.abs(v_res))),
                "dyn_p_max": float(np.max(np.abs(p_res))),
                "terminal_alt_violation": float(alt_res),
                "terminal_xy_violation": float(xy_res),
                "soc_violation": float(soc_res),
                "vz_positive": float(vz_pos),
            }
        return Solution(
            success=success,
            status=prob.status,
            p=p_val,
            v=v_val,
            objective=prob.value,
            infeasibility=float(infeas),
            diagnostics=diagnostics,
        )

    # Primary solve: prefer ECOS (handles QP and SOCP)
    try:
        ecos_opts = {"abstol": 1e-7, "reltol": 1e-7, "feastol": 1e-7}
        if opts.ecos_opts:
            ecos_opts.update(opts.ecos_opts)
        prob.solve(solver=cp.ECOS, warm_start=warm_start, **ecos_opts)
        sol = _extract()
        if opts.tighten_tol and sol.status == cp.OPTIMAL_INACCURATE:
            ecos_opts_t = dict(ecos_opts)
            ecos_opts_t["abstol"] *= 0.1
            ecos_opts_t["reltol"] *= 0.1
            ecos_opts_t["feastol"] *= 0.1
            prob.solve(solver=cp.ECOS, warm_start=warm_start, **ecos_opts_t)
            sol = _extract()
        if sol.success and sol.p is not None:
            return sol
    except Exception:
        pass
    # Secondary: OSQP for QP
    try:
        osqp_opts = {"eps_abs": 1e-7, "eps_rel": 1e-7}
        if opts.osqp_opts:
            osqp_opts.update(opts.osqp_opts)
        prob.solve(solver=cp.OSQP, warm_start=warm_start, **osqp_opts)
        sol = _extract()
        if opts.tighten_tol and sol.status == cp.OPTIMAL_INACCURATE:
            osqp_opts_t = dict(osqp_opts)
            osqp_opts_t["eps_abs"] *= 0.1
            osqp_opts_t["eps_rel"] *= 0.1
            prob.solve(solver=cp.OSQP, warm_start=warm_start, **osqp_opts_t)
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
