from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from .model import ProblemData, Scaling, SolverOptions, Solution, solve_inner


@dataclass
class OuterConfig:
    k_min: float = 0.02
    k_max: float = 0.5
    k_grid_points: int = 21
    tol_k: float = 1e-3
    tol_J: float = 1e-3
    max_iters: int = 30


@dataclass
class OuterResult:
    k_best: float
    J_best: float
    sol_best: Solution
    history: List[Dict]


def logspace_grid(kmin: float, kmax: float, n: int) -> np.ndarray:
    return np.exp(np.linspace(math.log(kmin), math.log(kmax), n))


def evaluate_k(
    k: float,
    pd: ProblemData,
    scaling: Scaling,
    opts: SolverOptions,
    warm_start_solution: Optional[Solution] = None,
) -> Tuple[float, Solution, bool]:
    init = None
    if warm_start_solution and warm_start_solution.p is not None:
        init = {"p": warm_start_solution.p, "v": warm_start_solution.v}
    sol = solve_inner(pd, k, scaling, opts, warm_start=True, init=init)
    feasible = sol.success and sol.p is not None
    J = float(sol.objective) if feasible else math.inf
    return J, sol, feasible


def parabola_vertex(x1: float, y1: float, x2: float, y2: float, x3: float, y3: float) -> Optional[float]:
    denom = (x1 - x2) * (x1 - x3) * (x2 - x3)
    if abs(denom) < 1e-16:
        return None
    a = (x3 * (y2 - y1) + x2 * (y1 - y3) + x1 * (y3 - y2)) / denom
    b = (x3**2 * (y1 - y2) + x2**2 * (y3 - y1) + x1**2 * (y2 - y3)) / denom
    if a <= 0:
        return None
    xv = -b / (2 * a)
    return xv


def outer_search(
    pd: ProblemData,
    scaling: Scaling,
    opts: SolverOptions,
    ocfg: OuterConfig,
) -> OuterResult:
    history: List[Dict] = []
    K = logspace_grid(ocfg.k_min, ocfg.k_max, ocfg.k_grid_points)
    best_sol: Optional[Solution] = None
    best_J = math.inf
    best_k = None

    # Initial sweep
    prev_sol: Optional[Solution] = None
    feas_list: List[Tuple[float, float, Solution]] = []
    for k in K:
        J, sol, feas = evaluate_k(k, pd, scaling, opts, warm_start_solution=prev_sol)
        history.append({"k": k, "J": J, "feasible": bool(feas), "stage": "grid"})
        if feas:
            feas_list.append((k, J, sol))
            if J < best_J:
                best_J, best_k, best_sol = J, k, sol
            prev_sol = sol

    # If too few feasible, widen bracket heuristic
    if len(feas_list) < 3:
        # try a wider bracket by doubling range once
        K2 = logspace_grid(max(ocfg.k_min * 0.5, 1e-3), ocfg.k_max * 2.0, ocfg.k_grid_points)
        for k in K2:
            J, sol, feas = evaluate_k(k, pd, scaling, opts, warm_start_solution=prev_sol)
            history.append({"k": k, "J": J, "feasible": bool(feas), "stage": "grid_widen"})
            if feas:
                feas_list.append((k, J, sol))
                if J < best_J:
                    best_J, best_k, best_sol = J, k, sol
                prev_sol = sol

    if best_k is None:
        # Return best infeasible attempt
        return OuterResult(k_best=float("nan"), J_best=float("inf"), sol_best=prev_sol or Solution(False, "infeasible", None, None, None, None), history=history)

    # Refinement iterations
    feas_list.sort(key=lambda t: t[0])
    for it in range(ocfg.max_iters):
        # choose best triplet around best_k
        ks = [k for k, _, _ in feas_list]
        idx = int(np.argmin([abs(k - best_k) for k in ks]))
        # choose neighbors
        j1 = max(0, idx - 1)
        j3 = min(len(ks) - 1, idx + 1)
        if j3 - j1 < 2:
            # pick widest available triplet
            if len(ks) >= 3:
                j1, j3 = 0, 2
            else:
                break
        k1, J1, _ = feas_list[j1]
        k2, J2, _ = feas_list[idx]
        k3, J3, _ = feas_list[j3]
        x1, x2, x3 = math.log(k1), math.log(k2), math.log(k3)
        xv = parabola_vertex(x1, J1, x2, J2, x3, J3)
        if xv is None:
            break
        k_new = float(math.exp(xv))
        if k_new <= 0:
            break
        # bracket within min/max of current neighbors
        k_low = min(k1, k3)
        k_high = max(k1, k3)
        k_new = min(max(k_new, k_low), k_high)

        J_new, sol_new, feas_new = evaluate_k(k_new, pd, scaling, opts, warm_start_solution=best_sol)
        history.append({"k": k_new, "J": J_new, "feasible": bool(feas_new), "stage": "refine", "iter": it})

        if feas_new:
            feas_list.append((k_new, J_new, sol_new))
            if J_new < best_J - ocfg.tol_J:
                best_J, best_k, best_sol = J_new, k_new, sol_new
            # stopping criteria
            if abs(k_new - k2) < ocfg.tol_k and abs(J_new - J2) < ocfg.tol_J:
                break
        else:
            # Move towards feasible side: shrink towards best_k
            if k_new > best_k:
                k_high = k_new
            else:
                k_low = k_new
            # If too many infeasible steps, break
    return OuterResult(k_best=best_k, J_best=best_J, sol_best=best_sol, history=history)
