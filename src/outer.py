from __future__ import annotations

import json
import math
import os
from concurrent.futures import ThreadPoolExecutor
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
    n_workers: int = 1
    bracket_factor: float = 2.0
    k_min_hard: float = 1e-3
    k_max_hard: float = 1.0
    trust_region: float = 1.0  # log10 distance
    cache_path: Optional[str] = None
    plot_dir: Optional[str] = None


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


def _append_cache(path: str, entry: Dict, sol: Solution) -> None:
    data = dict(entry)
    if sol.p is not None and sol.v is not None:
        data["p"] = sol.p.tolist()
        data["v"] = sol.v.tolist()
    if sol.diagnostics:
        data["diagnostics"] = sol.diagnostics
    with open(path, "a") as f:
        f.write(json.dumps(data) + "\n")


def _load_cache(path: str) -> Tuple[List[Dict], List[Tuple[float, float, Solution]], Optional[float], float, Optional[Solution]]:
    history: List[Dict] = []
    feas_list: List[Tuple[float, float, Solution]] = []
    best_J = math.inf
    best_k = None
    best_sol: Optional[Solution] = None
    with open(path, "r") as f:
        for line in f:
            rec = json.loads(line)
            history.append({
                "k": rec["k"],
                "J": rec["J"],
                "feasible": rec["feasible"],
                "stage": rec.get("stage", "cache"),
                "diagnostics": rec.get("diagnostics"),
            })
            if rec.get("feasible") and rec.get("p") is not None:
                sol = Solution(True, "cached", np.array(rec["p"]), np.array(rec["v"]), rec["J"], None, rec.get("diagnostics"))
                feas_list.append((rec["k"], rec["J"], sol))
                if rec["J"] < best_J:
                    best_J = rec["J"]
                    best_k = rec["k"]
                    best_sol = sol
    return history, feas_list, best_k, best_J, best_sol


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
    feas_list: List[Tuple[float, float, Solution]] = []
    best_sol: Optional[Solution] = None
    best_J = math.inf
    best_k = None

    # Load cache if available
    if ocfg.cache_path and os.path.exists(ocfg.cache_path):
        history, feas_list, best_k, best_J, best_sol = _load_cache(ocfg.cache_path)

    k_min, k_max = ocfg.k_min, ocfg.k_max

    def _log_infeasible(k: float, diagnostics: Optional[Dict]) -> None:
        if not diagnostics:
            return
        items = sorted(diagnostics.items(), key=lambda kv: abs(kv[1]), reverse=True)
        msg = ", ".join(f"{n}={v:.3g}" for n, v in items[:3])
        print(f"Infeasible solve at k={k:.3g}: {msg}")

    def _eval_segment(ks: np.ndarray) -> Tuple[List[Dict], List[Tuple[float, float, Solution]]]:
        seg_history: List[Dict] = []
        seg_feas: List[Tuple[float, float, Solution]] = []
        prev: Optional[Solution] = None
        for k in ks:
            J, sol, feas = evaluate_k(k, pd, scaling, opts, warm_start_solution=prev)
            entry = {
                "k": k,
                "J": J,
                "feasible": bool(feas),
                "stage": "grid",
                "diagnostics": sol.diagnostics,
            }
            seg_history.append(entry)
            if ocfg.cache_path:
                _append_cache(ocfg.cache_path, entry, sol)
            if feas:
                seg_feas.append((k, J, sol))
                prev = sol
            else:
                _log_infeasible(k, sol.diagnostics)
        return seg_history, seg_feas

    # Initial sweep with bracket expansion
    while True:
        K = logspace_grid(k_min, k_max, ocfg.k_grid_points)
        if ocfg.n_workers > 1:
            segments = np.array_split(K, ocfg.n_workers)
            with ThreadPoolExecutor(max_workers=ocfg.n_workers) as ex:
                futures = [ex.submit(_eval_segment, seg) for seg in segments]
                for fut in futures:
                    seg_hist, seg_feas = fut.result()
                    history.extend(seg_hist)
                    for k, J, sol in seg_feas:
                        feas_list.append((k, J, sol))
                        if J < best_J:
                            best_J, best_k, best_sol = J, k, sol
        else:
            seg_hist, seg_feas = _eval_segment(K)
            history.extend(seg_hist)
            for k, J, sol in seg_feas:
                feas_list.append((k, J, sol))
                if J < best_J:
                    best_J, best_k, best_sol = J, k, sol
        if len(feas_list) >= 3 or (k_min <= ocfg.k_min_hard and k_max >= ocfg.k_max_hard):
            break
        k_min = max(k_min / ocfg.bracket_factor, ocfg.k_min_hard)
        k_max = min(k_max * ocfg.bracket_factor, ocfg.k_max_hard)

    if best_k is None:
        return OuterResult(
            k_best=float("nan"),
            J_best=float("inf"),
            sol_best=best_sol or Solution(False, "infeasible", None, None, None, None),
            history=history,
        )

    # Refinement iterations
    feas_list.sort(key=lambda t: t[0])
    k_br_low = min(k for k, _, _ in feas_list)
    k_br_high = max(k for k, _, _ in feas_list)
    for it in range(ocfg.max_iters):
        ks = [k for k, _, _ in feas_list]
        idx = int(np.argmin([abs(k - best_k) for k in ks]))
        j1 = max(0, idx - 1)
        j3 = min(len(ks) - 1, idx + 1)
        if j3 - j1 < 2:
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
        k_low = min(k1, k3)
        k_high = max(k1, k3)
        k_new = min(max(k_new, k_low), k_high)
        logk = math.log10(k_new)
        log_best = math.log10(best_k)
        logk = min(max(logk, log_best - ocfg.trust_region), log_best + ocfg.trust_region)
        k_new = 10 ** logk
        k_new = min(max(k_new, k_br_low), k_br_high)

        J_new, sol_new, feas_new = evaluate_k(k_new, pd, scaling, opts, warm_start_solution=best_sol)
        entry = {
            "k": k_new,
            "J": J_new,
            "feasible": bool(feas_new),
            "stage": "refine",
            "iter": it,
            "diagnostics": sol_new.diagnostics,
        }
        history.append(entry)
        if ocfg.cache_path:
            _append_cache(ocfg.cache_path, entry, sol_new)

        if feas_new:
            feas_list.append((k_new, J_new, sol_new))
            if J_new < best_J - ocfg.tol_J:
                best_J, best_k, best_sol = J_new, k_new, sol_new
            k_br_low = min(k_br_low, k_new)
            k_br_high = max(k_br_high, k_new)
            if abs(k_new - k2) < ocfg.tol_k and abs(J_new - J2) < ocfg.tol_J:
                break
        else:
            _log_infeasible(k_new, sol_new.diagnostics)
            if k_new < best_k:
                k_br_low = max(k_br_low, k_new)
            else:
                k_br_high = min(k_br_high, k_new)

    if ocfg.plot_dir:
        from .figures import fig_outer_landscape

        fig_outer_landscape(history, ocfg.plot_dir)

    return OuterResult(k_best=best_k, J_best=best_J, sol_best=best_sol, history=history)
