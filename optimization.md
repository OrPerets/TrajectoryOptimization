## Optimization and Stabilization Plan for s6 Pipeline

This document instructs the implementation work needed to stabilize and improve the optimization pipeline that produced `outputs/s6_run6`. Follow the steps in order. Use small, verifiable edits with unit/integration tests at each step.

### Scope
- Fix SOC path producing NaNs and ensure both QP and SOC run end-to-end.
- Prevent infinite/NaN objectives; harden numerics and scaling.
- Expand and de-leak hyperparameter search for `k` with per-sortie evaluation.
- Produce correct, per-sortie metrics and aggregate summaries with statistical tests.
- Improve diagnostics (logs/JSONL) and figures for quick failure introspection.
- Keep runtime reasonable and reproducible.

## 1) Data, Units, and Preprocessing
- Validate units used for positions/angles and ensure consistent SI units end-to-end.
  - If inputs are in kilometers or degrees, convert to meters and radians before optimization.
  - Add a single utility `normalize_inputs(data, config)` that applies: centering, scaling by typical magnitude, and time alignment.
- Implement optional normalization for optimization variables and matrices:
  - Scale state/control variables to O(1). Denormalize outputs before metric computation.
  - Add config flags: `general.enable_scaling: true`, `general.scale_position_m: 1e3` (example), `general.scale_velocity_mps: 10`.
- Verify time alignment and remove obvious outliers before fitting. Log number of removed points per sortie.

Acceptance:
- A script/step prints a unit check summary per sortie with min/median/max magnitudes in normalized and raw units.
- No metric computation uses mixed units.

## 2) Solver Stability and Feasibility (QP and SOC)
- Ensure both QP (OSQP) and SOC (ECOS) branches run and return finite results.

- QP hardening (OSQP):
  - Add Hessian regularization: `H_reg = H + lambda_reg * I` with `lambda_reg ∈ [1e-8, 1e-4]` (configurable).
  - Add variable bounds where physically meaningful (position/velocity/angle/rate), avoid unbounded problems.
  - Initial settings: `eps_abs=1e-3`, `eps_rel=1e-3`, `max_iter=50000`, `scaled_termination=true`, `polish=true`.
  - Enable warm-start from previous sortie/previous timestep when applicable.
  - Check solver status; on failure, fall back to last feasible solution or raw trajectory and log a structured failure record.

- SOC hardening (ECOS):
  - Ensure SOC constraints are constructed with finite bounds; guard any sqrt/log/div by zero.
  - ECOS settings: `abstol=1e-6`, `reltol=1e-6`, `feastol=1e-7`, `maxit=10000`.
  - Implement feasibility repair (e.g., slack variables with small penalty) to avoid hard infeasibility.
  - If SOC still fails, return QP solution as fallback and mark `method_used: "QP_fallback"`.

- Objective safety:
  - Wrap objective computation in overflow-safe routines; clamp/guard against Inf/NaN.
  - If objective is Inf/NaN, mark run as `status: "objective_invalid"` and exclude from model selection; still store diagnostics.

Acceptance:
- No NaNs/Inf in metrics or `best_k.json`. Both branches produce outputs or a recorded fallback with reason.

## 3) Hyperparameter Search for k (Leak-free)
- Replace single-sample selection with cross-validated search across sorties.
  - Use leave-one-sortie-out (LOSO) CV: for each held-out sortie, train/select `k` on the remaining sorties, then evaluate on the held-out.
  - Search grid: log-spaced `k ∈ {1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1}`. Make grid configurable.
- Selection criterion:
  - Primary: median CEP50 across validation sorties.
  - Tiebreakers: CEP90, RMSE, stability flags (prefer settings with fewer solver failures).
- Persist artifacts per `k`:
  - `artifacts/hparam_search/run_id=<RUN>/k=<VAL>.jsonl` with one JSON per sortie: `[sortie_id, method, k, metrics..., objective, status, iters, pri_res, dua_res, time_s]`.
  - Aggregate per `k` CSV with medians/IQRs.
- Write `best_k.json` with finite `J_star_*` and selected `k_star_*` for both QP and SOC (if SOC is enabled).

Acceptance:
- `best_k.json` contains finite objectives and non-null `k_star_qp` and `k_star_soc` (when `enable_soc` true).
- `tables/T2_metrics_aggregate.csv` and `tables/T1_metrics_per_sortie.csv` reflect evaluations using cross-validated `k`.

## 4) Metrics: Per-sortie and Aggregates
- Compute metrics per sortie: `CEP50`, `CEP90`, `RMSE`, `TerminalMiss`.
  - Ensure metrics are defined in consistent units (meters) and are NaN-safe.
  - If a sortie fails to produce optimized output, use fallback result but flag it.
- Aggregation:
  - Compute median and 95% CI (via bootstrap or percentile bounds) across sorties for each method.
  - Produce `tables/T1_metrics_per_sortie.csv` and `tables/T2_metrics_aggregate.csv` without NaNs unless method truly absent.
- Statistical tests:
  - Use paired Wilcoxon signed-rank across sorties (Raw vs QP, Raw vs SOC, QP vs SOC) on a primary metric (CEP50).
  - Require N ≥ 10 sorties for the test to be emitted; otherwise, skip with a note.

Acceptance:
- `wilcoxon_tests.csv` includes valid rows for all method comparisons with reasonable p-values (no single-sample artifacts).

## 5) Diagnostics and Logging
- Add structured JSONL logging for every sortie and method:
  - Fields: `run_id, sortie_id, method, k, objective, objective_components, status, exit_flag, iter_count, pri_res, dua_res, time_s, scaling_used, bounds_used, regularization_lambda, notes`.
- Save per-run timing summary: `timing_summary.json` (already present) should also include OSQP/ECOS average iterations and failure counts.
- On failure or fallback, write a `diagnostics/<RUN>/sortie=<ID>_diagnostics.json` with matrices norms/condition numbers and constraint slack statistics.

Acceptance:
- `artifacts/` contains JSONL per method and `diagnostics/` exists for any failures with actionable fields.

## 6) Figures and Tables
- Ensure all figures handle multiple sorties and show distributions:
  - ECDF of errors (`F3`), CEP curves (`F4`), time-series error overlays (`F2`) sample a subset when N is large.
  - Add a small figure `F10_k_sensitivity.png`: metric vs `k` with medians and IQRs for QP and SOC.
- Update runtime breakdown (`F7`) to include solver iteration stats.
- Regenerate tables `T1`, `T2`, `T5`, `T6` with new fields where appropriate (e.g., failure counts).

Acceptance:
- Figures regenerate without errors; no NaNs in plotted arrays; titles/legends include units.

## 7) Runtime and Config Management
- Keep total runtime ≤ 2–5 minutes on the current dataset by default:
  - Limit grid size; enable early-stopping when metric improvements plateau.
  - Reduce `max_iter` by default; add `--fast`/`--full` modes.
- Add a single config for runs (YAML or JSON) with all above settings and seeds.

Acceptance:
- A single command switches between fast smoke-test and full evaluation, with deterministic seeds.

## 8) Reproducibility
- Set seeds for any randomized components and solvers where applicable.
- Capture environment: write `logs/env.txt` with package versions and solver versions.
- Optionally integrate MLflow or simple CSV run registry under `runs/registry.csv`.

Acceptance:
- Re-running with same seed produces identical `best_k.json` and near-identical metrics (within tolerance if parallelism involved).

## 9) Testing
- Unit tests:
  - Metrics correctness on synthetic data (known CEP/ RMSE outputs).
  - QP/SOC toy problems return finite objectives and respect constraints.
  - Scaling/denormalization round-trips.
- Integration tests:
  - End-to-end on 2–3 synthetic sorties completes without NaNs in < 15s.

Acceptance:
- All tests pass in CI; broken invariants fail loudly with actionable messages.

## 10) Deliverables
- Updated pipeline that produces:
  - Non-NaN `T1`, `T2`, `T5`, `T6` tables;
  - Valid `best_k.json` with finite `J_star_*` and chosen `k_star_*` for QP and SOC;
  - `wilcoxon_tests.csv` computed across ≥10 sorties or sensibly skipped;
  - `artifacts/*.jsonl` logs with statuses and iteration stats;
  - Regenerated figures `F1–F9` plus `F10_k_sensitivity`.

## 11) How to Run (expected commands)
Use or implement equivalent entry points; adjust paths to your repo structure.

```bash
# Fast smoke test (small grid, low iters)
python scripts/run_experiment.py \
  --config configs/s6.yaml \
  --run_id s6_smoke \
  --mode fast

# Full run with LOSO CV and full grid
python scripts/run_experiment.py \
  --config configs/s6.yaml \
  --run_id s6_run7 \
  --mode full

# Optional: hyperparameter search only
python scripts/run_hparam_search.py --config configs/s6.yaml --run_id s6_run7
```

Expected outputs land under `outputs/<run_id>/` with subfolders `figures/`, `tables/`, `artifacts/`, `logs/`, and `diagnostics/`.

## 12) Acceptance Criteria (quantitative)
- SOC branch completes without NaNs for ≥95% of sorties; remaining failures have diagnostics.
- `best_k.json` has finite `J_star_qp` and `J_star_soc` and non-null `k_star_*` when enabled.
- Compared to Raw, QP shows a meaningful improvement on CEP50 (target ≥ 5% relative) with p < 0.05 (Wilcoxon) across sorties. SOC is comparable or better.
- No table contains NaNs unless a method is truly absent by design.
- Total runtime within the targeted bounds for fast/full modes.

## 13) Order of Execution (recommended)
1. Implement structured logging and guards (Sections 2, 5).
2. Fix SOC branch and objective finiteness (Section 2).
3. Normalize inputs and ensure unit consistency (Section 1).
4. Implement LOSO CV grid search and write `best_k.json` (Section 3).
5. Rework metrics and aggregates with tests (Section 4, 9).
6. Regenerate figures/tables, add k-sensitivity (Section 6).
7. Tune runtime profiles and config management (Section 7, 8).
8. Run full evaluation and validate acceptance criteria (Section 12).

---

If any step uncovers systemic issues (e.g., widespread infeasibility), pause and add a focused diagnostic capturing the smallest reproducible example before proceeding.

