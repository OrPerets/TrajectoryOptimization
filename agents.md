## Project: Radar-only Trajectory Reconstruction (Convex QP/SOCP)

### Purpose
Reconstruct weapon trajectories from noisy airborne radar measurements using a convex optimization core (QP with optional SOCP terminal impact-angle constraint), a feasibility-aware outer loop to tune linear drag, and a reproducible experimental pipeline that outputs publication-quality figures and LaTeX tables.

### High-level Method
- Inner problem (QP/SOCP): states are per-sample position p_i and velocity v_i with exact discrete dynamics under linear drag k and gravity g. Optional terminal impact-angle bound is implemented as a second-order cone.
- Outer loop (feasibility-aware): tunes scalar drag k on a log grid and refines via parabolic interpolation in log k, respecting feasibility bracketing.
- Mapping: WGS‑84 flat-Earth approximation around a local origin for short ranges.
- Preprocessing: outlier masking (IQR on first diffs), resampling to 10 Hz, Savitzky–Golay smoothing, radar↔telemetry alignment (cross-correlation of Doppler proxy where available).
- Evaluation: CEP‑50/90, RMSE, terminal miss, paired Wilcoxon tests, plus ablations (SOC on/off, θ_max grid, k-grid density).

### Core Equations (SI units)
- Continuous: ṗ = v,  v̇ = −k v − g e_z,  g=9.80665 m/s².
- Exact discrete step over Δt:
  - v_{i+1} = e^{-kΔt} v_i − (1 − e^{-kΔt})(g/k) e_z
  - p_{i+1} = p_i + ((1 − e^{-kΔt})/k) v_i − (Δt − (1 − e^{-kΔt})/k)(g/k) e_z
  - Use series expansions for kΔt ≪ 1 for numerical stability.
- SOC impact-angle (optional, at terminal N): ||[v_xN, v_yN]||₂ ≤ tan(θ_max)(−v_zN), with v_zN ≤ 0.

### Software Architecture (src/)
- `model.py`: WGS‑84 mapping; exact discrete dynamics; CVXPY model builder; scaling; SOC; solver selection (ECOS→OSQP→SCS fallback).
- `outer.py`: log-grid sweep; feasibility filtering; parabolic refinement in log k; warm starts; history logging.
- `data.py`: CSV loading; robust column mapping heuristics; unit normalization (deg→rad, ft→m); resample/filter; alignment; caches.
- `utils.py`: plotting style, metrics (CEP/RMSE), Wilcoxon tests, I/O helpers.
- `figures.py`: figure helpers (overlay and error time-series scaffolded; extendable to F1–F9).
- `tables.py`: CSV + LaTeX tables (per-sortie and aggregate), assembly index.
- `main.py`: CLI runner, config management, outer search, metrics/tables saving.

### Current Status and Known Gaps
- Baseline QP pipeline runs end-to-end, with ECOS primary and OSQP/SCS fallbacks.
- Robust mapping for the provided files is in place (telemetry weapon geodetics; radar via weapon geodetics or target+offsets).
- Doppler-based alignment is scaffolded but may be bypassed if Doppler/LOS not present.
- Figures: only F1/F2 are implemented; remaining figure types are to be added.
- Tables: T1/T2 present; ablation tables and runtime breakdown pending.
- Some solver settings may need tuning per dataset to maximize feasibility/accuracy and reduce warnings.

## Sprint Plan and Refinements

### Sprint 1 (Data robustness & alignment)
 Goals: Make ingestion resilient across projects; enable reliable alignment; solid caches.
- [x] Add schema validation and clear errors for required columns; improve alias lists in `config/default.yaml`.
- [x] Add unit detection for alt/offsets (ft/m) with metadata logging and warnings.
- [x] Implement Doppler proxy alignment when Doppler is missing by constructing LOS from radar host states if available; else document fallback.
- [x] Add richer parquet/CSV cache policy: versioned file names with hash of inputs and config.
- [x] Write unit tests: time parsing (DOY:HH:MM:SS.sss), unit conversions, offset-to-local mapping.
- [x] Document alignment quality checks (lag < 50 ms) and auto-warnings in logs.

Deliverables: robust loader, alignment report per sortie, cached processed files with hashes.

### Sprint 2 (Inner problem stability & performance)
Goals: Improve numerical stability, exploit structure, reduce runtime.
- [x] Precompute and reuse step matrices for constant Δt; vectorize dynamics constraints generation.
- [x] Add banded/Toeplitz structure exploitation for second-difference operators; use CVXPY `cp.diff` where helpful.
- [x] Calibrate scaling (α_p, α_v) to keep residuals near O(1); auto-suggest values from data magnitudes.
- [x] Tighten solver tolerances adaptively (ECOS/OSQP) based on feasibility residuals; expose via config.
- [x] Add optional quadratic TV (QP) vs ℓ2-TV (SOCP) switch to speed QP path.
- [x] Add diagnostics: feasibility residuals by constraint family; terminal v_zN check when SOC active.

Deliverables: faster stable solves, diagnostic logs, config toggles for TV type and tolerances.

### Sprint 3 (Outer loop efficiency & reliability)
Goals: Make k tuning faster and more robust.
- [x] Parallelize initial log-grid evaluations with warm-start propagation (thread/process pool).
- [x] Implement bracket expansion policy (geometric growth) until ≥3 feasible points or hard bounds reached.
- [x] Add safeguarded parabolic step with trust region in log k; keep feasibility bracket invariant.
- [x] Cache (k, J, status, warm-start) to disk for resumable runs.
- [x] Plot outer landscape (F6) with feasible points, best k*, and refinement steps.

Deliverables: faster outer loop, landscape figure, resumable state.

### Sprint 4 (Evaluation, ablations, and figures)
Goals: Complete F1–F9 and ablations; statistical tests.
- [ ] Implement remaining figures:
  - [x] ECDF/hist (F3)
  - [ ] CEP curves (F4)
  - [ ] θ_max sensitivity (F5)
  - [ ] runtime (F7)
  - [ ] kinematics sanity (F8)
  - [ ] impact-angle hist (F9)
- [ ] Implement T3–T5 tables: ablations (SOC on/off; θ_max; grid density), runtime breakdown, solver settings.
- [ ] Add paired Wilcoxon tests for Raw vs QP and QP vs QP+SOC; annotate p-values on F4.
- [ ] Aggregate across sorties: median [IQR] with confidence intervals.
- [ ] Deterministic styles (fonts, DPI, colors) and consistent axis limits/annotations.

Deliverables: full figure suite and tables suitable for publication.

### Sprint 5 (Testing, CI, and reproducibility)
Goals: Confidence in results and portability.
- [ ] Expand unit tests: dynamics stepping (small/large kΔt), SOC feasibility edges, alignment, metrics.
- [ ] Add integration tests: small synthetic dataset with known k and trajectory.
- [ ] Pre-commit: black/ruff, type checks, basic lint.
- [ ] CI workflow (GitHub Actions): tests on 3.10–3.13; cache dependencies.
- [ ] Env capture: write `solver_versions.json`, `pip freeze`, file hashes.

Deliverables: green CI, higher test coverage, environment logs.

### Sprint 6 (Packaging, UX, and docs)
Goals: Easier onboarding and repeatability.
- [ ] Package as a Python module; console entry-point for `radar-traj-run`.
- [ ] Structured config with schema validation and helpful error messages.
- [ ] Extend README with troubleshooting, example notebooks, and figure catalog.
- [ ] Optional Dockerfile for fully pinned environment.
- [ ] CLI improvements: preset profiles (default, ablation-fast, soc-sweep), verbosity levels, dry-run mode.

Deliverables: packaged tool with strong docs and smoother CLI UX.

### Sprint 7 (Result validation & optimization)
Goals: Bring reconstruction accuracy to research quality and fix failure modes.
- [ ] Investigate unrealistic CEP values (~6.47e6 m) and correct unit scaling.
- [ ] Diagnose SOC path NaNs and validate impact-angle constraints.
- [ ] Cross-check telemetry vs radar mapping to catch frame/offset errors.
- [ ] Automate figure/table sanity checks to flag NaNs and extreme metrics.
- [ ] Re-run experiments with fixes and update OUTPUT_SUMMARY with normalized results.

Deliverables: verified metrics within expected ranges and functional SOC outputs.

## Backlog (Prioritized)
- Parallel outer loop on multi-core.
- Per-axis weighting (anisotropic sensor noise) and altitude-weight emphasis near terminal.
- Variable Δt support and missing-data handling in inner problem.
- LOS/Doppler model from host attitude and antenna geometry (when available).
- Curvature/turn-rate regularization alternatives; adaptive smoothing.
- Export KML/GeoJSON of trajectories for GIS tools.
- JAX/NumPyro probabilistic extension to quantify drag uncertainty (future work).

## Ownership and Roles
- Research Engineer: methods, solver tuning, figures/tables.
- Data Engineer: mapping, robust ingestion, caches.
- MLE/Infra: CI, packaging, Docker, reproducibility.

## Success Criteria
- Reproducible runs across environments; no silent failures.
- Stronger accuracy vs raw (CEP‑50/90, terminal miss) with significance.
- Clean, publication-quality figures and LaTeX tables.


