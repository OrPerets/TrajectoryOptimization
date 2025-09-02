# Radar-only Trajectory Reconstruction with Drag Tuning and Impact-Angle SOC

This repository implements the convex optimization-based reconstruction of weapon trajectories from airborne radar, with exact linear-drag dynamics, feasibility-aware drag tuning, and an optional terminal impact-angle second-order cone constraint.

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Data
Place the provided CSVs in the project root (or pass full paths):
- `S1_Hatala_83_sufa_trimmed (1).csv`
- `Telemetric Output Conversion.csv`

## Run (single sortie)

```bash
python main.py \
  --radar "S1_Hatala_83_sufa_trimmed (1).csv" \
  --telemetry "Telemetric Output Conversion.csv" \
  --config config/default.yaml \
  --out_dir outputs/s2_run2 \
  --enable_soc true \
  --theta_max 15
```

Ablation (no SOC, coarse grid):
```bash
python main.py \
  --radar "S1_Hatala_83_sufa_trimmed (1).csv" \
  --telemetry "Telemetric Output Conversion.csv" \
  --config config/default.yaml \
  --out_dir outputs/s1_ablate \
  --enable_soc false \
  --k_grid_points 9
```

## Outputs
- `outputs/<run_id>/solution_traj.parquet`: estimated trajectory (local frame)
- `outputs/<run_id>/best_k.json`: tuned drag parameter and objective
- `outputs/<run_id>/artifacts/processed/*.parquet`: preprocessed streams
- `outputs/<run_id>/tables/*.csv|.tex`: tables suitable for LaTeX, with an includer `results_tables.tex`

## Reproducibility
- Fixed seeds, environment logs, and deterministic plots. Edit `config/default.yaml` for hyperparameters and mappings.

## Notes
- The implementation uses exact discrete-time linear-drag dynamics with series expansions for small `kΔt`.
- Optional SOC at terminal enforces an impact angle bound.
- Preprocessing: outlier masking (IQR on first differences), resampling to 10 Hz, Savitzky–Golay smoothing, and radar/telemetry alignment via cross-correlation of Doppler proxies.
