from __future__ import annotations

import os
from typing import Dict, List

import numpy as np
import pandas as pd


def _save_csv_tex(df: pd.DataFrame, out_dir: str, name: str, caption: str, label: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, f"{name}.csv")
    tex_path = os.path.join(out_dir, f"{name}.tex")
    df.to_csv(csv_path, index=False)
    with open(tex_path, "w") as f:
        f.write("\\begin{table}[t]\n")
        f.write("\\centering\n")
        f.write(df.to_latex(index=False, escape=False))
        f.write(f"\\caption{{{caption}}}\n")
        f.write(f"\\label{{{label}}}\n")
        f.write("\\end{table}\n")


def table_per_sortie(metrics: List[Dict], out_dir: str) -> None:
    df = pd.DataFrame(metrics)
    _save_csv_tex(df, out_dir, "T1_metrics_per_sortie", "Per-sortie metrics", "tab:per_sortie")


def table_aggregate(metrics: List[Dict], out_dir: str) -> None:
    df = pd.DataFrame(metrics)
    # Aggregate with median and IQR
    def med_iqr(x: pd.Series) -> str:
        med = np.median(x)
        q25, q75 = np.quantile(x, [0.25, 0.75])
        return f"{med:.2f} [{q25:.2f}, {q75:.2f}]"

    cols = [
        "CEP50_raw",
        "CEP50_qp",
        "CEP50_soc",
        "CEP90_raw",
        "CEP90_qp",
        "CEP90_soc",
        "RMSE_qp",
        "RMSE_soc",
        "TerminalMiss_qp",
        "TerminalMiss_soc",
    ]
    agg = pd.DataFrame({c: [med_iqr(df[c])] for c in cols if c in df.columns})
    _save_csv_tex(agg, out_dir, "T2_metrics_aggregate", "Aggregate metrics (median [IQR])", "tab:aggregate")


def write_results_index(out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "results_tables.tex"), "w") as f:
        f.write("% Auto-generated table assembly\n")
        for name in [
            "T1_metrics_per_sortie",
            "T2_metrics_aggregate",
        ]:
            f.write(f"\\input{{{name}.tex}}\n")
