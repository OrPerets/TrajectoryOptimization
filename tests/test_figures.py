import matplotlib
matplotlib.use('Agg')

import numpy as np
from src.figures import (
    fig_error_ecdf_hist, 
    fig_cep_curves, 
    fig_theta_sensitivity,
    fig_runtime_breakdown,
    fig_kinematics_sanity,
    fig_impact_angle_hist
)


def test_fig_error_ecdf_hist(tmp_path):
    rng = np.random.default_rng(0)
    d_raw = np.abs(rng.normal(size=100))
    d_qp = np.abs(rng.normal(scale=0.8, size=100))
    d_soc = np.abs(rng.normal(scale=0.5, size=100))
    fig_error_ecdf_hist(d_raw, d_qp, d_soc, str(tmp_path))
    assert (tmp_path / 'fig_F3_error_ecdf_hist.png').exists()
    assert (tmp_path / 'fig_F3_error_ecdf_hist.pdf').exists()


def test_fig_cep_curves(tmp_path):
    rng = np.random.default_rng(0)
    d_raw = np.abs(rng.normal(size=100))
    d_qp = np.abs(rng.normal(scale=0.8, size=100))
    d_soc = np.abs(rng.normal(scale=0.5, size=100))
    fig_cep_curves(d_raw, d_qp, d_soc, str(tmp_path))
    assert (tmp_path / 'fig_F4_cep_curves.png').exists()
    assert (tmp_path / 'fig_F4_cep_curves.pdf').exists()


def test_fig_theta_sensitivity(tmp_path):
    rng = np.random.default_rng(0)
    d_raw = np.abs(rng.normal(size=100))
    d_qp = np.abs(rng.normal(scale=0.8, size=100))
    theta_values = [10.0, 15.0, 20.0]
    d_soc_by_theta = [np.abs(rng.normal(scale=0.5, size=100)) for _ in theta_values]
    fig_theta_sensitivity(d_raw, d_qp, theta_values, d_soc_by_theta, str(tmp_path))
    assert (tmp_path / 'fig_F5_theta_sensitivity.png').exists()
    assert (tmp_path / 'fig_F5_theta_sensitivity.pdf').exists()


def test_fig_runtime_breakdown(tmp_path):
    runtime_data = {
        "data_loading": 1.5,
        "optimization": 10.2,
        "figure_generation": 2.1,
        "table_generation": 0.8
    }
    fig_runtime_breakdown(runtime_data, str(tmp_path))
    assert (tmp_path / 'fig_F7_runtime_breakdown.png').exists()
    assert (tmp_path / 'fig_F7_runtime_breakdown.pdf').exists()


def test_fig_kinematics_sanity(tmp_path):
    rng = np.random.default_rng(0)
    n_samples = 50
    t = np.linspace(0, 5, n_samples)
    v_qp = rng.normal(size=(n_samples, 3))
    v_soc = rng.normal(size=(n_samples, 3))
    fig_kinematics_sanity(t, v_qp, v_soc, str(tmp_path))
    assert (tmp_path / 'fig_F8_kinematics_sanity.png').exists()
    assert (tmp_path / 'fig_F8_kinematics_sanity.pdf').exists()


def test_fig_impact_angle_hist(tmp_path):
    rng = np.random.default_rng(0)
    impact_angles_qp = rng.uniform(0, 45, 100)
    impact_angles_soc = rng.uniform(0, 15, 100)
    fig_impact_angle_hist(impact_angles_qp, impact_angles_soc, str(tmp_path))
    assert (tmp_path / 'fig_F9_impact_angle_hist.png').exists()
    assert (tmp_path / 'fig_F9_impact_angle_hist.pdf').exists()
