import matplotlib
matplotlib.use('Agg')

import numpy as np
from src.figures import fig_error_ecdf_hist


def test_fig_error_ecdf_hist(tmp_path):
    rng = np.random.default_rng(0)
    d_raw = np.abs(rng.normal(size=100))
    d_qp = np.abs(rng.normal(scale=0.8, size=100))
    d_soc = np.abs(rng.normal(scale=0.5, size=100))
    fig_error_ecdf_hist(d_raw, d_qp, d_soc, str(tmp_path))
    assert (tmp_path / 'fig_F3_error_ecdf_hist.png').exists()
    assert (tmp_path / 'fig_F3_error_ecdf_hist.pdf').exists()
