import numpy as np
from src.model import discrete_step_matrices, G0


def test_small_k_series_matches_limit():
    k = 1e-8
    dt = 0.1
    A, B, Cmag = discrete_step_matrices(k, dt)
    # As k->0, e^{-k dt} ~ 1 - k dt
    assert np.allclose(np.diag(A), 1 - k * dt, atol=1e-6)
    # (1 - e^{-k dt})/k ~ dt
    assert np.isclose(B[0, 0], dt, atol=1e-6)
    # Cmag ~ 0.5 g dt^2
    assert np.isclose(Cmag, 0.5 * G0 * dt * dt, atol=1e-6)
