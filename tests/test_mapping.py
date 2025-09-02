import numpy as np
import pandas as pd
from src.model import geodetic_to_local


def test_flat_earth_linear():
    # 1 degree in lon at equator ~ 111319.49 m
    a = 6378137.0
    phi0 = 0.0
    lam0 = 0.0
    h0 = 0.0
    phis = np.array([0.0, 0.0])
    lams = np.array([0.0, np.deg2rad(1.0)])
    hs = np.array([0.0, 0.0])
    xyz = geodetic_to_local(phis, lams, hs, phi0, lam0, h0)
    assert np.isclose(xyz[1,0], 111319.49, rtol=1e-3)
