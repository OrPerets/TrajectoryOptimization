import numpy as np
import pandas as pd
import pytest

from src.data import _parse_time_col, convert_units, Streams, resample_and_filter, load_config
from src.utils import normalize_inputs


def test_parse_time_doy_format():
    s = pd.Series(["123:01:02:03.000", "123:01:02:03.100", "123:01:02:03.200"])
    t = _parse_time_col(s)
    assert np.allclose(t, [0.0, 0.1, 0.2])


def test_convert_units_feet_to_meters():
    vals = np.array([2000.0, 2000.0])
    conv = convert_units("Altitude", vals, "altitude")
    assert np.allclose(conv, vals * 0.3048)


def test_offset_to_local_mapping(tmp_path):
    telem = pd.DataFrame({
        "Time": [0.0, 0.1, 0.2, 0.3],
        "Weapon LAT (deg)": [0.0, 0.0, 0.0, 0.0],
        "Weapon LON (deg)": [0.0, 0.0, 0.0, 0.0],
        "Weapon ALT (ft)": [0.0, 0.0, 0.0, 0.0],
        "Target LAT (deg)": [0.0, 0.0, 0.0, 0.0],
        "Target LON (deg)": [0.0, 0.0, 0.0, 0.0],
        "Target ALT (ft)": [0.0, 0.0, 0.0, 0.0],
    })
    radar = pd.DataFrame({
        "Time": [0.0, 0.1, 0.2, 0.3],
        "Tgt Offset North": [100.0, 100.0, 100.0, 100.0],
        "Tgt Offset East": [200.0, 200.0, 200.0, 200.0],
        "Tgt Offset Down": [50.0, 50.0, 50.0, 50.0],
    })
    streams = Streams(radar=radar, telemetry=telem, radar_path="", telemetry_path="")
    cfg = load_config("config/default.yaml")
    cfg["preprocessing"]["savgol_window"] = 3
    cfg["preprocessing"]["savgol_polyorder"] = 1
    proc = resample_and_filter(streams, cfg, out_dir=str(tmp_path))
    assert np.allclose(proc.radar_xyz[0], [200.0, 100.0, -50.0])


def test_normalize_inputs_scaling():
    data = {
        "positions": np.array([[0.0, 0.0, 0.0], [1000.0, 0.0, 0.0]]),
        "velocities": np.array([[100.0, 0.0, 0.0], [100.0, 0.0, 0.0]]),
    }
    cfg = {
        "general": {
            "enable_scaling": True,
            "scale_position_m": 1000.0,
            "scale_velocity_mps": 100.0,
        }
    }
    norm, summary = normalize_inputs(data, cfg)
    assert np.allclose(norm["positions"], [[-0.5, 0.0, 0.0], [0.5, 0.0, 0.0]])
    assert np.allclose(norm["velocities"], [[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    assert summary["position_raw"]["max"] == pytest.approx(1000.0)
    assert summary["position_norm"]["max"] == pytest.approx(0.5)

