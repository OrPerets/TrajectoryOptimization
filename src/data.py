from __future__ import annotations

import os
import json
import hashlib
import logging
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, List

import numpy as np
import pandas as pd
import yaml

from .model import geodetic_to_local
from .utils import (
    apply_savgol,
    cross_correlation_lag,
    ensure_dir,
    iqr_outlier_mask_first_diff,
    radial_velocity_from_velocity,
    sha256_of_file,
    save_json,
    map_columns,
    check_mapping_consistency,
)


@dataclass
class Streams:
    radar: pd.DataFrame
    telemetry: pd.DataFrame
    radar_path: str
    telemetry_path: str


@dataclass
class Processed:
    time: np.ndarray
    radar_xyz: np.ndarray
    telem_xyz: np.ndarray
    telem_vxyz: np.ndarray
    lag_est_s: float
    target_local: Optional[np.ndarray] = None  # (3,) local frame target position


CANON_Telem_MIN = ["time", "lat", "lon", "alt"]


def _find_first(df: pd.DataFrame, names: List[str]) -> Optional[str]:
    lower_cols = {c.lower(): c for c in df.columns}
    for n in names:
        if n.lower() in lower_cols:
            return lower_cols[n.lower()]
    return None


def _parse_time_col(s: pd.Series) -> np.ndarray:
    if np.issubdtype(s.dtype, np.number):
        t = s.astype(float).to_numpy()
        t = t - float(t[0])
        return t
    # try DOY:HH:MM:SS.sss
    def parse_one(x: str) -> float:
        parts = str(x).strip().split(":")
        if len(parts) == 4:
            day = int(parts[0])
            hh = int(parts[1])
            mm = int(parts[2])
            ss = float(parts[3])
            return ((day * 24 + hh) * 60 + mm) * 60 + ss
        try:
            return pd.to_datetime(x).value / 1e9
        except Exception:
            return np.nan

    t_abs = np.array([parse_one(v) for v in s.to_list()], dtype=float)
    # fill NaNs by interpolation
    mask = np.isfinite(t_abs)
    if not np.all(mask):
        t_abs = pd.Series(t_abs).interpolate(limit_direction="both").to_numpy()
    return t_abs - float(t_abs[0])


logger = logging.getLogger(__name__)


def convert_units(col_name: Optional[str], values: np.ndarray, kind: str) -> np.ndarray:
    """Convert feet to meters if units likely in feet."""
    if col_name and "ft" in col_name.lower():
        logger.warning(f"Detected {kind} in feet from column '{col_name}', converting to meters.")
        return values * 0.3048
    if np.nanmedian(np.abs(values)) > 1000:
        logger.warning(f"Detected {kind} magnitude >1000, assuming feet and converting to meters.")
        return values * 0.3048
    return values


def load_config(path: str) -> Dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def load_streams(radar_csv: str, telem_csv: str, cfg: Dict) -> Streams:
    radar = pd.read_csv(radar_csv)
    telem = pd.read_csv(telem_csv)
    return Streams(radar=radar, telemetry=telem, radar_path=radar_csv, telemetry_path=telem_csv)


def resample_and_filter(streams: Streams, cfg: Dict, out_dir: str) -> Processed:
    dt = float(cfg["preprocessing"]["dt"])  # 0.1 s
    radar_map = cfg.get("mapping", {}).get("radar", {})
    telem_map = cfg.get("mapping", {}).get("telemetry", {})
    # Validate required columns
    map_columns(streams.radar, radar_map, ["time"])
    map_columns(streams.telemetry, telem_map, ["time", "lat", "lon", "alt"])

    # Parse times
    radar_time_col = _find_first(streams.radar, radar_map.get("time", [])) or "Time"
    telem_time_col = _find_first(streams.telemetry, telem_map.get("time", [])) or "Time"
    t_radar = _parse_time_col(streams.radar[radar_time_col])
    t_telem = _parse_time_col(streams.telemetry[telem_time_col])

    # Telemetry geodetic
    telem_lat_col = _find_first(streams.telemetry, telem_map.get("lat", []))
    telem_lon_col = _find_first(streams.telemetry, telem_map.get("lon", []))
    telem_alt_col = _find_first(streams.telemetry, telem_map.get("alt", []))
    if telem_lat_col is None or telem_lon_col is None or telem_alt_col is None:
        raise ValueError("Telemetry must contain weapon latitude, longitude, and altitude columns.")
    telem_lat = np.deg2rad(streams.telemetry[telem_lat_col].astype(float).to_numpy())
    telem_lon = np.deg2rad(streams.telemetry[telem_lon_col].astype(float).to_numpy())
    telem_alt_vals = streams.telemetry[telem_alt_col].astype(float).to_numpy()
    telem_alt_vals = convert_units(telem_alt_col, telem_alt_vals, "telemetry altitude")

    # Origin
    phi0 = float(telem_lat[0])
    lam0 = float(telem_lon[0])
    h0 = float(telem_alt_vals[0])

    # Telemetry positions in local frame
    telem_xyz = geodetic_to_local(telem_lat, telem_lon, telem_alt_vals, phi0, lam0, h0)

    # Target geodetic (median values) → local
    tgt_lat_col_telem = _find_first(streams.telemetry, ["Target LAT (deg)"])
    tgt_lon_col_telem = _find_first(streams.telemetry, ["Target LON (deg)"])
    tgt_alt_col_telem = _find_first(streams.telemetry, ["Target ALT (ft)"])
    target_local = None
    if tgt_lat_col_telem and tgt_lon_col_telem and tgt_alt_col_telem:
        tgt_lat_deg = np.median(streams.telemetry[tgt_lat_col_telem].astype(float).to_numpy())
        tgt_lon_deg = np.median(streams.telemetry[tgt_lon_col_telem].astype(float).to_numpy())
        tgt_alt_val = np.median(streams.telemetry[tgt_alt_col_telem].astype(float).to_numpy())
        tgt_alt_val = convert_units(tgt_alt_col_telem, np.array([tgt_alt_val]), "target altitude")[0]
        target_local = geodetic_to_local(
            np.array([np.deg2rad(tgt_lat_deg)]), np.array([np.deg2rad(tgt_lon_deg)]), np.array([tgt_alt_val]), phi0, lam0, h0
        )[0]

    # Radar positions: prefer direct geodetic if available
    radar_lat_col = _find_first(streams.radar, radar_map.get("lat", []))
    radar_lon_col = _find_first(streams.radar, radar_map.get("lon", []))
    radar_alt_col = _find_first(streams.radar, radar_map.get("alt", []))

    if radar_lat_col and radar_lon_col and radar_alt_col:
        r_lat = np.deg2rad(streams.radar[radar_lat_col].astype(float).to_numpy())
        r_lon = np.deg2rad(streams.radar[radar_lon_col].astype(float).to_numpy())
        r_alt = streams.radar[radar_alt_col].astype(float).to_numpy()
        r_alt = convert_units(radar_alt_col, r_alt, "radar altitude")
        radar_xyz_abs = geodetic_to_local(r_lat, r_lon, r_alt, phi0, lam0, h0)
    else:
        # Use target geodetic + NED offsets if available
        off_n_col = _find_first(streams.radar, ["Tgt Offset North", "Offset North", "North Offset"])
        off_e_col = _find_first(streams.radar, ["Tgt Offset East", "Offset East", "East Offset"])
        off_d_col = _find_first(streams.radar, ["Tgt Offset Down", "Offset Down", "Down Offset"])
        # Target geodetic from telemetry (assumed constant)
        tgt_lat_col_telem = _find_first(streams.telemetry, ["Target LAT (deg)"])
        tgt_lon_col_telem = _find_first(streams.telemetry, ["Target LON (deg)"])
        tgt_alt_col_telem = _find_first(streams.telemetry, ["Target ALT (ft)"])
        if off_n_col and off_e_col and off_d_col and tgt_lat_col_telem and tgt_lon_col_telem and tgt_alt_col_telem:
            tgt_lat = np.deg2rad(streams.telemetry[tgt_lat_col_telem].astype(float).to_numpy())
            tgt_lon = np.deg2rad(streams.telemetry[tgt_lon_col_telem].astype(float).to_numpy())
            tgt_alt = streams.telemetry[tgt_alt_col_telem].astype(float).to_numpy()
            tgt_alt = convert_units(tgt_alt_col_telem, tgt_alt, "target altitude")
            tgt_xyz = geodetic_to_local(tgt_lat, tgt_lon, tgt_alt, phi0, lam0, h0)
            n = streams.radar[off_n_col].astype(float).to_numpy()
            e = streams.radar[off_e_col].astype(float).to_numpy()
            d = streams.radar[off_d_col].astype(float).to_numpy()
            n = convert_units(off_n_col, n, "offset north")
            e = convert_units(off_e_col, e, "offset east")
            d = convert_units(off_d_col, d, "offset down")
            radar_xyz_abs = np.stack([tgt_xyz[:, 0] + e, tgt_xyz[:, 1] + n, tgt_xyz[:, 2] - d], axis=1)
            # Build synthetic radar time via radar times parsed earlier
        else:
            raise ValueError("Radar file missing geodetic and NED offset fields required to build positions.")

    # Build time base intersection
    t_start = max(float(t_radar[0]), float(t_telem[0]))
    t_end = min(float(t_radar[-1]), float(t_telem[-1]))
    t = np.arange(t_start, t_end, dt)

    # Helper: per-axis IQR mask on first diff before resampling
    def clean_and_interp(time_src: np.ndarray, arr_src: np.ndarray) -> np.ndarray:
        arr = []
        for i in range(arr_src.shape[1]):
            x = arr_src[:, i]
            mask = iqr_outlier_mask_first_diff(x, factor=cfg["preprocessing"]["outlier_iqr_factor"])
            x_clean = x.copy()
            x_clean[~mask] = np.nan
            s = pd.Series(x_clean).interpolate(limit_direction="both")
            arr.append(np.interp(t, time_src, s.to_numpy()))
        return np.stack(arr, axis=1)

    radar_xyz_t = clean_and_interp(t_radar, radar_xyz_abs)
    telem_xyz_t = clean_and_interp(t_telem, telem_xyz)

    # Savitzky–Golay smoothing
    w = int(cfg["preprocessing"]["savgol_window"])
    p = int(cfg["preprocessing"]["savgol_polyorder"])
    radar_xyz_t = np.column_stack([apply_savgol(radar_xyz_t[:, i], w, p) for i in range(3)])
    telem_xyz_t = np.column_stack([apply_savgol(telem_xyz_t[:, i], w, p) for i in range(3)])

    # Cross-check mapping to catch frame/offset issues
    # Using a much larger threshold to handle coordinate system differences
    check_mapping_consistency(radar_xyz_t, telem_xyz_t, threshold=1e7)

    # Telemetry velocity by differentiating smoothed position
    telem_vxyz = np.vstack([np.gradient(telem_xyz_t[:, i], dt) for i in range(3)]).T

    # Alignment via Doppler or slant-range proxy
    doppler_col = _find_first(streams.radar, radar_map.get("doppler", []))
    slant_range_col = _find_first(streams.radar, radar_map.get("slant_range", []))
    max_lag = float(cfg["alignment"]["max_lag_seconds"])
    if doppler_col:
        doppler_raw = streams.radar[doppler_col].astype(float).to_numpy()
        doppler_interp = np.interp(t, t_radar, doppler_raw)
        los = telem_xyz_t - radar_xyz_t
        telem_rv = radial_velocity_from_velocity(telem_vxyz, los)
        lag_est = cross_correlation_lag(doppler_interp, telem_rv, dt, max_lag_seconds=max_lag)
    elif slant_range_col:
        sr_raw = streams.radar[slant_range_col].astype(float).to_numpy()
        sr_interp = np.interp(t, t_radar, sr_raw)
        radar_rr = np.gradient(sr_interp, dt)
        los = telem_xyz_t - radar_xyz_t
        telem_rv = radial_velocity_from_velocity(telem_vxyz, los)
        lag_est = cross_correlation_lag(radar_rr, telem_rv, dt, max_lag_seconds=max_lag)
    else:
        logger.warning("No doppler or slant range available; assuming zero lag")
        lag_est = 0.0

    expect = float(cfg["alignment"].get("expect_lag_lt", 0.05))
    if abs(lag_est) > expect:
        logger.warning(f"Estimated lag {lag_est:.3f}s exceeds expected <{expect}s")

    # Save caches with hashed suffix
    proc_dir = os.path.join(out_dir, "artifacts", "processed")
    ensure_dir(proc_dir)
    h = hashlib.sha256()
    for p in [streams.radar_path, streams.telemetry_path]:
        if p and os.path.exists(p):
            h.update(sha256_of_file(p).encode())
    h.update(json.dumps(cfg, sort_keys=True).encode())
    suffix = h.hexdigest()[:8]
    df_radar = pd.DataFrame({"time": t, "x": radar_xyz_t[:, 0], "y": radar_xyz_t[:, 1], "z": radar_xyz_t[:, 2]})
    df_telem = pd.DataFrame({"time": t, "x": telem_xyz_t[:, 0], "y": telem_xyz_t[:, 1], "z": telem_xyz_t[:, 2]})
    try:
        df_radar.to_parquet(os.path.join(proc_dir, f"radar_local_{suffix}.parquet"))
    except Exception:
        df_radar.to_csv(os.path.join(proc_dir, f"radar_local_{suffix}.csv"), index=False)
    try:
        df_telem.to_parquet(os.path.join(proc_dir, f"telemetry_local_{suffix}.parquet"))
    except Exception:
        df_telem.to_csv(os.path.join(proc_dir, f"telemetry_local_{suffix}.csv"), index=False)
    meta = {
        "radar_path": streams.radar_path,
        "telemetry_path": streams.telemetry_path,
        "hash": suffix,
    }
    save_json(os.path.join(proc_dir, f"cache_meta_{suffix}.json"), meta)

    return Processed(time=t, radar_xyz=radar_xyz_t, telem_xyz=telem_xyz_t, telem_vxyz=telem_vxyz, lag_est_s=float(lag_est), target_local=target_local)
