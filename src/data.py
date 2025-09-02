from __future__ import annotations

import os
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
)


@dataclass
class Streams:
    radar: pd.DataFrame
    telemetry: pd.DataFrame


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


def load_config(path: str) -> Dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def load_streams(radar_csv: str, telem_csv: str, cfg: Dict) -> Streams:
    radar = pd.read_csv(radar_csv)
    telem = pd.read_csv(telem_csv)
    return Streams(radar=radar, telemetry=telem)


def resample_and_filter(streams: Streams, cfg: Dict, out_dir: str) -> Processed:
    dt = float(cfg["preprocessing"]["dt"])  # 0.1 s

    # Parse times
    radar_time_col = _find_first(streams.radar, cfg.get("mapping", {}).get("radar", {}).get("time", ["Time", "time"])) or "Time"
    telem_time_col = _find_first(streams.telemetry, cfg.get("mapping", {}).get("telemetry", {}).get("time", ["Time", "time"])) or "Time"
    t_radar = _parse_time_col(streams.radar[radar_time_col])
    t_telem = _parse_time_col(streams.telemetry[telem_time_col])

    # Telemetry geodetic
    telem_lat_col = _find_first(streams.telemetry, ["Weapon LAT (deg)", "lat", "Latitude", "Lat"])
    telem_lon_col = _find_first(streams.telemetry, ["Weapon LON (deg)", "lon", "Longitude", "Lon"])
    telem_alt_col = _find_first(streams.telemetry, ["Weapon ALT (ft)", "alt", "Altitude"])
    if telem_lat_col is None or telem_lon_col is None or telem_alt_col is None:
        raise ValueError("Telemetry must contain weapon latitude, longitude, and altitude columns.")
    telem_lat = np.deg2rad(streams.telemetry[telem_lat_col].astype(float).to_numpy())
    telem_lon = np.deg2rad(streams.telemetry[telem_lon_col].astype(float).to_numpy())
    telem_alt_vals = streams.telemetry[telem_alt_col].astype(float).to_numpy()
    # feet to meters if header indicates ft
    if "(ft)" in telem_alt_col or "ft" in telem_alt_col.lower():
        telem_alt_vals = telem_alt_vals * 0.3048

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
        if "(ft)" in tgt_alt_col_telem or "ft" in tgt_alt_col_telem.lower():
            tgt_alt_val = tgt_alt_val * 0.3048
        target_local = geodetic_to_local(
            np.array([np.deg2rad(tgt_lat_deg)]), np.array([np.deg2rad(tgt_lon_deg)]), np.array([tgt_alt_val]), phi0, lam0, h0
        )[0]

    # Radar positions: prefer direct geodetic if available
    radar_lat_col = _find_first(streams.radar, ["Weapon LAT (deg)", "lat", "Latitude"])
    radar_lon_col = _find_first(streams.radar, ["Weapon LON (deg)", "lon", "Longitude"])
    radar_alt_col = _find_first(streams.radar, ["Weapon ALT (ft)", "alt", "Altitude"])

    if radar_lat_col and radar_lon_col and radar_alt_col:
        r_lat = np.deg2rad(streams.radar[radar_lat_col].astype(float).to_numpy())
        r_lon = np.deg2rad(streams.radar[radar_lon_col].astype(float).to_numpy())
        r_alt = streams.radar[radar_alt_col].astype(float).to_numpy()
        if "(ft)" in radar_alt_col or "ft" in radar_alt_col.lower():
            r_alt = r_alt * 0.3048
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
            if "(ft)" in tgt_alt_col_telem or "ft" in tgt_alt_col_telem.lower():
                tgt_alt = tgt_alt * 0.3048
            tgt_xyz = geodetic_to_local(tgt_lat, tgt_lon, tgt_alt, phi0, lam0, h0)
            n = streams.radar[off_n_col].astype(float).to_numpy()
            e = streams.radar[off_e_col].astype(float).to_numpy()
            d = streams.radar[off_d_col].astype(float).to_numpy()
            # feet to meters if likely in feet (assume offsets in meters if large?)
            # Heuristic: if median(|n|) > 1000, assume feet
            scale = 0.3048 if np.nanmedian(np.abs(n)) > 1000 else 1.0
            n = n * scale
            e = e * scale
            d = d * scale
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

    # Telemetry velocity by differentiating smoothed position
    telem_vxyz = np.vstack([np.gradient(telem_xyz_t[:, i], dt) for i in range(3)]).T

    # Alignment via Doppler proxy (if available): skip for now due to unknown doppler column
    lag_est = 0.0

    # Save caches
    proc_dir = os.path.join(out_dir, "artifacts", "processed")
    ensure_dir(proc_dir)
    df_radar = pd.DataFrame({"time": t, "x": radar_xyz_t[:, 0], "y": radar_xyz_t[:, 1], "z": radar_xyz_t[:, 2]})
    df_telem = pd.DataFrame({"time": t, "x": telem_xyz_t[:, 0], "y": telem_xyz_t[:, 1], "z": telem_xyz_t[:, 2]})
    try:
        df_radar.to_parquet(os.path.join(proc_dir, "radar_local.parquet"))
    except Exception:
        df_radar.to_csv(os.path.join(proc_dir, "radar_local.csv"), index=False)
    try:
        df_telem.to_parquet(os.path.join(proc_dir, "telemetry_local.parquet"))
    except Exception:
        df_telem.to_csv(os.path.join(proc_dir, "telemetry_local.csv"), index=False)

    return Processed(time=t, radar_xyz=radar_xyz_t, telem_xyz=telem_xyz_t, telem_vxyz=telem_vxyz, lag_est_s=float(lag_est), target_local=target_local)
