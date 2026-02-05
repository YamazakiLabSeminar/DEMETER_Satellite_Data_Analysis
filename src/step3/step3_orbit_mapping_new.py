from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Step3MappingConfig:
    lead_hours: float = 4.0
    max_dist_km: float = 330.0
    orbit_index_name: str = "step3_orbit_index.csv"
    orbit_map_name: str = "step3_orbit_map.csv"


def _log_info(logger, msg: str) -> None:
    if logger is None:
        print(msg)
    else:
        logger.info(msg)


def _log_warn(logger, msg: str) -> None:
    if logger is None:
        print(f"[WARN] {msg}")
    else:
        logger.warning(msg)


def _parse_datetime_str(dt_str: str) -> pd.Timestamp:
    ts = pd.to_datetime(dt_str, errors="raise")
    if getattr(ts, "tzinfo", None) is not None:
        ts = ts.tz_localize(None)
    return ts


def get_first_last_datetime_csv(
    csv_path: Path, datetime_col: str = "datetime"
) -> Tuple[pd.Timestamp, pd.Timestamp]:
    with csv_path.open("r", encoding="utf-8", errors="ignore") as f:
        header = f.readline().rstrip("\n")
        cols = header.split(",")
        if datetime_col not in cols:
            raise ValueError(f"{csv_path.name}: '{datetime_col}' not found.")
        dt_idx = cols.index(datetime_col)

        first_line = ""
        while True:
            line = f.readline()
            if not line:
                break
            line = line.strip()
            if line:
                first_line = line
                break
        if not first_line:
            raise ValueError(f"{csv_path.name}: no data line.")
        first_dt = _parse_datetime_str(first_line.split(",")[dt_idx])

    with csv_path.open("rb") as fb:
        fb.seek(0, 2)
        size = fb.tell()
        if size == 0:
            raise ValueError(f"{csv_path.name}: empty file.")

        block = b""
        pos = size
        while pos > 0:
            read_size = min(4096, pos)
            pos -= read_size
            fb.seek(pos)
            block = fb.read(read_size) + block
            if b"\n" in block and block.count(b"\n") >= 2:
                break

        lines = block.splitlines()
        last_line_bytes = b""
        for i in range(len(lines) - 1, -1, -1):
            if lines[i].strip():
                last_line_bytes = lines[i]
                break
        if not last_line_bytes:
            raise ValueError(f"{csv_path.name}: could not find last line.")

        last_line = last_line_bytes.decode("utf-8", errors="ignore")
        if last_line.startswith("datetime") or last_line.startswith(header):
            raise ValueError(f"{csv_path.name}: invalid last line.")
        last_dt = _parse_datetime_str(last_line.split(",")[dt_idx])

    return first_dt, last_dt


def hubeny_distance_km(
    lat1_deg: np.ndarray, lon1_deg: np.ndarray, lat2_deg: float, lon2_deg: float
) -> np.ndarray:
    a = 6378137.0
    f = 1 / 298.257223563
    b = a * (1 - f)
    e2 = (a**2 - b**2) / a**2

    lat1 = np.deg2rad(lat1_deg)
    lon1 = np.deg2rad(lon1_deg)
    lat2 = np.deg2rad(lat2_deg)
    lon2 = np.deg2rad(lon2_deg)

    latm = (lat1 + lat2) / 2.0
    dlat = lat1 - lat2
    dlon = lon1 - lon2

    sin_latm = np.sin(latm)
    w = np.sqrt(1.0 - e2 * (sin_latm**2))
    m = a * (1.0 - e2) / (w**3)
    n = a / w

    dy = m * dlat
    dx = n * np.cos(latm) * dlon
    return np.sqrt(dx**2 + dy**2) / 1000.0


def find_segment_around_min(mask: np.ndarray, idx_min: int) -> Tuple[int, int]:
    start = idx_min
    end = idx_min
    while start - 1 >= 0 and mask[start - 1]:
        start -= 1
    while end + 1 < len(mask) and mask[end + 1]:
        end += 1
    return start, end


def build_orbit_index(step2_dir: Path, out_csv: Path, logger=None) -> pd.DataFrame:
    rows = []
    csv_files = sorted(step2_dir.glob("*.csv"))
    total_files = len(csv_files)
    progress_every = max(1, total_files // 20)

    for i, csv_path in enumerate(csv_files, start=1):
        first_dt, last_dt = get_first_last_datetime_csv(csv_path, datetime_col="datetime")
        rows.append(
            {
                "orbit_file": csv_path.name,
                "orbit_start_time": first_dt.isoformat(),
                "orbit_end_time": last_dt.isoformat(),
            }
        )
        if (i == 1) or (i % progress_every == 0) or (i == total_files):
            pct = (i / total_files) * 100.0 if total_files > 0 else 100.0
            _log_info(logger, f"[INDEX] {i}/{total_files} ({pct:.1f}%)")

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_df = pd.DataFrame(rows)
    out_df.to_csv(out_csv, index=False, encoding="utf-8")
    return out_df


def run_step3_orbit_mapping_new(
    eq_catalog_path: Path,
    step2_dir: Path,
    tables_dir: Path,
    cfg: Optional[Step3MappingConfig] = None,
    logger=None,
) -> None:
    if cfg is None:
        cfg = Step3MappingConfig()

    if not eq_catalog_path.exists():
        raise FileNotFoundError(f"EQ catalog not found: {eq_catalog_path}")
    if not step2_dir.exists():
        raise FileNotFoundError(f"Step2 directory not found: {step2_dir}")

    tables_dir.mkdir(parents=True, exist_ok=True)
    orbit_index_path = tables_dir / cfg.orbit_index_name
    orbit_map_path = tables_dir / cfg.orbit_map_name

    _log_info(logger, f"Read earthquake catalog: {eq_catalog_path}")
    eq_df = pd.read_csv(eq_catalog_path)

    required_eq_cols = ["latitude", "longitude", "mag", "time"]
    missing_cols = [c for c in required_eq_cols if c not in eq_df.columns]
    if missing_cols:
        raise ValueError(f"eq_catalog missing columns: {missing_cols}")

    eq_df = eq_df[required_eq_cols].copy()
    eq_df["time"] = pd.to_datetime(eq_df["time"], errors="raise", format="mixed")
    if getattr(eq_df["time"].dt, "tz", None) is not None:
        eq_df["time"] = eq_df["time"].dt.tz_localize(None)

    if orbit_index_path.exists():
        _log_info(logger, f"Load orbit index: {orbit_index_path}")
        orbit_index = pd.read_csv(orbit_index_path)
    else:
        _log_info(logger, f"Build orbit index from Step2 outputs: {step2_dir}")
        orbit_index = build_orbit_index(step2_dir=step2_dir, out_csv=orbit_index_path, logger=logger)

    orbit_index["orbit_start_time"] = pd.to_datetime(
        orbit_index["orbit_start_time"], errors="raise", format="mixed"
    )
    orbit_index["orbit_end_time"] = pd.to_datetime(
        orbit_index["orbit_end_time"], errors="raise", format="mixed"
    )
    if getattr(orbit_index["orbit_start_time"].dt, "tz", None) is not None:
        orbit_index["orbit_start_time"] = orbit_index["orbit_start_time"].dt.tz_localize(None)
    if getattr(orbit_index["orbit_end_time"].dt, "tz", None) is not None:
        orbit_index["orbit_end_time"] = orbit_index["orbit_end_time"].dt.tz_localize(None)

    lead_td = timedelta(hours=cfg.lead_hours)
    out_rows: List[dict] = []
    n_no_match = 0
    n_multi_candidates = 0

    total_eq = len(eq_df)
    progress_every_eq = max(1, total_eq // 20)

    def report_progress(i: int) -> None:
        if (i == 1) or (i % progress_every_eq == 0) or (i == total_eq):
            pct = (i / total_eq) * 100.0 if total_eq > 0 else 100.0
            _log_info(
                logger,
                f"[MAP] {i}/{total_eq} ({pct:.1f}%) linked={len(out_rows)} "
                f"no_link={n_no_match} multi_cand={n_multi_candidates}",
            )

    for i, (_, eq) in enumerate(eq_df.iterrows(), start=1):
        eq_time = eq["time"]
        eq_lat = float(eq["latitude"])
        eq_lon = float(eq["longitude"])

        window_start = eq_time - lead_td
        window_end = eq_time

        cand = orbit_index[
            (orbit_index["orbit_end_time"] > window_start)
            & (orbit_index["orbit_start_time"] < window_end)
        ].copy()

        if cand.empty:
            n_no_match += 1
            report_progress(i)
            continue

        if len(cand) > 1:
            n_multi_candidates += 1

        best_row: Optional[dict] = None
        best_closest = np.inf

        for _, ob in cand.iterrows():
            orbit_file = ob["orbit_file"]
            orbit_start = ob["orbit_start_time"]
            orbit_end = ob["orbit_end_time"]

            orbit_path = step2_dir / orbit_file
            if not orbit_path.exists():
                _log_warn(logger, f"Orbit file not found: {orbit_path}")
                continue

            try:
                df = pd.read_csv(orbit_path, usecols=["datetime", "lat", "lon", "mlat", "mlon"])
            except Exception as e:
                _log_warn(logger, f"Failed reading {orbit_file}: {e}")
                continue

            try:
                df["datetime"] = pd.to_datetime(df["datetime"], errors="raise", format="mixed")
                if getattr(df["datetime"].dt, "tz", None) is not None:
                    df["datetime"] = df["datetime"].dt.tz_localize(None)
            except Exception as e:
                _log_warn(logger, f"{orbit_file}: datetime parse failed: {e}")
                continue

            df = df[(df["datetime"] >= window_start) & (df["datetime"] <= window_end)].copy()
            if df.empty:
                continue

            df = df.dropna(subset=["lat", "lon", "mlat"])
            if df.empty:
                continue

            lat_arr = df["lat"].to_numpy(dtype="float64")
            lon_arr = df["lon"].to_numpy(dtype="float64")
            mlat_arr = df["mlat"].to_numpy(dtype="float64")
            dt_arr = df["datetime"].to_numpy()

            dist_km = hubeny_distance_km(lat_arr, lon_arr, eq_lat, eq_lon)
            mask = dist_km <= cfg.max_dist_km
            if not np.any(mask):
                continue

            idx_true = np.where(mask)[0]
            idx_min = idx_true[np.argmin(dist_km[idx_true])]
            seg_start, seg_end = find_segment_around_min(mask, idx_min)

            pass_time_start = pd.Timestamp(dt_arr[seg_start]).isoformat()
            pass_time_end = pd.Timestamp(dt_arr[seg_end]).isoformat()
            closest_in_seg = float(np.min(dist_km[seg_start : seg_end + 1]))

            if closest_in_seg < best_closest:
                best_closest = closest_in_seg
                best_row = {
                    "eq_time": eq_time.isoformat(),
                    "eq_lat": eq_lat,
                    "eq_lon": eq_lon,
                    "orbit_file": orbit_file,
                    "orbit_start_time": orbit_start.isoformat(),
                    "orbit_end_time": orbit_end.isoformat(),
                    "pass_time_start": pass_time_start,
                    "pass_time_end": pass_time_end,
                    "closest_dis_km": closest_in_seg,
                }

        if best_row is None:
            n_no_match += 1
            report_progress(i)
            continue

        out_rows.append(best_row)
        report_progress(i)

    out_df = pd.DataFrame(out_rows)
    columns = [
        "eq_time",
        "eq_lat",
        "eq_lon",
        "orbit_file",
        "orbit_start_time",
        "orbit_end_time",
        "pass_time_start",
        "pass_time_end",
        "closest_dis_km",
    ]
    for c in columns:
        if c not in out_df.columns:
            out_df[c] = np.nan
    out_df = out_df[columns]
    out_df.to_csv(orbit_map_path, index=False, encoding="utf-8")

    _log_info(logger, f"Saved orbit index: {orbit_index_path}")
    _log_info(logger, f"Saved orbit map: {orbit_map_path} (rows={len(out_df)})")
    _log_info(logger, f"No-link events count: {n_no_match}")
    _log_info(logger, f"Events with multiple time-overlap candidates: {n_multi_candidates}")


# 既存呼び出し名との互換
run_step3_orbit_mapping = run_step3_orbit_mapping_new
