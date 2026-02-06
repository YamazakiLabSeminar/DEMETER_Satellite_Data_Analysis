from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def standardize_eq_catalog(
    in_csv: Path,
    out_csv: Path,
) -> pd.DataFrame:
    """
    地震カタログから ["time","latitude","longitude","depth","mag"] を抽出し、
    新ヘッダ ["eq_id","datetime","lat","lon","depth","mag"] に整形して保存する。
    """
    if not in_csv.exists():
        raise FileNotFoundError(f"Input CSV not found: {in_csv}")

    df = pd.read_csv(in_csv)

    required = ["time", "latitude", "longitude", "depth", "mag"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Input CSV missing columns: {missing}")

    out = pd.DataFrame(
        {
            "datetime": pd.to_datetime(df["time"], errors="coerce", format="mixed"),
            "lat": pd.to_numeric(df["latitude"], errors="coerce"),
            "lon": pd.to_numeric(df["longitude"], errors="coerce"),
            "depth": pd.to_numeric(df["depth"], errors="coerce"),
            "mag": pd.to_numeric(df["mag"], errors="coerce"),
        }
    )

    out = out.dropna(subset=["datetime", "lat", "lon", "depth", "mag"]).copy()
    out = out.sort_values("datetime").reset_index(drop=True)
    out.insert(0, "eq_id", np.arange(1, len(out) + 1, dtype=int))

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_csv, index=False, encoding="utf-8")
    return out
