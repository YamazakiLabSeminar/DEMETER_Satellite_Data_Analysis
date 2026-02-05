from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd


REQUIRED_COLS = ["latitude", "longitude", "magnitude", "datetime", "event_id"]
COL_CANDIDATES = {
    "latitude": ["latitude", "lat", "eq_lat"],
    "longitude": ["longitude", "lon", "eq_lon"],
    "magnitude": ["magnitude", "mag", "mw", "ml"],
    "datetime": ["datetime", "time", "origin_time", "ot"],
    "event_id": ["event_id", "id", "eq_id"],
}


@dataclass(frozen=True)
class EqDeclusterConfig:
    fs_time_prop: float = 1.0
    keep_only_mainshock: bool = True


def _log_info(logger, msg: str) -> None:
    if logger is None:
        print(msg)
    else:
        logger.info(msg)


def _pick_column(df: pd.DataFrame, canonical: str) -> str:
    for cand in COL_CANDIDATES[canonical]:
        if cand in df.columns:
            return cand
    raise ValueError(
        f"Input CSV missing '{canonical}' column. "
        f"Tried: {COL_CANDIDATES[canonical]}, have: {list(df.columns)}"
    )


def _normalize_columns(df: pd.DataFrame, logger=None) -> pd.DataFrame:
    """
    列名の揺れを吸収して、REQUIRED_COLS を持つDataFrameに正規化する。
    """
    picked = {k: _pick_column(df, k) for k in REQUIRED_COLS}
    _log_info(
        logger,
        "Column mapping: "
        + ", ".join([f"{k}<-{v}" for k, v in picked.items()]),
    )
    out = df.copy()
    for canonical, src in picked.items():
        if canonical != src:
            out[canonical] = out[src]
    return out


def _to_hmtk_catalog(df: pd.DataFrame):
    """
    DataFrame -> OpenQuake HMTK Catalogue へ変換する。
    """
    try:
        from openquake.hmtk.seismicity.catalogue import Catalogue
    except Exception as e:
        raise ImportError(
            "openquake-engine が未導入のため declustering を実行できません。"
            " 例: pip install openquake-engine"
        ) from e

    dt = pd.to_datetime(df["datetime"], utc=True, errors="raise", format="mixed")
    # HMTKは second を float で受け取れるため、小数秒を保持する
    second = dt.dt.second.astype(float) + dt.dt.microsecond.astype(float) / 1_000_000.0

    data = {
        "eventID": df["event_id"].astype(str).to_numpy(),
        "longitude": pd.to_numeric(df["longitude"], errors="raise").to_numpy(),
        "latitude": pd.to_numeric(df["latitude"], errors="raise").to_numpy(),
        "magnitude": pd.to_numeric(df["magnitude"], errors="raise").to_numpy(),
        "year": dt.dt.year.astype(int).to_numpy(),
        "month": dt.dt.month.astype(int).to_numpy(),
        "day": dt.dt.day.astype(int).to_numpy(),
        "hour": dt.dt.hour.astype(int).to_numpy(),
        "minute": dt.dt.minute.astype(int).to_numpy(),
        "second": second.to_numpy(),
    }
    return Catalogue.make_from_dict(data)


def run_gardner_knopoff_declustering(
    in_csv: Path,
    out_csv: Path,
    cfg: Optional[EqDeclusterConfig] = None,
    logger=None,
) -> pd.DataFrame:
    """
    地震カタログCSVに GardnerKnopoffType1 を適用し、結果CSVを保存する。

    付与列:
    - cluster_index
    - cluster_flag
    - is_mainshock (cluster_flag == 0)
    """
    if cfg is None:
        cfg = EqDeclusterConfig()

    if not in_csv.exists():
        raise FileNotFoundError(f"Input CSV not found: {in_csv}")

    df = pd.read_csv(in_csv)
    df = _normalize_columns(df, logger=logger)

    # 入力順依存を避けるため時刻で並べる
    df = df.copy()
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True, errors="raise", format="mixed")
    df = df.sort_values("datetime").reset_index(drop=True)

    hmtk_catalog = _to_hmtk_catalog(df)

    try:
        from openquake.hmtk.seismicity.declusterer.dec_gardner_knopoff import (
            GardnerKnopoffType1,
        )
        from openquake.hmtk.seismicity.declusterer.distance_time_windows import (
            GardnerKnopoffWindow,
        )
    except Exception as e:
        raise ImportError(
            "openquake-engine の declusterer モジュール読み込みに失敗しました。"
        ) from e

    declusterer = GardnerKnopoffType1()
    decluster_cfg = {
        "time_distance_window": GardnerKnopoffWindow(),
        "fs_time_prop": float(cfg.fs_time_prop),
    }

    _log_info(logger, f"Declustering start: rows={len(df)}, fs_time_prop={cfg.fs_time_prop}")
    cluster_index, cluster_flag = declusterer.decluster(hmtk_catalog, decluster_cfg)

    df["cluster_index"] = cluster_index
    df["cluster_flag"] = cluster_flag
    df["is_mainshock"] = df["cluster_flag"] == 0

    out_df = df[df["is_mainshock"]].copy() if cfg.keep_only_mainshock else df
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_csv, index=False, encoding="utf-8")

    _log_info(logger, f"Declustering done: total={len(df)}, saved={len(out_df)}")
    _log_info(logger, f"Saved: {out_csv}")
    return out_df


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="EQ catalog preprocessing with GardnerKnopoffType1")
    p.add_argument("--in_csv", type=str, required=True, help="Input earthquake catalog CSV path")
    p.add_argument("--out_csv", type=str, required=True, help="Output CSV path")
    p.add_argument("--fs_time_prop", type=float, default=1.0, help="OpenQuake fs_time_prop")
    p.add_argument(
        "--keep_all",
        action="store_true",
        help="Keep all events (default is keep only mainshocks)",
    )
    return p


def main() -> None:
    args = _build_argparser().parse_args()
    cfg = EqDeclusterConfig(
        fs_time_prop=args.fs_time_prop,
        keep_only_mainshock=not args.keep_all,
    )
    run_gardner_knopoff_declustering(
        in_csv=Path(args.in_csv),
        out_csv=Path(args.out_csv),
        cfg=cfg,
        logger=None,
    )


if __name__ == "__main__":
    main()
