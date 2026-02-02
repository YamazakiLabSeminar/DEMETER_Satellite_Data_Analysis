from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import logging

import numpy as np
import pandas as pd

# =====================================================
# Step2: Kp merge (nearest) + binning + CDF normalization (3-pass)
#
# 入力: Step1出力CSV群（1軌道ごと）
#   必須列: datetime, lat, lon, mlat, mlon, E_1700band_mean, is_filled
#
# 入力: Kp CSV（UTC）
#   例: year,month,day,hour,minute,sec(or second),milsec(or milsecond),kp
#
# 出力: Step2出力CSV群（1軌道ごと）
#   追加列: kp_str, kp_num, kp_cat, season, mlat_bin, mlon_bin, bin_id, E_norm
# =====================================================

STEP1_REQUIRED_COLS = [
    "datetime",
    "lat",
    "lon",
    "mlat",
    "mlon",
    "E_1700band_mean",
    "is_filled",
]

@dataclass(frozen=True)
class Step2IO:
    step1_dir: Path
    kp_csv_path: Path
    out_dir: Path
    tables_dir: Path
    checkpoint_path: Path
    bin_stats_path: Path