from __future__ import annotations

import shutil
from pathlib import Path
from typing import Optional

import pandas as pd

#=======================================
# 動作：
# - orbit_indexを読み込み、in_eq_window == TRUEの行だけ抽出してDataFrameで返す。
# - interim/step2_normalizedの中から"df(orbit_index)["orbit_file"]と一致するファイルを選び、
# 　新規フォルダstep3_candidateにコピーする。
#=======================================

def extract_candidate_orbits(
    orbit_index_csv: Path,
    step2_dir: Path,
    out_dir: Optional[Path] = None,
) -> pd.DataFrame:
    """
    orbit_index を読み込み、in_eq_window が TRUE の行のみ抽出し DataFrame として返す。
    抽出された orbit_file と一致するファイルを step2_dir から探し、
    step3_candidate フォルダにコピーする。
    """
    if not orbit_index_csv.exists():
        raise FileNotFoundError(f"orbit_index CSV not found: {orbit_index_csv}")
    if not step2_dir.exists():
        raise FileNotFoundError(f"step2 folder not found: {step2_dir}")

    df = pd.read_csv(orbit_index_csv)
    if "in_eq_window" not in df.columns:
        raise ValueError("orbit_index CSV missing column: in_eq_window")
    if "orbit_file" not in df.columns:
        raise ValueError("orbit_index CSV missing column: orbit_file")

    mask = df["in_eq_window"].astype(str).str.upper().isin(["TRUE", "1", "YES"])
    cand_df = df[mask].copy()

    if out_dir is None:
        out_dir = step2_dir.parent / "step3_candidate"
    out_dir.mkdir(parents=True, exist_ok=True)

    file_set = set(cand_df["orbit_file"].dropna().astype(str))
    for f in step2_dir.iterdir():
        if f.is_file() and f.name in file_set:
            shutil.copy2(f, out_dir / f.name)

    return cand_df

