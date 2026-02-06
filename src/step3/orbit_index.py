from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional


#===========================================
# 軌道データのファイル名、軌道データの開始/終了時刻の表を作成する。
# ヘッダ: ["orbit_file","orbit_start_time","orbit_end_time"]
# file_name: step3_orbit_index.csv
#============================================


@dataclass(frozen=True)
class OrbitIndexRow:
    orbit_file: str
    orbit_start_time: str
    orbit_end_time: str


def _read_first_last_datetime(csv_path: Path) -> tuple[Optional[str], Optional[str]]:
    """
    CSVファイルの "datetime" 列について、最初行と最後行の値を取得する。
    空行はスキップする。
    """
    encodings = ("utf-8-sig", "utf-8", "cp932")
    last_error: Optional[Exception] = None

    for enc in encodings:
        try:
            with csv_path.open("r", encoding=enc, newline="") as f:
                reader = csv.reader(f)
                header = next(reader, None)
                if header is None:
                    return None, None

                header = [h.strip() for h in header]
                if "datetime" not in header:
                    raise ValueError(f"'datetime' column not found: {csv_path}")
                dt_idx = header.index("datetime")

                first: Optional[str] = None
                last: Optional[str] = None
                for row in reader:
                    if not row or all(not c.strip() for c in row):
                        continue
                    if dt_idx >= len(row):
                        continue
                    dt_val = row[dt_idx].strip()
                    if first is None:
                        first = dt_val
                    last = dt_val
                return first, last
        except Exception as e:
            last_error = e
            continue

    raise RuntimeError(f"Failed to read CSV: {csv_path}") from last_error


def build_orbit_index(
    in_dir: Path,
    out_csv: Optional[Path] = None,
    pattern: str = "*.csv",
    print_lines: bool = True,
) -> list[OrbitIndexRow]:
    """
    フォルダ内CSVのファイル名と、各ファイルの開始/終了 datetime を取得し、
    新規CSV(step3_orbit_index.csv)として保存する。
    """
    if not in_dir.exists():
        raise FileNotFoundError(f"Input folder not found: {in_dir}")

    files = sorted([p for p in in_dir.glob(pattern) if p.is_file()])
    rows: list[OrbitIndexRow] = []

    for p in files:
        start_dt, end_dt = _read_first_last_datetime(p)
        if start_dt is None or end_dt is None:
            continue
        row = OrbitIndexRow(
            orbit_file=p.name,
            orbit_start_time=str(start_dt),
            orbit_end_time=str(end_dt),
        )
        rows.append(row)
        if print_lines:
            print(f"{row.orbit_file}, {row.orbit_start_time}, {row.orbit_end_time}")

    if out_csv is None:
        out_csv = in_dir / "step3_orbit_index.csv"

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["orbit_file", "orbit_start_time", "orbit_end_time"])
        for r in rows:
            writer.writerow([r.orbit_file, r.orbit_start_time, r.orbit_end_time])

    return rows


