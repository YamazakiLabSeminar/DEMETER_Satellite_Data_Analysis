from __future__ import annotations

from pathlib import Path
import logging

from src.paths import INTERIM_DIR, TABLES_DIR
from src.step1.step1_demeter import step1_process_one_file


def process_one_csv(csv_path: Path, logger: logging.Logger) -> None:
    """
    Step1: 1ファイル処理の呼び出し口。
    """
    out_dir = INTERIM_DIR / "step1_extracted"              # 出力フォルダ
    summary_path = TABLES_DIR / "step1_summary.csv"        # サマリ（追記）

    # Step1処理（例外は呼び出し側の safe_run_one がログに落として止めずに進む）
    step1_process_one_file(
        csv_path=csv_path,
        out_dir=out_dir,
        summary_path=summary_path,
        logger=logger,
        f_low=1621.09375,
        f_high=1718.75,
    )
