from __future__ import annotations

import argparse
from pathlib import Path


def build_argparser() -> argparse.ArgumentParser:
    """
    コマンドライン引数を定義する関数。
    例：
      python -m src.step3.main
      python -m src.step3.main --eq_catalog all_eq_declustring_30day_30km.csv
      python -m src.step3.main --step2_dir step2_normalized_test
    """
    p = argparse.ArgumentParser(description="Step3-1: Earthquake-orbit mapping (index + orbit_map)")
    p.add_argument(
        "--eq_catalog",
        type=str,
        default="earthquake_catalog\eq_m4.8above_depth40kmbelow_2004-2010_declustred.csv",
        help="Earthquake catalog path relative to EXTERNAL_DIR, or absolute path.",
    )
    p.add_argument(
        "--step2_dir",
        type=str,
        default="step2_normalized",
        help="Step2 output directory name under INTERIM_DIR, or absolute path.",
    )
    return p


def resolve_path(arg: str, base_dir: Path) -> Path:
    """
    引数で渡されたパスを確定する。
    - 絶対パスならそのまま使う
    - 相対パスなら base_dir / arg として解釈する
    """
    p = Path(arg)
    if p.is_absolute():
        return p
    return base_dir / p


def main() -> None:
    # ===== 0) 引数を読む =====
    args = build_argparser().parse_args()

    # ===== 1) paths.py からSSD上の標準パスを読む =====
    # EXTERNAL_DIR: F:\external
    # INTERIM_DIR : E:\interim
    # TABLES_DIR  : E:\tables
    # LOGS_DIR    : E:\logs
    from ..paths import EXTERNAL_DIR, INTERIM_DIR, TABLES_DIR, LOGS_DIR, ensure_dirs

    # 必要フォルダを作る（無ければ作成、既にあれば何もしない）
    ensure_dirs()

    # ===== 2) loggerを作る =====
    from ..logger_setup import setup_logger
    logger = setup_logger(LOGS_DIR, name="step3")

    # ===== 3) 入力パスを決める =====
    # 地震カタログ： EXTERNAL_DIR を基準に解釈
    eq_catalog_path = resolve_path(args.eq_catalog, INTERIM_DIR)

    # Step2出力： INTERIM_DIR を基準に解釈
    step2_dir = resolve_path(args.step2_dir, INTERIM_DIR)

    # ===== 4) 実行情報をログに出す =====
    logger.info("=== Step3-1 start: orbit mapping ===")
    logger.info(f"eq_catalog_path = {eq_catalog_path}")
    logger.info(f"step2_dir       = {step2_dir}")
    logger.info(f"tables_dir      = {TABLES_DIR}")
    logger.info(f"logs_dir        = {LOGS_DIR}")

    # ===== 5) Step3-1 本体を呼ぶ =====
    from .step3_orbit_mapping_new import run_step3_orbit_mapping_new

    run_step3_orbit_mapping_new(
        eq_catalog_path=eq_catalog_path,
        step2_dir=step2_dir,
        tables_dir=TABLES_DIR,
        cfg=None,          # ここは今はデフォルト設定でOK
        logger=logger,
    )

    logger.info("=== Step3-1 end ===")


if __name__ == "__main__":
    main()
