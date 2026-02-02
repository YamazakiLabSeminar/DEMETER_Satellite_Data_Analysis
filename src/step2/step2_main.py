from __future__ import annotations

import argparse
from pathlib import Path


def build_argparser() -> argparse.ArgumentParser:
    """
    コマンドライン引数を定義する。
    例：
      python -m src.step2.main
      python -m src.step2.main --config step2_normalization.yaml
      python -m src.step2.main --config C:\\path\\to\\step2_normalization.yaml
    """
    p = argparse.ArgumentParser(description="Step2: Kp merge + binning + CDF normalization")
    p.add_argument(
        "--config",
        type=str,
        default="step2_normalization.yaml",
        help="Step2 YAML config path. If relative, it is resolved under CONFIG_DIR.",
    )
    return p


def resolve_config_path(cfg_arg: str, config_dir: Path) -> Path:
    """
    --config で渡されたパスを確定する。
    - 絶対パスならそのまま
    - 相対パスなら CONFIG_DIR/相対パス として解釈する
    """
    p = Path(cfg_arg)
    if p.is_absolute():
        return p
    return config_dir / p


def main() -> None:
    # ===== 0) 引数を読む =====
    args = build_argparser().parse_args()

    # ===== 1) SSD入出力パス（paths.py）を使う =====
    # ※ここが今回の修正ポイント：inputs/outputs ではなく paths.py をソースオブトゥルースにする
    from ..paths import (
        CONFIG_DIR,
        EXTERNAL_DIR,
        INTERIM_DIR,
        TABLES_DIR,
        LOGS_DIR,
        PROJECT_ROOT,
        ensure_dirs,
    )

    # 必要フォルダを作る（存在してもOK）
    ensure_dirs()

    # ===== 2) config を読む（CONFIG_DIR 配下がデフォルト） =====
    from ..config_loader import load_yaml_config

    cfg_path = resolve_config_path(args.config, CONFIG_DIR)
    cfg = load_yaml_config(cfg_path)

    # ===== 3) logger を作る（ログもSSD2へ） =====
    from ..logger_setup import setup_logger

    logger = setup_logger(LOGS_DIR, name="step2")

    # ===== 4) Step2 の入出力パスを config から決める =====
    # Step1出力（入力扱い）は SSD2 の INTERIM_DIR にある想定
    step1_dirname = cfg.get("io", {}).get("step1_dirname", "step1_extracted")
    step2_dirname = cfg.get("io", {}).get("step2_dirname", "step2_normalized")

    step1_dir = INTERIM_DIR / step1_dirname
    step2_out_dir = INTERIM_DIR / step2_dirname

    # Kp は SSD1 側（EXTERNAL_DIR）にある想定
    kp_csv_filename = cfg.get("kp", {}).get("csv_filename", "kpデータ_ALL(csv).csv")
    kp_csv_path = EXTERNAL_DIR / kp_csv_filename

    # ===== 5) 実行情報をログに残す（トラブル時に超重要） =====
    logger.info("=== Step2 start ===")
    logger.info(f"PROJECT_ROOT = {PROJECT_ROOT}")
    logger.info(f"CONFIG_DIR   = {CONFIG_DIR}")
    logger.info(f"config_path  = {cfg_path}")

    logger.info(f"EXTERNAL_DIR = {EXTERNAL_DIR}")
    logger.info(f"INTERIM_DIR  = {INTERIM_DIR}")
    logger.info(f"TABLES_DIR   = {TABLES_DIR}")
    logger.info(f"LOGS_DIR     = {LOGS_DIR}")

    logger.info(f"step1_dir    = {step1_dir}")
    logger.info(f"kp_csv_path  = {kp_csv_path}")
    logger.info(f"step2_outdir = {step2_out_dir}")

    # ===== 6) Step2 本体を呼ぶ =====
    from .step2_normalization import run_step2

    run_step2(
        step1_dir=step1_dir,
        kp_csv_path=kp_csv_path,
        out_dir=step2_out_dir,
        tables_dir=TABLES_DIR,
        cfg=cfg,
        logger=logger,
    )

    logger.info("=== Step2 end ===")


if __name__ == "__main__":
    main()
