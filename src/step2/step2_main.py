from __future__ import annotations

import argparse
from pathlib import Path


def build_argparser() -> argparse.ArgumentParser:
    """
    コマンドライン引数（オプション）を定義する。
    例：
      python -m src.step2.main
      python -m src.step2.main --config configs/step2_normalization.yaml
    """
    p = argparse.ArgumentParser(description="Step2: Kp merge + binning + CDF normalization")
    p.add_argument(
        "--config",
        type=str,
        default="configs/step2_normalization.yaml",
        help="Path to Step2 YAML config (relative to project root allowed).",
    )
    return p


def main() -> None:
    # ===== 0) 引数を読む =====
    args = build_argparser().parse_args()

    # ===== 1) プロジェクトルート（卒研解析/）を決める =====
    # このファイルは src/step2/main.py にある。
    # main.py → step2 → src → (プロジェクトルート) なので parents[2]。
    project_root = Path(__file__).resolve().parents[2]

    # ===== 2) あなたのフォルダ構成に合わせて主要パスを組む =====
    configs_dir = project_root / "configs"
    inputs_dir = project_root / "inputs"
    outputs_dir = project_root / "outputs"

    external_dir = inputs_dir / "external"
    interim_dir = inputs_dir / "interim"

    logs_dir = outputs_dir / "logs"
    tables_dir = outputs_dir / "tables"

    # ===== 3) configファイルのパスを確定 =====
    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        cfg_path = project_root / cfg_path

    # ===== 4) YAMLを読む（上流：config_loader） =====
    # ※あなたが持っている src/config_loader_setup.py を使う
    from ..config_loader_setup import load_yaml_config
    cfg = load_yaml_config(cfg_path)

    # ===== 5) loggerを作る（上流：logger_setup） =====
    from ..logger_setup import setup_logger
    logger = setup_logger(logs_dir, name="step2")

    # ===== 6) Step2入出力パスを config から決める =====
    # Step1出力は inputs/interim/ にある前提
    step1_dirname = cfg.get("io", {}).get("step1_dirname", "step1_extracted")
    step2_dirname = cfg.get("io", {}).get("step2_dirname", "step2_normalized")

    step1_dir = interim_dir / step1_dirname
    step2_out_dir = interim_dir / step2_dirname

    kp_csv_filename = cfg.get("kp", {}).get("csv_filename", "kpデータ_ALL(csv).csv")
    kp_csv_path = external_dir / kp_csv_filename

    # ===== 7) 実行情報をログに残す =====
    logger.info("=== Step2 start ===")
    logger.info(f"project_root = {project_root}")
    logger.info(f"config_path  = {cfg_path}")
    logger.info(f"step1_dir    = {step1_dir}")
    logger.info(f"kp_csv_path  = {kp_csv_path}")
    logger.info(f"step2_outdir = {step2_out_dir}")
    logger.info(f"tables_dir   = {tables_dir}")
    logger.info(f"logs_dir     = {logs_dir}")

    # ===== 8) Step2本体を呼ぶ =====
    from .step2_normalization import run_step2

    run_step2(
        step1_dir=step1_dir,
        kp_csv_path=kp_csv_path,
        out_dir=step2_out_dir,
        tables_dir=tables_dir,
        cfg=cfg,
        logger=logger,
    )

    logger.info("=== Step2 end ===")


if __name__ == "__main__":
    main()
