from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path


def setup_logger(log_dir: Path, name: str = "demeter") -> logging.Logger:
    """
    log_dir 配下にログファイルを作り、コンソール(画面)にも表示するロガーを返す。
    """
    log_dir.mkdir(parents=True, exist_ok=True)

    # 例: run_20260126_153012.log みたいなファイル名になる
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = log_dir / f"run_{timestamp}.log"

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # 二重に同じログが出ないようにする（再実行時に重要）
    if logger.handlers:
        return logger

    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # 1) ログファイルへ出力する設定
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(fmt)

    # 2) 画面（ターミナル）へ出力する設定
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(fmt)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    logger.info("Logger initialized")
    logger.info(f"log file: {log_path}")

    return logger
