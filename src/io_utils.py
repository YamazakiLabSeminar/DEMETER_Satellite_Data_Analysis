from __future__ import annotations

from pathlib import Path
from typing import Iterable
import logging
import traceback


def iter_csv_files(folder: Path) -> list[Path]:
    """
    folder直下のCSVをファイル名昇順で返す。
    （膨大な数でも、まずは一覧を固定したいので list で返す）
    """
    return sorted(folder.glob("*.csv"))


def load_checkpoint(path: Path) -> set[str]:
    """
    既に処理済みのファイル名一覧を読み込む。
    （チェックポイントが無ければ空集合）
    """
    if not path.exists():
        return set()

    done = set()
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            name = line.strip()
            if name:
                done.add(name)
    return done


def append_checkpoint(path: Path, filename: str) -> None:
    """
    1件処理できたら、そのファイル名をチェックポイントに追記する。
    """
    with path.open("a", encoding="utf-8") as f:
        f.write(filename + "\n")


def safe_run_one(logger: logging.Logger, func, *args, **kwargs) -> bool:
    """
    関数を安全に実行。失敗しても止めず、例外をログに残す。
    """
    try:
        func(*args, **kwargs)
        return True
    except Exception as e:
        logger.error("Exception occurred: %s", e)
        logger.error(traceback.format_exc())
        return False
