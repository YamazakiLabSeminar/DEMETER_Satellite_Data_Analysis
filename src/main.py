from __future__ import annotations

from paths import RAW_DIR, LOGS_DIR, TABLES_DIR, ensure_dirs
from logger_setup import setup_logger
from io_utils import iter_csv_files, load_checkpoint, append_checkpoint, safe_run_one
from process_one_file import process_one_csv


def main():
    ensure_dirs()
    logger = setup_logger(LOGS_DIR)

    logger.info("=== Step1 start ===")
    logger.info(f"RAW_DIR = {RAW_DIR}")

    csv_files = iter_csv_files(RAW_DIR)
    logger.info(f"Found {len(csv_files)} csv files.")

    # Step1用のチェックポイント（Step0.5とファイル名を分ける）
    checkpoint_path = TABLES_DIR / "checkpoint_step1_done.txt"
    done = load_checkpoint(checkpoint_path)
    logger.info(f"Already done (checkpoint): {len(done)} files.")

    ok = 0
    ng = 0
    skipped = 0
    total = len(csv_files)

    for i, csv_path in enumerate(csv_files, start=1):
        name = csv_path.name

        if name in done:
            skipped += 1
            continue

        logger.info(f"[{i}/{total}] Step1 processing: {name}")

        # safe_run_one が例外を捕まえてログに残し、止めずに次へ進む
        success = safe_run_one(logger, process_one_csv, csv_path, logger)

        if success:
            ok += 1
            append_checkpoint(checkpoint_path, name)
            done.add(name)
        else:
            ng += 1

        # ログが多すぎないように100件ごとにまとめ表示
        if (ok + ng) % 100 == 0:
            logger.info(f"Progress: success={ok}, failed={ng}, skipped={skipped}")

    logger.info(f"=== Step1 done === success={ok}, failed={ng}, skipped={skipped}")


if __name__ == "__main__":
    main()
