from paths import LOGS_DIR, ensure_dirs
from logger_setup import setup_logger


def main():
    ensure_dirs()
    logger = setup_logger(LOGS_DIR)

    logger.info("Start analysis (step0)")
    logger.info("This is a test log message.")
    logger.warning("This is a warning example (not an error).")
    logger.info("Finish step0")


if __name__ == "__main__":
    main()
