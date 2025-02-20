# filepath: logging_config.py
import logging
import logging.handlers
from pathlib import Path
import sys


def setup_logging():
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # Create logs directory
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    console_handler.setLevel(logging.INFO)
    logger.addHandler(console_handler)

    # File handler for general logs
    file_handler = logging.handlers.RotatingFileHandler(
        "logs/app.log", maxBytes=10485760, backupCount=5, encoding="utf-8"
    )
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.INFO)
    logger.addHandler(file_handler)

    # File handler for error logs
    error_file_handler = logging.handlers.RotatingFileHandler(
        "logs/error.log", maxBytes=10485760, backupCount=5, encoding="utf-8"
    )
    error_file_handler.setFormatter(formatter)
    error_file_handler.setLevel(logging.ERROR)
    logger.addHandler(error_file_handler)

    return logger


logger = setup_logging()