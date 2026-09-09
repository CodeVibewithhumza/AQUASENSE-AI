"""Centralized logging configuration for AquaSense AI."""
import logging
import sys
from pathlib import Path
from src.utils.config import BASE_DIR

LOG_DIR = BASE_DIR / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOG_DIR / "aquasense.log"


def get_logger(name: str = "AquaSense") -> logging.Logger:
    """Configures and returns a structured logger instance."""
    logger_instance = logging.getLogger(name)

    if not logger_instance.handlers:
        logger_instance.setLevel(logging.INFO)

        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)-8s | %(name)s:%(funcName)s:%(lineno)d - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )

        # Console Handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        logger_instance.addHandler(console_handler)

        # File Handler
        file_handler = logging.FileHandler(LOG_FILE, encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logger_instance.addHandler(file_handler)

        logger_instance.propagate = False

    return logger_instance


logger = get_logger("AquaSenseAI")
