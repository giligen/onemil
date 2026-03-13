"""
Logging configuration for OneMil scanner.

Sets up logging with:
- Console output with colored formatting (UTF-8 on Windows)
- File logging with rotation
"""

import logging
import sys
from pathlib import Path
from typing import Optional
from logging.handlers import RotatingFileHandler


class LogColors:
    """ANSI color codes for log levels."""
    RESET = "\033[0m"
    DEBUG = "\033[36m"     # Cyan
    INFO = "\033[32m"      # Green
    WARNING = "\033[33m"   # Yellow
    ERROR = "\033[31m"     # Red
    CRITICAL = "\033[35m"  # Magenta


class ColoredFormatter(logging.Formatter):
    """Custom formatter that adds colors to log output."""

    LEVEL_COLORS = {
        logging.DEBUG: LogColors.DEBUG,
        logging.INFO: LogColors.INFO,
        logging.WARNING: LogColors.WARNING,
        logging.ERROR: LogColors.ERROR,
        logging.CRITICAL: LogColors.CRITICAL,
    }

    def format(self, record: logging.LogRecord) -> str:
        """Format log record with colors for terminal output."""
        color = self.LEVEL_COLORS.get(record.levelno, LogColors.RESET)
        formatted = super().format(record)
        return f"{color}{formatted}{LogColors.RESET}"


def setup_logging(
    log_level: str = "INFO",
    log_dir: Optional[str] = None,
    log_file: str = "onemil.log",
    verbose: bool = False
) -> logging.Logger:
    """
    Set up logging for the application.

    Args:
        log_level: Logging level string (DEBUG, INFO, WARNING, ERROR)
        log_dir: Directory for log files (defaults to ./logs)
        log_file: Name of log file
        verbose: If True, override log_level to DEBUG

    Returns:
        Root logger configured with handlers
    """
    if verbose:
        log_level = "DEBUG"

    level = getattr(logging, log_level.upper(), logging.INFO)

    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    root_logger.handlers = []

    detailed_format = "%(asctime)s | %(levelname)-8s | %(name)s:%(lineno)d | %(message)s"
    date_format = "%Y-%m-%d %H:%M:%S"

    # Console handler (colored, UTF-8 for Windows emoji support)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.stream = open(sys.stdout.fileno(), mode='w', encoding='utf-8', closefd=False)
    console_formatter = ColoredFormatter(detailed_format, datefmt=date_format)
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)

    # File handler (rotating, UTF-8)
    if log_dir is None:
        log_dir = Path(__file__).parent.parent / "logs"
    else:
        log_dir = Path(log_dir)

    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / log_file

    file_handler = RotatingFileHandler(
        log_path,
        maxBytes=10 * 1024 * 1024,  # 10 MB
        backupCount=5,
        encoding='utf-8'
    )
    file_handler.setLevel(level)
    file_formatter = logging.Formatter(detailed_format, datefmt=date_format)
    file_handler.setFormatter(file_formatter)
    root_logger.addHandler(file_handler)

    # Quiet noisy third-party loggers
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("yfinance").setLevel(logging.WARNING)
    logging.getLogger("peewee").setLevel(logging.WARNING)

    logging.info(f"Logging initialized at {log_level.upper()} level, file: {log_path}")

    return root_logger


def get_logger(name: str) -> logging.Logger:
    """Get a logger with the specified name."""
    return logging.getLogger(name)
