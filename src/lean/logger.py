"""
Centralized logging configuration for LEAN.

Provides structured logging with file output and optional console output.
"""

import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional


def setup_logger(
    name: str = "lean",
    log_dir: str = "./logs",
    log_level: int = logging.INFO,
    console_output: bool = False
) -> logging.Logger:
    """Configure and return a logger instance.

    Args:
        name: Logger name (typically module name)
        log_dir: Directory for log files
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        console_output: Whether to also output to console

    Returns:
        Configured logger instance
    """
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(log_level)

    # Avoid duplicate handlers if logger already exists
    if logger.handlers:
        return logger

    # Create log directory
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)

    # Create formatters
    detailed_formatter = logging.Formatter(
        fmt='%(asctime)s | %(name)-15s | %(levelname)-8s | %(funcName)-20s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    simple_formatter = logging.Formatter(
        fmt='%(asctime)s | %(levelname)-8s | %(message)s',
        datefmt='%H:%M:%S'
    )

    # File handler - detailed logs
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = log_path / f"lean_{timestamp}.log"
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(log_level)
    file_handler.setFormatter(detailed_formatter)
    logger.addHandler(file_handler)

    # File handler - errors only
    error_file = log_path / f"lean_errors_{timestamp}.log"
    error_handler = logging.FileHandler(error_file, encoding='utf-8')
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(detailed_formatter)
    logger.addHandler(error_handler)

    # Console handler (optional)
    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.WARNING)  # Only warnings and above to console
        console_handler.setFormatter(simple_formatter)
        logger.addHandler(console_handler)

    # Log initialization
    logger.info(f"Logger initialized: {name}")
    logger.info(f"Log file: {log_file}")
    logger.info(f"Error log: {error_file}")

    return logger


def get_logger(name: str) -> logging.Logger:
    """Get an existing logger or create a new one with default settings.

    Args:
        name: Logger name (typically __name__)

    Returns:
        Logger instance
    """
    logger = logging.getLogger(name)

    # If logger has no handlers, set it up with defaults
    if not logger.handlers:
        return setup_logger(name)

    return logger
