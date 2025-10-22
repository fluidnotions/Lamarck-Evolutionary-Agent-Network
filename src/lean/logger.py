"""
Centralized logging configuration for LEAN using loguru.

Provides clean, structured logging with daily rotation and easy-to-read format.
"""

import sys
from pathlib import Path
from datetime import datetime
from loguru import logger


def setup_logger(log_dir: str = "./logs") -> None:
    """Configure loguru logger with daily rotation and clean format.

    Args:
        log_dir: Directory for log files

    Features:
        - Single log file per day with all levels
        - Automatic rotation at midnight
        - Clean format showing module, level, and message
        - Colored output for terminal
    """
    # Remove default handler
    logger.remove()

    # Create log directory
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)

    # Get today's date for log filename
    today = datetime.now().strftime('%Y-%m-%d')
    log_file = log_path / f"lean_{today}.log"

    # Add file handler with daily rotation
    logger.add(
        log_file,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
        level="DEBUG",
        rotation="00:00",  # Rotate at midnight
        retention="30 days",  # Keep logs for 30 days
        compression="zip",  # Compress old logs
        enqueue=True,  # Thread-safe
    )

    # Add console handler for warnings and above (optional)
    # Uncomment if you want console output:
    # logger.add(
    #     sys.stderr,
    #     format="<level>{level: <8}</level> | <cyan>{name}:{function}</cyan> - {message}",
    #     level="WARNING",
    #     colorize=True,
    # )

    logger.info(f"Logger initialized - logging to {log_file}")


def get_logger(name: str):
    """Get logger instance with context.

    Args:
        name: Module name (typically __name__)

    Returns:
        Logger bound to the module name for context
    """
    return logger.bind(name=name)


# Initialize logger on import
setup_logger()
