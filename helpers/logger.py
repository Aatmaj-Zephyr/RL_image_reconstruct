"""Setup logger for the project. No need to modify this file."""
import sys
from loguru import logger

from helpers.config import config

log = logger.opt(colors=True)


def _log_format(run_id: str) -> str:
    return (
        f"[{run_id}] "
        "<cyan>{time:YYYY-MM-DD HH:mm:ss}</cyan> | "
        "<level>{level}</level> | "
        "<cyan>{name}</cyan>:<cyan>{line}</cyan> - "
        "{message}"
    )


def _configure_logger(level: str, run_id: str) -> None:
    logger.remove()

    # Console handler
    logger.add(
        sys.stdout,
        level=level,
        format=_log_format(run_id),
        colorize=True,
    )

    # File handler
    logger.add(
        config.DEBUG_LOG_FILE_PATH,
        level=level,
        format=_log_format(run_id),
        rotation="10 MB",
        colorize=False,
    )


def setup_logger() -> None:
    """
    Configure logger using explicit config object.
    """
    assert config.runtime.RUN_ID is not None, "RUN_ID must be set in runtime before setting up logger."
    level = "DEBUG" if config.IS_DEBUG else "INFO"

    _configure_logger(level=level, run_id=config.runtime.RUN_ID)


def setup_worker_logger(run_id: str, level: str = "INFO") -> None:
    """Configure logger for worker processes spawned outside main startup path."""
    _configure_logger(level=level, run_id=run_id)
