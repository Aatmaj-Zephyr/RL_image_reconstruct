"""Setup logger for the project. No need to modify this file."""
import sys
from loguru import logger

from helpers.config import config

log = logger.opt(colors=True)

def setup_logger() -> None:
    """
    Configure logger using explicit config object.
    """
    assert config.runtime.RUN_ID is not None, "RUN_ID must be set in runtime before setting up logger."

    logger.remove()

    LOG_FORMAT = (
        f"[{config.runtime.RUN_ID}] "
        "<cyan>{time:YYYY-MM-DD HH:mm:ss}</cyan> | "
        "<level>{level}</level> | "
        "<cyan>{name}</cyan>:<cyan>{line}</cyan> - "
        "{message}"
    )

    level = "DEBUG" if config.IS_DEBUG else "INFO"

    # Console handler
    logger.add(
        sys.stdout,
        level=level,
        format=LOG_FORMAT,
        colorize=True,
        
    )

    # File handler
    logger.add(
        config.DEBUG_LOG_FILE_PATH,
        level=level,
        format=LOG_FORMAT,
        rotation="10 MB",
        colorize=False
    )
    
