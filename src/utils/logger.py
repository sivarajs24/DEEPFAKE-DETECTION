"""
DeepGuard-X Core Utilities
Logger configuration with rich formatting and file output
"""

import logging
import sys
from pathlib import Path
from typing import Optional
from loguru import logger
from rich.console import Console
from rich.logging import RichHandler

console = Console()


def setup_logger(
    name: str = "deepguard-x",
    log_file: Optional[str] = None,
    log_level: str = "INFO",
    rich_output: bool = True
) -> logging.Logger:
    """
    Set up a sophisticated logger with rich formatting
    
    Args:
        name: Logger name
        log_file: Path to log file (optional)
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        rich_output: Whether to use rich formatting for console output
        
    Returns:
        Configured logger instance
    """
    # Remove default loguru handlers
    logger.remove()
    
    # Console handler with rich formatting
    if rich_output:
        logger.add(
            sys.stderr,
            format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
            level=log_level,
            colorize=True,
        )
    else:
        logger.add(
            sys.stderr,
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
            level=log_level,
        )
    
    # File handler if specified
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        logger.add(
            log_file,
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
            level=log_level,
            rotation="100 MB",  # Rotate after 100MB
            retention="30 days",  # Keep logs for 30 days
            compression="zip",  # Compress rotated logs
        )
    
    return logger


def get_logger(name: str = "deepguard-x") -> logging.Logger:
    """
    Get a logger instance
    
    Args:
        name: Logger name
        
    Returns:
        Logger instance
    """
    return logger.bind(name=name)


# Global logger instance
log = setup_logger()
