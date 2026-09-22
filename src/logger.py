"""Centralized logging configuration for the Music Mood Predictor.

Provides :func:`get_logger`, which returns a logger that writes to both the
console and a rotating-free ``logs/app.log`` file. Every record carries a
timestamp and severity level (Requirements 12.2, 12.3). Repeated calls for the
same logger name reuse the existing handlers instead of attaching duplicates.
"""

from __future__ import annotations

import logging
import os

# Log line format: ISO-like timestamp | severity | logger name | message.
LOG_FORMAT: str = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"

# Directory and file where logs are persisted.
LOG_DIR: str = "logs"
LOG_FILE: str = os.path.join(LOG_DIR, "app.log")

# Default logging level for configured loggers.
DEFAULT_LEVEL: int = logging.INFO


def get_logger(name: str, level: int = DEFAULT_LEVEL) -> logging.Logger:
    """Return a configured logger with console and file handlers.

    The logger emits records formatted with a timestamp and severity level to
    both stdout and ``logs/app.log``. Calling this function multiple times with
    the same ``name`` will not attach duplicate handlers.

    Args:
        name: Logger name, typically ``__name__`` of the calling module.
        level: Logging level for the logger. Defaults to ``INFO``.

    Returns:
        A configured :class:`logging.Logger` instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Avoid propagating to the root logger, which could cause duplicate output.
    logger.propagate = False

    # Only configure handlers once per logger name to prevent duplicates on
    # repeated calls.
    if logger.handlers:
        return logger

    formatter = logging.Formatter(LOG_FORMAT)

    # Console handler.
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler. Ensure the log directory exists before creating the file
    # handler; failures here should not prevent console logging from working.
    try:
        os.makedirs(LOG_DIR, exist_ok=True)
        file_handler = logging.FileHandler(LOG_FILE, encoding="utf-8")
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    except OSError:
        # If the file handler cannot be created (e.g., permissions), continue
        # with console logging only rather than crashing the caller.
        logger.warning("Could not create file handler at %s", LOG_FILE)

    return logger
