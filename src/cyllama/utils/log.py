"""
Logging configuration module with colored output.

Uses the color.py module for terminal colors.
"""
import datetime
import logging
import os
import sys

from .color import (
    white, grey, green, cyan, yellow, red, bold,
    use_color, END
)

# ----------------------------------------------------------------------------
# env helpers


def getenv(key: str, default: bool = False) -> bool:
    """Convert '0','1' env values to bool {True, False}"""
    return bool(int(os.getenv(key, default)))


# ----------------------------------------------------------------------------
# constants

PY_VER_MINOR = sys.version_info.minor
DEBUG = getenv("DEBUG", default=True)
COLOR = getenv("COLOR", default=True)

# ----------------------------------------------------------------------------
# logging config


class CustomFormatter(logging.Formatter):
    """Custom logging formatting class using color.py for terminal colors."""

    fmt = "%(delta)s - %(levelname)s - %(name)s.%(funcName)s - %(message)s"

    def __init__(self, use_color_flag=COLOR):
        self.use_color_flag = use_color_flag
        self._build_formats()

    def _build_formats(self):
        """Build format strings for each log level."""
        if self.use_color_flag and use_color():
            # Build colored format strings
            base_fmt = "{delta} - {level} - {name} - {msg}"

            self.formats = {
                logging.DEBUG: base_fmt.format(
                    delta=white("%(delta)s"),
                    level=grey("%(levelname)s"),
                    name=white("%(name)s.%(funcName)s"),
                    msg=grey("%(message)s")
                ),
                logging.INFO: base_fmt.format(
                    delta=white("%(delta)s"),
                    level=green("%(levelname)s"),
                    name=white("%(name)s.%(funcName)s"),
                    msg=grey("%(message)s")
                ),
                logging.WARNING: base_fmt.format(
                    delta=white("%(delta)s"),
                    level=yellow("%(levelname)s"),
                    name=white("%(name)s.%(funcName)s"),
                    msg=grey("%(message)s")
                ),
                logging.ERROR: base_fmt.format(
                    delta=white("%(delta)s"),
                    level=red("%(levelname)s"),
                    name=white("%(name)s.%(funcName)s"),
                    msg=grey("%(message)s")
                ),
                logging.CRITICAL: base_fmt.format(
                    delta=white("%(delta)s"),
                    level=bold(red("%(levelname)s")),
                    name=white("%(name)s.%(funcName)s"),
                    msg=grey("%(message)s")
                ),
            }
        else:
            # No color - use plain format for all levels
            self.formats = {level: self.fmt for level in [
                logging.DEBUG, logging.INFO, logging.WARNING,
                logging.ERROR, logging.CRITICAL
            ]}

    def format(self, record):
        """Custom logger formatting method"""
        if PY_VER_MINOR > 10:
            duration = datetime.datetime.fromtimestamp(
                record.relativeCreated / 1000, datetime.UTC
            )
        else:
            duration = datetime.datetime.utcfromtimestamp(
                record.relativeCreated / 1000
            )
        record.delta = duration.strftime("%H:%M:%S")

        log_fmt = self.formats.get(record.levelno, self.fmt)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)


def config(name: str) -> logging.Logger:
    """Configure and return a logger with custom formatting.

    Args:
        name: Logger name (typically __name__)

    Returns:
        Configured logger instance
    """
    strm_handler = logging.StreamHandler()
    strm_handler.setFormatter(CustomFormatter())
    logging.basicConfig(
        level=logging.DEBUG if DEBUG else logging.INFO,
        handlers=[strm_handler],
    )
    return logging.getLogger(name)
