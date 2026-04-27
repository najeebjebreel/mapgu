"""Centralised logging setup for MAPGU."""

from __future__ import annotations

import logging
import sys

from tqdm import tqdm


class _TqdmHandler(logging.StreamHandler):
    """Logging handler that routes through tqdm.write to avoid clobbering progress bars."""

    def emit(self, record: logging.LogRecord) -> None:
        try:
            tqdm.write(self.format(record), file=sys.stdout)
            self.flush()
        except Exception:
            self.handleError(record)


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Return (or create) a named logger with a single stdout handler.

    Calling this multiple times with the same *name* is idempotent — handlers
    are only attached once.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = _TqdmHandler(sys.stdout)
        handler.setFormatter(
            logging.Formatter("[%(levelname)s][%(name)s] %(message)s")
        )
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger
