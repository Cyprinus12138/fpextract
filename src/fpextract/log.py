"""Logging: clean stdout plus an append-mode file with ISO timestamps.

Each run writes a delimiter line to the file so consecutive runs can be
told apart when auditing a long-lived log.
"""
from __future__ import annotations

import datetime as dt
import logging
from pathlib import Path


def setup_logger(name: str, log_path: Path | None = None,
                 level: int = logging.INFO) -> logging.Logger:
    """Return a logger with a plain stdout handler and, when ``log_path`` is
    given, an append-mode file handler. Idempotent per ``name``."""
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(level)
    logger.propagate = False

    stdout_h = logging.StreamHandler()
    stdout_h.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(stdout_h)

    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_h = logging.FileHandler(log_path, mode="a", encoding="utf-8")
        file_h.setFormatter(logging.Formatter(
            fmt="%(asctime)s %(levelname)-7s %(message)s",
            datefmt="%Y-%m-%dT%H:%M:%S",
        ))
        logger.addHandler(file_h)
        started = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
        file_h.stream.write(f"\n==== {name} run started {started} ====\n")
        file_h.stream.flush()
    return logger


def log_run_finished(logger: logging.Logger) -> None:
    """Write the matching end-of-run delimiter to the file handler(s)."""
    finished = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
    for h in logger.handlers:
        if isinstance(h, logging.FileHandler):
            h.stream.write(f"==== {logger.name} run finished {finished} ====\n")
            h.stream.flush()
