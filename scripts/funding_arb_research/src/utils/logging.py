"""Centralized logging. JSON-friendly format, deterministic timestamps."""
from __future__ import annotations

import logging
import sys
from logging import Logger
from pathlib import Path
from typing import Optional

_DEFAULT_FMT = "%(asctime)s | %(levelname)-7s | %(name)-28s | %(message)s"


def get_logger(
    name: str,
    level: int = logging.INFO,
    log_file: Optional[Path] = None,
) -> Logger:
    """Return a logger that emits to stdout and optionally to a file.

    Logger names are namespaced (e.g. ``funding_arb.collectors.binance``)
    so log filtering remains simple.
    """
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(level)
    formatter = logging.Formatter(_DEFAULT_FMT, datefmt="%Y-%m-%dT%H:%M:%S%z")

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    logger.propagate = False
    return logger


def attach_run_logfile(logger: Logger, log_file: Path) -> None:
    """Attach an additional file handler to an existing logger (per-run logs)."""
    log_file.parent.mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setFormatter(logging.Formatter(_DEFAULT_FMT, datefmt="%Y-%m-%dT%H:%M:%S%z"))
    logger.addHandler(fh)
