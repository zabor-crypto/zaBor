"""Shared Binance FAPI REST utilities: endpoint rotation and 403 handling."""

from __future__ import annotations

import itertools
import threading
import time
import urllib.error
import urllib.request
from typing import Any

# fapi1/2/3 return 302 redirects from Contabo/EU IPs as of May 2026 — use only the main endpoint.
_BASES = [
    "https://fapi.binance.com",
]
_cycle = itertools.cycle(_BASES)
_lock = threading.Lock()


def next_base() -> str:
    """Return the next FAPI base URL in round-robin order."""
    with _lock:
        return next(_cycle)


def safe_urlopen(req: urllib.request.Request, timeout: float) -> Any:
    """Open a URL, sleeping 60 s on 403 before re-raising.

    Sleeping instead of immediately retrying avoids making a ban worse
    by flooding Binance with requests right after the IP is flagged.
    """
    try:
        return urllib.request.urlopen(req, timeout=timeout)
    except urllib.error.HTTPError as exc:
        if exc.code == 403:
            time.sleep(60)
        raise
