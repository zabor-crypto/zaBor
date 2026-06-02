"""Common HTTP plumbing: rate limiting, retries, schema-issue tracking.

Collectors are async; they share a single client and respect a per-venue rps.
"""
from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any, Optional

import httpx
from tenacity import (
    AsyncRetrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from ..normalization.schema import SchemaIssue
from ..utils.logging import get_logger


class RateLimiter:
    """Token-bucket-ish limiter; coarse but adequate for research collection."""

    def __init__(self, rps: float):
        self.min_interval = 1.0 / max(rps, 0.001)
        self._last = 0.0
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        async with self._lock:
            now = time.monotonic()
            wait = self._last + self.min_interval - now
            if wait > 0:
                await asyncio.sleep(wait)
            self._last = time.monotonic()


@dataclass
class CollectorContext:
    """Shared state and config for a collector run."""
    venue: str
    rest_base: str
    rate_limit_rps: float = 5.0
    timeout_s: float = 20.0
    max_retries: int = 5
    issues: list[SchemaIssue] = field(default_factory=list)
    client: Optional[httpx.AsyncClient] = None
    logger: Any = None

    def __post_init__(self):
        if self.logger is None:
            self.logger = get_logger(f"funding_arb.collectors.{self.venue}")
        self.limiter = RateLimiter(self.rate_limit_rps)

    async def __aenter__(self):
        self.client = httpx.AsyncClient(
            base_url=self.rest_base,
            timeout=self.timeout_s,
            headers={"User-Agent": "funding-arb-research/0.1"},
        )
        return self

    async def __aexit__(self, *exc):
        if self.client is not None:
            await self.client.aclose()
            self.client = None


class TransientHTTPError(Exception):
    pass


async def request_json(
    ctx: CollectorContext,
    method: str,
    path: str,
    *,
    params: Optional[dict] = None,
    json_body: Optional[dict] = None,
) -> Any:
    """HTTP request with rate limiting and exponential backoff retry.

    Retries on:
      - network errors,
      - HTTP 5xx,
      - HTTP 429 (with brief backoff).
    Does not retry 4xx (other than 429) — these indicate a programmer bug.
    """
    assert ctx.client is not None, "CollectorContext must be entered (async with)"

    async for attempt in AsyncRetrying(
        stop=stop_after_attempt(ctx.max_retries),
        wait=wait_exponential(multiplier=0.5, min=0.5, max=20.0),
        retry=retry_if_exception_type((TransientHTTPError, httpx.TransportError)),
        reraise=True,
    ):
        with attempt:
            await ctx.limiter.acquire()
            try:
                resp = await ctx.client.request(method, path, params=params, json=json_body)
            except httpx.TransportError as e:
                ctx.logger.warning("transport error %s %s: %s", method, path, e)
                raise

            if resp.status_code == 429 or resp.status_code >= 500:
                ctx.logger.warning(
                    "transient HTTP %d on %s %s -> backing off",
                    resp.status_code, method, path,
                )
                raise TransientHTTPError(f"{resp.status_code}")

            if resp.status_code >= 400:
                ctx.logger.error(
                    "non-retryable HTTP %d on %s %s: %s",
                    resp.status_code, method, path, resp.text[:200],
                )
                resp.raise_for_status()

            try:
                return resp.json()
            except ValueError as e:
                ctx.logger.error("non-JSON response on %s %s: %s", method, path, e)
                raise
    return None  # unreachable
