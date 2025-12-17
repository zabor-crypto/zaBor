from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BitgetCredentials:
    key: str
    secret: str
    password: str


class BitgetClient:
    """
    Placeholder for a real Bitget client implementation.

    Rules:
    - Do not log secrets
    - Add retries + rate limit handling
    - Add idempotency for order placement
    - Add request signing and strict response validation
    """

    def __init__(self, creds: BitgetCredentials) -> None:
        self._creds = creds

    # Implement:
    # - get_market_rules(symbol)
    # - get_ticker(symbol)
    # - place_order(...)
    # - get_order(...)
    # - cancel_order(...)
