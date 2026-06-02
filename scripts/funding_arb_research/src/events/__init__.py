"""Event-driven backtest plumbing.

Three triggers — Bitget extreme funding, new-listing spike, cross-DEX
dispersion — plug into one engine. The trigger emits a ``FundingEvent``
when its condition fires; a hedge router maps the event to a venue
pair; the position manager opens, holds, and closes the position with
honest costs and funding settlement.
"""
