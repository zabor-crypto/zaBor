"""Per-venue async WebSocket clients.

Each client subscribes to its venue's public streams, normalizes
messages into the canonical recorder rows, and hands them to a writer.
Network IO and venue-specific protocol live here; nothing else in the
recorder package speaks the venue dialect.
"""
