"""Self-hosted WebSocket recorder.

Long-lived async process that subscribes to public market-data streams
on each configured venue and writes them to compressed parquet,
partitioned by ``date / venue / coin / channel``.

The recorder is the realism boundary for backtests: trades whose
timestamps fall inside the recorder coverage window can use real depth
and trade tape; outside the window, the cost model falls back to
configured bps. Each trade record carries ``slippage_source`` so a
report can show which trades were realism-grade.
"""
