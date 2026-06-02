import time

from v1.contracts.event_types import MANDATORY_STREAMS, STREAM_FORCE_ORDER, RawMarketEvent
from v1.ingest.stream_supervisor import StreamSupervisor


def _event(stream: str, recv_mono_ns: int, parse_status: str = "ok") -> RawMarketEvent:
    return RawMarketEvent(
        event_id=f"e-{stream}",
        schema_version="v1",
        venue="binance_um_futures",
        stream=stream,
        symbol="BTCUSDT",
        conn_id="c",
        exchange_ts_ms=1,
        recv_wall_ts_ns=recv_mono_ns,
        recv_mono_ts_ns=recv_mono_ns,
        ingest_seq=1,
        stream_local_seq=1,
        payload_json="{}",
        payload_hash="h",
        parse_status=parse_status,
        gap_flag=False,
    )


def test_supervisor_reports_no_data_for_missing_streams() -> None:
    sup = StreamSupervisor(symbols=["BTCUSDT"], stale_after_ms=1000)
    snap = sup.health_snapshot()
    assert snap.healthy is False
    assert len(snap.unhealthy_reasons) == len(MANDATORY_STREAMS) - 1
    assert any(reason.endswith(":no_data") for reason in snap.unhealthy_reasons)
    assert not any(STREAM_FORCE_ORDER in reason for reason in snap.unhealthy_reasons)


def test_supervisor_reports_stale_after_timeout() -> None:
    sup = StreamSupervisor(symbols=["BTCUSDT"], stale_after_ms=5)
    now_ns = time.monotonic_ns()
    for stream in MANDATORY_STREAMS:
        sup.observe(_event(stream, now_ns))

    fresh = sup.health_snapshot()
    assert fresh.healthy is True

    time.sleep(0.01)
    stale = sup.health_snapshot()
    assert stale.healthy is False
    assert any(":stale_" in reason for reason in stale.unhealthy_reasons)
    assert not any(STREAM_FORCE_ORDER in reason for reason in stale.unhealthy_reasons)


def test_supervisor_counts_parse_errors() -> None:
    sup = StreamSupervisor(symbols=["BTCUSDT"], stale_after_ms=1000)
    now_ns = time.monotonic_ns()
    sup.observe(_event("aggTrade", now_ns, parse_status="parse_error"))
    assert sup._health[("BTCUSDT", "aggTrade")].parse_errors == 1
