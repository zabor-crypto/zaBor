"""Telegram notification module for KillSwitch.

Sends alerts via Telegram Bot API (no extra dependencies — uses stdlib urllib).
All methods return True on success, False on failure (failures are logged but never raise).

Configure via environment variables:
  TELEGRAM_BOT_TOKEN  — bot token from BotFather
  TELEGRAM_CHAT_ID    — chat/user ID to send messages to
"""

import json
import time
import urllib.request
import urllib.error
from datetime import datetime, timezone
from typing import Optional


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


class TelegramNotifier:
    def __init__(self, bot_token: str, chat_id: str):
        self._url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
        self._chat_id = chat_id
        # Dedup: track last message hash per key to avoid spamming repeated blocked alerts
        self._last_sent: dict = {}

    # ------------------------------------------------------------------
    # Core send
    # ------------------------------------------------------------------

    def send(self, text: str, parse_mode: str = "HTML", dedup_key: Optional[str] = None,
             dedup_ttl: int = 600) -> bool:
        """Send a message. If dedup_key is set, suppress identical key within dedup_ttl seconds."""
        if dedup_key:
            last = self._last_sent.get(dedup_key, 0)
            if time.time() - last < dedup_ttl:
                return True  # suppressed — same alert sent recently
            self._last_sent[dedup_key] = time.time()

        payload = json.dumps({
            "chat_id": self._chat_id,
            "text": text,
            "parse_mode": parse_mode,
            "disable_web_page_preview": True,
        }).encode()
        try:
            req = urllib.request.Request(
                self._url,
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=10) as resp:
                return resp.status == 200
        except urllib.error.HTTPError as e:
            print(f"[TELEGRAM] HTTP {e.code}: {e.read().decode()[:200]}")
            return False
        except Exception as e:
            print(f"[TELEGRAM] Send failed: {e}")
            return False

    # ------------------------------------------------------------------
    # Typed notifications
    # ------------------------------------------------------------------

    def notify_startup(self, config_path: str, dry_run: bool, exchanges: list) -> bool:
        mode = "DRY RUN 🟡" if dry_run else "LIVE 🔴"
        text = (
            f"🟢 <b>KillSwitch started</b>\n\n"
            f"Mode: <b>{mode}</b>\n"
            f"Config: <code>{config_path}</code>\n"
            f"Exchanges: {', '.join(exchanges)}\n"
            f"Time: {_utc_now()}"
        )
        return self.send(text)

    def notify_shutdown(self, reason: str = "manual stop") -> bool:
        text = (
            f"🔴 <b>KillSwitch stopped</b>\n\n"
            f"Reason: {reason}\n"
            f"Time: {_utc_now()}"
        )
        return self.send(text)

    def notify_stage_executed(
        self, scope: str, stage_num: int, mode: str,
        dry_run: bool, attr_source: Optional[str] = None,
        positions_closed: Optional[int] = None, dd_str: str = "",
    ) -> bool:
        icon = {1: "⚡", 2: "🔶", 3: "🔴"}.get(stage_num, "⚠️")
        dry = " [DRY RUN]" if dry_run else ""
        text = (
            f"{icon} <b>KillSwitch{dry}: Stage {stage_num} triggered</b>\n\n"
            f"Exchange: <code>{scope}</code>\n"
            f"Mode: <code>{mode}</code>\n"
        )
        if attr_source:
            text += f"Loss source: <b>{attr_source}</b>\n"
        if positions_closed is not None:
            text += f"Positions closed: <b>{positions_closed}</b>\n"
        if dd_str:
            text += f"Drawdown: {dd_str}\n"
        text += f"Time: {_utc_now()}"
        return self.send(text)

    def notify_drawdown_warning(
        self, scope: str, stage_num: int, window: str,
        dd_pct: float, threshold_pct: float,
        confirmations: int, required: int,
    ) -> bool:
        text = (
            f"⚠️ <b>KillSwitch: Drawdown confirmed</b>\n\n"
            f"Exchange: <code>{scope}</code>\n"
            f"Drawdown: <b>{dd_pct:.1%}</b> (threshold: {threshold_pct:.1%})\n"
            f"Window: <code>{window}</code> | Confirmations: {confirmations}/{required}\n"
            f"Target stage: {stage_num}\n"
            f"Time: {_utc_now()}"
        )
        dedup = f"warn|{scope}|stage{stage_num}|{window}"
        return self.send(text, dedup_key=dedup, dedup_ttl=300)

    def notify_stage_blocked(self, scope: str, stage_num: int, reason: str) -> bool:
        text = (
            f"⏸ <b>KillSwitch: Stage {stage_num} blocked</b>\n\n"
            f"Exchange: <code>{scope}</code>\n"
            f"Reason: {reason}\n"
            f"Time: {_utc_now()}"
        )
        dedup = f"blocked|{scope}|stage{stage_num}"
        return self.send(text, dedup_key=dedup, dedup_ttl=1800)

    def notify_trading_lock(
        self, scope: str, stage_num: int, expires_ts: int
    ) -> bool:
        expires_str = datetime.fromtimestamp(expires_ts, tz=timezone.utc).strftime(
            "%Y-%m-%d %H:%M UTC"
        )
        text = (
            f"🔒 <b>KillSwitch: Trading locked</b>\n\n"
            f"Exchange: <code>{scope}</code>\n"
            f"Stage: <b>{stage_num}</b>\n"
            f"Locked until: <b>{expires_str}</b>\n\n"
            f"⚠️ Review positions manually before clearing the lock"
        )
        return self.send(text)

    def notify_error(self, scope: str, message: str) -> bool:
        text = (
            f"❌ <b>KillSwitch: Execution error</b>\n\n"
            f"Exchange: <code>{scope}</code>\n"
            f"Error: {message[:400]}\n"
            f"Time: {_utc_now()}"
        )
        dedup = f"err|{scope}|{message[:60]}"
        return self.send(text, dedup_key=dedup, dedup_ttl=300)

    def notify_circuit_breaker(self, scope: str) -> bool:
        text = (
            f"🔌 <b>KillSwitch: Circuit breaker tripped</b>\n\n"
            f"Exchange: <code>{scope}</code>\n"
            f"Polling halted until connectivity recovers.\n"
            f"Time: {_utc_now()}"
        )
        dedup = f"cb|{scope}"
        return self.send(text, dedup_key=dedup, dedup_ttl=3600)


def build_notifier(bot_token_env: str = "TELEGRAM_BOT_TOKEN",
                   chat_id_env: str = "TELEGRAM_CHAT_ID") -> Optional[TelegramNotifier]:
    """Build TelegramNotifier from environment variables, or return None if not configured."""
    import os
    token = os.environ.get(bot_token_env, "").strip()
    chat_id = os.environ.get(chat_id_env, "").strip()
    if token and chat_id:
        notifier = TelegramNotifier(token, chat_id)
        print(f"[TELEGRAM] Notifier active (chat_id={chat_id[:6]}...)")
        return notifier
    print(f"[TELEGRAM] Not configured — set {bot_token_env} and {chat_id_env} to enable")
    return None
