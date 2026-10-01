"""
Telegram notification service for OneMil day trading system.

Sends notifications via Telegram for all trading events:
- Scanner startup
- Stock qualified by scanner
- Bull flag pattern detected
- Trade plan created
- Bracket order submitted / filled
- Position closed (P&L)
- End-of-day detailed report
- Errors (NO SILENT FAILURES)

Uses aiohttp for async HTTP requests to Telegram API.

Flood control (added 2026-09-29): the engine pushed 144 identical ERROR
lines to Telegram in 30s, drawing 68x HTTP 429 "Too Many Requests" before
this existed. Every send now goes through two gates before it touches the
network — see `_dedup_gate` and `_rate_cap_gate` — and a 429 response is
honoured once, never looped on.
"""

import logging
import asyncio
import html
import time
import aiohttp
from collections import deque
from typing import Optional, Dict, List, Any
from datetime import datetime, timezone, date

logger = logging.getLogger(__name__)


class TelegramNotifier:
    """
    Telegram notification service for OneMil trading bot.

    Sends formatted HTML messages to Telegram for various trading events.
    NO SILENT FAILURES - all errors are logged and reported.

    Every send passes through flood control before it reaches the network:
    exact-duplicate text within `_dedup_window_s` is suppressed, and no
    more than `_rate_limit_max` sends go out per rolling
    `_rate_limit_window_s`. Both gates are shared by every caller of
    `send_message` / `send_message_sync`, including `TelegramErrorHandler`
    (monitoring/telegram_error_handler.py), which delegates its transport
    to an internal instance of this class for that reason.
    """

    def __init__(
        self,
        bot_token: str,
        chat_id: str,
        enabled: bool = True,
    ):
        """
        Initialize Telegram notifier.

        Args:
            bot_token: Telegram bot token from BotFather
            chat_id: Telegram chat ID to send messages to
            enabled: Master switch for notifications
        """
        self.bot_token = bot_token
        self.chat_id = chat_id
        self.enabled = enabled

        self.api_url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"

        # --- Flood control state ---
        # Dedup: suppress a message whose text matches the last SENT message
        # within this window. Anchored on the last actual send, not on each
        # suppressed attempt, so a message repeating faster than the window
        # still gets a heartbeat send roughly once per window.
        self._dedup_window_s: float = 60.0
        self._last_text: Optional[str] = None
        self._last_text_time: float = 0.0
        self._suppressed_dup_count: int = 0

        # Rate cap: at most N sends per rolling window. Beyond the cap,
        # sends are dropped (not queued) and counted; a single summary line
        # goes out ahead of the next message once the window frees capacity.
        self._rate_limit_max: int = 20
        self._rate_limit_window_s: float = 60.0
        self._send_times: deque = deque()
        self._rate_suppressed_count: int = 0

        if self.enabled:
            if not self.bot_token or not self.chat_id:
                logger.error("Telegram enabled but bot_token or chat_id not configured")
                self.enabled = False
            else:
                logger.info("TelegramNotifier initialized successfully")
        else:
            logger.info("Telegram notifications disabled")

    # =========================================================================
    # Flood control
    # =========================================================================

    def _dedup_gate(self, message: str, now: float) -> Optional[str]:
        """
        Suppress exact-duplicate text sent within the dedup window.

        Returns None if `message` is identical to the last SENT message and
        less than `_dedup_window_s` has elapsed (the send is suppressed).
        Otherwise returns the text to actually send: `message` itself, or
        `message` with a "(+N identical suppressed)" suffix if earlier
        duplicates were folded into this one.
        """
        if self._last_text == message and (now - self._last_text_time) < self._dedup_window_s:
            self._suppressed_dup_count += 1
            logger.warning(
                f"Telegram dedup: suppressing duplicate #{self._suppressed_dup_count} "
                f"(identical to message sent {now - self._last_text_time:.1f}s ago)"
            )
            return None

        suffix_count = self._suppressed_dup_count
        self._last_text = message
        self._last_text_time = now
        self._suppressed_dup_count = 0

        if suffix_count:
            return f"{message} (+{suffix_count} identical suppressed)"
        return message

    def _rate_cap_gate(self, message: str, now: float) -> List[str]:
        """
        Cap actual sends to `_rate_limit_max` per rolling `_rate_limit_window_s`.

        Beyond the cap, the send is dropped (not queued) and counted. Once
        the window frees enough room for a send, a single summary line
        ("suppressed N messages in the last minute") is emitted ahead of
        the next real message.

        Returns the message texts to transmit, in order: empty (this send
        was capped), one entry (just `message`), or two entries (the
        summary line, then `message`).
        """
        while self._send_times and (now - self._send_times[0]) >= self._rate_limit_window_s:
            self._send_times.popleft()

        if len(self._send_times) >= self._rate_limit_max:
            self._rate_suppressed_count += 1
            logger.warning(
                f"Telegram rate cap hit ({self._rate_limit_max}/"
                f"{int(self._rate_limit_window_s)}s) — dropping send "
                f"(#{self._rate_suppressed_count} suppressed this window)"
            )
            return []

        outgoing: List[str] = []
        if self._rate_suppressed_count:
            outgoing.append(
                f"⚠️ Telegram: suppressed {self._rate_suppressed_count} "
                f"messages in the last minute"
            )
            self._send_times.append(now)
            self._rate_suppressed_count = 0

        outgoing.append(message)
        self._send_times.append(now)
        return outgoing

    # =========================================================================
    # Core Send
    # =========================================================================

    async def send_message(self, message: str, parse_mode: str = "HTML") -> bool:
        """
        Send a message to Telegram, subject to flood control.

        Args:
            message: Message text (supports HTML formatting)
            parse_mode: Telegram parse mode ('HTML' or 'Markdown')

        Returns:
            True if the last message this call transmitted got a 200 from
            Telegram. False if nothing was transmitted (disabled, or fully
            suppressed by dedup/rate-cap) or the API call failed.
        """
        if not self.enabled:
            logger.debug(f"Telegram disabled, would have sent:\n{message}")
            return False

        now = time.monotonic()
        gated = self._dedup_gate(message, now)
        if gated is None:
            return False

        outgoing = self._rate_cap_gate(gated, now)
        if not outgoing:
            return False

        ok = False
        for text in outgoing:
            ok = await self._post_message(text, parse_mode)
        return ok

    async def _post_message(self, message: str, parse_mode: str = "HTML") -> bool:
        """
        Transmit one message to the Telegram API.

        No dedup/rate-cap here — `send_message` is responsible for flood
        control; this method only handles wire format, length truncation,
        the HTML-parse-error plain-text fallback, and a single 429 retry.
        """
        try:
            # Telegram max message length is 4096 chars
            if len(message) > 4096:
                logger.warning(f"Telegram message truncated ({len(message)} -> 4096 chars)")
                message = message[:4090] + "\n..."

            async with aiohttp.ClientSession() as session:
                payload = {
                    "chat_id": self.chat_id,
                    "text": message,
                    "disable_web_page_preview": True,
                }
                if parse_mode:
                    payload["parse_mode"] = parse_mode

                timeout = aiohttp.ClientTimeout(total=10)
                async with session.post(self.api_url, json=payload, timeout=timeout) as response:
                    if response.status == 200:
                        logger.debug("Telegram message sent successfully")
                        return True
                    elif response.status == 429:
                        return await self._handle_rate_limited(session, payload, timeout, response)
                    elif response.status == 400 and "parse entities" in (await response.text()):
                        # HTML parse error — retry without parse_mode (plain text)
                        logger.warning("Telegram HTML parse error — retrying as plain text")
                        payload.pop("parse_mode", None)
                        async with session.post(self.api_url, json=payload, timeout=timeout) as retry:
                            if retry.status == 200:
                                logger.debug("Telegram message sent (plain text fallback)")
                                return True
                            else:
                                logger.error(f"Telegram plain text fallback also failed: {retry.status}")
                                return False
                    else:
                        error_text = await response.text()
                        logger.error(f"Telegram API error {response.status}: {error_text}")
                        return False

        except asyncio.TimeoutError:
            logger.error("Telegram API request timed out")
            return False
        except aiohttp.ClientError as e:
            logger.error(f"Telegram HTTP error: {e}")
            return False
        except RuntimeError as e:
            # CPython's concurrent.futures.thread atexit guard: the process's
            # main thread has already returned (interpreter shutting down)
            # and aiohttp's DNS/connection setup tried to use the dead
            # thread-pool executor. A process-exit condition, not a
            # retryable send failure — one WARNING, never ERROR (2026-10-01
            # 20:04-20:10 UTC incident, docs/orb_shutdown_hygiene_20261001.md).
            if 'interpreter shutdown' in str(e):
                logger.warning(
                    "Telegram send: interpreter shutting down, message dropped"
                )
                return False
            logger.error(f"Unexpected error sending Telegram message: {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error sending Telegram message: {e}")
            return False

    async def _handle_rate_limited(self, session, payload, timeout, response) -> bool:
        """
        Honour a Telegram 429 "Too Many Requests" response exactly once.

        Reads `retry_after` from the JSON body (Telegram's documented
        field) falling back to the `Retry-After` header. Sleeps at most 5s
        and retries a single time; if `retry_after` exceeds 5s, or the
        retry also fails, the message is dropped with one WARNING. Never
        loops on 429 — this is what turned into 68 repeated 429s on
        2026-09-29.
        """
        retry_after = None
        try:
            body = await response.json()
            retry_after = (body or {}).get("parameters", {}).get("retry_after")
        except Exception:
            retry_after = None

        if retry_after is None:
            header_val = response.headers.get("Retry-After")
            if header_val is not None:
                try:
                    retry_after = float(header_val)
                except ValueError:
                    retry_after = None

        if retry_after is None or retry_after > 5:
            logger.warning(
                f"Telegram 429 Too Many Requests — retry_after="
                f"{retry_after if retry_after is not None else 'unknown'}s "
                f"exceeds 5s cap, dropping message"
            )
            return False

        logger.warning(
            f"Telegram 429 Too Many Requests — sleeping {retry_after}s then retrying once"
        )
        await asyncio.sleep(retry_after)

        async with session.post(self.api_url, json=payload, timeout=timeout) as retry_response:
            if retry_response.status == 200:
                logger.debug("Telegram message sent after 429 retry")
                return True
            else:
                logger.warning(
                    f"Telegram send dropped after 429 retry (status {retry_response.status})"
                )
                return False

    def send_message_sync(self, message: str, parse_mode: str = "HTML") -> bool:
        """
        Synchronous wrapper for send_message.

        Uses asyncio.run() to send from sync context.
        """
        try:
            return asyncio.run(self.send_message(message, parse_mode))
        except RuntimeError:
            loop = asyncio.get_event_loop()
            return loop.run_until_complete(self.send_message(message, parse_mode))

    # =========================================================================
    # Scanner Events
    # =========================================================================

    def notify_scanner_started(self, universe_size: int, trading_enabled: bool,
                                mode: str = "paper") -> None:
        """Notify that scanner has started."""
        mode_label = "PAPER" if mode == "paper" else "LIVE"
        trading_status = "ACTIVE" if trading_enabled else "OFF"

        msg = (
            f"🚀 <b>OneMil Scanner Started</b>\n\n"
            f"📊 Universe: <b>{universe_size}</b> stocks\n"
            f"💰 Trading: <b>{trading_status}</b> ({mode_label})\n"
            f"⏰ {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}"
        )
        self.send_message_sync(msg)

    def notify_stock_qualified(self, symbol: str, price: float, change_pct: float,
                                relative_volume: float, headline: Optional[str] = None) -> None:
        """Notify that a stock has been qualified by the scanner."""
        news_line = f"📰 <i>{html.escape(headline)}</i>" if headline else "📰 No headline"
        msg = (
            f"🎯 <b>Stock Qualified: {html.escape(symbol)}</b>\n\n"
            f"💲 Price: <b>${price:.2f}</b> ({change_pct:+.1f}%)\n"
            f"📈 Relative Volume: <b>{relative_volume:.1f}x</b>\n"
            f"{news_line}"
        )
        self.send_message_sync(msg)

    def notify_premarket_gaps(self, gaps: List[Dict]) -> None:
        """Notify about pre-market gap-ups detected.

        Cap at top 30 by gap_pct — full list lives in console + DB.
        Telegram ceiling is 4096 chars; 375-row dump (5/8 incident) hit
        14K chars, truncated mid-<b> tag, broke HTML parse.
        """
        if not gaps:
            return

        TOP_N = 30
        sorted_gaps = sorted(gaps, key=lambda x: x.get('gap_pct', 0), reverse=True)
        shown = sorted_gaps[:TOP_N]

        header = f"🌅 <b>Pre-Market Gap-Ups: {len(gaps)} found</b>"
        if len(gaps) > TOP_N:
            header += f" (showing top {TOP_N})"
        lines = [header + "\n"]

        for g in shown:
            symbol = html.escape(g.get('symbol', ''))
            gap_pct = g.get('gap_pct', 0)
            price = g.get('current_price', 0)
            prev = g.get('prev_close', 0)
            lines.append(
                f"  {symbol}: ${prev:.2f} → ${price:.2f} (<b>+{gap_pct:.1f}%</b>)"
            )

        self.send_message_sync("\n".join(lines))

    # =========================================================================
    # Trading Events
    # =========================================================================

    def notify_pattern_detected(self, symbol: str, pole_gain_pct: float,
                                 retracement_pct: float, breakout_level: float) -> None:
        """Notify that a bull flag pattern was detected."""
        msg = (
            f"🏁 <b>Bull Flag Detected: {html.escape(symbol)}</b>\n\n"
            f"📊 Pole Gain: <b>{pole_gain_pct:.1f}%</b>\n"
            f"📉 Retracement: <b>{retracement_pct:.1f}%</b>\n"
            f"🎯 Breakout Level: <b>${breakout_level:.2f}</b>"
        )
        self.send_message_sync(msg)

    def notify_trade_planned(self, symbol: str, entry: float, stop: float,
                              target: float, shares: int, risk_reward: float) -> None:
        """Notify that a trade plan was created."""
        risk = entry - stop
        total_risk = risk * shares
        msg = (
            f"📋 <b>Trade Plan: {html.escape(symbol)}</b>\n\n"
            f"▶️ Entry: <b>${entry:.2f}</b>\n"
            f"🛑 Stop: <b>${stop:.2f}</b> (risk: ${risk:.2f}/share)\n"
            f"🎯 Target: <b>${target:.2f}</b>\n"
            f"📊 R:R = <b>{risk_reward:.1f}:1</b>\n"
            f"📦 Shares: <b>{shares}</b> (total risk: ${total_risk:.2f})"
        )
        self.send_message_sync(msg)

    def notify_order_submitted(self, symbol: str, order_id: str, shares: int,
                                entry: float) -> None:
        """Notify that a bracket order was submitted."""
        msg = (
            f"📤 <b>[Bull Flag] Order Submitted: {html.escape(symbol)}</b>\n\n"
            f"🆔 Order: <code>{html.escape(order_id)}</code>\n"
            f"📦 {shares} shares @ ${entry:.2f}\n"
            f"⏰ {datetime.now(timezone.utc).strftime('%H:%M:%S UTC')}"
        )
        self.send_message_sync(msg)

    def notify_order_filled(self, symbol: str, shares: int, fill_price: float,
                             order_id: str) -> None:
        """Notify that an order was filled."""
        msg = (
            f"✅ <b>[Bull Flag] Order Filled: {html.escape(symbol)}</b>\n\n"
            f"📦 {shares} shares @ <b>${fill_price:.2f}</b>\n"
            f"🆔 <code>{html.escape(order_id)}</code>"
        )
        self.send_message_sync(msg)

    def notify_position_closed(self, symbol: str, entry_price: float,
                                exit_price: float, shares: int,
                                pnl: float, exit_reason: str) -> None:
        """Notify that a position was closed."""
        pnl_emoji = "💰" if pnl >= 0 else "💸"
        pnl_sign = "+" if pnl >= 0 else ""
        pnl_pct = ((exit_price - entry_price) / entry_price) * 100

        reason_map = {
            'take_profit': '🎯 Take Profit',
            'stop_loss': '🛑 Stop Loss',
            'trail_stop': '📈 Trailing Stop',
            'stop_loss_fallback': '🛑 Stop (Fallback)',
            'force_close': '🕐 Force Close',
            'eod_close': '🕐 End of Day',
            'exhaustion_partial': '⚡ Exhaustion Partial',
            'exhaust+trail_stop': '⚡📈 Exhaust + Trail',
            'exhaust+stop_loss': '⚡🛑 Exhaust + Stop',
            'exhaust+force_close': '⚡🕐 Exhaust + Close',
        }
        reason_label = reason_map.get(exit_reason, exit_reason)

        msg = (
            f"{pnl_emoji} <b>[Bull Flag] Position Closed: {html.escape(symbol)}</b>\n\n"
            f"📊 Entry: ${entry_price:.2f} → Exit: ${exit_price:.2f}\n"
            f"📈 P&L: <b>{pnl_sign}${pnl:.2f}</b> ({pnl_sign}{pnl_pct:.1f}%)\n"
            f"📦 Shares: {shares}\n"
            f"📌 Reason: {reason_label}"
        )
        self.send_message_sync(msg)

    def notify_error(self, error_msg: str, component: str = "System") -> None:
        """Notify about an error."""
        msg = (
            f"🚨 <b>ERROR — {html.escape(component)}</b>\n\n"
            f"<code>{html.escape(error_msg[:2000])}</code>\n"
            f"⏰ {datetime.now(timezone.utc).strftime('%H:%M:%S UTC')}"
        )
        self.send_message_sync(msg)

    # =========================================================================
    # End-of-Day Report
    # =========================================================================

    def send_daily_report(self, report: Dict[str, Any]) -> None:
        """
        Send the detailed end-of-day trading report.

        Args:
            report: Dict containing all daily trading data:
                - trade_date: str
                - universe_size: int
                - premarket_gaps: list of gap-up dicts
                - qualified_stocks: list of qualified stock dicts
                - patterns_detected: int
                - patterns_detected_details: list of pattern dicts
                - trades: list of trade dicts
                - total_trades: int
                - winning_trades: int
                - losing_trades: int
                - gross_pnl: float
                - open_positions: int
        """
        trade_date = report.get('trade_date', date.today().isoformat())
        universe_size = report.get('universe_size', 0)
        premarket_gaps = report.get('premarket_gaps', [])
        qualified_stocks = report.get('qualified_stocks', [])
        patterns_detected = report.get('patterns_detected', 0)
        pattern_details = report.get('patterns_detected_details', [])
        trades = report.get('trades', [])
        total_trades = report.get('total_trades', 0)
        winning_trades = report.get('winning_trades', 0)
        losing_trades = report.get('losing_trades', 0)
        gross_pnl = report.get('gross_pnl', 0.0)
        open_positions = report.get('open_positions', 0)

        # Header
        pnl_emoji = "💰" if gross_pnl >= 0 else "💸"
        pnl_sign = "+" if gross_pnl >= 0 else ""
        lines = [
            f"📊 <b>OneMil Daily Report — {trade_date}</b>",
            f"{'═' * 35}",
            "",
        ]

        # P&L Summary
        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
        lines.extend([
            f"{pnl_emoji} <b>P&L: {pnl_sign}${gross_pnl:.2f}</b>",
            f"📈 Trades: {total_trades} ({winning_trades}W / {losing_trades}L)",
            f"🎯 Win Rate: {win_rate:.0f}%",
            "",
        ])

        # Scanner Summary
        lines.extend([
            f"<b>🔍 Scanner</b>",
            f"  Universe: {universe_size} stocks",
            f"  Pre-market gaps: {len(premarket_gaps)}",
        ])
        for g in premarket_gaps[:10]:
            symbol = html.escape(str(g.get('symbol', '')))
            gap = g.get('gap_pct', 0)
            price = g.get('current_price', 0)
            lines.append(f"    {symbol}: +{gap:.1f}% (${price:.2f})")

        lines.append(f"  Qualified intraday: {len(qualified_stocks)}")
        for q in qualified_stocks[:10]:
            symbol = html.escape(str(q.get('symbol', '')))
            change = q.get('intraday_change_pct', q.get('change_pct', 0))
            rvol = q.get('relative_volume', 0)
            headline = q.get('news_headline', q.get('headline', ''))
            news_str = f' — "{html.escape(str(headline))}"' if headline else ''
            lines.append(
                f"    {symbol}: {change:+.1f}%, {rvol:.1f}x vol{news_str}"
            )
        lines.append("")

        # Pattern Detection
        lines.extend([
            f"<b>🏁 Pattern Detection</b>",
            f"  Patterns found: {patterns_detected}",
        ])
        for p in pattern_details[:10]:
            symbol = html.escape(str(p.get('symbol', '')))
            pole = p.get('pole_gain_pct', 0)
            retrace = p.get('retracement_pct', 0)
            lines.append(
                f"    {symbol}: pole +{pole:.1f}%, retrace {retrace:.1f}%"
            )
        lines.append("")

        # Trade Details
        if trades:
            lines.append(f"<b>💼 Trades</b>")
            for t in trades:
                symbol = html.escape(str(t.get('symbol', '')))
                entry = t.get('entry_price', 0)
                exit_p = t.get('exit_price')
                pnl = t.get('pnl')
                status = t.get('order_status', '')
                reason = t.get('exit_reason', '')

                if exit_p and pnl is not None:
                    pnl_s = f"+${pnl:.2f}" if pnl >= 0 else f"-${abs(pnl):.2f}"
                    result_emoji = "✅" if pnl >= 0 else "❌"
                    lines.append(
                        f"  {result_emoji} {symbol}: ${entry:.2f} → ${exit_p:.2f} "
                        f"({pnl_s}) [{reason}]"
                    )
                else:
                    lines.append(
                        f"  ⏳ {symbol}: ${entry:.2f} (status: {status})"
                    )
            lines.append("")

        # Open Positions
        if open_positions > 0:
            lines.append(f"⚠️ Open positions: {open_positions}")
            lines.append("")

        # Footer
        lines.append(f"<i>Generated {datetime.now(timezone.utc).strftime('%H:%M UTC')}</i>")

        self.send_message_sync("\n".join(lines))
