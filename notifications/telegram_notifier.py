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
"""

import logging
import asyncio
import html
import aiohttp
from typing import Optional, Dict, List, Any
from datetime import datetime, timezone, date

logger = logging.getLogger(__name__)


class TelegramNotifier:
    """
    Telegram notification service for OneMil trading bot.

    Sends formatted HTML messages to Telegram for various trading events.
    NO SILENT FAILURES - all errors are logged and reported.
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

        if self.enabled:
            if not self.bot_token or not self.chat_id:
                logger.error("Telegram enabled but bot_token or chat_id not configured")
                self.enabled = False
            else:
                logger.info("TelegramNotifier initialized successfully")
        else:
            logger.info("Telegram notifications disabled")

    # =========================================================================
    # Core Send
    # =========================================================================

    async def send_message(self, message: str, parse_mode: str = "HTML") -> bool:
        """
        Send a message to Telegram.

        Args:
            message: Message text (supports HTML formatting)
            parse_mode: Telegram parse mode ('HTML' or 'Markdown')

        Returns:
            True if sent successfully, False otherwise
        """
        if not self.enabled:
            logger.debug(f"Telegram disabled, would have sent:\n{message}")
            return False

        try:
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
        except Exception as e:
            logger.error(f"Unexpected error sending Telegram message: {e}")
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
        """Notify about pre-market gap-ups detected."""
        if not gaps:
            return

        lines = [f"🌅 <b>Pre-Market Gap-Ups: {len(gaps)} found</b>\n"]
        for g in sorted(gaps, key=lambda x: x.get('gap_pct', 0), reverse=True):
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
            f"📤 <b>Order Submitted: {html.escape(symbol)}</b>\n\n"
            f"🆔 Order: <code>{html.escape(order_id)}</code>\n"
            f"📦 {shares} shares @ ${entry:.2f}\n"
            f"⏰ {datetime.now(timezone.utc).strftime('%H:%M:%S UTC')}"
        )
        self.send_message_sync(msg)

    def notify_order_filled(self, symbol: str, shares: int, fill_price: float,
                             order_id: str) -> None:
        """Notify that an order was filled."""
        msg = (
            f"✅ <b>Order Filled: {html.escape(symbol)}</b>\n\n"
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
            'eod_close': '🕐 End of Day',
        }
        reason_label = reason_map.get(exit_reason, exit_reason)

        msg = (
            f"{pnl_emoji} <b>Position Closed: {html.escape(symbol)}</b>\n\n"
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
