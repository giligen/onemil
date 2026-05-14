"""
Order executor for submitting bracket orders via Alpaca.

Handles:
- Bracket order submission (entry + stop loss + take profit)
- Order status tracking
- Trade record creation in database
"""

import json
import logging
from datetime import date, datetime, timezone
from typing import Any, Dict, Optional

from data_sources.alpaca_client import AlpacaClient
from persistence.database import Database
from trading.trade_planner import TradePlan

logger = logging.getLogger(__name__)

# Telemetry threshold: WARN when loop_processed_at → order_submitted_at exceeds
# this. Historical norm (April 2026, paper API): 220-450ms. The 2026-04-15
# Anthropic+Alpaca cloud incident pushed this to 3.3s — that's the kind of
# anomaly we want to surface in real time. See research/macd_wave_latency.md
# (or memory:project_prod_timing) for the underlying analysis.
_SUBMIT_LATENCY_WARN_MS = 1000


class OrderExecutor:
    """
    Executes trade plans by submitting bracket orders to Alpaca.

    Each bracket order consists of:
    - Entry: limit buy at entry_price
    - Stop loss: sell stop at stop_loss_price
    - Take profit: sell limit at take_profit_price
    All with TimeInForce.DAY (expire at close).
    """

    def __init__(
        self,
        alpaca_client: AlpacaClient,
        db: Database,
        order_stream: Optional[Any] = None,
    ):
        """
        Initialize OrderExecutor.

        Args:
            alpaca_client: Alpaca API client for order submission
            db: Database for trade record persistence
            order_stream: Optional OrderStreamWatcher for fast wash-trade check.
                When attached and healthy, conflict check uses the in-memory
                cache (~microseconds) instead of a REST call (~200-400 ms).
        """
        self.alpaca = alpaca_client
        self.db = db
        self.order_stream = order_stream

        # Marketable-limit fallback config (IREZ+TTGT post-mortem 2026-05-08).
        # Read once at construction so submit_buy_stop_order's hot path doesn't
        # touch yaml. Defaults to enabled — see
        # docs/irez_ttgt_paper_vs_prod_divergence.md.
        try:
            from config import Config as _Config
            self._marketable_limit_fallback_cfg = _Config().marketable_limit_fallback_cfg
        except Exception as e:
            logger.warning(
                f"OrderExecutor: failed to load marketable_limit_fallback_cfg: {e} "
                f"— defaulting to enabled"
            )
            self._marketable_limit_fallback_cfg = {"enabled": True}

    def _persist_submit_timing(self, trade_id: int, symbol: str,
                                strategy: str,
                                pipeline_timing: Optional[Dict[str, Any]]) -> None:
        """Persist pipeline timing telemetry for a trade. Call AFTER save_trade.

        Updates: order_submitted_at, bar_close_at, loop_processed_at, and
        derived bar_close_to_loop_ms + quote_to_submit_ms. Persisted via
        update_trade because save_trade's INSERT only covers a fixed set of
        columns — mirrors macd_wave_engine's pattern.

        For bull_flag, loop_to_quote_ms stays NULL (no quote-fetch step);
        quote_to_submit_ms carries the entire loop→submit-ack interval and
        is the primary anomaly signal. Logs WARN when it exceeds
        _SUBMIT_LATENCY_WARN_MS so we get early notice of Alpaca/cloud
        degradation days (see 2026-04-15 incident).
        """
        order_submitted_at = datetime.now(timezone.utc)
        pt = pipeline_timing or {}
        bar_close_at = pt.get('bar_close_at')
        loop_processed_at = pt.get('loop_processed_at')

        updates: Dict[str, Any] = {'order_submitted_at': order_submitted_at}
        if bar_close_at is not None:
            updates['bar_close_at'] = bar_close_at
        if loop_processed_at is not None:
            updates['loop_processed_at'] = loop_processed_at

        if bar_close_at is not None and loop_processed_at is not None:
            updates['bar_close_to_loop_ms'] = int(
                (loop_processed_at - bar_close_at).total_seconds() * 1000
            )

        if loop_processed_at is not None:
            loop_to_submit_ms = int(
                (order_submitted_at - loop_processed_at).total_seconds() * 1000
            )
            updates['quote_to_submit_ms'] = loop_to_submit_ms
            if loop_to_submit_ms > _SUBMIT_LATENCY_WARN_MS:
                logger.warning(
                    f"{symbol}: SLOW SUBMIT — loop→submit "
                    f"{loop_to_submit_ms}ms > {_SUBMIT_LATENCY_WARN_MS}ms "
                    f"threshold (strategy={strategy}). "
                    f"Likely Alpaca/cloud-provider degradation."
                )

        try:
            self.db.update_trade(trade_id, updates)
        except Exception as e:
            logger.warning(
                f"{symbol}: failed to persist timing telemetry: {e}"
            )

    def _has_conflicting_orders(self, symbol: str) -> bool:
        """Check if symbol has existing open orders on Alpaca (any strategy).

        Prevents wash trades when two strategies target the same symbol.

        Fast path: if OrderStreamWatcher is attached AND healthy, read from its
        push-updated in-memory set. O(1). No network.
        Slow path: REST get_orders fallback (legacy behaviour).
        """
        # Fast path via the shared OrderStreamWatcher cache.
        if self.order_stream is not None and self.order_stream.is_healthy():
            try:
                open_symbols = self.order_stream.get_open_order_symbols()
                if symbol in open_symbols:
                    logger.warning(
                        f"{symbol}: BLOCKED — open order detected via "
                        f"order-stream cache (would cause wash trade)"
                    )
                    return True
                return False
            except Exception as e:
                logger.warning(
                    f"{symbol}: order-stream conflict check failed, "
                    f"falling back to REST: {e}"
                )
                # fall through to REST

        # Slow path: REST (unchanged from legacy).
        try:
            from alpaca.trading.requests import GetOrdersRequest
            from alpaca.trading.enums import QueryOrderStatus
            req = GetOrdersRequest(status=QueryOrderStatus.OPEN, symbols=[symbol])
            existing = self.alpaca.trading_client.get_orders(filter=req)
            if existing:
                sides = [o.side.value for o in existing]
                logger.warning(
                    f"{symbol}: BLOCKED — {len(existing)} existing order(s) "
                    f"({', '.join(sides)}) would cause wash trade"
                )
                return True
        except Exception as e:
            logger.warning(f"{symbol}: Failed to check existing orders: {e}")
        return False

    def submit_bracket_order(self, plan: TradePlan) -> Optional[Dict[str, Any]]:
        """
        Submit a bracket order for a trade plan.

        Args:
            plan: TradePlan with entry, stop, target, and sizing

        Returns:
            Dict with order details if successful, None on failure
        """
        logger.info(
            f"{plan.symbol}: Submitting bracket order — "
            f"BUY {plan.shares} @ ${plan.entry_price:.2f}, "
            f"SL ${plan.stop_loss_price:.2f}, TP ${plan.take_profit_price:.2f}"
        )

        try:
            order = self.alpaca.submit_bracket_order(
                symbol=plan.symbol,
                qty=plan.shares,
                side='buy',
                limit_price=plan.entry_price,
                tp_price=plan.take_profit_price,
                sl_price=plan.stop_loss_price,
            )
        except Exception as e:
            logger.error(f"{plan.symbol}: Bracket order submission failed: {e}")
            return None

        if order is None:
            logger.error(f"{plan.symbol}: Bracket order returned None")
            return None

        order_id = order.get('id', '')
        order_status = order.get('status', 'unknown')

        logger.info(
            f"{plan.symbol}: Bracket order submitted — "
            f"ID: {order_id}, status: {order_status}"
        )

        # Save trade record to database
        pattern_data = json.dumps({
            'pole_start_idx': plan.pattern.pole_start_idx,
            'pole_end_idx': plan.pattern.pole_end_idx,
            'flag_start_idx': plan.pattern.flag_start_idx,
            'flag_end_idx': plan.pattern.flag_end_idx,
            'pole_low': plan.pattern.pole_low,
            'pole_high': plan.pattern.pole_high,
            'pole_height': plan.pattern.pole_height,
            'pole_gain_pct': plan.pattern.pole_gain_pct,
            'flag_low': plan.pattern.flag_low,
            'flag_high': plan.pattern.flag_high,
            'retracement_pct': plan.pattern.retracement_pct,
            'pullback_candle_count': plan.pattern.pullback_candle_count,
            'avg_pole_volume': plan.pattern.avg_pole_volume,
            'avg_flag_volume': plan.pattern.avg_flag_volume,
            'breakout_level': plan.pattern.breakout_level,
        })

        now = datetime.now(timezone.utc)
        trade_record = {
            'trade_date': date.today().isoformat(),
            'symbol': plan.symbol,
            'side': 'buy',
            'entry_price': plan.entry_price,
            'stop_loss_price': plan.stop_loss_price,
            'take_profit_price': plan.take_profit_price,
            'shares': plan.shares,
            'risk_per_share': plan.risk_per_share,
            'total_risk': plan.total_risk,
            'risk_reward_ratio': plan.risk_reward_ratio,
            'order_id': order_id,
            'order_status': order_status,
            'fill_price': None,
            'filled_at': None,
            'exit_price': None,
            'exit_reason': None,
            'exited_at': None,
            'pnl': None,
            'pnl_pct': None,
            'pattern_data': pattern_data,
            'strategy': 'bull_flag',
            'created_at': now,
            'updated_at': now,
        }

        try:
            trade_id = self.db.save_trade(trade_record)
            logger.info(f"{plan.symbol}: Trade record saved (id={trade_id})")
        except Exception as e:
            logger.error(f"{plan.symbol}: Failed to save trade record: {e}")

        return {
            'order_id': order_id,
            'status': order_status,
            'symbol': plan.symbol,
            'shares': plan.shares,
            'entry_price': plan.entry_price,
            'stop_loss_price': plan.stop_loss_price,
            'take_profit_price': plan.take_profit_price,
        }

    def submit_buy_stop_bracket_order(
        self, plan: TradePlan, slippage_pct: float = 0.02,
        sl_override: Optional[float] = None,
        pipeline_timing: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Submit a buy-stop bracket order for a trade plan.

        Places a stop-limit order that triggers at breakout_level and fills
        at breakout_level * (1 + slippage_pct) maximum.

        Slippage limit set to 2% based on 15-month backtest analysis (636 trades,
        Jan 2025 — Mar 2026). Gap-over trades where entry > breakout_level:
          - 0-0.5% gap: 109 trades, 42% win, +$475 avg → profitable
          - 0.5-1% gap:  82 trades, 39% win, +$318 avg → profitable
          - 1-2% gap:    98 trades, 41% win, +$314 avg → profitable
          - 2-5% gap:    43 trades, 23% win, -$134 avg → NET LOSERS
          - >5% gap:      7 trades, 14% win, -$281 avg → garbage
        At 2% cap: $254K total PnL (peak). At 0.5% cap: $198K (-$56K left on table).
        Gap-fill adjustment maintains planned dollar risk regardless of entry gap.

        Args:
            plan: TradePlan with entry (breakout_level), stop, target, sizing
            slippage_pct: Maximum slippage above stop_price (default 2.0%)
            sl_override: If set, use this SL price on the bracket instead of
                plan.stop_loss_price. Used by self-managed stops to place a
                wide safety-net SL while keeping plan.stop_loss_price (and
                risk_per_share/total_risk) at the real stop level.

        Returns:
            Dict with order details if successful, None on failure
        """
        if self._has_conflicting_orders(plan.symbol):
            return None

        stop_price = plan.entry_price
        limit_price = round(stop_price * (1 + slippage_pct), 2)
        bracket_sl = sl_override if sl_override is not None else plan.stop_loss_price

        logger.info(
            f"{plan.symbol}: Submitting buy-stop bracket order — "
            f"BUY {plan.shares} stop @ ${stop_price:.2f}, "
            f"limit ${limit_price:.2f}, "
            f"SL ${bracket_sl:.2f}, TP ${plan.take_profit_price:.2f}"
            f"{' (safety-net SL)' if sl_override else ''}"
        )

        try:
            order = self.alpaca.submit_stop_bracket_order(
                symbol=plan.symbol,
                qty=plan.shares,
                side='buy',
                stop_price=stop_price,
                limit_price=limit_price,
                tp_price=plan.take_profit_price,
                sl_price=bracket_sl,
            )
        except Exception as e:
            logger.error(f"{plan.symbol}: Buy-stop order submission failed: {e}")
            return None

        if order is None:
            logger.error(f"{plan.symbol}: Buy-stop order returned None")
            return None

        order_id = order.get('id', '')
        order_status = order.get('status', 'unknown')

        logger.info(
            f"{plan.symbol}: Buy-stop bracket order submitted — "
            f"ID: {order_id}, status: {order_status}"
        )

        # Save trade record to database (same as bracket order)
        pattern_data = json.dumps({
            'pole_start_idx': plan.pattern.pole_start_idx,
            'pole_end_idx': plan.pattern.pole_end_idx,
            'flag_start_idx': plan.pattern.flag_start_idx,
            'flag_end_idx': plan.pattern.flag_end_idx,
            'pole_low': plan.pattern.pole_low,
            'pole_high': plan.pattern.pole_high,
            'pole_height': plan.pattern.pole_height,
            'pole_gain_pct': plan.pattern.pole_gain_pct,
            'flag_low': plan.pattern.flag_low,
            'flag_high': plan.pattern.flag_high,
            'retracement_pct': plan.pattern.retracement_pct,
            'pullback_candle_count': plan.pattern.pullback_candle_count,
            'avg_pole_volume': plan.pattern.avg_pole_volume,
            'avg_flag_volume': plan.pattern.avg_flag_volume,
            'breakout_level': plan.pattern.breakout_level,
        })

        now = datetime.now(timezone.utc)
        trade_record = {
            'trade_date': date.today().isoformat(),
            'symbol': plan.symbol,
            'side': 'buy',
            'entry_price': plan.entry_price,
            'stop_loss_price': plan.stop_loss_price,
            'take_profit_price': plan.take_profit_price,
            'shares': plan.shares,
            'risk_per_share': plan.risk_per_share,
            'total_risk': plan.total_risk,
            'risk_reward_ratio': plan.risk_reward_ratio,
            'order_id': order_id,
            'order_status': order_status,
            'fill_price': None,
            'filled_at': None,
            'exit_price': None,
            'exit_reason': None,
            'exited_at': None,
            'pnl': None,
            'pnl_pct': None,
            'pattern_data': pattern_data,
            'strategy': 'bull_flag',
            'created_at': now,
            'updated_at': now,
        }

        try:
            trade_id = self.db.save_trade(trade_record)
            logger.info(f"{plan.symbol}: Trade record saved (id={trade_id})")
            # Persist pipeline timing telemetry via UPDATE (save_trade INSERT
            # doesn't cover these columns).
            self._persist_submit_timing(
                trade_id, plan.symbol, 'bull_flag', pipeline_timing,
            )
        except Exception as e:
            logger.error(f"{plan.symbol}: Failed to save trade record: {e}")

        return {
            'order_id': order_id,
            'status': order_status,
            'symbol': plan.symbol,
            'shares': plan.shares,
            'stop_price': stop_price,
            'limit_price': limit_price,
            'stop_loss_price': plan.stop_loss_price,
            'take_profit_price': plan.take_profit_price,
            'order_type': 'stop_bracket',
        }

    def submit_buy_stop_order(
        self, plan: TradePlan, slippage_pct: float = 0.02,
        pipeline_timing: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Submit a simple stop-limit buy order (no bracket legs).

        Uses less margin than bracket orders — no TP/SL legs reserved on Alpaca.
        Safety-net SL is submitted separately after fill detection.
        Used when self_managed_stops is enabled (StopMonitor handles real stop).

        Args:
            plan: TradePlan with entry (breakout_level), stop, target, sizing
            slippage_pct: Maximum slippage above stop_price (default 2.0%)

        Returns:
            Dict with order details if successful, None on failure
        """
        if self._has_conflicting_orders(plan.symbol):
            return None

        stop_price = plan.entry_price
        limit_price = round(stop_price * (1 + slippage_pct), 2)

        logger.info(
            f"{plan.symbol}: Submitting buy-stop order (simple) — "
            f"BUY {plan.shares} stop @ ${stop_price:.2f}, "
            f"limit ${limit_price:.2f}"
        )

        # Pre-flight quote check (IREZ+TTGT post-mortem 2026-05-08; extended
        # 2026-05-14 after KPTI/TRT). Alpaca LIVE rejects a buy stop-limit
        # whenever stop_price <= current ASK — the order is immediately
        # marketable, not a real stop (rejected ~4ms after submit, no
        # reject_reason in the REST response). Paper Alpaca does NOT enforce
        # this — the parity gap that lost TTGT, KPTI, TRT and ~24 other prod
        # orders. The earlier revision of this code checked the BID; that is
        # the wrong side of the spread and let the straddle case through.
        #   bid >= stop       → breakout fully confirmed → marketable LIMIT.
        #   bid < stop <= ask → spread straddles the breakout level → re-bump
        #                       the stop to ask + rebump_buffer so it stays a
        #                       real stop (only fills on a genuine upward
        #                       print). If the bumped stop would exceed
        #                       limit_price the breakout has run past our max
        #                       fill price → skip (return None; the engine
        #                       leaves the symbol un-traded and retries it on
        #                       the next bar).
        #   ask < stop        → normal stop-limit (unchanged).
        # BT has no bid/ask so it cannot hit this rejection — the rebump is a
        # live-microstructure adaptation, not a parity divergence; BT's 2%
        # gap-over cap (backtest.py max_gap_pct) mirrors limit_price here.
        # See docs/irez_ttgt_paper_vs_prod_divergence.md.
        _mlf_cfg = self._marketable_limit_fallback_cfg or {}
        _mlf_enabled = _mlf_cfg.get('enabled', True)
        _rebump_buffer = _mlf_cfg.get('rebump_buffer', 0.02)
        _used_marketable_limit = False
        _used_stop_rebump = False
        _submitted_stop = stop_price
        order = None
        if _mlf_enabled:
            try:
                _q = self.alpaca.get_latest_quote(plan.symbol)
                _bid = float((_q or {}).get('bid_price', 0) or 0)
                _ask = float((_q or {}).get('ask_price', 0) or 0)
            except Exception as _quote_err:
                logger.warning(
                    f"{plan.symbol}: pre-flight quote fetch failed: "
                    f"{_quote_err} — proceeding with stop-limit "
                    f"(defensive fallback)"
                )
                _bid = _ask = 0.0

            if _bid > 0 and _bid >= stop_price:
                # Whole spread already above the breakout level — confirmed.
                logger.info(
                    f"{plan.symbol}: STOP ALREADY TRIGGERED "
                    f"(bid ${_bid:.2f} >= stop ${stop_price:.2f}) — "
                    f"submitting as marketable LIMIT @ ${limit_price:.2f} "
                    f"to avoid Alpaca live rejection"
                )
                try:
                    order = self.alpaca.submit_limit_buy_order(
                        symbol=plan.symbol,
                        qty=plan.shares,
                        limit_price=limit_price,
                    )
                    _used_marketable_limit = True
                except Exception as e:
                    logger.error(
                        f"{plan.symbol}: marketable-limit fallback failed: {e}"
                    )
                    return None
            elif _ask > 0 and _ask >= stop_price:
                # Spread straddles the breakout level (bid < stop <= ask). A
                # native stop here is immediately marketable → Alpaca rejects
                # it. Re-bump the stop just above the ask so it stays a real
                # stop that only fills on a genuine upward print.
                _new_stop = round(_ask + _rebump_buffer, 2)
                if _new_stop > limit_price:
                    # Ask has already run past our max fill price — the
                    # breakout is too extended. Don't chase.
                    logger.warning(
                        f"{plan.symbol}: BUY-STOP SKIPPED — breakout "
                        f"extended past limit (ask ${_ask:.2f} + buffer "
                        f"${_rebump_buffer:.2f} = ${_new_stop:.2f} > limit "
                        f"${limit_price:.2f}). Not chasing."
                    )
                    return None
                logger.info(
                    f"{plan.symbol}: STOP STRADDLED BY SPREAD "
                    f"(bid ${_bid:.2f} < stop ${stop_price:.2f} <= ask "
                    f"${_ask:.2f}) — re-bumping stop ${stop_price:.2f} → "
                    f"${_new_stop:.2f} (limit ${limit_price:.2f} unchanged) "
                    f"to avoid Alpaca live rejection"
                )
                try:
                    order = self.alpaca.submit_stop_limit_order(
                        symbol=plan.symbol,
                        qty=plan.shares,
                        side='buy',
                        stop_price=_new_stop,
                        limit_price=limit_price,
                    )
                    _used_stop_rebump = True
                    _submitted_stop = _new_stop
                except Exception as e:
                    logger.error(
                        f"{plan.symbol}: stop-rebump submission failed: {e}"
                    )
                    return None

        if order is None:
            try:
                order = self.alpaca.submit_stop_limit_order(
                    symbol=plan.symbol,
                    qty=plan.shares,
                    side='buy',
                    stop_price=stop_price,
                    limit_price=limit_price,
                )
            except Exception as e:
                logger.error(f"{plan.symbol}: Buy-stop order submission failed: {e}")
                return None

        if order is None:
            logger.error(f"{plan.symbol}: Buy-stop order returned None")
            return None

        order_id = order.get('id', '')
        order_status = order.get('status', 'unknown')

        logger.info(
            f"{plan.symbol}: Buy-stop order submitted (simple) — "
            f"ID: {order_id}, status: {order_status}"
        )

        # Save trade record to database (same fields as bracket order)
        pattern_data = json.dumps({
            'pole_start_idx': plan.pattern.pole_start_idx,
            'pole_end_idx': plan.pattern.pole_end_idx,
            'flag_start_idx': plan.pattern.flag_start_idx,
            'flag_end_idx': plan.pattern.flag_end_idx,
            'pole_low': plan.pattern.pole_low,
            'pole_high': plan.pattern.pole_high,
            'pole_height': plan.pattern.pole_height,
            'pole_gain_pct': plan.pattern.pole_gain_pct,
            'flag_low': plan.pattern.flag_low,
            'flag_high': plan.pattern.flag_high,
            'retracement_pct': plan.pattern.retracement_pct,
            'pullback_candle_count': plan.pattern.pullback_candle_count,
            'avg_pole_volume': plan.pattern.avg_pole_volume,
            'avg_flag_volume': plan.pattern.avg_flag_volume,
            'breakout_level': plan.pattern.breakout_level,
        })

        now = datetime.now(timezone.utc)
        trade_record = {
            'trade_date': date.today().isoformat(),
            'symbol': plan.symbol,
            'side': 'buy',
            'entry_price': plan.entry_price,
            'stop_loss_price': plan.stop_loss_price,
            'take_profit_price': plan.take_profit_price,
            'shares': plan.shares,
            'risk_per_share': plan.risk_per_share,
            'total_risk': plan.total_risk,
            'risk_reward_ratio': plan.risk_reward_ratio,
            'order_id': order_id,
            'order_status': order_status,
            'fill_price': None,
            'filled_at': None,
            'exit_price': None,
            'exit_reason': None,
            'exited_at': None,
            'pnl': None,
            'pnl_pct': None,
            'pattern_data': pattern_data,
            'strategy': 'bull_flag',
            'created_at': now,
            'updated_at': now,
        }

        try:
            trade_id = self.db.save_trade(trade_record)
            logger.info(f"{plan.symbol}: Trade record saved (id={trade_id})")
            # Persist pipeline timing telemetry via UPDATE (save_trade INSERT
            # doesn't cover these columns).
            self._persist_submit_timing(
                trade_id, plan.symbol, 'bull_flag', pipeline_timing,
            )
        except Exception as e:
            logger.error(f"{plan.symbol}: Failed to save trade record: {e}")

        return {
            'order_id': order_id,
            'status': order_status,
            'symbol': plan.symbol,
            'shares': plan.shares,
            # The actually-submitted stop — equals stop_price normally, or the
            # re-bumped ask+buffer when the spread straddled the breakout.
            'stop_price': _submitted_stop,
            'limit_price': limit_price,
            'stop_loss_price': plan.stop_loss_price,
            'take_profit_price': plan.take_profit_price,
            # Tag distinguishes the submit paths so post-hoc DB queries can
            # count fallback / rebump fills vs. baseline. See
            # marketable_limit_fallback in config.yaml.
            'order_type': (
                'marketable_limit_fallback' if _used_marketable_limit
                else 'stop_rebump' if _used_stop_rebump
                else 'stop_simple'
            ),
        }

