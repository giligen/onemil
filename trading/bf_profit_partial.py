"""Bull-flag PROFIT PARTIAL — one spec for backtest and live (2026-09-06).

The practitioner consistency machinery (Warrior/BBT/Bullish Bears, and the
scaled-exit backtests: WR +16pp, MDD −5.7pp, return −3.5pp): sell a fraction
of the position at +N R, move the stop to breakeven, let the remainder run
on the existing trail. Owner 9/6: "shaving the P&L significantly but gaining
WR and consistency is the right move."

Contract (both sides call these functions — parity by construction):
  * level   = r_baseline + r_multiple × r_unit, with r_baseline/r_unit from
              trading/bf_trail.r_baseline_and_unit (the same plan-R knob as
              the trail: trading.trailing_stop.r_basis).
  * trigger = a CLOSED 1-min bar whose HIGH >= level (BT: bar['high'];
              live: StopMonitor.highest_since_entry, which advances only on
              closed bars). Never a tick.
  * fill    = BT: the trigger bar's close through the stop-fill model
              (market sell after the bar closes); live: the existing
              StopMonitor.execute_partial_exit limit-sell path. The
              DECISION is identical; the fill is measured live.
  * after   = fraction of the ACTIVE shares sold; stop raised to the FILL
              price (true breakeven on the remainder) if move_to_breakeven;
              trail/exhaustion continue on the remainder; one partial per
              trade (the exhaustion partial is skipped once this fired).
Config: trading.profit_partial.{enabled, r_multiple, fraction,
move_to_breakeven}. Default OFF — flag flip only after the shadow window.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class ProfitPartialConfig:
    enabled: bool = False
    r_multiple: float = 1.5
    fraction: float = 0.5
    move_to_breakeven: bool = True
    # fill='close': market sell after the trigger bar closes (BT: bar close via
    # the stop-fill model; live: execute_partial_exit). fill='level': a RESTING
    # LIMIT at the level (BT: fills at the level when the bar's high touches;
    # live: a limit order placed at arming — needs the limit-leg plumbing).
    fill: str = 'close'
    # runner_target_r > 0: after the partial the remainder exits at
    # r_baseline + runner_target_r × R via a resting limit (touch fill), else
    # it keeps the unified trail.
    runner_target_r: float = 0.0


DISABLED = ProfitPartialConfig()


def load_profit_partial_config(trading_cfg: Optional[dict]) -> ProfitPartialConfig:
    """From the `trading:` block of config.yaml (missing block => disabled)."""
    pp = (trading_cfg or {}).get('profit_partial') or {}
    cfg = ProfitPartialConfig(
        enabled=bool(pp.get('enabled', False)),
        r_multiple=float(pp.get('r_multiple', 1.5)),
        fraction=float(pp.get('fraction', 0.5)),
        move_to_breakeven=bool(pp.get('move_to_breakeven', True)),
        fill=str(pp.get('fill', 'close')),
        runner_target_r=float(pp.get('runner_target_r', 0.0)),
    )
    if cfg.fill not in ('close', 'level'):
        raise ValueError(f"trading.profit_partial.fill must be close|level, got {cfg.fill!r}")
    if cfg.enabled and not (0.0 < cfg.fraction < 1.0 and cfg.r_multiple > 0):
        raise ValueError(f"trading.profit_partial: fraction must be in (0,1) and r_multiple > 0, got {cfg}")
    return cfg


def partial_level(r_baseline: float, r_unit: float, r_multiple: float) -> float:
    """Price at which the partial fires: r_baseline + r_multiple × R."""
    return r_baseline + r_multiple * r_unit


def profit_partial_fires(bar_high: float, level: float) -> bool:
    """Closed-bar rule: the bar's high reached the level."""
    return level > 0 and bar_high >= level


def partial_shares(active_shares: int, fraction: float) -> int:
    """Shares to sell; 0 when the split would leave no runner or no partial."""
    sell = int(active_shares * fraction)
    if sell < 1 or active_shares - sell < 1:
        return 0
    return sell


def breakeven_stop(current_stop: float, fill_price: float) -> float:
    """Stop after the partial: never below the fill (true breakeven), never lowered."""
    return max(current_stop, fill_price)
