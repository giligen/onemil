"""Bull-flag per-trade risk cap — ONE clamp for the backtest and the live engine.

Why (2026-09-06, research/bf_consistency/README.md §6a/§6d): the sizing
stack multiplies base risk by conviction × risk-tier × MACD-zone × regime
(observed per-trade risk on the honest book: median 1.8×, max 4× base).
Those multipliers are the fitted layers; in 2026 they amplified losses.
Bounding the FINAL risk at `max_risk_mult × risk_per_trade` keeps every
multiplier's ordering (bigger conviction still sizes bigger, up to the
cap) while bounding what one trade can lose — the month-variance lever.

Contract: cap_usd = risk_per_trade × max_risk_mult; shares are clamped
to floor(cap_usd / risk_per_share) AFTER every multiplier and BEFORE the
buying-power ceiling. Live: `TradingEngine._apply_risk_cap`. BT Stage-2:
`batch_backtest.filter_bull_flag_trades` (after the regime multiplier,
pnl rescaled by the same share ratio — the cache's pnl is linear in
shares). Config: `trading.risk_cap.{enabled, max_risk_mult}`, default OFF.
"""
import logging
from dataclasses import dataclass
from typing import Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RiskCapConfig:
    enabled: bool = False
    max_risk_mult: float = 2.0


DISABLED = RiskCapConfig()


def load_risk_cap_config(trading_cfg: Optional[dict]) -> RiskCapConfig:
    """Parse `trading.risk_cap` (missing → disabled). max_risk_mult must be > 0."""
    rc = (trading_cfg or {}).get('risk_cap') or {}
    mult = float(rc.get('max_risk_mult', 2.0))
    enabled = bool(rc.get('enabled', False))
    if enabled and mult <= 0:
        raise ValueError(f"trading.risk_cap.max_risk_mult must be > 0, got {mult}")
    return RiskCapConfig(enabled=enabled, max_risk_mult=mult)


def cap_usd(cfg: RiskCapConfig, risk_per_trade: float) -> Optional[float]:
    """Dollar cap for one trade, or None when the cap is off/undefined."""
    if not cfg.enabled or risk_per_trade <= 0:
        return None
    return float(risk_per_trade) * cfg.max_risk_mult


def capped_shares(shares: int, risk_per_share: float,
                  cap: Optional[float]) -> Tuple[int, float]:
    """Clamp `shares` so shares × risk_per_share <= cap.

    Returns (new_shares, scale) with scale = new/old (1.0 when untouched).
    A position that is already at/under the cap, a disabled cap, or a
    degenerate risk_per_share (<= 0) is returned unchanged. Never returns
    0 for a positive input (floor of 1 share, matching the sizing chain).
    """
    if cap is None or shares <= 0 or risk_per_share <= 0:
        return int(shares), 1.0
    risk = shares * risk_per_share
    if risk <= cap:
        return int(shares), 1.0
    new = max(1, int(cap // risk_per_share))
    return new, new / shares
