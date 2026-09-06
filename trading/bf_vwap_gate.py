"""Bull-flag above-VWAP gate — ONE decision for the backtest and the live engine.

Rule (2026-09-06, research/bf_consistency/README.md §6): a bull flag whose
breakout level sits AT or BELOW the session VWAP is not traded. Mechanism:
the flag is a continuation pattern; below VWAP the average participant is
under water and the breakout is a bounce into supply, not a continuation.
Every BF practitioner's rule book says "long only above VWAP" — we had
only an UPPER bound (quality_filter.max_vwap_distance_pct, disabled).

Evidence (regen-7, the honest cache, 896 raw rows): the `vwap_dist <= 0`
bucket is the WORST bucket in BOTH years (2025 mean −0.29R / WR 35%,
2026 −0.41R / WR 31%; n=92/94). On the Stage-2 book it removes 16 of 79
trades (11 losers, 5 winners, net −$40.7K): $107,351 → $148,089; 2026
−$10.9K → +$15.9K; MDD −$27.5K → −$15.9K. Era-consistent on the raw
cache, not fitted on the 79.

The feature is the same number on both sides:
  BT   backtest._compute_qf_features → qf_vwap_dist_pct
       = (breakout_level − vwap_through_setup_bar) / vwap × 100
  live trading_engine: (setup.breakout_level − _compute_vwap(bars,
       flag_end_idx)) / vwap × 100
BT applies the gate at Stage-2 from the cached feature (the cache stays
broad for research); live applies it before conviction/sizing.
"""
import logging
from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple

logger = logging.getLogger(__name__)

FEATURE = 'qf_vwap_dist_pct'


@dataclass(frozen=True)
class VwapGateConfig:
    """`trading.bull_flag.vwap_gate` — keep only setups with
    vwap_dist_pct > min_dist_pct (0.0 = strictly above VWAP)."""
    enabled: bool = False
    min_dist_pct: float = 0.0


DISABLED = VwapGateConfig()


def load_vwap_gate_config(bull_flag_cfg: Optional[dict]) -> VwapGateConfig:
    """Parse `bull_flag.vwap_gate` (missing block → disabled)."""
    vg = (bull_flag_cfg or {}).get('vwap_gate') or {}
    return VwapGateConfig(
        enabled=bool(vg.get('enabled', False)),
        min_dist_pct=float(vg.get('min_dist_pct', 0.0)),
    )


def passes_vwap_gate(vwap_dist_pct: Optional[float],
                     cfg: VwapGateConfig) -> Tuple[bool, str]:
    """The decision. Unknown distance (no VWAP) FAILS OPEN — keep, but say so:
    a missing feature must never silently veto (CLAUDE.md fallback rule)."""
    if not cfg.enabled:
        return True, 'vwap_gate disabled'
    if vwap_dist_pct is None:
        logger.warning("vwap_gate: vwap distance unknown — fail-open (kept)")
        return True, 'vwap unknown (fail-open)'
    if vwap_dist_pct > cfg.min_dist_pct:
        return True, f'vwap_dist {vwap_dist_pct:+.2f}% > {cfg.min_dist_pct:+.2f}%'
    return False, (f'vwap_dist {vwap_dist_pct:+.2f}% <= {cfg.min_dist_pct:+.2f}% '
                   f'(breakout at/below VWAP)')


def _feature_value(trade: dict) -> Optional[float]:
    raw = trade.get(FEATURE)
    if raw in (None, '', 'None'):
        return None
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None


def filter_trades(trades: Iterable[dict], cfg: VwapGateConfig) -> List[dict]:
    """Stage-2: drop cached trades whose setup-time VWAP distance fails the gate."""
    trades = list(trades)
    if not cfg.enabled:
        return trades
    kept, removed = [], 0
    for t in trades:
        keep, _ = passes_vwap_gate(_feature_value(t), cfg)
        if keep:
            kept.append(t)
        else:
            removed += 1
    if removed:
        logger.info(f"VWAP gate: {len(trades)} → {len(kept)} trades "
                    f"({removed} removed at/below VWAP+{cfg.min_dist_pct:.2f}%)")
    return kept
