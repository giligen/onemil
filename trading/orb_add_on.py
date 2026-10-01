"""ORB position ADD-ON: one more entry after +R (owner 2026-10-01 ask).

Pure sizing/trigger/stop-mode module — no I/O, no broker calls, no logging.
Every decision here is a plain function of its inputs so BT and live can
call the EXACT same code (CLAUDE.md "ONE spec ... mechanism + evidence +
explicit code", parity by construction). All stateful work (submitting the
buy, resizing the StopMonitor watch + broker legs, persisting) lives in
trading/orb_engine.py, which calls these functions and nothing else.

Config: `exit.add_on` (production) and the SAME inner keys per add-on pool
via `exit_add_on` in that pool's dict (the per-pool exit-override mechanism
shipped in commit d9b6fbc, trading/orb_addon_gates.resolve_pool_exit_params
— add_on is NOT merged into that function: it is position-management, not
a pool-admission gate, and its values are not all floats).

    enabled: bool = False        # master switch. False everywhere this
                                  # module is consulted means the engine
                                  # takes none of the add-on branches —
                                  # byte-identical pre-existing behaviour.
    at_r: float = 1.0             # R multiple that triggers the add.
    units: float = 1.0            # add_shares = round(base_shares * units);
                                   # 1.0 = the same share count as the base.
    stop_mode: str = 'original'   # 'original' | 'add_breakeven' | 'live_lock'
    max_adds: int = 1             # cap on ADD events per position.
    applies_to: List[str] = ['production']   # pool_id allow-list.

R-unit convention (parity with the existing scale-out/lock mechanisms,
trading/orb_engine.py ORBEngine._maybe_arm_scale / _confirm_fill's
`lock_r_unit=range_size` comment "BT-parity: 1R = range_size, not
risk_per_share"): R here is the OPENING RANGE SIZE (range_high - range_low)
captured once at range-close, NOT entry-to-stop distance (which drifts once
a lock/trail ratchets pos.stop_price). Reusing it means the add trigger
needs no new "original risk" field and cannot be corrupted by a later stop
ratchet.
"""
import logging
from dataclasses import dataclass, field
from datetime import datetime, time as dt_time
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

STOP_MODES: Tuple[str, ...] = ('original', 'add_breakeven', 'live_lock')

DEFAULT_ADD_ON: Dict = {
    'enabled': False,
    'at_r': 1.0,
    'units': 1.0,
    'stop_mode': 'original',
    'max_adds': 1,
    'applies_to': ['production'],
}

# "No add after 15:00 ET" (owner spec, literal) — a rail, not a tunable.
ADD_CUTOFF_ET = dt_time(15, 0)


def resolve_add_on_params(cfg: Optional[Dict]) -> Dict:
    """Resolve one `add_on` dict (production `exit.add_on`, or a pool's
    `exit_add_on` override) into a validated PARTIAL dict.

    Mirrors resolve_pool_exit_params's contract exactly: a key absent from
    `cfg`, or present but invalid, is simply ABSENT from the returned dict
    (never defaulted here) — every consumer's own
    `override.get(key, production_default)` is the ONE place production
    defaults live. cfg=None/{} => {} => every consumer falls back to
    production, byte-identical. An invalid value is dropped with a
    WARNING (never crashes the engine, never arms a nonsense order).
    """
    out: Dict = {}
    if not cfg:
        return out
    if not isinstance(cfg, dict):
        logger.warning("ORB ADD-ON: add_on config is %r, not a dict — ignoring entirely", type(cfg))
        return out

    def _bad(key, raw, why):
        logger.warning("ORB ADD-ON: add_on.%s=%r %s — ignoring, falls back to production", key, raw, why)

    if 'enabled' in cfg and cfg.get('enabled') is not None:
        raw = cfg['enabled']
        if isinstance(raw, bool):
            out['enabled'] = raw
        else:
            _bad('enabled', raw, 'is not a bool')

    if 'at_r' in cfg and cfg.get('at_r') is not None:
        raw = cfg['at_r']
        try:
            val = float(raw)
        except (TypeError, ValueError):
            _bad('at_r', raw, 'is not numeric')
        else:
            if val > 0.0:
                out['at_r'] = val
            else:
                _bad('at_r', raw, 'must be > 0')

    if 'units' in cfg and cfg.get('units') is not None:
        raw = cfg['units']
        try:
            val = float(raw)
        except (TypeError, ValueError):
            _bad('units', raw, 'is not numeric')
        else:
            if val > 0.0:
                out['units'] = val
            else:
                _bad('units', raw, 'must be > 0')

    if 'stop_mode' in cfg and cfg.get('stop_mode') is not None:
        raw = cfg['stop_mode']
        val = str(raw).strip().lower()
        if val in STOP_MODES:
            out['stop_mode'] = val
        else:
            _bad('stop_mode', raw, f'is not one of {STOP_MODES}')

    if 'max_adds' in cfg and cfg.get('max_adds') is not None:
        raw = cfg['max_adds']
        try:
            val = int(raw)
        except (TypeError, ValueError):
            _bad('max_adds', raw, 'is not an int')
        else:
            if val >= 1:
                out['max_adds'] = val
            else:
                _bad('max_adds', raw, 'must be >= 1')

    if 'applies_to' in cfg and cfg.get('applies_to') is not None:
        raw = cfg['applies_to']
        if isinstance(raw, (list, tuple)) and all(isinstance(p, str) for p in raw):
            out['applies_to'] = list(raw)
        else:
            _bad('applies_to', raw, 'is not a list of pool-id strings')

    return out


def add_on_param(override: Optional[Dict], production: Optional[Dict], key: str):
    """ONE place every consumer reads an add_on value: pool override (if it
    set this key) -> production config (if it set this key) -> hard
    default. Mirrors `(pos.pool_exit or {}).get(key, production_default)`
    used throughout orb_engine.py for the sibling pool_exit mechanism."""
    override = override or {}
    production = production or {}
    if key in override:
        return override[key]
    if key in production:
        return production[key]
    return DEFAULT_ADD_ON[key]


def is_past_add_cutoff(now_et: datetime) -> bool:
    """True once the ET wall clock is at/after 15:00 — 'no add after 15:00
    ET' is a literal rail, not derived from any other config."""
    return now_et.time() >= ADD_CUTOFF_ET


def is_pool_eligible(pool_id: str, applies_to: List[str]) -> bool:
    """`applies_to` allow-list gate. production default is ['production']
    so a pool trade (pool_id != 'production') is excluded unless the config
    explicitly lists it."""
    return (pool_id or 'production') in (applies_to or [])


def adds_remaining(add_count: int, max_adds: int) -> bool:
    """True while another add is still allowed on this position."""
    return int(add_count or 0) < int(max_adds or 0)


def has_reached_add_trigger(bar_close_price: float, entry_price: float,
                             range_size: float, at_r: float) -> bool:
    """True the FIRST bar CLOSE at/above entry_price + at_r * range_size.

    range_size <= 0 (degenerate opening range) or at_r <= 0 fails CLOSED —
    never triggers — mirroring _maybe_arm_scale's own degenerate-range
    guard (trading/orb_engine.py)."""
    if range_size <= 0.0 or at_r <= 0.0 or entry_price <= 0.0:
        return False
    return float(bar_close_price) >= entry_price + at_r * range_size


def compute_add_qty(base_shares: int, units: float) -> int:
    """add_shares = round(base_shares * units); never negative, 0 if
    base_shares/units is non-positive (caller must treat 0 as 'no add')."""
    if base_shares is None or base_shares <= 0 or units is None or units <= 0:
        return 0
    return max(0, round(base_shares * units))


def compute_combined_position(base_shares: int, base_price: float,
                               add_shares: int, add_price: float) -> Tuple[int, float]:
    """Weighted-average cost basis across the base fill and the add fill.
    add_shares <= 0 is a no-op (returns base unchanged) — defends against a
    caller mistakenly invoking this for a zero-qty add."""
    if add_shares is None or add_shares <= 0:
        return int(base_shares), float(base_price)
    combined_qty = int(base_shares) + int(add_shares)
    combined_avg = ((base_shares * base_price) + (add_shares * add_price)) / combined_qty
    return combined_qty, combined_avg


def resolve_add_breakeven_leg(stop_mode: str, add_price: float, add_qty: int) -> Optional[Dict]:
    """The separate partial-exit level to track for `stop_mode='add_breakeven'`
    — a stop for JUST the add qty, at the add's own fill price (breakeven
    for that leg only; the base qty keeps its original stop untouched).

    None for 'original' (the position's stop is unchanged over the combined
    qty — no second leg) and 'live_lock' (the position's existing lock rule
    already governs the combined qty via pos.shares/pos.entry_price — no
    new leg needed). Returned as a plain dict (not a dataclass) so it
    round-trips through pattern_data JSON with no extra (de)serialization
    code."""
    if stop_mode == 'add_breakeven' and add_qty and add_qty > 0:
        return {'price': float(add_price), 'qty': int(add_qty), 'done': False}
    return None


def add_breakeven_triggered(bar_close_price: float, leg: Optional[Dict]) -> bool:
    """True the first bar CLOSE at/below the add-breakeven leg's price — a
    STOP, so it fires on a drop back to the add's own entry, never on a
    rise. False once `leg['done']` or leg is None/absent."""
    if not leg or leg.get('done'):
        return False
    return float(bar_close_price) <= float(leg['price'])


def build_add_record(at_r: float, px: float, qty: int, ts: datetime) -> Dict:
    """ONE shape for an `adds` ledger entry — pattern_data.add_on.adds[] —
    so every writer and every reader (restart rehydration, Telegram, tests)
    agrees on the same keys."""
    return {
        'at_r': float(at_r),
        'px': float(px),
        'qty': int(qty),
        'ts': ts.isoformat() if hasattr(ts, 'isoformat') else str(ts),
    }
