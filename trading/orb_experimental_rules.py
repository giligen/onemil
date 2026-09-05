"""ORB experimental entry/exit rules — the 2026-09-05 signal study.

Pure functions behind env flags, default OFF (the pipeline is byte-identical
with every flag unset — tests/test_orb_experimental_rules.py pins it). Each
rule cites its source in research/orb_entry_signals_web_sweep_sep2026.md and
is evaluated by the pre-registered protocol in
research/orb_signal_study/DESIGN.md. Nothing here is live; a rule that
PROPOSES graduates to its own shared module + parity test before any ship.

Flags (study_orb_pipeline_static_lock.py reads them):
  ORB_EXP_RVOL_VETO=<t>      C1a  pre-ranking: drop candidates with rvol_open5 < t (NaN kept)
  ORB_EXP_RVOL_RANK=1        C1b  pre-ranking: order the day's candidates by rvol_open5 desc first
  ORB_EXP_RCP_GATE=pre|post  C2   range-candle direction gate, pre-ranking or post-selection (no refill)
  ORB_EXP_RCP_FORM=green|upper    green: range_return_pct > 0 ; upper: range_close_position >= 0.5
  ORB_EXP_RATR_MIN / _MAX    C5   pre-ranking veto on range/ATR14 (NaN kept)
  ORB_EXP_MID_KILL=1         C3   exit: first closed 1-min bar with close < range midpoint
                                  before +0.5R was touched -> exit next open
  ORB_EXP_REARM=1            C4   entry: one re-arm of the stop-limit after a tag/mid-kill exit
                                  inside the 60-min window
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

import pandas as pd


def _env(name: str) -> str:
    return (os.environ.get(name) or '').strip()


def _env_on(name: str) -> bool:
    return _env(name).lower() in ('1', 'true', 'yes', 'on')


@dataclass(frozen=True)
class ExpFlags:
    """Snapshot of the study flags (read once per process)."""
    rvol_veto: Optional[float]
    rvol_rank: bool
    rcp_gate: Optional[str]       # 'pre' | 'post' | None
    rcp_form: str                 # 'green' | 'upper'
    ratr_min: Optional[float]
    ratr_max: Optional[float]
    mid_kill: bool
    rearm: bool

    @property
    def any_on(self) -> bool:
        return any([self.rvol_veto is not None, self.rvol_rank, self.rcp_gate,
                    self.ratr_min is not None, self.ratr_max is not None,
                    self.mid_kill, self.rearm])

    def describe(self) -> str:
        return (f"rvol_veto={self.rvol_veto} rvol_rank={self.rvol_rank} rcp_gate={self.rcp_gate}"
                f"/{self.rcp_form} ratr=[{self.ratr_min},{self.ratr_max}] mid_kill={self.mid_kill} "
                f"rearm={self.rearm}")


def load_flags() -> ExpFlags:
    """Read ORB_EXP_* from the environment; anything unset = OFF."""
    def _f(name):
        v = _env(name)
        return float(v) if v else None
    gate = _env('ORB_EXP_RCP_GATE').lower() or None
    if gate not in (None, 'pre', 'post'):
        raise ValueError(f"ORB_EXP_RCP_GATE must be pre|post, got {gate!r}")
    form = _env('ORB_EXP_RCP_FORM').lower() or 'green'
    if form not in ('green', 'upper'):
        raise ValueError(f"ORB_EXP_RCP_FORM must be green|upper, got {form!r}")
    return ExpFlags(rvol_veto=_f('ORB_EXP_RVOL_VETO'), rvol_rank=_env_on('ORB_EXP_RVOL_RANK'),
                    rcp_gate=gate, rcp_form=form, ratr_min=_f('ORB_EXP_RATR_MIN'),
                    ratr_max=_f('ORB_EXP_RATR_MAX'), mid_kill=_env_on('ORB_EXP_MID_KILL'),
                    rearm=_env_on('ORB_EXP_REARM'))


# ---------------------------------------------------------------------------
# C1 — opening relative volume (Zarattini/Barbon/Aziz "stocks in play")
# ---------------------------------------------------------------------------

def rvol_keep_mask(rvol: pd.Series, threshold: float) -> pd.Series:
    """True = keep. NaN rvol (no 14-day history) is kept — fail-open, like
    every other ORB gate; the caller counts the NaNs."""
    return rvol.isna() | (rvol >= threshold)


def rvol_rank_key(rvol: pd.Series) -> pd.Series:
    """Sort key for 'top-N by RVOL' (descending; NaN last)."""
    return (-rvol).fillna(float('inf'))


# ---------------------------------------------------------------------------
# C2 — range-candle direction (TradingStats: the strongest single filter)
# ---------------------------------------------------------------------------

def range_direction_keep_mask(df: pd.DataFrame, form: str) -> pd.Series:
    """True = keep. green: the 5-min range closed above its open
    (range_return_pct > 0); upper: it closed in its upper half
    (range_close_position >= 0.5). NaN feature -> kept (fail-open)."""
    if form == 'green':
        col = df['range_return_pct']
        return col.isna() | (col > 0)
    if form == 'upper':
        col = df['range_close_position']
        return col.isna() | (col >= 0.5)
    raise ValueError(form)


# ---------------------------------------------------------------------------
# C5 — range width vs ATR14 tier
# ---------------------------------------------------------------------------

def ratr_keep_mask(ratr: pd.Series, lo: Optional[float], hi: Optional[float]) -> pd.Series:
    """True = keep rows whose range/ATR14 is inside [lo, hi]; NaN kept."""
    keep = pd.Series(True, index=ratr.index)
    if lo is not None:
        keep &= ratr.isna() | (ratr >= lo)
    if hi is not None:
        keep &= ratr.isna() | (ratr <= hi)
    return keep


def ratr_tier(ratr: float) -> str:
    """TradingStats tiers: narrow < 0.3, normal 0.3-0.6, wide > 0.6 (NaN -> 'unknown')."""
    if ratr is None or pd.isna(ratr):
        return 'unknown'
    return 'narrow' if ratr < 0.3 else ('normal' if ratr <= 0.6 else 'wide')


# ---------------------------------------------------------------------------
# C3 — midpoint-reversal kill (NQ MBO study: 71% -> 22.7% continuation)
# ---------------------------------------------------------------------------

def midpoint_kill_fires(bar_close: float, range_high: float, range_low: float,
                        entry_price: float, max_high_since_entry: float) -> bool:
    """A CLOSED post-entry bar closing below the range midpoint while the
    trade has not yet touched entry + 0.5R (R = range size) => kill.
    Evaluated on closed bars only (BT and any live twin must agree)."""
    r = range_high - range_low
    mid = (range_high + range_low) / 2.0
    half_r_touched = max_high_since_entry >= entry_price + 0.5 * r
    return (not half_r_touched) and bar_close < mid


# ---------------------------------------------------------------------------
# C4 — one re-arm after an early exit (second-episode breakout)
# ---------------------------------------------------------------------------

REARM_REASONS = ('tag_bb', 'tag_b1', 'mid_kill')


def rearm_allowed(exit_reason: str, exit_ts, window_end_ts, rearmed_already: bool) -> bool:
    """One re-arm, same slot, same day, only after an early tag/mid-kill exit
    and only while the 60-min entry window is still open."""
    if rearmed_already or exit_reason not in REARM_REASONS:
        return False
    return exit_ts is not None and window_end_ts is not None and exit_ts < window_end_ts
