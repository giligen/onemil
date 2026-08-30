"""Ignition strategy rules — SINGLE SOURCE OF TRUTH (2026-07-24).

Born from the 7/24 parity audit: the shadow and the BT replay each
carried their own copies of the gates and drifted (shadow floored on
sighting price vs book's day-open, counted catalyst cohort over raw
sightings vs book's triggers, had no day-dollar rule, gated on
chg-at-sighting vs book's level-crossed). Every gate the validated
book applies lives HERE and both consumers import it:
  - trading/ignition_shadow.py   (live S1 measurement)
  - research/scripts/ignition_bt_replay.py (nightly/on-demand replay)

Gate provenance — extracted from the research artifacts and verified
against the 1,331-trade book (all values audited 2026-07-24):
  universe: day_open >= $2.00 (book open min 2.0), TRUE open gap < 5%
    (universe gap max 4.9999 — ORB disjointness), day dollar volume
    >= $2M (universe dollar min 2,000,069 — EOD figure, BT-ONLY
    LOOKAHEAD: live cannot know it at 9:35; the live proxy is the
    participation cap + pos floor, and the nightly replay applies the
    real gate so the parity line quantifies the residual gap)
  trigger:  first 1-min bar HIGH >= day_open*1.10, minute 575..630
  entry:    next bar open * 1.003; CHASE GUARD entry <= level*1.05
            (book max entry/level = 1.0030)
  stop:     min(pre-30min low, entry*0.99); R >= 5% of entry
            (book R min 5.04)
  sizing:   min($3,000 / R%, $25,000, 15% x next-bar dollar volume);
            skip if < $2,000 (book pos min 2,022)
  exits:    static lock ARM +1.75R -> stop to +0.5R; 15:45 flat
  catalyst: own-ticker news OR >= 2 same-day TRIGGERS sharing the
            underlying anchor (book `uc` counts candidate triggers,
            NOT raw sightings)
"""
from __future__ import annotations

from typing import Optional

PRICE_FLOOR = 2.0            # on DAY OPEN (not sighting price)
OPEN_GAP_MAX_PCT = 5.0       # ORB disjointness, true open vs prev close
DAY_DOLLAR_MIN = 2_000_000   # BT-only (EOD lookahead) — see docstring
TRIGGER_PCT = 10.0
TRIGGER_MIN_START = 575      # 9:35 ET
TRIGGER_MIN_END = 630        # 10:30 ET
CHASE_MAX_RATIO = 1.05       # entry <= level * this
ENTRY_SLIP = 1.003
PRE_BARS_MIN = 5
R_MIN_PCT = 5.0
RISK_USD = 3000.0
POS_CAP_USD = 25000.0
PARTICIPATION = 0.15
POS_MIN_USD = 2000.0
ARM_R = 1.75
LOCK_R = 0.5
EOD_FLAT_MIN = 945           # 15:45 ET
MIN_COHORT = 2
FRICTION_BPS = 0.0012        # participation-scaled slip adder


def universe_reject(day_open: float,
                    prev_close: Optional[float]) -> Optional[str]:
    """Universe gates computable live at trigger time. Returns a skip
    reason or None. (day-dollar is deliberately NOT here — lookahead.)"""
    if day_open < PRICE_FLOOR:
        return 'skip_price_floor'
    if prev_close and prev_close > 0:
        gap = (day_open - prev_close) / prev_close * 100.0
        if gap >= OPEN_GAP_MAX_PCT:
            return 'skip_gap_orb_territory'
    return None


def level(day_open: float) -> float:
    return day_open * (1 + TRIGGER_PCT / 100.0)


def level_crossed(window_highs, day_open: float) -> bool:
    """The book's trigger: any bar HIGH in the 9:35-10:30 window touched
    the +10% level — NOT price-at-sighting (a pullback after the cross
    still counts; 7/21 BIYA parity miss)."""
    lv = level(day_open)
    return any(h >= lv for h in window_highs)


def chase_reject(entry: float, day_open: float) -> bool:
    return entry > level(day_open) * CHASE_MAX_RATIO


def r_pct_from_stop(entry: float, stop: float) -> float:
    return (entry - stop) / entry * 100.0


def stop_from_pre_lows(pre_low: float, entry: float) -> float:
    return min(pre_low, entry * 0.99)


def position_usd(r_pct: float, bar_dollar_vol: float) -> float:
    """BT sizing: risk-parity capped by account cap AND participation."""
    pos = min(RISK_USD / (r_pct / 100.0), POS_CAP_USD)
    return min(pos, PARTICIPATION * bar_dollar_vol)


def position_reject(pos_usd: float) -> bool:
    return pos_usd < POS_MIN_USD


def catalyst_confirmed(has_news: Optional[bool], anchor: Optional[str],
                       trigger_cohort: int) -> bool:
    """news True confirms; else >= MIN_COHORT same-day TRIGGERS sharing
    the anchor. has_news None (fetch failed) never confirms by itself —
    complex confirmation is still available."""
    if has_news is True:
        return True
    return bool(anchor) and trigger_cohort >= MIN_COHORT


def trigger_entry_stop(g, day_open: float) -> dict:
    """Reconstruct the BT trigger mechanics from an intraday df `g`
    (columns m/open/high/low/close/volume, sorted, index reset, from
    the 9:30 open). Returns a dict with either {'reject': reason} or
    {'trigger_m','entry','stop','r_pct','next_idx','bar_dollar'}.

    THE single source of trigger truth (2026-08-14): the shadow used to
    compute chase/stop/R from the scanner SIGHTING price/minute, which
    mis-evaluated late-sighted movers (8/13 CRWU/CWVX chase-skipped,
    SMCL/SMCX r-too-small — 4 BT-kept trades the shadow missed; they
    happened to sum −$4,896 that day, but the bug is DECISION drift, not
    the day's sign — 8/14 audit corrected the earlier "+$8K monsters"
    narrative). The BT
    keys everything to the ACTUAL trigger bar (first window bar whose
    high crossed the +10% level), not to when the scanner noticed."""
    lvl = level(day_open)
    trig = g[(g['high'] >= lvl) & (g['m'] >= TRIGGER_MIN_START)
             & (g['m'] <= TRIGGER_MIN_END)]
    if trig.empty:
        return {'reject': 'skip_level_not_crossed'}
    ti = trig.index[0]
    nxt = g[g.index > ti]
    if nxt.empty:
        return {'reject': 'no_next_bar'}
    nb = nxt.iloc[0]
    entry = float(nb['open']) * ENTRY_SLIP
    if chase_reject(entry, day_open):
        return {'reject': 'skip_chase_guard'}
    pre = g[(g['m'] >= g.loc[ti, 'm'] - 30) & (g['m'] < g.loc[ti, 'm'])]
    if len(pre) < PRE_BARS_MIN:
        return {'reject': 'skip_pre_bars'}
    stop = stop_from_pre_lows(float(pre['low'].min()), entry)
    rp = r_pct_from_stop(entry, stop)
    if rp < R_MIN_PCT:
        return {'reject': 'skip_r_too_small'}
    # bar_dollar for the participation/illiquidity gate: use the TRIGGER
    # bar's volume — complete when the entry bar opens. 8/30 audit: this
    # used the ENTRY bar's full volume, unknowable at its open (lookahead
    # selecting the book via skip_illiquid; same class the at-fill gates
    # already guard against at :205-216).
    return {'trigger_m': int(g.loc[ti, 'm']), 'entry': entry,
            'stop': stop, 'r_pct': rp, 'next_idx': nb.name,
            'bar_dollar': float(g.loc[ti, 'volume']) * entry}


def structure_gates_at_fill(bars_to_fill, fill_price: float) -> dict:
    """AT-FILL structure validation for a pre-staged stop-limit fill
    (prestage P0-1/P0-5, 2026-08-22 — SINGLE SOURCE shared by
    trading/ignition_prestage.py and any BT/replay consumer).

    A staged order fills on EVERY universe crosser; the book keeps only
    structure-passers. This helper re-runs the SAME gates the book runs
    (`trigger_entry_stop` mechanics), keyed to the ACTUAL fill:
      - trigger bar = first window bar whose high crossed the +10% level
        (no such bar => 'stage_fill_no_trigger': odd-print/pre-window
        election — §B7 flag class)
      - chase guard on the FILL price (fill <= level * CHASE_MAX_RATIO)
      - pre-bars count >= PRE_BARS_MIN in the 30min before the trigger
      - stop = min(pre-30min low, fill*0.99); R >= R_MIN_PCT
      - participation sizing floor (POS_MIN_USD) from the trigger's
        next-bar dollar volume when available

    Args:
        bars_to_fill: intraday df with columns m/open/high/low/close/
            volume, sorted by m, index reset, FROM the 9:30 open UP TO
            (and including) the fill minute's bar.
        fill_price: actual broker fill price.

    Returns:
        {'reject': reason} on any gate failure, else
        {'ok': True, 'stop': float, 'r_pct': float, 'trigger_m': int,
         'level': float, 'pos_cap_usd': float}. Catalyst state is NOT
        checked here — the caller owns news/anchor-cohort state.
    """
    g = bars_to_fill
    if g is None or len(g) < 2:
        return {'reject': 'no_bars'}
    day_open = float(g.iloc[0]['open'])
    if day_open <= 0:
        return {'reject': 'no_bars'}
    lvl = level(day_open)
    if chase_reject(fill_price, day_open):
        return {'reject': 'skip_chase_guard'}
    trig = g[(g['high'] >= lvl) & (g['m'] >= TRIGGER_MIN_START)
             & (g['m'] <= TRIGGER_MIN_END)]
    if trig.empty:
        return {'reject': 'stage_fill_no_trigger'}
    ti = trig.index[0]
    trig_m = int(g.loc[ti, 'm'])
    pre = g[(g['m'] >= trig_m - 30) & (g['m'] < trig_m)]
    if len(pre) < PRE_BARS_MIN:
        return {'reject': 'skip_pre_bars'}
    stop = stop_from_pre_lows(float(pre['low'].min()), fill_price)
    rp = r_pct_from_stop(fill_price, stop)
    if rp < R_MIN_PCT:
        return {'reject': 'skip_r_too_small'}
    # participation floor: book proxy for the EOD day-dollar gate. Uses
    # the bar AFTER the trigger bar when present (BT convention); when
    # the fill IS the trigger bar's minute (no next bar yet) the check
    # is skipped — the caller's sizing is already frozen at $risk.
    nxt = g[g.index > ti]
    pos_cap = float('inf')
    if not nxt.empty:
        bar_dollar = float(nxt.iloc[0]['volume']) * fill_price
        pos = position_usd(rp, bar_dollar)
        pos_cap = PARTICIPATION * bar_dollar
        if position_reject(pos):
            return {'reject': 'skip_illiquid'}
    return {'ok': True, 'stop': stop, 'r_pct': rp, 'trigger_m': trig_m,
            'level': lvl, 'pos_cap_usd': pos_cap}


def resim_exit(bars, entry: float, stop: float, entry_min: int):
    """Harness exit physics on a df with columns m/open/high/low/close.
    Conservative: stop before arm; gap-down fills at min(stop, open)."""
    cur = stop
    armed = False
    R = entry - stop
    post = bars[bars['m'] > entry_min]
    for _, r in post.iterrows():
        if r['m'] >= EOD_FLAT_MIN:
            return (r['open'] - entry) / R, 'eod'
        if r['low'] <= cur:
            fill = min(cur, r['open'])
            return (fill * 0.999 - entry) / R, 'lock' if armed else 'stop'
        if not armed and r['high'] >= entry + ARM_R * R:
            armed = True
            cur = entry + LOCK_R * R
    if len(post):
        return (post.iloc[-1]['close'] - entry) / R, 'eod'
    return 0.0, 'none'
