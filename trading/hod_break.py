"""HOD-break — ONE spec for the backtest and the live engine (research/bf_zero, 2026-09-13).

The book that survived the clean-sheet study (`research/bf_zero/REPORT.md`): a stock that is
already >= 5% above its 9:30 open, with relative volume in a band, consolidates for >= K
one-minute bars all within X% of the running high-of-day, then a bar's high reaches the HOD.
Entry is a capped limit that fills at the NEXT bar's open if that open is at or under
level x (1 + cap) — no chase. Stop = consolidation low. Target = entry + target_r x R as a
resting limit that fills when a bar CLOSES at or above it (no wick fills). Flat at 15:55 ET.

Every function here is pure (arrays in, numbers out) and every quantity is computed on bars
at or before the decision bar. The backtest walks bars through these functions; the live
engine feeds them the same closed bars. Parity is by construction and enforced by
`tests/test_hod_break.py`.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np

OPEN_MINUTE = 570            # 09:30 ET as minutes since midnight
FLAT_MINUTE = 955            # 15:55 ET — no trade survives the close
STOP_FILL_SLIP = 0.001       # a stop is filled 10 bps through (or at a gap-through open)

# Same-clock cumulative-volume checkpoints (minutes since midnight ET) and the market-wide median
# fraction of a day's volume traded by each, measured on the study's 559K symbol-days
# (research/bf_zero/pass2.log). Used by rv_profile when a per-symbol same-clock baseline is absent.
VP_CHECKPOINTS: tuple = (575, 585, 600, 630, 660, 720, 780, 840, 900)
VP_FRACTION: dict = {575: 0.02, 585: 0.051, 600: 0.097, 630: 0.188, 660: 0.269,
                     720: 0.411, 780: 0.529, 840: 0.642, 900: 0.76}


@dataclass(frozen=True)
class HodBreakParams:
    """The knobs — defaults are the study's surviving configuration."""
    consol_bars: int = 5             # K: bars that must sit within consol_pct of the HOD
    consol_pct: float = 0.04         # X: how far below the HOD the consolidation lows may sit
    min_dist_open_pct: float = 5.0   # entry level must be >= this % above the 09:30 open (causal floor)
    rv_lo: float = 1.0               # relative-volume band [rv_lo, rv_hi) at the signal bar
    rv_hi: float = 5.0
    min_r_pct: float = 1.0           # stop distance as % of entry — tighter stops are noise + spread
    cap: float = 0.006               # limit cap above the HOD level (60 bps); no fill above it
    entry_limit_pct: float = 0.0015  # resting-order limit above trigger (15 bps); cell 1,427/1,438 (docs/hod_resting_entry_spec_20260925.md)
    target_r: float = 2.0            # fixed take-profit in R
    max_per_day: int = 8             # first-come cap on fills per session
    max_concurrent: int = 4
    last_entry_minute: int = 930     # no new entries after 15:30 ET
    flat_minute: int = FLAT_MINUTE


@dataclass(frozen=True)
class HodBreakSignal:
    """A confirmed break on a CLOSED bar (index `bar_idx`), before any fill is known."""
    bar_idx: int
    level: float            # the HOD that was broken = the limit's reference
    stop: float             # consolidation low
    dist_open_pct: float
    rv_profile: float


@dataclass(frozen=True)
class HodBreakTrade:
    entry_idx: int
    entry: float
    stop: float
    target: float
    r_per_share: float
    exit_idx: int
    exit_price: float
    reason: str             # 'stop' | 'target' | 'eod'
    rr: float


def profile_fraction(minute: int) -> float:
    """Median fraction of a day's volume traded by `minute` (last checkpoint at or before it)."""
    ck = [c for c in VP_CHECKPOINTS if c <= minute]
    return VP_FRACTION[ck[-1]] if ck else VP_FRACTION[VP_CHECKPOINTS[0]]


def rv_profile(cum_volume: float, adv20: float, minute: int) -> float:
    """Relative volume: cumulative volume so far ÷ (20-day ADV × the fraction of a day normally
    traded by this clock time). >1 means the stock is ahead of its own normal pace."""
    if not adv20 or adv20 <= 0:
        return float('nan')
    return float(cum_volume) / (float(adv20) * profile_fraction(minute))


def consolidation_low(l: Sequence[float], h: Sequence[float], j: int, p: HodBreakParams) -> Optional[float]:
    """If bars j-K+1..j all hold within consol_pct of the running HOD at j, return their min low;
    else None. `j` is the last CLOSED bar before the break bar."""
    if j + 1 < p.consol_bars + 1:
        return None
    hod = float(np.max(h[: j + 1]))
    lows = np.asarray(l[j + 1 - p.consol_bars: j + 1], dtype=float)
    lo = float(lows.min())
    if lo >= hod * (1.0 - p.consol_pct) and lo < hod:
        return lo
    return None


def arm_state(o: Sequence[float], h: Sequence[float], l: Sequence[float], v: Sequence[float], m: Sequence[int],
              j: int, adv20: float, p: HodBreakParams = HodBreakParams()) -> Optional[dict]:
    """Resting buy-stop-limit arming at the close of bar j, for the order that rests through bar j+1
    (docs/hod_resting_entry_spec_20260925.md; cell 1,438, research/hod_entry/causal_arming.py's `arm_state`
    is the research build of this same rule — PARITY enforced by tests/test_hod_resting_entry.py). Uses only
    bars 0..j (closed data) — never anything from bar j+1 itself. Returns
    dict(level, trigger, limit, stop) or None."""
    o = np.asarray(o, dtype=float); h = np.asarray(h, dtype=float); l = np.asarray(l, dtype=float); v = np.asarray(v, dtype=float)
    if int(m[j]) + 1 > p.last_entry_minute:   # RTH minutes are consecutive — next minute is derived, never read from bar j+1
        return None
    stop = consolidation_low(l, h, j, p)
    if stop is None:
        return None
    level = float(np.max(h[: j + 1]))
    if level < float(o[0]) * (1.0 + p.min_dist_open_pct / 100.0):
        return None
    rv = rv_profile(float(np.sum(v[: j + 1])), adv20, int(m[j]))
    if not (p.rv_lo <= rv < p.rv_hi):
        return None
    trigger = round(level + 0.01, 6)
    return dict(level=level, trigger=trigger, limit=round(level * (1.0 + p.entry_limit_pct), 6), stop=stop)


def resting_entry_fill(ask: float, arm: dict) -> Optional[float]:
    """Fill for a resting buy-stop-limit once a print/bar reaches `arm['trigger']`: the ask if it is at or
    under `arm['limit']`, else no fill (no chase — the order simply keeps resting/re-arming)."""
    return float(ask) if float(ask) <= arm['limit'] + 1e-9 else None


def resting_order_qty(risk_usd: float, arm: dict) -> int:
    """Live order size for a resting arm (docs/hod_live_resting_orders_spec_20260925.md item 1): floor(risk /
    (trigger - stop)) — the SAME formula as `shares_for`, keyed on the order's trigger (the broker's stop price,
    known at arm time) rather than the tape's expected ask (only known after a fill). ONE helper so the live
    engine never re-derives this arithmetic."""
    return shares_for(risk_usd, arm['trigger'], arm['stop'])


def detect(o: Sequence[float], h: Sequence[float], l: Sequence[float], v: Sequence[float], m: Sequence[int],
           adv20: float, p: HodBreakParams = HodBreakParams(), start_idx: int = 0) -> Optional[HodBreakSignal]:
    """First break on or after `start_idx`: bar i whose high reaches the HOD of bars[:i] after a
    valid consolidation ending at i-1, with the causal filters. Bars are CLOSED 1-min RTH bars in
    order. Returns the signal or None. Pure; call on every closed bar in live with start_idx=i."""
    h = np.asarray(h, dtype=float); l = np.asarray(l, dtype=float); v = np.asarray(v, dtype=float)
    n = len(h)
    if n < p.consol_bars + 2:
        return None
    o0 = float(o[0])
    hod = np.maximum.accumulate(h)
    cumv = np.cumsum(v)
    for i in range(max(start_idx, p.consol_bars + 1), n):
        if int(m[i]) > p.last_entry_minute:
            return None
        level = float(hod[i - 1])
        if h[i] < level:
            continue
        stop = consolidation_low(l, h, i - 1, p)
        if stop is None:
            continue
        dist = (level / o0 - 1.0) * 100.0
        if dist < p.min_dist_open_pct:
            continue
        rv = rv_profile(float(cumv[i]), adv20, int(m[i]))
        if not (p.rv_lo <= rv < p.rv_hi):
            continue
        return HodBreakSignal(bar_idx=i, level=level, stop=stop, dist_open_pct=dist, rv_profile=rv)
    return None


def entry_fill(next_open: float, level: float, p: HodBreakParams) -> Optional[float]:
    """Capped limit: fills at the next bar's open iff it is at or under level x (1 + cap)."""
    limit = level * (1.0 + p.cap)
    return float(next_open) if next_open <= limit else None


def walk_exit(o: Sequence[float], h: Sequence[float], l: Sequence[float], c: Sequence[float], m: Sequence[int],
              entry_idx: int, entry: float, stop: float, target: float, p: HodBreakParams) -> tuple:
    """From the bar AFTER entry: stop (low <= stop) fills at min(stop, open) x (1 - slip); target fills
    when a bar closes at/above it; flat at the first bar >= flat_minute. Stop wins a same-bar tie.
    Returns (exit_idx, exit_price, reason)."""
    o = np.asarray(o, dtype=float); h = np.asarray(h, dtype=float); l = np.asarray(l, dtype=float); c = np.asarray(c, dtype=float)
    for k in range(entry_idx + 1, len(o)):
        if int(m[k]) >= p.flat_minute:
            return k, float(o[k]), 'eod'
        if l[k] <= stop:
            return k, float(min(stop, o[k]) * (1.0 - STOP_FILL_SLIP)), 'stop'
        if c[k] >= target:
            return k, float(target), 'target'
    return len(o) - 1, float(c[-1]), 'eod'


def simulate(o, h, l, c, v, m, adv20: float, p: HodBreakParams = HodBreakParams()) -> Optional[HodBreakTrade]:
    """The whole trade on one symbol-day of closed bars: detect → next-open capped fill → walk."""
    sig = detect(o, h, l, v, m, adv20, p)
    if sig is None or sig.bar_idx + 1 >= len(o):
        return None
    entry = entry_fill(float(o[sig.bar_idx + 1]), sig.level, p)
    if entry is None or sig.stop >= entry:
        return None
    r = entry - sig.stop
    if r / entry * 100.0 < p.min_r_pct:
        return None
    target = entry + p.target_r * r
    k, px, why = walk_exit(o, h, l, c, m, sig.bar_idx + 1, entry, sig.stop, target, p)
    return HodBreakTrade(entry_idx=sig.bar_idx + 1, entry=entry, stop=sig.stop, target=target, r_per_share=r,
                         exit_idx=k, exit_price=px, reason=why, rr=(px - entry) / r)


def run_book(rows, max_per_day: int, max_concurrent: int):
    """THE executable-book rule, ONE copy for the backtest (spec_sim/book_sim), the EOD check and the live engine's
    semantics. `rows`: iterable of tuples (day, entry_m, exit_m, symbol, payload...) — a trade per symbol-day whose
    entry fills at the open of bar `entry_m` and exits during bar `exit_m`. First-come by entry minute, ties broken by
    SYMBOL (alphabetical: the live engine evaluates a minute's bars in symbol order), at most `max_per_day` fills a
    day, at most `max_concurrent` open at once. CAUSAL freeing: a slot is free for an entry at bar k only if the exit
    happened on a bar STRICTLY BEFORE k (exit_m < entry_m). An exit during bar k itself is after that bar's open — the
    old `>` rule freed it in hindsight (found 2026-09-15: 5% of the study's trades were admitted that way; live
    cannot). Returns the taken rows in order."""
    taken = []
    by_day: dict = {}
    for r in rows:
        by_day.setdefault(r[0], []).append(r)
    for day in sorted(by_day):
        open_exits = []; n_day = 0
        for r in sorted(by_day[day], key=lambda x: (int(x[1]), str(x[3]))):
            entry_m, exit_m = int(r[1]), int(r[2])
            open_exits = [e for e in open_exits if e >= entry_m]
            if n_day >= max_per_day or len(open_exits) >= max_concurrent:
                continue
            taken.append(r); open_exits.append(exit_m); n_day += 1
    return taken


def shares_for(risk_usd: float, entry: float, stop: float) -> int:
    """Position size from dollar risk; 0 when the stop is not below the entry."""
    r = entry - stop
    return int(risk_usd / r) if r > 0 else 0
