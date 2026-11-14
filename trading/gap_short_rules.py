"""Pure rule helpers for the gap-fade short paper instrument (cells 1,773 -> 1,776; spec docs/gap_short_paper_instrument_20261114.md).

No I/O, no broker, no clock: the cron script `scripts/gap_short_paper.py` and the parity test against
`research/signal_search/1776R_build.py` both import these functions, so back-test and live share ONE definition.
ONE deliberate deviation from the back-test: L1 is summed over 04:00-09:19 ET live (PM_END_HM) because Alpaca rejects `opg`
orders after 09:28:00 ET, so the pick cron runs at 09:20; the BT used 04:00-09:25 (BT_PM_END_HM, used by the parity test).
POPULATION = cell 1,428 (research/hod_entry/cell_1428.py): gap >= 5 %, price $3-30, prior-day volume >= 500 K shares, test tickers
excluded. The ONE unavoidable live deviation: the BT gaps at the 09:30 open, live at 09:19 uses the pre-market last trade.
Order of operations is the BACK-TEST's (1776R_build.load_universe/__main__): rank by gap, take the top N, THEN keep
those whose pre-market dollar volume L1 clears the cutoff.
"""
from __future__ import annotations

import math
import re
from datetime import datetime, timedelta
from typing import Iterable, Optional, Sequence
from zoneinfo import ZoneInfo

ET = ZoneInfo("America/New_York")
# FROZEN strategy constants (asserted by tests/test_gap_short_rules.py; changing one needs a PREREG, never a silent edit)
L1_MIN_USD = 7_224_000           # the BT's exact TRAIN top-tercile cutoff (1776R min 7,224,163), fixed
TOP_K = 10
COVER_TIME = "10:30"
NOTIONAL_PER_NAME = 300
MAX_NAMES = 10
GAP_MIN = 0.05                   # cell 1,428: gap >= 5 % (BT: 09:30 open vs prior close; live proxy: 09:19 pre-market last)
PRICE_LO, PRICE_HI = 3.0, 30.0   # cell 1,428: price in [$3, $30] (BT: the open; live: pre-market last)
PRIOR_DAY_VOL_MIN = 500_000      # cell 1,428: single prior-day volume (shares) from daily_bars
GAP_MAX = 2.0                    # drop gap > 200 %
L1_CUTOFF = float(L1_MIN_USD)
MIN_GAP, MAX_GAP = GAP_MIN, GAP_MAX
MIN_PRICE = 1.0                  # prev close floor
TOP_N = TOP_K
NOTIONAL = float(NOTIONAL_PER_NAME)
GROSS_CAP = 3000.0
DAY_KILL_USD = -300.0
WEEK_PAUSE_USD = -300.0
COVER_ASK_BP = 10.0              # buy-to-cover limit = ask + 10 bp
BACKSTOP_MULT = 2.0              # +100 % against us -> cover
FEES_BP, LOCATE_BP, BORROW_BP = 4.0, 9.0, 27.8   # BT cost lines (bp round trip, advisory locate, 100 %/360 borrow)
BT_NET_BP = 145.0                # rebuilt VAL line the weekly scorer compares to
BT_WEEKLY_P10_R = -0.66          # weekly P10 of the BT at the $150 unit
UNIT_RISK_USD = 150.0
EXCLUDE_RE = re.compile(r"^Z[A-Z]ZZT$")   # NASDAQ test tickers
PM_START_HM, PM_END_HM = (4, 0), (9, 19)   # DEVIATION from the BT (09:25): Alpaca rejects opg after 09:28, so pick runs 09:20
BT_PM_END_HM = (9, 25)
OPG_LAST_SUBMIT_HM = (9, 27, 30)


def is_todays_premarket_trade(ts: datetime, day: str) -> bool:
    """True when a latest-trade timestamp falls on `day` (ET date) at or after 04:00 ET (else the name is not a live gapper)."""
    if ts is None:
        return False
    t = ts.astimezone(ET)
    return t.strftime("%Y-%m-%d") == day and (t.hour, t.minute) >= PM_START_HM


def gap_fraction(pm_last: float, prev_close: float) -> Optional[float]:
    """Gap = pre-market last / prev close - 1; None when either input is missing or non-positive."""
    if not pm_last or not prev_close or pm_last <= 0 or prev_close <= 0:
        return None
    return pm_last / prev_close - 1.0


def premarket_dollar_volume(bars: Iterable[dict], end_hm: tuple = PM_END_HM) -> float:
    """L1 = sum(close * volume) over 1-min bars that start >= 04:00 ET and END <= end_hm (09:19 ET live; the BT used 09:25).

    Each bar is a dict with tz-aware `t` (start of the minute), `c`, `v`. The end-of-bar test (start + 1 min) is what the
    back-test used, so a bar starting 09:25 never leaks the 09:25 minute into the sum.
    """
    total = 0.0
    for b in bars:
        t = b["t"].astimezone(ET)
        midnight = t.replace(hour=0, minute=0, second=0, microsecond=0)
        lo = midnight + timedelta(hours=PM_START_HM[0], minutes=PM_START_HM[1])
        hi = midnight + timedelta(hours=end_hm[0], minutes=end_hm[1])
        if t >= lo and t + timedelta(minutes=1) <= hi:
            total += float(b["c"]) * float(b["v"])
    return total


def price_ok(price) -> bool:
    """Cell 1,428 price band: PRICE_LO <= price <= PRICE_HI."""
    return bool(price) and PRICE_LO <= price <= PRICE_HI


def is_excluded_symbol(symbol: str) -> bool:
    """True for NASDAQ test tickers (`^Z[A-Z]ZZT$`)."""
    return bool(EXCLUDE_RE.match(symbol))


def select_picks(cands: Sequence[dict], l1_cutoff: float = L1_CUTOFF, min_gap: Optional[float] = MIN_GAP,
                 max_gap: Optional[float] = MAX_GAP, top_n: int = TOP_N) -> list:
    """Rules 1-3: gap window and symbol exclusion -> rank by gap desc -> top N -> keep L1 >= cutoff.

    `cands` are dicts with `symbol`, `gap` (fraction) and `l1` (None/NaN when unknown -> never picked). min_gap/max_gap None
    disables that bound (the parity test uses None for the floor: the BT ranked a scanner population, not gap >= 10 %).
    Ties on gap break by symbol so the result is deterministic.
    """
    pool = [c for c in cands if c.get("gap") is not None and not is_excluded_symbol(c["symbol"])
            and (min_gap is None or c["gap"] >= min_gap) and (max_gap is None or c["gap"] <= max_gap)]
    pool.sort(key=lambda c: (-c["gap"], c["symbol"]))
    top = pool[:top_n]
    return [c for c in top if c.get("l1") is not None and not math.isnan(c["l1"]) and c["l1"] >= l1_cutoff]


def position_qty(pm_last: float, notional: float = NOTIONAL) -> int:
    """Shares = floor(notional / pre-market last); 0 when the price is unusable or exceeds the notional."""
    if not pm_last or pm_last <= 0:
        return 0
    return int(math.floor(notional / pm_last))


def apply_gross_cap(picks: Sequence[dict], cap: float = GROSS_CAP) -> list:
    """Keep picks in order while cumulative qty * pm_last stays <= cap (each pick needs `qty`, `pm_last`)."""
    out, gross = [], 0.0
    for p in picks:
        v = p["qty"] * p["pm_last"]
        if p["qty"] > 0 and gross + v <= cap:
            out.append(p)
            gross += v
    return out


def cover_limit_price(ask: float, bp: float = COVER_ASK_BP) -> float:
    """Marketable buy-to-cover limit: ask + bp, rounded to the cent (sub-$1 names are not traded)."""
    return round(ask * (1.0 + bp / 1e4), 2)


def backstop_hit(entry_price: float, last_price: float, mult: float = BACKSTOP_MULT) -> bool:
    """True when the short is up `mult` x against us (halt-up backstop)."""
    return bool(entry_price and last_price and last_price >= entry_price * mult)


def net_bp(entry: float, cover: float, locate_bp: float = LOCATE_BP, fees_bp: float = FEES_BP,
           borrow_bp: float = BORROW_BP) -> Optional[float]:
    """Short return in bp net of cost lines: (entry/cover - 1) * 1e4 - fees - locate - borrow (BT definition r_bp - costs)."""
    if not entry or not cover or entry <= 0 or cover <= 0:
        return None
    return (entry / cover - 1.0) * 1e4 - fees_bp - locate_bp - borrow_bp


def day_killed(realized_usd_today: float) -> bool:
    """Paper day kill: realised P&L today at or below -$300."""
    return realized_usd_today <= DAY_KILL_USD


def week_paused(realized_usd_week: float) -> bool:
    """No new picks when the ledger's realised week is at or below -$300."""
    return realized_usd_week <= WEEK_PAUSE_USD


def two_red_weeks_below_p10(weekly_r: Sequence[float], p10: float = BT_WEEKLY_P10_R) -> bool:
    """Pause rule: the last two weekly results are both below the BT weekly P10 (in R at the $150 unit)."""
    return len(weekly_r) >= 2 and weekly_r[-1] < p10 and weekly_r[-2] < p10


def client_order_id(day: str, symbol: str, leg: str = "") -> str:
    """Deterministic order id: gapshort-<day>-<sym> (entry) or gapshort-<leg>-<day>-<sym>."""
    return f"gapshort-{leg + '-' if leg else ''}{day}-{symbol}"
