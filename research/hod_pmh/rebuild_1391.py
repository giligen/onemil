#!/usr/bin/env python3
"""Independent rebuild of cell 1,391 (PMH break, P3 exit) from the PREREG prose only.

P3 exit rule: breakeven lock at +1R, no target, stop, flat at the 15:55 open.

Signal (frozen, per research/hod_exit_lab/PREREG_PMH.md):
  * Universe = the HOD-break universe's symbol-days (features.csv day/symbol pairs).
  * PMH = max(high) of pre-market bars 04:00-09:29 ET (data/cache.db::intraday_bars_1min),
    only if pre-market volume >= 50,000 shares, and only if >= 20 pre-market bars with prints.
  * Break = first regular-session 1-min bar (09:31..11:30 ET, 09:30 excluded) whose CLOSE > PMH.
  * Entry = OPEN of the next bar.
  * Stop = low of the last 15 minutes before the break bar, floored at 1.5% of price.
  * One trade per symbol-day. Flat at the 15:55 open.
  * Cost = measured NBBO spread at the HOD signal minute of the same name-day (nbbo.csv),
    charged as a half-spread on each side (one full spread round trip), expressed in R.

Both bar tables store `t` / `timestamp` as full ISO UTC strings; converted to ET here.
"""
import csv
import math
import os
import sqlite3
import sys
from collections import defaultdict
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

ROOT = "/home/ec2-user/onemil"
CACHE_DB = os.path.join(ROOT, "data/cache.db")
SIP_DB = os.path.join(ROOT, "research/bf_zero/bars_sip.db")
FEATURES = os.path.join(ROOT, "research/bf_zero/causal_filter/features.csv")
NBBO = os.path.join(ROOT, "research/bf_zero/causal_filter/nbbo.csv")
OUT = os.path.join(ROOT, "research/hod_pmh/rebuild_1391_trades.csv")

ET = ZoneInfo("America/New_York")
UTC = ZoneInfo("UTC")

PM_START, PM_END = 4 * 60, 9 * 60 + 29          # 04:00 .. 09:29 ET inclusive
BREAK_FIRST, BREAK_LAST = 9 * 60 + 31, 11 * 60 + 30
FLAT_M = 15 * 60 + 55
PM_MIN_BARS = 20
PM_MIN_VOL = 50_000
STOP_FLOOR_PCT = 0.015
LOOKBACK_MIN = 15


def et_minute(iso_ts):
    """ISO UTC timestamp string -> minutes since midnight ET (and the ET date)."""
    dt = datetime.fromisoformat(iso_ts)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    e = dt.astimezone(ET)
    return e.hour * 60 + e.minute, e.strftime("%Y-%m-%d")


def utc_window(day, m_from, m_to):
    """ET day + minute range -> (iso_lo, iso_hi) UTC strings for a BETWEEN filter."""
    d = datetime.strptime(day, "%Y-%m-%d")
    lo = (d + timedelta(minutes=m_from)).replace(tzinfo=ET).astimezone(UTC)
    hi = (d + timedelta(minutes=m_to)).replace(tzinfo=ET).astimezone(UTC)
    return lo.strftime("%Y-%m-%dT%H:%M:00+00:00"), hi.strftime("%Y-%m-%dT%H:%M:00+00:00")


def load_universe():
    """First HOD signal per (day, symbol): the symbol-day universe + its signal minute."""
    uni = {}
    with open(FEATURES) as fh:
        for row in csv.DictReader(fh):
            key = (row["day"], row["symbol"])
            em = int(row["entry_m"])
            if key not in uni or em < uni[key]:
                uni[key] = em
    return uni


def load_nbbo():
    """(day, symbol) -> mean NBBO spread in dollars at the HOD signal minute."""
    out = {}
    with open(NBBO) as fh:
        for row in csv.DictReader(fh):
            try:
                sp = float(row["spread_mean"])
            except (TypeError, ValueError):
                continue
            if not math.isfinite(sp):
                continue
            key = (row["day"], row["symbol"])
            if key not in out:
                out[key] = sp
    return out


def split_of(day):
    if day < "2026-01-01":
        return "TRAIN"
    if day < "2026-06-01":
        return "VAL"
    return "TEST"


def simulate(bars, entry, stop, r_dollars):
    """Walk bars from the entry bar. Returns (exit_price, exit_minute, reason).

    Conservative intrabar order: the stop in force at the start of the bar is tested
    first (gap-through fills at the open), then the +1R breakeven lock is armed."""
    lock = entry + r_dollars
    cur_stop = stop
    for m, o, h, l, c in bars:
        if m >= FLAT_M:
            return o, m, "flat1555"
        if l <= cur_stop:
            return (o if o <= cur_stop else cur_stop), m, "stop"
        if h >= lock and cur_stop < entry:
            cur_stop = entry
    m, o, h, l, c = bars[-1]
    return c, m, "eod_last_bar"


def main():
    uni = load_universe()
    nbbo = load_nbbo()
    print(f"universe symbol-days: {len(uni)}   nbbo rows: {len(nbbo)}", flush=True)

    cache = sqlite3.connect(f"file:{CACHE_DB}?mode=ro", uri=True)
    sip = sqlite3.connect(f"file:{SIP_DB}?mode=ro", uri=True)

    rows = []
    n_avail_ok = n_vol_ok = n_break = 0
    n_total = 0
    for i, ((day, sym), sig_m) in enumerate(sorted(uni.items())):
        if split_of(day) == "TEST":
            continue
        n_total += 1
        if i % 1000 == 0:
            print(f"  .. {i}/{len(uni)}  {day} {sym}  trades={len(rows)}", flush=True)

        lo, hi = utc_window(day, PM_START, PM_END)
        pm = cache.execute(
            "SELECT timestamp, high, volume FROM intraday_bars_1min "
            "WHERE symbol=? AND timestamp BETWEEN ? AND ? ORDER BY timestamp",
            (sym, lo, hi)).fetchall()
        pm = [(t, h, v) for (t, h, v) in pm if h is not None and v is not None]
        if len(pm) < PM_MIN_BARS:
            continue
        n_avail_ok += 1
        pmvol = sum(v for _, _, v in pm)
        if pmvol < PM_MIN_VOL:
            continue
        n_vol_ok += 1
        pmh = max(h for _, h, _ in pm)

        raw = sip.execute(
            "SELECT t, o, h, l, c FROM bars WHERE symbol=? AND day=? ORDER BY t",
            (sym, day)).fetchall()
        bars = []
        for t, o, h, l, c in raw:
            m, edate = et_minute(t)
            if edate != day:
                continue
            if 9 * 60 + 30 <= m <= 16 * 60:
                bars.append((m, o, h, l, c))
        bars.sort()
        if len(bars) < 2:
            continue

        bidx = None
        for j, (m, o, h, l, c) in enumerate(bars):
            if m < BREAK_FIRST:
                continue
            if m > BREAK_LAST:
                break
            if c > pmh:
                bidx = j
                break
        if bidx is None or bidx + 1 >= len(bars):
            continue
        n_break += 1

        bm = bars[bidx][0]
        entry = bars[bidx + 1][1]
        if entry is None or entry <= 0:
            continue
        window = [b for b in bars if bm - LOOKBACK_MIN <= b[0] <= bm - 1]
        low15 = min(b[3] for b in window) if window else bars[bidx][3]
        r_dollars = max(entry - low15, STOP_FLOOR_PCT * entry)
        stop = entry - r_dollars

        exit_px, exit_m, reason = simulate(bars[bidx + 1:], entry, stop, r_dollars)
        gross = (exit_px - entry) / r_dollars
        sp = nbbo.get((day, sym))
        cost_r = (sp / r_dollars) if sp is not None else None
        net = gross - cost_r if cost_r is not None else None
        rows.append(dict(
            day=day, symbol=sym, split=split_of(day), pmh=round(pmh, 4),
            pm_bars=len(pm), pm_vol=int(pmvol), break_m=bm, entry_m=bars[bidx + 1][0],
            entry=round(entry, 4), stop=round(stop, 4), r_dollars=round(r_dollars, 4),
            r_pct=round(100 * r_dollars / entry, 4), exit_m=exit_m,
            exit=round(exit_px, 4), reason=reason, gross_R=round(gross, 5),
            cost_R=(round(cost_r, 5) if cost_r is not None else ""),
            net_R=(round(net, 5) if net is not None else ""),
            hod_sig_m=sig_m))

    with open(OUT, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    print(f"\nsymbol-days scanned (TRAIN+VAL): {n_total}")
    print(f"availability rail ok (>=20 pm bars): {n_avail_ok} "
          f"({100.0*n_avail_ok/max(n_total,1):.1f}%)")
    print(f"pm volume rail ok (>=50k): {n_vol_ok}")
    print(f"breaks with an entry bar: {n_break}   trades written: {len(rows)}")
    for sp_name in ("TRAIN", "VAL"):
        g = [r for r in rows if r["split"] == sp_name]
        gr = [r["gross_R"] for r in g]
        nr = [r["net_R"] for r in g if r["net_R"] != ""]
        cov = 100.0 * len(nr) / max(len(g), 1)
        print(f"{sp_name}: n={len(g)} mean_gross={sum(gr)/max(len(gr),1):+.4f} "
              f"mean_net={sum(nr)/max(len(nr),1):+.4f} nbbo_cov={cov:.1f}%")
    print(f"written: {OUT}")


if __name__ == "__main__":
    main()
