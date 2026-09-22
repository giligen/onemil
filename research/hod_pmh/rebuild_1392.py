#!/usr/bin/env python3
"""Independent rebuild of PMH cell 1,392 (P4) from the PREREG prose only.

P4 = P1 (exit: target +2R / stop / flat at the 15:55 open) restricted to breaks
whose break bar falls in 09:31-10:00 ET.

Signal (frozen, per research/hod_exit_lab/PREREG_PMH.md):
  * Universe  : the 15,656 HOD-universe symbol-days (research/hod_exit_lab/pm_candidates.csv).
  * PMH       : max(high) over pre-market bars 04:00-09:29 ET (data/cache.db::intraday_bars_1min),
                requires total pre-market volume >= 50,000 shares, else no level.
  * Rail      : symbol-day usable only if it has >= 20 pre-market bars with prints.
  * Break     : first regular-session 1-min bar from 09:31 ET (09:30 excluded) up to and
                including 11:30 ET whose CLOSE > PMH  (research/bf_zero/bars_sip.db).
  * Entry     : OPEN of the next bar.
  * Stop      : low of the last 15 minutes before the break bar, floored at 1.5% of price
                (i.e. risk is at least 1.5% of entry).
  * Exit      : stop, or target = entry + 2R, else the OPEN of the 15:55 bar. One trade per
                symbol-day. Stop is resolved before target inside a bar (pessimistic).
  * Cost      : measured NBBO half-spread at the HOD signal minute of the same name-day
                (research/bf_zero/causal_filter/nbbo.csv), charged on entry and on exit:
                    half   = 0.5 * spread_pct / max(r_pct, 0.05)
                    net_R  = gross_R - 2 * half
Splits: TRAIN day < 2026-01-01, VAL 2026-01-01 <= day < 2026-06-01, TEST sealed.

Both bar stores keep the timestamp as an ISO-8601 UTC string; all windows below are
expressed in Eastern time and converted per calendar day (DST aware).
"""

import csv
import sqlite3
import sys
from collections import defaultdict
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

ROOT = "/home/ec2-user/onemil"
CACHE_DB = f"{ROOT}/data/cache.db"
SIP_DB = f"{ROOT}/research/bf_zero/bars_sip.db"
CANDIDATES = f"{ROOT}/research/hod_exit_lab/pm_candidates.csv"
NBBO = f"{ROOT}/research/bf_zero/causal_filter/nbbo.csv"
STOP_WINDOW = (sys.argv[1] if len(sys.argv) > 1 else "all")  # "all" = prose-literal (pre-market
                                                # bars may enter the 15-min lookback); "rth" =
                                                # regular-session bars only.
OUT = f"{ROOT}/research/hod_pmh/rebuild_1392_trades_{STOP_WINDOW}.csv"

ET = ZoneInfo("America/New_York")
UTC = ZoneInfo("UTC")

PM_START, PM_END = 4 * 60, 9 * 60 + 29      # 04:00 .. 09:29 ET inclusive
BREAK_FIRST, BREAK_LAST = 9 * 60 + 31, 11 * 60 + 30
P4_LAST = 10 * 60                            # 09:31-10:00 ET window for cell 1,392
FLAT_MIN = 15 * 60 + 55                      # 15:55 ET
MIN_PM_BARS = 20
MIN_PM_VOL = 50_000
STOP_FLOOR = 0.015
LOOKBACK = 15                                # minutes before the break bar
TARGET_R = 2.0


def et_window_utc(day, start_min, end_min):
    """UTC ISO bounds [lo, hi) for an Eastern-time minute window on a calendar day."""
    d = datetime.strptime(day, "%Y-%m-%d")
    lo = (d + timedelta(minutes=start_min)).replace(tzinfo=ET).astimezone(UTC)
    hi = (d + timedelta(minutes=end_min)).replace(tzinfo=ET).astimezone(UTC)
    return lo.isoformat(), hi.isoformat()


def et_minute(ts_iso):
    """Minutes since Eastern midnight for an ISO UTC timestamp string."""
    e = _to_et(ts_iso)
    return e.hour * 60 + e.minute


def _to_et(ts_iso):
    return datetime.fromisoformat(ts_iso).astimezone(ET)


def load_candidates():
    with open(CANDIDATES) as fh:
        rows = list(csv.DictReader(fh))
    by_day = defaultdict(list)
    for r in rows:
        by_day[r["bar_date"]].append(r["symbol"])
    return by_day, len(rows)


def load_nbbo():
    """(day, symbol) -> mean NBBO spread in dollars at that day's HOD signal minute."""
    out = {}
    with open(NBBO) as fh:
        for r in csv.DictReader(fh):
            try:
                out[(r["day"], r["symbol"])] = float(r["spread_mean"])
            except (ValueError, KeyError):
                continue
    return out


def day_bars(cur_cache, cur_sip, day, symbols):
    """Return {symbol: {et_minute: (o,h,l,c,v)}} for one day, PM bars + RTH bars merged."""
    lo_pm, hi_pm = et_window_utc(day, 0, 9 * 60 + 30)          # 00:00-09:30 ET
    lo_rth, hi_rth = et_window_utc(day, 9 * 60 + 30, 16 * 60)  # 09:30-16:00 ET
    want = set(symbols)
    pm = defaultdict(dict)
    rth = defaultdict(dict)

    # one query per symbol: the PK (symbol, timestamp) is the only usable index here,
    # a bar_date-only predicate would scan the whole 80M-row table for every day.
    for sym in want:
        cur_cache.execute(
            "SELECT timestamp,open,high,low,close,volume FROM intraday_bars_1min "
            "WHERE symbol=? AND timestamp>=? AND timestamp<?",
            (sym, lo_pm, hi_pm),
        )
        for ts, o, h, l, c, v in cur_cache.fetchall():
            pm[sym][et_minute(ts)] = (o, h, l, c, float(v))

    cur_sip.execute(
        "SELECT symbol,t,o,h,l,c,v FROM bars WHERE day=? AND t>=? AND t<?",
        (day, lo_rth, hi_rth),
    )
    for sym, ts, o, h, l, c, v in cur_sip.fetchall():
        if sym in want:
            rth[sym][et_minute(ts)] = (o, h, l, c, float(v or 0.0))

    return pm, rth


def split_of(day):
    if day < "2026-01-01":
        return "TRAIN"
    if day < "2026-06-01":
        return "VAL"
    return "TEST"


def main():
    by_day, n_cand = load_candidates()
    nbbo = load_nbbo()
    con_c = sqlite3.connect(f"file:{CACHE_DB}?mode=ro", uri=True)
    con_s = sqlite3.connect(f"file:{SIP_DB}?mode=ro", uri=True)
    cur_c, cur_s = con_c.cursor(), con_s.cursor()

    trades = []
    n_rail_fail = n_vol_fail = n_no_break = n_late = n_no_entry = 0
    days = sorted(by_day)
    for i, day in enumerate(days):
        pm_all, rth_all = day_bars(cur_c, cur_s, day, by_day[day])
        for sym in by_day[day]:
            pm = pm_all.get(sym, {})
            pm_win = {m: b for m, b in pm.items() if PM_START <= m <= PM_END}
            n_prints = sum(1 for b in pm_win.values() if b[4] > 0)
            if n_prints < MIN_PM_BARS:
                n_rail_fail += 1
                continue
            if sum(b[4] for b in pm_win.values()) < MIN_PM_VOL:
                n_vol_fail += 1
                continue
            pmh = max(b[1] for b in pm_win.values())

            rth = rth_all.get(sym, {})
            if not rth:
                n_no_break += 1
                continue
            brk = None
            for m in range(BREAK_FIRST, BREAK_LAST + 1):
                b = rth.get(m)
                if b is not None and b[3] > pmh:
                    brk = m
                    break
            if brk is None:
                n_no_break += 1
                continue
            if brk > P4_LAST:                      # cell 1,392 condition
                n_late += 1
                continue

            # entry = open of the next bar that exists
            nxt = next((m for m in range(brk + 1, FLAT_MIN + 1) if m in rth), None)
            if nxt is None:
                n_no_entry += 1
                continue
            entry = rth[nxt][0]

            # stop = low of the last 15 minutes before the break bar (PM bars included
            # when the break is early), floored so risk >= 1.5% of entry
            lows = [b[2] for m, b in rth.items() if brk - LOOKBACK <= m < brk]
            if STOP_WINDOW == "all":
                lows += [b[2] for m, b in pm.items() if brk - LOOKBACK <= m < brk]
            raw_stop = min(lows) if lows else entry
            stop = min(raw_stop, entry * (1.0 - STOP_FLOOR))
            risk = entry - stop
            if risk <= 0:
                n_no_entry += 1
                continue
            target = entry + TARGET_R * risk

            exit_px, exit_m, why = None, None, None
            for m in range(nxt, FLAT_MIN):
                b = rth.get(m)
                if b is None:
                    continue
                if b[2] <= stop:                   # stop resolved before target
                    exit_px, exit_m, why = stop, m, "stop"
                    break
                if b[1] >= target:
                    exit_px, exit_m, why = target, m, "target"
                    break
            if exit_px is None:
                flat = next((m for m in range(FLAT_MIN, 16 * 60) if m in rth), None)
                if flat is not None:
                    exit_px, exit_m, why = rth[flat][0], flat, "eod"
                else:
                    last = max(m for m in rth if m < FLAT_MIN)
                    exit_px, exit_m, why = rth[last][3], last, "eod_last"

            gross = (exit_px - entry) / risk
            r_pct = 100.0 * risk / entry
            sp = nbbo.get((day, sym))
            sp_pct = (100.0 * sp / entry) if sp is not None else None
            half = 0.5 * sp_pct / max(r_pct, 0.05) if sp_pct is not None else None
            net = gross - 2 * half if half is not None else None
            trades.append(
                dict(day=day, symbol=sym, split=split_of(day), break_m=brk, entry_m=nxt,
                     exit_m=exit_m, pmh=round(pmh, 4), entry=round(entry, 4),
                     stop=round(stop, 4), target=round(target, 4), r_pct=round(r_pct, 4),
                     why=why, gross_R=round(gross, 6),
                     sp_pct=None if sp_pct is None else round(sp_pct, 6),
                     net_R=None if net is None else round(net, 6))
            )
        if i % 50 == 0:
            print(f"[{i}/{len(days)}] {day} trades={len(trades)}", flush=True)

    with open(OUT, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(trades[0].keys()))
        w.writeheader()
        w.writerows(trades)

    print(f"\ncandidates={n_cand} rail_fail={n_rail_fail} pmvol_fail={n_vol_fail} "
          f"no_break={n_no_break} late_break(dropped cohort)={n_late} no_entry={n_no_entry}")
    cov = sum(1 for t in trades if t["net_R"] is not None)
    print(f"trades={len(trades)} nbbo_coverage={cov}/{len(trades)}")
    for sp in ("TRAIN", "VAL"):
        s = [t for t in trades if t["split"] == sp]
        g = [t["gross_R"] for t in s]
        n = [t["net_R"] for t in s if t["net_R"] is not None]
        if not s:
            continue
        print(f"{sp}: n={len(s)} mean_gross={sum(g)/len(g):.4f} "
              f"n_net={len(n)} mean_net={sum(n)/len(n):.4f}")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    sys.exit(main())
