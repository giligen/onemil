#!/usr/bin/env python3
"""Independent rebuild of cell 1,390 (PMH break, P2 exit: no target, stop, flat 15:55).

Built ONLY from the prose in research/hod_exit_lab/PREREG_PMH.md and FACTS.md
(with the correction that bars_sip.db `t` is a full ISO UTC timestamp, not HH:MM ET).
No production PMH code was read.

Signal (frozen, per PREREG_PMH.md):
  PMH  = max high of pre-market bars 04:00-09:29 ET (data/cache.db::intraday_bars_1min),
         only if pre-market volume >= 50,000 shares.
  Rail = symbol-day usable only with >= 20 pre-market bars with prints (04:00-09:35 ET).
  Break= first regular-session bar 09:31..11:30 ET whose CLOSE > PMH (09:30 excluded).
  Entry= OPEN of the next bar.
  Stop = low of the last 15 minutes before the break bar, floored at 1.5% of price.
  Exit (P2) = stop, else flat at the 15:55 open. One trade per symbol-day.
  Cost = measured NBBO half-spread at the HOD signal minute of the same name-day
         (research/bf_zero/causal_filter/nbbo.csv), charged per FACTS sec.3:
           half = 0.5 * sp_pct / max(r_pct, 0.05);  net = rr - half - half*ratio
           ratio = {'stop': 0.875, 'eod': 0.412}
Splits: TRAIN day < 2026-01-01, VAL 2026-01-01..2026-05-31, TEST sealed (not scored).
"""

import csv
import sqlite3
import sys
from collections import defaultdict
from datetime import datetime
from zoneinfo import ZoneInfo

ROOT = "/home/ec2-user/onemil"
CACHE_DB = f"{ROOT}/data/cache.db"
SIP_DB = f"{ROOT}/research/bf_zero/bars_sip.db"
CANDS = f"{ROOT}/research/hod_exit_lab/pm_candidates.csv"
NBBO = f"{ROOT}/research/bf_zero/causal_filter/nbbo.csv"
OUT = f"{ROOT}/research/hod_pmh/rebuild_1390_trades.csv"

ET = ZoneInfo("America/New_York")
RATIO = {"stop": 0.875, "eod": 0.412}
PM_START, PM_END = 4 * 60, 9 * 60 + 29        # 04:00 .. 09:29 ET, level window
RAIL_END = 9 * 60 + 35                         # 04:00 .. 09:35 ET, availability rail
BRK_START, BRK_END = 9 * 60 + 31, 11 * 60 + 30  # break search window
FLAT_M = 15 * 60 + 55                          # 15:55 ET flat
STOP_FLOOR = 0.015                             # 1.5% of price


def et_minute(ts: str) -> int:
    """Minutes-since-midnight ET for an ISO UTC timestamp string."""
    return (lambda d: d.hour * 60 + d.minute)(
        datetime.fromisoformat(ts).astimezone(ET)
    )


def load_candidates():
    """(symbol, day) universe = the HOD-break universe's symbol-days."""
    out = defaultdict(list)
    with open(CANDS) as f:
        for row in csv.DictReader(f):
            out[row["bar_date"]].append(row["symbol"])
    return out


def load_nbbo():
    """First (lowest entry_m) HOD signal-minute NBBO spread per (day, symbol)."""
    best = {}
    with open(NBBO) as f:
        for row in csv.DictReader(f):
            if not row["spread_mean"]:
                continue
            k = (row["day"], row["symbol"])
            m = int(row["entry_m"])
            if k not in best or m < best[k][0]:
                best[k] = (m, float(row["spread_mean"]))
    return {k: v[1] for k, v in best.items()}


def pm_bars(cur, symbol, day):
    """Pre-market 1-min bars for a symbol-day as {et_minute: (o,h,l,c,v)}."""
    rows = cur.execute(
        "SELECT timestamp, open, high, low, close, volume FROM intraday_bars_1min "
        "WHERE symbol=? AND bar_date=?", (symbol, day)).fetchall()
    out = {}
    for ts, o, h, l, c, v in rows:
        m = et_minute(ts)
        if PM_START <= m <= RAIL_END:
            out[m] = (o, h, l, c, v)
    return out


def rth_day(cur, day, symbols):
    """All regular-session bars for the given symbols on one day."""
    out = defaultdict(dict)
    qmarks = ",".join("?" * len(symbols))
    rows = cur.execute(
        f"SELECT symbol, t, o, h, l, c, v FROM bars WHERE day=? AND symbol IN ({qmarks})",
        [day] + list(symbols)).fetchall()
    for sym, t, o, h, l, c, v in rows:
        m = et_minute(t)
        if 9 * 60 + 30 <= m <= 16 * 60:
            out[sym][m] = (o, h, l, c, v)
    return out


def simulate(pm, rth):
    """Return a trade dict for one symbol-day, or (None, reason)."""
    if len(pm) < 20:
        return None, "rail_pm_bars"
    lvl = [b for m, b in pm.items() if PM_START <= m <= PM_END]
    if not lvl:
        return None, "no_pm_level_bars"
    pmvol = sum(b[4] for b in lvl)
    if pmvol < 50_000:
        return None, "pm_volume"
    pmh = max(b[1] for b in lvl)
    if not rth:
        return None, "no_rth"

    brk = None
    for m in sorted(rth):
        if m < BRK_START:
            continue
        if m > BRK_END:
            break
        if rth[m][3] > pmh:
            brk = m
            break
    if brk is None:
        return None, "no_break"

    nxt = [m for m in sorted(rth) if m > brk]
    if not nxt:
        return None, "no_entry_bar"
    em = nxt[0]
    entry = rth[em][0]

    pre = [rth[m][2] for m in rth if brk - 15 <= m <= brk - 1]
    if not pre:
        return None, "no_stop_window"
    stop = min(min(pre), entry * (1.0 - STOP_FLOOR))
    risk = entry - stop
    if risk <= 0:
        return None, "bad_risk"

    exit_px, why, xm = None, None, None
    for m in sorted(rth):
        if m < em or m >= FLAT_M:
            continue
        o, h, l, c, v = rth[m]
        if l <= stop:
            exit_px, why, xm = (o if o <= stop else stop), "stop", m
            break
    if exit_px is None:
        if FLAT_M in rth:
            exit_px, why, xm = rth[FLAT_M][0], "eod", FLAT_M
        else:
            tail = [m for m in rth if em <= m < FLAT_M]
            if not tail:
                return None, "no_exit_bar"
            xm = max(tail)
            exit_px, why = rth[xm][3], "eod"

    return {
        "entry_m": em, "exit_m": xm, "entry": entry, "stop": stop,
        "pmh": pmh, "pm_vol": pmvol, "pm_bars": len(pm),
        "r_pct": risk / entry * 100.0, "rr": (exit_px - entry) / risk,
        "why": why, "exit_px": exit_px,
    }, None


def main():
    cands = load_candidates()
    nb = load_nbbo()
    cc = sqlite3.connect(f"file:{CACHE_DB}?mode=ro", uri=True)
    cs = sqlite3.connect(f"file:{SIP_DB}?mode=ro", uri=True)
    cur_c, cur_s = cc.cursor(), cs.cursor()

    trades, drops = [], defaultdict(int)
    days = sorted(cands)
    for i, day in enumerate(days):
        syms = sorted(set(cands[day]))
        rth_all = rth_day(cur_s, day, syms)
        for sym in syms:
            pm = pm_bars(cur_c, sym, day)
            tr, why = simulate(pm, rth_all.get(sym, {}))
            if tr is None:
                drops[why] += 1
                continue
            tr["day"], tr["symbol"] = day, sym
            sp = nb.get((day, sym))
            if sp is None:
                tr["sp_pct"], tr["net"] = "", ""
            else:
                sp_pct = sp / tr["entry"] * 100.0
                half = 0.5 * sp_pct / max(tr["r_pct"], 0.05)
                tr["sp_pct"] = sp_pct
                tr["net"] = tr["rr"] - half - half * RATIO[tr["why"]]
            tr["split"] = ("TRAIN" if day < "2026-01-01"
                           else "VAL" if day < "2026-06-01" else "TEST")
            trades.append(tr)
        if i % 50 == 0:
            print(f"[{i}/{len(days)}] {day} trades={len(trades)}", flush=True)

    cols = ["day", "symbol", "split", "entry_m", "exit_m", "entry", "stop", "exit_px",
            "pmh", "pm_vol", "pm_bars", "r_pct", "rr", "why", "sp_pct", "net"]
    with open(OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for t in trades:
            w.writerow({c: t[c] for c in cols})

    print("drops:", dict(sorted(drops.items(), key=lambda kv: -kv[1])), flush=True)
    for sp in ("TRAIN", "VAL"):
        s = [t for t in trades if t["split"] == sp]
        net = [t["net"] for t in s if t["net"] != ""]
        gross = [t["rr"] for t in s]
        print(f"{sp}: n={len(s)} cov={len(net)/max(len(s),1):.3f} "
              f"gross={sum(gross)/max(len(gross),1):.4f} "
              f"net={sum(net)/max(len(net),1):.4f}", flush=True)
    print("wrote", OUT, flush=True)


if __name__ == "__main__":
    sys.exit(main())
