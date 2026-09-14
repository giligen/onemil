#!/usr/bin/env python3
"""HOD-break end-of-day check: live (or dry-run) signals vs the exact spec on the day's bars.

For today's session:
  1. journal → every `[HOD DRY] WOULD BUY` / `[HOD] BUY` / `FILLED` / `EXIT` / `FORCE CLOSE` / ERROR line
  2. for each signalled symbol, fetch today's RTH 1-min bars (Alpaca) and run `trading.hod_break.simulate`
     with the shipped params → the spec's own signal minute, level, stop, target, exit and R
  3. parity: live level/stop vs spec level/stop (should match to the cent when the bar streams agree);
     signal-minute drift; symbols the spec would NOT have traded (a live-side defect) and vice versa
  4. the would-be book P&L at risk_usd (dry run) or the realized P&L (live) + the rolling weekly tally

Usage: python3 scripts/hod_break_eod_check.py [YYYY-MM-DD]   → prints a report; exit 0 always
(this is a reporting tool; it never modifies anything).
"""
import os
import re
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT); os.chdir(ROOT)

from config import Config                                  # noqa: E402
from data_sources.alpaca_client import AlpacaClient        # noqa: E402
from trading.hod_break import HodBreakParams, simulate, OPEN_MINUTE  # noqa: E402

ET = ZoneInfo('America/New_York')
RX_DRY = re.compile(r'\[HOD DRY\] WOULD BUY (\S+) level ([\d.]+) limit ([\d.]+) stop ([\d.]+) target ([\d.]+) R ([\d.]+) \(([\d.]+)%\) x(\d+) \| \+([\d.]+)% from open, rv ([\d.]+), spread (\d+) bps')
RX_BUY = re.compile(r'\[HOD\] (BUY|ENTRY SUBMITTED) (\S+) level ([\d.]+) limit ([\d.]+) stop ([\d.]+) target ([\d.]+)')


def journal(day: str) -> list:
    out = subprocess.run(['journalctl', '-u', 'onemil-trader', '--since', f'{day} 09:00', '--until', f'{day} 23:59', '--no-pager', '-o', 'short-iso'],
                         capture_output=True, text=True).stdout.splitlines()
    return [ln for ln in out if '[HOD' in ln or ('hod_break' in ln and ('ERROR' in ln or 'Traceback' in ln))]


def bars_for(alpaca: AlpacaClient, symbols: list, day: str) -> dict:
    """RTH 1-min bars for `day` as arrays (o,h,l,c,v,m) per symbol."""
    got = alpaca.get_1min_bars_multi(symbols, lookback_minutes=420) if symbols else {}
    out = {}
    for sym, df in got.items():
        if df is None or not len(df):
            continue
        ts = df['timestamp'] if 'timestamp' in df.columns else df.index.to_series()
        t = [x.to_pydatetime() if hasattr(x, 'to_pydatetime') else x for x in ts]
        rows = []
        for i, tt in enumerate(t):
            if tt.tzinfo is None: tt = tt.replace(tzinfo=timezone.utc)
            e = tt.astimezone(ET)
            if e.strftime('%Y-%m-%d') != day: continue
            m = e.hour * 60 + e.minute
            if OPEN_MINUTE <= m < 960:
                rows.append((m, float(df.iloc[i]['open']), float(df.iloc[i]['high']), float(df.iloc[i]['low']), float(df.iloc[i]['close']), float(df.iloc[i].get('volume', 0) or 0)))
        if rows:
            rows.sort(); a = np.array(rows, dtype=float)
            out[sym] = (a[:, 1], a[:, 2], a[:, 3], a[:, 4], a[:, 5], a[:, 0].astype(int))
    return out


def main() -> int:
    day = sys.argv[1] if len(sys.argv) > 1 else datetime.now(timezone.utc).astimezone(ET).strftime('%Y-%m-%d')
    cfg = Config(); hb = cfg.hod_break_cfg; p = HodBreakParams(**hb['params']); risk = hb['risk_usd']
    lines = journal(day)
    dry = {}; live = {}
    for ln in lines:                                    # FIRST signal per symbol (later re-breaks are by-products)
        m = RX_DRY.search(ln)
        if m and m.group(1) not in dry: dry[m.group(1)] = m
        m = RX_BUY.search(ln)
        if m and m.group(2) not in live: live[m.group(2)] = m
    errors = [ln for ln in lines if 'ERROR' in ln or 'Traceback' in ln or 'queue full' in ln]
    fills = [ln for ln in lines if '[HOD] FILLED' in ln]; exits = [ln for ln in lines if '[HOD] EXIT' in ln]
    print(f"HOD-BREAK EOD {day} — mode {'DRY' if hb['dry_run'] else 'LIVE'} | dry signals {len(dry)} | live orders {len(live)} | fills {len(fills)} | exits {len(exits)} | errors {len(errors)}")
    for ln in errors[:8]: print('  ERR', ln[-200:])
    syms = sorted(set(dry) | set(live))
    if not syms:
        print('  no signals today'); return 0
    alpaca = AlpacaClient(cfg.alpaca_api_key, cfg.alpaca_api_secret, paper=cfg.alpaca_paper)
    B = bars_for(alpaca, syms, day)
    adv = {r['symbol']: float(r.get('avg_volume_daily') or 0) for r in __import__('persistence.database', fromlist=['Database']).Database().get_active_universe()}
    tot_r = 0.0; n = 0; mism = 0; book = []
    print(f"  {'sym':6s} {'live_lvl':>8s} {'spec_lvl':>8s} {'live_stop':>9s} {'spec_stop':>9s} {'spec_min':>8s} {'exit':>6s} {'R':>6s}  note")
    for sym in syms:
        m = dry.get(sym) or live.get(sym)
        live_level = float(m.group(2) if sym in dry else m.group(3)); live_stop = float(m.group(4) if sym in dry else m.group(5))
        arr = B.get(sym)
        if arr is None:
            print(f"  {sym:6s} {live_level:8.2f} {'?':>8s} — no bars fetched"); continue
        t = simulate(*arr, adv.get(sym, 0.0), p)
        if t is None:
            from trading.hod_break import detect, entry_fill
            sig = detect(*arr[:3], arr[4], arr[5], adv.get(sym, 0.0), p)
            if sig is None: why = 'NO QUALIFYING BREAK on REST bars (rv/floor/consolidation differ from the engine bars)'
            else:
                nxt = arr[0][sig.bar_idx + 1] if sig.bar_idx + 1 < len(arr[0]) else None
                why = f"first break {int(arr[5][sig.bar_idx]) // 60:02d}:{int(arr[5][sig.bar_idx]) % 60:02d} lvl {sig.level:.2f}: " + ('no next bar' if nxt is None else (f'NO-CHASE, next open {nxt:.2f} > cap' if entry_fill(nxt, sig.level, p) is None else 'stop >= entry or r_min'))
            mism += 1; print(f"  {sym:6s} {live_level:8.2f} {'none':>8s} {live_stop:9.2f} {'':>9s} {'':>8s} {'':>6s} {'':>6s}  SPEC NO TRADE: {why}"); continue
        o, h, l, c, v, mm = arr
        note = '' if abs(t.stop - live_stop) < 0.011 else 'STOP MISMATCH'
        tot_r += t.rr; n += 1; book.append((int(mm[t.entry_idx]), int(mm[t.exit_idx]), sym, t.rr))
        print(f"  {sym:6s} {live_level:8.2f} {t.entry / 1.0:8.2f} {live_stop:9.2f} {t.stop:9.2f} {mm[t.entry_idx] // 60:02d}:{mm[t.entry_idx] % 60:02d} {t.reason:>6s} {t.rr:+6.2f}  {note}")
    # ---- the DRY-RUN book: the engine's OWN logged signals (level/limit/stop/target) walked on today's bars ----
    from trading.hod_break import STOP_FILL_SLIP
    dbook = []; gated = []
    print("\n  DRY-RUN BOOK — the engine's own signals, filled at the next open if <= the logged limit, logged stop/target walked forward:")
    for sym, m in dry.items():
        arr = B.get(sym)
        if arr is None: continue
        o, h, l, c, v, mm = arr
        level, limit, stop, target = (float(m.group(k)) for k in (2, 3, 4, 5))
        hh, mn = int(m.string.split(' | ')[0].split(' ')[1][:2]) if False else (0, 0), 0
        ts = re.search(r'(\d{2}):(\d{2}):\d{2} \|', m.string); sig_min = int(ts.group(1)) * 60 + int(ts.group(2)) - 4 * 60   # journal is UTC (ET+4 in Sept)
        idx = np.flatnonzero(mm >= sig_min)                   # the first bar at/after the signal minute = the next open
        if not len(idx): print(f"  {sym:6s} no bars after the signal"); continue
        i = int(idx[0]); nxt = float(o[i])
        if nxt > limit: print(f"  {sym:6s} {level:8.2f} next open {nxt:.2f} > limit {limit:.2f} — no fill, no chase"); continue
        entry = nxt; r = entry - stop
        if r <= 0: print(f"  {sym:6s} stop {stop:.2f} >= fill {entry:.2f} — no trade"); continue
        why, px, k = 'OPEN', float(c[-1]), len(o) - 1
        for j in range(i + 1, len(o)):
            if int(mm[j]) >= p.flat_minute: why, px, k = 'eod', float(o[j]), j; break
            if l[j] <= stop: why, px, k = 'stop', float(min(stop, o[j]) * (1 - STOP_FILL_SLIP)), j; break
            if c[j] >= target: why, px, k = 'target', target, j; break
        shares = int(m.group(8)); usd = shares * (px - entry)          # dollars = shares x move (the logged size), not R x risk
        rr = (px - entry) / r; dbook.append((int(mm[i]), int(mm[k]), sym, rr, usd))
        spread_bps_logged = float(m.group(11)); r_pct_logged = float(m.group(7)); gated.append((int(mm[i]), int(mm[k]), sym, rr, usd, spread_bps_logged / (r_pct_logged * 100.0)))
        print(f"  {sym:6s} {level:8.2f} fill {entry:6.2f} stop {stop:6.2f} target {target:6.2f} {int(mm[i]) // 60:02d}:{int(mm[i]) % 60:02d} {why:>6s} {rr:+6.2f}")
    if dbook:
        taken = []; open_exits = []
        for em, xm, sym, rr, usd in sorted(dbook):
            open_exits = [e for e in open_exits if e > em]
            if len(taken) >= p.max_per_day or len(open_exits) >= p.max_concurrent: continue
            taken.append((sym, rr, usd)); open_exits.append(xm)
        dr = sum(r for _, r, _ in taken); dusd = sum(u for _, _, u in taken); allr = sum(r for *_, r, _ in dbook); allusd = sum(u for *_, u in dbook)
        print(f"  DRY-RUN all filled signals: {len(dbook)}, {allr:+.1f}R, ${allusd:+,.0f} at the logged sizes")
        print(f"  DRY-RUN EXECUTABLE book (first {p.max_per_day}/day, {p.max_concurrent} concurrent, logged sizes): {len(taken)} trades, {dr:+.1f}R, ${dusd:+,.0f} | {[(s_, round(r, 2)) for s_, r, _ in taken]}")
        print(f"  GATE 6 on the DRY-RUN book: {'PASS' if dr > 0 else 'FAIL'}")
        for frac in (0.15, 0.10):
            taken = []; open_exits = []
            for em, xm, sym, rr, usd, sf in sorted(gated):
                if sf > frac: continue
                open_exits = [e for e in open_exits if e > em]
                if len(taken) >= p.max_per_day or len(open_exits) >= p.max_concurrent: continue
                taken.append((sym, rr, usd)); open_exits.append(xm)
            print(f"  DRY-RUN book with spread <= {frac:.0%} of R: {len(taken)} trades, {sum(r for _, r, _ in taken):+.1f}R, ${sum(u for _, _, u in taken):+,.0f} | signals passing the gate {sum(1 for g in gated if g[5] <= frac)}/{len(gated)}")
    if n:
        print(f"\n  all spec trades on signalled symbols: {n}, {tot_r:+.1f}R = ${tot_r * risk:+,.0f} | spec-has-no-trade {mism}")
        # the EXECUTABLE would-be book: first-come, max_per_day, max_concurrent (dry mode never counts entries)
        taken = []; open_exits = []
        for em, xm, sym, rr in sorted(book):
            open_exits = [e for e in open_exits if e > em]
            if len(taken) >= p.max_per_day or len(open_exits) >= p.max_concurrent: continue
            taken.append((sym, rr)); open_exits.append(xm)
        br = sum(r for _, r in taken)
        print(f"  EXECUTABLE would-be book (first {p.max_per_day}/day, {p.max_concurrent} concurrent, ${risk:.0f} risk): {len(taken)} trades, {br:+.1f}R = ${br * risk:+,.0f} | {[(s_, round(r, 2)) for s_, r in taken]}")
        print(f"  GATE 6 (positive would-be day): {'PASS' if br > 0 else 'FAIL'}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
