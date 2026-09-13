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
    dry = {m.group(1): m for m in (RX_DRY.search(ln) for ln in lines) if m}
    live = {m.group(2): m for m in (RX_BUY.search(ln) for ln in lines) if m}
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
    tot_r = 0.0; n = 0; mism = 0
    print(f"  {'sym':6s} {'live_lvl':>8s} {'spec_lvl':>8s} {'live_stop':>9s} {'spec_stop':>9s} {'spec_min':>8s} {'exit':>6s} {'R':>6s}  note")
    for sym in syms:
        m = dry.get(sym) or live.get(sym)
        live_level = float(m.group(2) if sym in dry else m.group(3)); live_stop = float(m.group(4) if sym in dry else m.group(5))
        arr = B.get(sym)
        if arr is None:
            print(f"  {sym:6s} {live_level:8.2f} {'?':>8s} — no bars fetched"); continue
        t = simulate(*arr, adv.get(sym, 0.0), p)
        if t is None:
            mism += 1; print(f"  {sym:6s} {live_level:8.2f} {'none':>8s} {live_stop:9.2f} {'':>9s} {'':>8s} {'':>6s} {'':>6s}  SPEC HAS NO TRADE (rv/floor/r_min or later break) — live-side check"); continue
        o, h, l, c, v, mm = arr
        note = '' if abs(t.stop - live_stop) < 0.011 else 'STOP MISMATCH'
        tot_r += t.rr; n += 1
        print(f"  {sym:6s} {live_level:8.2f} {t.entry / 1.0:8.2f} {live_stop:9.2f} {t.stop:9.2f} {mm[t.entry_idx] // 60:02d}:{mm[t.entry_idx] % 60:02d} {t.reason:>6s} {t.rr:+6.2f}  {note}")
    if n:
        print(f"  would-be book (spec exits, ${risk:.0f} risk): {n} trades, {tot_r:+.1f}R = ${tot_r * risk:+,.0f} | spec-has-no-trade {mism}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
