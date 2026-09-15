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
from trading.hod_break import HodBreakParams, simulate, OPEN_MINUTE, run_book  # noqa: E402

ET = ZoneInfo('America/New_York')
RX_DRY = re.compile(r'\[HOD DRY\] WOULD BUY (\S+) level ([\d.]+) limit ([\d.]+) stop ([\d.]+) target ([\d.]+) R ([\d.]+) \(([\d.]+)%\) x(\d+) \| \+([\d.]+)% from open, rv ([\d.]+), spread (\d+) bps')
RX_BUY = re.compile(r'\[HOD\] (BUY|ENTRY SUBMITTED) (\S+) level ([\d.]+) limit ([\d.]+) stop ([\d.]+) target ([\d.]+)')


def journal(day: str) -> list:
    out = subprocess.run(['journalctl', '-u', 'onemil-trader', '--since', f'{day} 09:00', '--until', f'{day} 23:59', '--no-pager', '-o', 'short-iso'],
                         capture_output=True, text=True).stdout.splitlines()
    return [ln for ln in out if '[HOD' in ln or ('hod_break' in ln and ('ERROR' in ln or 'Traceback' in ln))]


def bars_for(alpaca: AlpacaClient, symbols: list, day: str) -> dict:
    """RTH 1-min bars for `day` as arrays (o,h,l,c,v,m) per symbol — explicit window (works for past days and after the close)."""
    d0 = datetime.strptime(day, '%Y-%m-%d').replace(tzinfo=ET)
    start = d0.replace(hour=9, minute=30).astimezone(timezone.utc); end = d0.replace(hour=16, minute=0).astimezone(timezone.utc)
    got = {}
    for sym in symbols:
        try: got[sym] = alpaca.get_historical_1min_bars(sym, start, end)
        except Exception as e: print(f'  {sym}: bar fetch failed ({e})')
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


MEASURABLES_SINCE = '2026-09-14'                       # first dry-run session: the rolling window of the DB-sourced rows
RX_SPREAD = re.compile(r'(WOULD BUY|ENTRY SUBMITTED) (\S+) level ([\d.]+) .*spread (\d+) bps')
JOURNAL_COUNTS = {'gate': r'of R [\d.]+ > \d+% — skip', 'nochase': r'NO CHASE', 'rmin': r'within [\d.]+% of the ask', 'spread100': r'spread \d+ bps > \d+ — skip',
                  'daycap': r'per-day cap', 'conc': r'concurrency cap', 'missed': r"MISSED the spec's break", 'notional': r'notional cap'}


def closed_trades(trades_db, since: str, day: str) -> list:
    """HOD-break rows from the trades DB, read-only, trade_date in [since, day]; dicts with pattern_data decoded."""
    import json, sqlite3
    try:
        con = sqlite3.connect(f'file:{trades_db}?mode=ro', uri=True, timeout=10); con.row_factory = sqlite3.Row
        rows = [dict(r) for r in con.execute("select * from trades where strategy='hod_break' and trade_date between ? and ?", (since, day))]; con.close()
    except Exception as e:
        print(f'  trades DB read failed: {e}'); return []
    for r in rows:
        try: r['pd'] = json.loads(r.get('pattern_data') or '{}')
        except Exception: r['pd'] = {}
    return rows


def _et_minute(iso: str):
    try: t = datetime.fromisoformat(str(iso).replace('Z', '+00:00')).astimezone(ET); return t.hour * 60 + t.minute
    except Exception: return None


def _bar_open(bars: dict, sym: str, minute) -> float:
    a = bars.get(sym)
    if a is None or minute is None: return 0.0
    idx = np.flatnonzero(a[5] == int(minute)); return float(a[0][idx[0]]) if len(idx) else 0.0


def live_measurables(lines: list, day: str, dry_mode: bool, trades_db, bars: dict, flat_minute: int = 955, since: str = MEASURABLES_SINCE) -> list:
    """REPORT §12 rows 1-16: [no, label, live value, spec band, verdict]. Journal rows are today's; DB rows carry the day and the
    rolling value since `since`. Dry mode: rows 3-13 are n/a. No parentheses in the text: the cron builds the telegram from it."""
    c = {k: sum(1 for ln in lines if re.search(rx, ln)) for k, rx in JOURNAL_COUNTS.items()}
    sig = {m.group(2): int(m.group(4)) for m in (RX_SPREAD.search(ln) for ln in lines) if m}
    n_ord = len(sig); reached = n_ord + c['gate'] + c['nochase'] + c['rmin'] + c['spread100'] + c['daycap'] + c['conc']
    n_gate = n_ord + c['gate']; pass_rate = n_ord / n_gate * 100 if n_gate else float('nan')
    spreads = sorted(sig.values()); med_spread = float(np.median(spreads)) if spreads else float('nan')
    rmin_rate = c['rmin'] / reached * 100 if reached else float('nan')
    rows = [[1, 'gate pass rate, $20+ signals', f'{n_ord}/{n_gate} = {pass_rate:.0f}%' if n_gate else 'no signal reached the gate', '43% · band 30-55% over >= 50 signals',
             'WATCH n<20' if n_gate < 20 else ('OK' if 30 <= pass_rate <= 55 else ('ESCALATE' if n_gate >= 50 else 'WATCH'))],
            [2, 'signals/day reaching the gates, $20+', f'{reached} today', 'median 11-25 · zero-signal days ~0', 'ESCALATE none reached' if reached == 0 else ('WATCH <5' if reached < 5 else 'OK')]]
    na = 'n/a dry'
    if dry_mode:
        rows += [[k, lab, na, band, 'n/a'] for k, lab, band in ((3, 'fills/day', '4.6-6.6 per day, 22-32 per wk'), (4, 'fill rate of submitted orders', '~100% of ask<=cap · alert <85%'),
                 (5, 'entry fill vs the open of the fill minute', 'median <= +8 bps, mean <= +15'), (6, 'ask at decision vs the next open', 'median +6 bps'),
                 (7, 'TP-fill rate', '33-39% · live >= spec'), (8, 'stop rate / eod rate', '41-45% / 20-22%'), (9, 'stop fill vs stop price', 'modeled -18 bps · alert mean worse than -40'),
                 (10, 'eod fill vs the 15:55 open', 'minus half spread, ~-8 bps'), (11, 'mean R per trade, net realized', '+0.25 to +0.35 · no verdict before 150 trades'),
                 (12, 'WR', '48-53%'), (13, 'weekly R', '+6 to +10 · worst -8 to -12'))]
    else:
        T = closed_trades(trades_db, since, day); D = [r for r in T if r['trade_date'] == day]
        filled = [r for r in T if r.get('fill_price')]; dfill = [r for r in D if r.get('fill_price')]
        sessions = max(1, int(np.busday_count(since, day)) + 1)
        closed = [r for r in filled if r.get('exit_price') and r.get('exit_reason')]
        for r in closed:
            fp, st = float(r['fill_price']), float(r['stop_loss_price'] or 0); q = float(r.get('filled_qty') or r.get('shares') or 0)
            r['rr'] = float(r['pnl'] or 0) / ((fp - st) * q) if fp > st and q > 0 else float('nan')
        rr = np.array([r['rr'] for r in closed if np.isfinite(r['rr'])]); n = len(rr)
        ent = [(float(r['fill_price']) / _bar_open(bars, r['symbol'], _et_minute(r.get('filled_at'))) - 1) * 1e4 for r in dfill if _bar_open(bars, r['symbol'], _et_minute(r.get('filled_at')))]
        ask = [(float(r['pd'].get('quote_ask') or 0) / _bar_open(bars, r['symbol'], _et_minute(r.get('filled_at'))) - 1) * 1e4 for r in dfill if r['pd'].get('quote_ask') and _bar_open(bars, r['symbol'], _et_minute(r.get('filled_at')))]
        stp = [(float(r['exit_price']) / float(r['stop_loss_price']) - 1) * 1e4 for r in closed if r['exit_reason'] == 'stop' and r.get('stop_loss_price')]
        eod = [(float(r['exit_price']) / _bar_open(bars, r['symbol'], flat_minute) - 1) * 1e4 for r in closed if r['exit_reason'] == 'eod' and r['trade_date'] == day and _bar_open(bars, r['symbol'], flat_minute)]
        share = lambda why: 100 * sum(1 for r in closed if r['exit_reason'] == why) / len(closed) if closed else float('nan')
        wk = {}
        for r in closed:
            if np.isfinite(r['rr']): wk[datetime.strptime(r['trade_date'], '%Y-%m-%d').isocalendar()[1]] = wk.get(datetime.strptime(r['trade_date'], '%Y-%m-%d').isocalendar()[1], 0.0) + r['rr']
        cur_wk = wk.get(datetime.strptime(day, '%Y-%m-%d').isocalendar()[1], 0.0); worst_wk = min(wk.values()) if wk else 0.0
        fr = 100 * len(dfill) / len(D) if D else float('nan'); per_day = len(filled) / sessions
        mean_r = rr.mean() if n else float('nan'); wr = 100 * (rr > 0).mean() if n else float('nan')
        mstat = lambda x: f'median {np.median(x):+.0f} mean {np.mean(x):+.0f} bps n={len(x)}' if x else 'no rows'
        rows += [[3, 'fills/day', f'{len(dfill)} today · rolling {per_day:.1f}/day over {sessions} sessions', '4.6-6.6 per day, 22-32 per wk', 'OK' if 3 <= per_day <= 9 else 'WATCH'],
                 [4, 'fill rate of submitted orders', f'{len(dfill)}/{len(D)} today = {fr:.0f}%' if D else 'no orders today', '~100% of ask<=cap · alert <85%', 'n/a' if not D else ('OK' if fr >= 95 else ('WATCH' if fr >= 85 else 'ESCALATE'))],
                 [5, 'entry fill vs the open of the fill minute', mstat(ent), 'median <= +8 bps, mean <= +15', 'n/a' if not ent else ('OK' if np.mean(ent) <= 15 else ('WATCH' if np.mean(ent) <= 30 else 'ESCALATE'))],
                 [6, 'ask at decision vs the next open', mstat(ask), 'median +6 bps', 'n/a' if not ask else ('OK' if np.mean(ask) <= 15 else 'WATCH')],
                 [7, 'TP-fill rate', f'{share("target"):.0f}% of {len(closed)} closed', '33-39% · live >= spec', 'WATCH n<20' if len(closed) < 20 else ('OK' if share('target') >= 30 else ('ESCALATE' if len(closed) >= 30 and share('target') < 25 else 'WATCH'))],
                 [8, 'stop rate / eod rate', f'{share("stop"):.0f}% / {share("eod"):.0f}%', '41-45% / 20-22%', 'WATCH n<20' if len(closed) < 20 else ('OK' if share('stop') <= 55 else 'WATCH')],
                 [9, 'stop fill vs stop price', mstat(stp), 'modeled -18 bps · alert mean worse than -40', 'n/a' if not stp else ('OK' if np.mean(stp) >= -25 else ('WATCH' if np.mean(stp) >= -40 else 'ESCALATE'))],
                 [10, 'eod fill vs the 15:55 open', mstat(eod), 'minus half spread, ~-8 bps', 'n/a' if not eod else ('OK' if np.mean(eod) >= -30 else 'WATCH')],
                 [11, 'mean R per trade, net realized', f'{mean_r:+.3f}R over {n} trades' if n else 'no closed trades', '+0.25 to +0.35 · no verdict before 150 trades',
                  'WATCH n<30' if n < 30 else ('ESCALATE' if mean_r < 0 or (n >= 150 and mean_r < 0.15) else ('OK' if mean_r >= 0.2 else 'WATCH'))],
                 [12, 'WR', f'{wr:.0f}% over {n}' if n else 'no closed trades', '48-53%', 'WATCH n<30' if n < 30 else ('ESCALATE' if n >= 50 and wr < 40 else ('OK' if wr >= 44 else 'WATCH'))],
                 [13, 'weekly R', f'this week {cur_wk:+.1f}R · worst week {worst_wk:+.1f}R', '+6 to +10 · worst -8 to -12', 'ESCALATE' if cur_wk <= -12 else ('WATCH' if cur_wk < 0 else 'OK')]]
    rows += [[14, 'miss rate vs spec', f"{c['missed']} MISSED lines today · full audit: scripts/hod_break_miss_audit.py", '0', 'ESCALATE' if c['missed'] else 'OK'],
             [15, 'spread at decision, passing $20+', f'median {med_spread:.0f} bps over {len(spreads)}' if spreads else 'no passing signal', 'median 17 bps · sfr median 0.082', 'n/a' if not spreads else ('OK' if med_spread <= 30 else 'WATCH')],
             [16, 'r_min reject rate, $20+ signals', f"{c['rmin']}/{reached} = {rmin_rate:.0f}%" if reached else 'no signal reached', 'a few % · live r = ask - stop is LOOSER than the spec, the rejects are stops within 1% that simulate rejects too',
              'n/a' if not reached else ('OK' if rmin_rate <= 25 else ('ESCALATE' if reached >= 10 and rmin_rate > 50 else 'WATCH'))]]
    return rows


def print_measurables(rows: list) -> None:
    print('\n  LIVE MEASURABLES vs SPEC, REPORT §12:')
    for no, lab, val, band, verdict in rows:
        print(f'  {no:2d} {verdict:18s} {lab}: {val} | spec {band}')


RX_REJECT = re.compile(r'\[HOD\] (\S+): (stop [\d.]+ within [\d.]+% of the ask|ask [\d.]+ above cap|spread \d+ bps = \d+% of R|spread \d+ bps > \d+)')


def rejection_parity(lines, day, cfg, p, hb) -> int:
    """The engine's fill-level rejections (r_min on the ask, no-chase, the spread gates) are decisions the spec makes on the
    NEXT OPEN (`simulate`: open <= cap, r/open >= min_r_pct; the spread gate on the historical NBBO). Re-run the spec for
    every rejected symbol: 'spec also no trade' = parity; 'spec HAD a trade' = a live-only rejection to count."""
    rej = {}
    for ln in lines:
        m = RX_REJECT.search(ln)
        if m and m.group(1) not in rej: rej[m.group(1)] = m.group(2).split(' ')[0] if not m.group(2).startswith('stop') else 'r_min'
    if not rej: return 0
    alpaca = AlpacaClient(cfg.alpaca_api_key, cfg.alpaca_api_secret, paper=cfg.alpaca_paper)
    B = bars_for(alpaca, sorted(rej), day)
    try:
        from trading.hod_break_engine import load_adv20_from_daily_bars
        adv, _ = load_adv20_from_daily_bars(__import__('persistence.database', fromlist=['Database']).Database()._cache_path)
    except Exception as e:
        print(f'  rejection parity: ADV map unavailable ({e})'); adv = {}
    n_dev = 0; print(f"\n  REJECTION PARITY, {len(rej)} symbols the engine rejected at the fill level:")
    for sym, why in sorted(rej.items()):
        arr = B.get(sym)
        if arr is None: print(f"  {sym:6s} {why:8s} no bars"); continue
        o, h, l, c, v, m = arr
        tr = simulate(o, h, l, c, v, m, adv.get(sym, 0.0), p)
        if tr is None: print(f"  {sym:6s} {why:8s} spec: no trade either  OK")
        else:
            n_dev += 1 if why in ('r_min', 'ask') else 0
            print(f"  {sym:6s} {why:8s} spec HAD a trade: fill {tr.entry:.2f} stop {tr.stop:.2f} {tr.reason} {tr.rr:+.2f}R  {'DEVIATION' if why in ('r_min', 'ask') else 'gate'}")
    print(f"  live-only fill rejections where the spec traded: {n_dev}")
    return n_dev


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
    Db = __import__('persistence.database', fromlist=['Database']).Database(); trades_db = Db._trades_path
    rejected = rejection_parity(lines, day, cfg, p, hb)
    if not syms:
        print('  no signals today')
        B = {} if hb['dry_run'] else bars_for(AlpacaClient(cfg.alpaca_api_key, cfg.alpaca_api_secret, paper=cfg.alpaca_paper), sorted({r['symbol'] for r in closed_trades(trades_db, day, day)}), day)
        print_measurables(live_measurables(lines, day, hb['dry_run'], trades_db, B, p.flat_minute)); return 0
    alpaca = AlpacaClient(cfg.alpaca_api_key, cfg.alpaca_api_secret, paper=cfg.alpaca_paper)
    B = bars_for(alpaca, syms, day)
    if not hb['dry_run']:
        B.update(bars_for(alpaca, sorted({r['symbol'] for r in closed_trades(trades_db, day, day)} - set(B)), day))
    adv = {r['symbol']: float(r.get('avg_volume_daily') or 0) for r in Db.get_active_universe()}
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
        ts = re.match(r'(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}[+-]\d{4})', m.string)
        t_et = datetime.strptime(ts.group(1), '%Y-%m-%dT%H:%M:%S%z').astimezone(ET); sig_min = t_et.hour * 60 + t_et.minute   # DST-safe
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
        taken = [(sym, rr, usd) for _, _, _, sym, rr, usd in run_book([(0, em, xm, sym, rr, usd) for em, xm, sym, rr, usd in dbook], p.max_per_day, p.max_concurrent)]
        dr = sum(r for _, r, _ in taken); dusd = sum(u for _, _, u in taken); allr = sum(r for *_, r, _ in dbook); allusd = sum(u for *_, u in dbook)
        print(f"  DRY-RUN all filled signals: {len(dbook)}, {allr:+.1f}R, ${allusd:+,.0f} at the logged sizes")
        print(f"  DRY-RUN EXECUTABLE book (first {p.max_per_day}/day, {p.max_concurrent} concurrent, logged sizes): {len(taken)} trades, {dr:+.1f}R, ${dusd:+,.0f} | {[(s_, round(r, 2)) for s_, r, _ in taken]}")
        print(f"  GATE 6 on the DRY-RUN book: {'PASS' if dr > 0 else 'FAIL'}")
        for frac, floor in ((0.15, 5.0), (0.15, 20.0), (0.10, 20.0)):
            rows = [(0, em, xm, sym, rr, usd) for em, xm, sym, rr, usd, sf in gated if sf <= frac and float(dry[sym].group(2)) >= floor]
            taken = [(sym, rr, usd) for _, _, _, sym, rr, usd in run_book(rows, p.max_per_day, p.max_concurrent)]
            print(f"  DRY-RUN book with spread <= {frac:.0%} of R and price >= ${floor:.0f}: {len(taken)} trades, {sum(r for _, r, _ in taken):+.1f}R, ${sum(u for _, _, u in taken):+,.0f} | signals passing {sum(1 for g in gated if g[5] <= frac and float(dry[g[2]].group(2)) >= floor)}/{len(gated)}")
    if n:
        print(f"\n  all spec trades on signalled symbols: {n}, {tot_r:+.1f}R = ${tot_r * risk:+,.0f} | spec-has-no-trade {mism}")
        # the EXECUTABLE would-be book: first-come, max_per_day, max_concurrent (dry mode never counts entries)
        taken = [(sym, rr) for _, _, _, sym, rr in run_book([(0, em, xm, sym, rr) for em, xm, sym, rr in book], p.max_per_day, p.max_concurrent)]
        br = sum(r for _, r in taken)
        print(f"  EXECUTABLE would-be book (first {p.max_per_day}/day, {p.max_concurrent} concurrent, ${risk:.0f} risk): {len(taken)} trades, {br:+.1f}R = ${br * risk:+,.0f} | {[(s_, round(r, 2)) for s_, r in taken]}")
        print(f"  GATE 6 (positive would-be day): {'PASS' if br > 0 else 'FAIL'}")
    print_measurables(live_measurables(lines, day, hb['dry_run'], trades_db, B, p.flat_minute))
    return 0


if __name__ == '__main__':
    sys.exit(main())
