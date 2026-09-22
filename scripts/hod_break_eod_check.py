#!/usr/bin/env python3
"""HOD-break / red-to-green end-of-day check: live (or dry-run) signals vs the exact spec on the day's bars.

For today's session:
  1. journal → every `[HOD DRY] WOULD BUY` / `[HOD] BUY` / `FILLED` / `EXIT` / `FORCE CLOSE` / ERROR line
  2. for each signalled symbol, fetch today's RTH 1-min bars (Alpaca) and run the book's spec
     (`trading.hod_break.simulate` / `trading.red_to_green.detect` + the shared fill/walk) with the shipped
     params → the spec's own signal minute, level, stop, target, exit and R
  3. parity: live level/stop vs spec level/stop (should match to the cent when the bar streams agree);
     signal-minute drift; symbols the spec would NOT have traded (a live-side defect) and vice versa
  4. the would-be book P&L at risk_usd (dry run) or the realized P&L (live) + the rolling weekly tally

Usage: python3 scripts/hod_break_eod_check.py [YYYY-MM-DD] [--book hod_break|red_to_green]
`--book red_to_green` reads `Config().red_to_green_cfg`, greps the `[R2G` journal lines, runs the F6-PDR spec
(prior close / prior-day range from daily_bars, the loader the engine itself uses) and prints the same sections
with an `[R2G EOD]` header. Default `hod_break` — byte-identical to the pre-option script.
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
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, 'scripts')); os.chdir(ROOT)

from book_spec import book_from_argv, load_book            # noqa: E402
from config import Config                                  # noqa: E402
from data_sources.alpaca_client import AlpacaClient        # noqa: E402
from trading.hod_break import OPEN_MINUTE, run_book        # noqa: E402

ET = ZoneInfo('America/New_York')


def rx_dry(tag: str = 'HOD'):
    """The book's `WOULD BUY` line: symbol, level, limit, stop, target, R, R%, shares, dist, rv, spread."""
    return re.compile(rf'\[{tag} DRY\] WOULD BUY (\S+) level ([\d.]+) limit ([\d.]+) stop ([\d.]+) target ([\d.]+) R ([\d.]+) \(([\d.]+)%\) x(\d+) \| \+([\d.]+)% from open, rv ([\d.]+), spread (\d+) bps')


def rx_buy(tag: str = 'HOD'):
    return re.compile(rf'\[{tag}\] (BUY|ENTRY SUBMITTED) (\S+) level ([\d.]+) limit ([\d.]+) stop ([\d.]+) target ([\d.]+)')


RX_DRY = rx_dry(); RX_BUY = rx_buy()


_ARCHIVE_RX = re.compile(r'^\w{3} +\d{1,2} \d{2}:\d{2}:\d{2} \S+ (\S+\[\d+\]): (\d{4}-\d{2}-\d{2}) (\d{2}:\d{2}:\d{2}) (.*)$')


def journal(day: str, book=None) -> list:
    """Session lines for `day`: journald (short-iso) UNIONED with logs/session_archive/<day>.log.

    2026-09-22 defect: journald's size cap had already dropped half of 2026-09-14's lines (20 left of 42
    archived), so the dry-run book for a past day changed between runs (12 trades → 9). The nightly archive
    (crontab 21:58) is the durable copy; its lines carry the default journald prefix, so they are rewritten
    to the short-iso prefix the timestamp parser below expects (the app's own body timestamp is UTC).
    De-duplicated on the message body.
    """
    out = subprocess.run(['journalctl', '-u', 'onemil-trader', '--since', f'{day} 09:00', '--until', f'{day} 23:59', '--no-pager', '-o', 'short-iso'],
                         capture_output=True, text=True).stdout.splitlines()
    arch = os.path.join(ROOT, 'logs', 'session_archive', f'{day}.log')
    if os.path.exists(arch):
        seen = {ln.split(']: ', 1)[-1] for ln in out}
        added = 0
        with open(arch, errors='replace') as fh:
            arch_lines = fh.read().splitlines()
        for ln in arch_lines:
            m = _ARCHIVE_RX.match(ln)
            if not m or m.group(2) != day:
                continue
            body = f'{m.group(2)} {m.group(3)} {m.group(4)}'
            if body in seen:
                continue
            seen.add(body)
            out.append(f'{m.group(2)}T{m.group(3)}+0000 archive {m.group(1)}: {body}')
            added += 1
        if added:
            print(f'[journal] {day}: {added} line(s) restored from session_archive (journald had rotated them)')
    if book is None:
        return [ln for ln in out if '[HOD' in ln or ('hod_break' in ln and ('ERROR' in ln or 'Traceback' in ln))]
    return [ln for ln in out if book.journal_line(ln)]


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
R2G_MEASURABLES_SINCE = '2026-09-18'                   # the red-to-green book's first dry-run session (the 12:30 UTC boot)
RX_SPREAD = re.compile(r'(WOULD BUY|ENTRY SUBMITTED) (\S+) level ([\d.]+) .*spread (\d+) bps')
JOURNAL_COUNTS = {'gate': r'of R [\d.]+ > \d+% — skip', 'nochase': r'NO CHASE', 'rmin': r'within [\d.]+% of the ask', 'spread100': r'spread \d+ bps > \d+ — skip',
                  'daycap': r'per-day cap', 'conc': r'concurrency cap', 'missed': r"MISSED the spec's break", 'notional': r'notional cap'}


def closed_trades(trades_db, since: str, day: str, strategy: str = 'hod_break') -> list:
    """The book's rows from the trades DB, read-only, trade_date in [since, day]; dicts with pattern_data decoded."""
    import json, sqlite3
    try:
        con = sqlite3.connect(f'file:{trades_db}?mode=ro', uri=True, timeout=10); con.row_factory = sqlite3.Row
        rows = [dict(r) for r in con.execute("select * from trades where strategy=? and trade_date between ? and ?", (strategy, since, day))]; con.close()
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


# The REPORT §12 bands are the HOD-break book's. The red-to-green book has its own declared numbers and no live
# history at all, so its rows carry the F6 figures where they exist and NO verdict where the band would be another
# book's. Execution-quality rows — fill rate, fill vs the next open, stop/eod slippage, the miss rate — are
# mechanical (they measure the engine, not the book) and keep their band and verdict.
R2G_KEEP_VERDICT = {2, 4, 5, 6, 9, 10, 14}
R2G_BANDS = {
    1: 'no R2G band yet, first dry session 2026-09-18',
    3: 'F6 declared ~1-3 fills a day at 12/day 4 concurrent',
    7: 'no R2G band yet, target_r 2.0 books ~35% TP in the F6 study',
    8: 'no R2G band yet',
    11: 'F6 declared +0.058 TRAIN / +0.110 VAL / +0.003 TEST R per trade at 2R',
    12: 'no R2G band yet',
    13: 'no R2G band yet, F6 worst week -10.7R on TRAIN at 1R risk units',
    15: 'no R2G band yet, the book is a $5+ universe not the HOD $20 floor',
    16: 'no R2G band yet',
}


def r2g_bands(rows: list) -> list:
    """Replace the HOD-break REPORT §12 bands with the F6 figures and drop the verdicts a HOD band would have set."""
    return [[no, lab, val, R2G_BANDS.get(no, band), (verdict if no in R2G_KEEP_VERDICT else 'n/a')]
            for no, lab, val, band, verdict in rows]


def live_measurables(lines: list, day: str, dry_mode: bool, trades_db, bars: dict, flat_minute: int = 955, since: str = MEASURABLES_SINCE,
                     strategy: str = 'hod_break', price_floor: float = 20.0) -> list:
    """REPORT §12 rows 1-16: [no, label, live value, spec band, verdict]. Journal rows are today's; DB rows carry the day and the
    rolling value since `since`. Dry mode: rows 3-13 are n/a. No parentheses in the text: the cron builds the telegram from it."""
    c = {k: sum(1 for ln in lines if re.search(rx, ln)) for k, rx in JOURNAL_COUNTS.items()}
    sig = {m.group(2): int(m.group(4)) for m in (RX_SPREAD.search(ln) for ln in lines) if m}
    n_ord = len(sig); reached = n_ord + c['gate'] + c['nochase'] + c['rmin'] + c['spread100'] + c['daycap'] + c['conc']
    n_gate = n_ord + c['gate']; pass_rate = n_ord / n_gate * 100 if n_gate else float('nan')
    spreads = sorted(sig.values()); med_spread = float(np.median(spreads)) if spreads else float('nan')
    rmin_rate = c['rmin'] / reached * 100 if reached else float('nan')
    fl = f'${price_floor:.0f}+'                          # the book's price floor: the rows below count signals above it
    rows = [[1, f'gate pass rate, {fl} signals', f'{n_ord}/{n_gate} = {pass_rate:.0f}%' if n_gate else 'no signal reached the gate', '43% · band 30-55% over >= 50 signals',
             'WATCH n<20' if n_gate < 20 else ('OK' if 30 <= pass_rate <= 55 else ('ESCALATE' if n_gate >= 50 else 'WATCH'))],
            [2, f'signals/day reaching the gates, {fl}', f'{reached} today', 'median 11-25 · zero-signal days ~0', 'ESCALATE none reached' if reached == 0 else ('WATCH <5' if reached < 5 else 'OK')]]
    na = 'n/a dry'
    if dry_mode:
        rows += [[k, lab, na, band, 'n/a'] for k, lab, band in ((3, 'fills/day', '4.6-6.6 per day, 22-32 per wk'), (4, 'fill rate of submitted orders', '~100% of ask<=cap · alert <85%'),
                 (5, 'entry fill vs the open of the fill minute', 'median <= +8 bps, mean <= +15'), (6, 'ask at decision vs the next open', 'median +6 bps'),
                 (7, 'TP-fill rate', '33-39% · live >= spec'), (8, 'stop rate / eod rate', '41-45% / 20-22%'), (9, 'stop fill vs stop price', 'modeled -18 bps · alert mean worse than -40'),
                 (10, 'eod fill vs the 15:55 open', 'minus half spread, ~-8 bps'), (11, 'mean R per trade, net realized', '+0.25 to +0.35 · no verdict before 150 trades'),
                 (12, 'WR', '48-53%'), (13, 'weekly R', '+6 to +10 · worst -8 to -12'))]
    else:
        T = closed_trades(trades_db, since, day, strategy); D = [r for r in T if r['trade_date'] == day]
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
    rows += [[14, 'miss rate vs spec', f"{c['missed']} MISSED lines today · full audit: scripts/hod_break_miss_audit.py" + ('' if strategy == 'hod_break' else f' --book {strategy}'), '0', 'ESCALATE' if c['missed'] else 'OK'],
             [15, f'spread at decision, passing {fl}', f'median {med_spread:.0f} bps over {len(spreads)}' if spreads else 'no passing signal', 'median 17 bps · sfr median 0.082', 'n/a' if not spreads else ('OK' if med_spread <= 30 else 'WATCH')],
             [16, f'r_min reject rate, {fl} signals', f"{c['rmin']}/{reached} = {rmin_rate:.0f}%" if reached else 'no signal reached', 'a few % · live r = ask - stop is LOOSER than the spec, the rejects are stops within 1% that simulate rejects too',
              'n/a' if not reached else ('OK' if rmin_rate <= 25 else ('ESCALATE' if reached >= 10 and rmin_rate > 50 else 'WATCH'))]]
    return rows


def print_measurables(rows: list) -> None:
    print('\n  LIVE MEASURABLES vs SPEC, REPORT §12:')
    for no, lab, val, band, verdict in rows:
        print(f'  {no:2d} {verdict:18s} {lab}: {val} | spec {band}')


def rx_reject(tag: str = 'HOD'):
    return re.compile(rf'\[{tag}\] (\S+): (stop [\d.]+ within [\d.]+% of the ask|ask [\d.]+ above cap|spread \d+ bps = \d+% of R|spread \d+ bps > \d+)')


RX_REJECT = rx_reject()


def rejection_parity(lines, day, cfg, book) -> int:
    """The engine's fill-level rejections (r_min on the ask, no-chase, the spread gates) are decisions the spec makes on the
    NEXT OPEN (`simulate`: open <= cap, r/open >= min_r_pct; the spread gate on the historical NBBO). Re-run the spec for
    every rejected symbol: 'spec also no trade' = parity; 'spec HAD a trade' = a live-only rejection to count."""
    rej = {}
    for ln in lines:
        m = rx_reject(book.tag).search(ln)
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
        tr = book.simulate(sym, o, h, l, c, v, m, adv.get(sym, 0.0))
        if tr is None: print(f"  {sym:6s} {why:8s} spec: no trade either  OK")
        else:
            n_dev += 1 if why in ('r_min', 'ask') else 0
            print(f"  {sym:6s} {why:8s} spec HAD a trade: fill {tr.entry:.2f} stop {tr.stop:.2f} {tr.reason} {tr.rr:+.2f}R  {'DEVIATION' if why in ('r_min', 'ask') else 'gate'}")
    print(f"  live-only fill rejections where the spec traded: {n_dev}")
    return n_dev


def append_to_pool(day: str, taken_rows: list, dry_run: bool, is_r2g: bool) -> str:
    """Feed the day's EXECUTABLE would-be book into the pooled ramp instrument (frames11 F36).

    `taken_rows`: `run_book` output for the DRY-RUN EXECUTABLE book — the ENGINE's own logged
    signals walked forward on the day's bars, tuples (day_key, entry_m, exit_m, symbol, rr, usd).
    That is the stream frames10 F33 replayed (`r33.log`: 9/14 11 trades, 9/16 7, 9/17 7, 9/18 6),
    and it is the right one: it measures what the engine actually signalled, not what the spec
    would have signalled on REST bars.

    Only the HOD-break book in DRY mode writes: `trading.ramp_pool` treats the dry stream as n and
    band width ONLY (never a P&L clause), and a LIVE HOD book would enter the pool through
    trades.db instead. The red-to-green book has its own reference and is not pooled.
    Idempotent on (day, symbol, entry minute) — re-running this check appends nothing.
    """
    if is_r2g:
        return 'POOL: red_to_green is not a pooled book — nothing appended'
    if not dry_run:
        return 'POOL: live mode — the pool reads trades.db, nothing appended from the sim book'
    try:
        from trading import ramp_pool
        rows = [(r[3], r[4], r[1]) for r in taken_rows]     # (symbol, R, entry minute)
        added = ramp_pool.append_dry_trades(day, rows)
        return (f"POOL: appended {added} of {len(rows)} dry trade(s) to "
                f"{ramp_pool.DRY_POOL_PATH.name} (idempotent on day+symbol+entry minute)")
    except Exception as e:  # noqa: BLE001 - a reporting tool must never break on the pool write
        return f'POOL: append FAILED ({e}) — the pooled ramp line will be short this session'


def main() -> int:
    argv = list(sys.argv[1:]); book_name = book_from_argv(argv)
    day = argv[0] if argv else datetime.now(timezone.utc).astimezone(ET).strftime('%Y-%m-%d')
    cfg = Config()
    Db = __import__('persistence.database', fromlist=['Database']).Database(); trades_db = Db._trades_path
    book = load_book(book_name, cache_path=getattr(Db, '_cache_path', None))
    hb = book.cfg; p = book.params; risk = hb['risk_usd']
    since = R2G_MEASURABLES_SINCE if book.is_r2g else MEASURABLES_SINCE
    lines = journal(day, book)
    dry = {}; live = {}
    RX_D = rx_dry(book.tag); RX_B = rx_buy(book.tag)
    for ln in lines:                                    # FIRST signal per symbol (later re-breaks are by-products)
        m = RX_D.search(ln)
        if m and m.group(1) not in dry: dry[m.group(1)] = m
        m = RX_B.search(ln)
        if m and m.group(2) not in live: live[m.group(2)] = m
    errors = [ln for ln in lines if 'ERROR' in ln or 'Traceback' in ln or 'queue full' in ln]
    fills = [ln for ln in lines if f'{book.live_tag} FILLED' in ln]; exits = [ln for ln in lines if f'{book.live_tag} EXIT' in ln]
    header = 'HOD-BREAK EOD' if not book.is_r2g else '[R2G EOD]'
    print(f"{header} {day} — mode {'DRY' if hb['dry_run'] else 'LIVE'} | dry signals {len(dry)} | live orders {len(live)} | fills {len(fills)} | exits {len(exits)} | errors {len(errors)}")
    if book.is_r2g:
        print(f"  book red_to_green (F6-PDR): prior day from daily_bars for {len(book.prev_day)} symbols | pdr >= {p.pdr_min_pct:.0f}% | level = prior close x {1 + p.level_buffer:.3f} | "
              f"range floor {p.range_floor_pct:.0f}% | target {p.target_r}R | {p.max_per_day}/day {p.max_concurrent} concurrent | last entry {p.last_entry_minute // 60:02d}:{p.last_entry_minute % 60:02d}")
    for ln in errors[:8]: print('  ERR', ln[-200:])
    syms = sorted(set(dry) | set(live))
    rejected = rejection_parity(lines, day, cfg, book)
    if not syms:
        print('  no signals today')
        B = {} if hb['dry_run'] else bars_for(AlpacaClient(cfg.alpaca_api_key, cfg.alpaca_api_secret, paper=cfg.alpaca_paper), sorted({r['symbol'] for r in closed_trades(trades_db, day, day, book.strategy)}), day)
        rows = live_measurables(lines, day, hb['dry_run'], trades_db, B, p.flat_minute, since=since, strategy=book.strategy, price_floor=hb['min_price'])
        print_measurables(r2g_bands(rows) if book.is_r2g else rows); return 0
    alpaca = AlpacaClient(cfg.alpaca_api_key, cfg.alpaca_api_secret, paper=cfg.alpaca_paper)
    B = bars_for(alpaca, syms, day)
    if not hb['dry_run']:
        B.update(bars_for(alpaca, sorted({r['symbol'] for r in closed_trades(trades_db, day, day, book.strategy)} - set(B)), day))
    adv = {r['symbol']: float(r.get('avg_volume_daily') or 0) for r in Db.get_active_universe()}
    tot_r = 0.0; n = 0; mism = 0; specbook = []
    print(f"  {'sym':6s} {'live_lvl':>8s} {'spec_lvl':>8s} {'live_stop':>9s} {'spec_stop':>9s} {'spec_min':>8s} {'exit':>6s} {'R':>6s}  note")
    for sym in syms:
        m = dry.get(sym) or live.get(sym)
        live_level = float(m.group(2) if sym in dry else m.group(3)); live_stop = float(m.group(4) if sym in dry else m.group(5))
        arr = B.get(sym)
        if arr is None:
            print(f"  {sym:6s} {live_level:8.2f} {'?':>8s} — no bars fetched"); continue
        t = book.simulate(sym, *arr, adv.get(sym, 0.0))
        if t is None:
            from trading.hod_break import entry_fill
            sig = book.detect(sym, *arr[:3], arr[4], arr[5], adv.get(sym, 0.0))
            if sig is None: why = ('NO QUALIFYING BREAK on REST bars (the gap-down/pdr precondition, the range floor or the level differ from the engine bars)'
                                   if book.is_r2g else 'NO QUALIFYING BREAK on REST bars (rv/floor/consolidation differ from the engine bars)')
            else:
                nxt = arr[0][sig.bar_idx + 1] if sig.bar_idx + 1 < len(arr[0]) else None
                why = f"first break {int(arr[5][sig.bar_idx]) // 60:02d}:{int(arr[5][sig.bar_idx]) % 60:02d} lvl {sig.level:.2f}: " + ('no next bar' if nxt is None else (f'NO-CHASE, next open {nxt:.2f} > cap' if entry_fill(nxt, sig.level, p) is None else 'stop >= entry or r_min'))
            mism += 1; print(f"  {sym:6s} {live_level:8.2f} {'none':>8s} {live_stop:9.2f} {'':>9s} {'':>8s} {'':>6s} {'':>6s}  SPEC NO TRADE: {why}"); continue
        o, h, l, c, v, mm = arr
        note = '' if abs(t.stop - live_stop) < 0.011 else 'STOP MISMATCH'
        tot_r += t.rr; n += 1; specbook.append((int(mm[t.entry_idx]), int(mm[t.exit_idx]), sym, t.rr))
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
        taken_rows = run_book([(0, em, xm, sym, rr, usd) for em, xm, sym, rr, usd in dbook], p.max_per_day, p.max_concurrent)
        taken = [(sym, rr, usd) for _, _, _, sym, rr, usd in taken_rows]
        dr = sum(r for _, r, _ in taken); dusd = sum(u for _, _, u in taken); allr = sum(r for *_, r, _ in dbook); allusd = sum(u for *_, u in dbook)
        print(f"  DRY-RUN all filled signals: {len(dbook)}, {allr:+.1f}R, ${allusd:+,.0f} at the logged sizes")
        print(f"  DRY-RUN EXECUTABLE book (first {p.max_per_day}/day, {p.max_concurrent} concurrent, logged sizes): {len(taken)} trades, {dr:+.1f}R, ${dusd:+,.0f} | {[(s_, round(r, 2)) for s_, r, _ in taken]}")
        print(f"  GATE 6 on the DRY-RUN book: {'PASS' if dr > 0 else 'FAIL'}")
        print('  ' + append_to_pool(day, taken_rows, dry_run=hb['dry_run'], is_r2g=book.is_r2g))
        for frac, floor in (() if book.is_r2g else ((0.15, 5.0), (0.15, 20.0), (0.10, 20.0))):   # the HOD spread study's gates; the F6 book has neither
            rows = [(0, em, xm, sym, rr, usd) for em, xm, sym, rr, usd, sf in gated if sf <= frac and float(dry[sym].group(2)) >= floor]
            taken = [(sym, rr, usd) for _, _, _, sym, rr, usd in run_book(rows, p.max_per_day, p.max_concurrent)]
            print(f"  DRY-RUN book with spread <= {frac:.0%} of R and price >= ${floor:.0f}: {len(taken)} trades, {sum(r for _, r, _ in taken):+.1f}R, ${sum(u for _, _, u in taken):+,.0f} | signals passing {sum(1 for g in gated if g[5] <= frac and float(dry[g[2]].group(2)) >= floor)}/{len(gated)}")
    if n:
        print(f"\n  all spec trades on signalled symbols: {n}, {tot_r:+.1f}R = ${tot_r * risk:+,.0f} | spec-has-no-trade {mism}")
        # the EXECUTABLE would-be book: first-come, max_per_day, max_concurrent (dry mode never counts entries)
        taken = [(sym, rr) for _, _, _, sym, rr in run_book([(0, em, xm, sym, rr) for em, xm, sym, rr in specbook], p.max_per_day, p.max_concurrent)]
        br = sum(r for _, r in taken)
        print(f"  EXECUTABLE would-be book (first {p.max_per_day}/day, {p.max_concurrent} concurrent, ${risk:.0f} risk): {len(taken)} trades, {br:+.1f}R = ${br * risk:+,.0f} | {[(s_, round(r, 2)) for s_, r in taken]}")
        print(f"  GATE 6 (positive would-be day): {'PASS' if br > 0 else 'FAIL'}")
    rows = live_measurables(lines, day, hb['dry_run'], trades_db, B, p.flat_minute, since=since, strategy=book.strategy, price_floor=hb['min_price'])
    print_measurables(r2g_bands(rows) if book.is_r2g else rows)
    return 0


if __name__ == '__main__':
    sys.exit(main())
