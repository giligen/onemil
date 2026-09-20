#!/usr/bin/env python3
"""exec_cost cell: MOC exit at 16:00 (no spread) vs marketable time-exit at 15:45/15:55
(pays half-spread), for the ORB honest book and the BF regen-7 Stage-2 book.
Scope declared in PREREG.md: BF gets the full bar-walk (has entry/exit time + stop_loss +
shares); ORB honest book has none of those columns -> spread-swap-only estimate, no
stop-window walk (the declared caveat).
"""
import csv
import sqlite3
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import numpy as np

ROOT = '/home/ec2-user/onemil/'
ET = ZoneInfo('America/New_York')
UTC = ZoneInfo('UTC')

con = sqlite3.connect('file:' + ROOT + 'data/cache.db?mode=ro', uri=True, timeout=30)
cur = con.cursor()

# ---- half-spread table (FULL spread as % of price; med column) ----
mt = {}
with open(ROOT + 'research/mature_method/frames14/f45_minute_table.csv') as f:
    for row in csv.DictReader(f):
        mt[int(row['clock_m'])] = float(row['med'])
hs_1545_pct = mt[945] / 2.0 / 100.0   # half-spread as fraction of price
hs_1555_pct = mt[955] / 2.0 / 100.0
print(f'half-spread @15:45 = {hs_1545_pct*1e4:.2f} bp of price; @15:55 = {hs_1555_pct*1e4:.2f} bp', flush=True)


def daily_close(symbol, date_str):
    cur.execute('SELECT close FROM daily_bars WHERE symbol=? AND bar_date=?', (symbol, date_str))
    r = cur.fetchone()
    return float(r[0]) if r else None


def bar_close_1559(symbol, date_str):
    d = datetime.strptime(date_str, '%Y-%m-%d').replace(tzinfo=ET, hour=15, minute=59)
    u0 = d.astimezone(UTC).isoformat()
    cur.execute("SELECT close FROM intraday_bars_1min WHERE symbol=? AND bar_date=? AND timestamp>=? "
                "ORDER BY timestamp ASC LIMIT 1", (symbol, date_str, u0))
    r = cur.fetchone()
    return float(r[0]) if r else None


def bars_between(symbol, date_str, t0_et_str, t1_et_str):
    """1-min bars strictly after t0 (HH:MM ET) up to and incl t1 (HH:MM ET), same date."""
    hh0, mm0 = map(int, t0_et_str.split(':')[:2])
    hh1, mm1 = map(int, t1_et_str.split(':')[:2])
    d0 = datetime.strptime(date_str, '%Y-%m-%d').replace(tzinfo=ET, hour=hh0, minute=mm0) + timedelta(minutes=1)
    d1 = datetime.strptime(date_str, '%Y-%m-%d').replace(tzinfo=ET, hour=hh1, minute=mm1)
    u0, u1 = d0.astimezone(UTC).isoformat(), d1.astimezone(UTC).isoformat()
    cur.execute('SELECT timestamp, open, high, low, close FROM intraday_bars_1min '
                'WHERE symbol=? AND bar_date=? AND timestamp>=? AND timestamp<=? ORDER BY timestamp ASC',
                (symbol, date_str, u0, u1))
    return cur.fetchall()


def split_of(date_str):
    if date_str < '2025-01-01':
        return None
    if date_str <= '2025-12-31':
        return '2025'
    if '2026-01-01' <= date_str <= '2026-05-31':
        return '2026H1(Jan-May)'
    return None  # TEST sealed, or beyond scope


def clustered_t(deltas, dates):
    by_day = {}
    for d, x in zip(dates, deltas):
        by_day.setdefault(d, []).append(x)
    day_means = np.array([np.mean(v) for v in by_day.values()])
    n = len(day_means)
    if n < 2:
        return float('nan'), n
    se = day_means.std(ddof=1) / np.sqrt(n)
    t = day_means.mean() / se if se > 0 else float('nan')
    return t, n


def mdd(pnls):
    c = np.cumsum(pnls)
    peak = np.maximum.accumulate(np.concatenate(([0.0], c)))[1:]
    dd = c - peak
    return float(dd.min()) if len(dd) else 0.0


report_lines = []
report_lines.append('# exec_cost REPORT.md — MOC exit cell (ORB + BF)\n')

# =========================== ORB ===========================
orb_rows = list(csv.DictReader(open(ROOT + 'analysis_results/orb_bplus_book.csv')))
orb_time = [r for r in orb_rows if r['exit_reason'] == 'eod']
orb_scale = [r for r in orb_rows if r['exit_reason'] == 'scale_eod']
print(f'ORB: {len(orb_time)} pure eod, {len(orb_scale)} scale_eod (excluded, see caveat)', flush=True)

orb_split_stats = {}
orb_close_diff_flags = []
for r in orb_time:
    sym, date_s = r['symbol'], r['date']
    sp = split_of(date_s)
    if sp is None:
        continue
    entry = float(r['entry_price'])
    pnl_pct = float(r['pnl_pct'])
    pnl_old = float(r['pnl'])
    if pnl_pct == 0:
        continue
    exit_px = entry * (1 + pnl_pct / 100.0)
    shares = pnl_old / (exit_px - entry)
    dclose = daily_close(sym, date_s)
    if dclose is None:
        continue
    c1559 = bar_close_1559(sym, date_s)
    if c1559 and c1559 != 0:
        if abs(dclose - c1559) / c1559 > 0.005:
            orb_close_diff_flags.append(1)
        else:
            orb_close_diff_flags.append(0)
    hs_dollars = exit_px * hs_1545_pct
    mid_at_1545 = exit_px + hs_dollars   # sell nets mid - half_spread -> mid = recorded + hs
    new_px = dclose  # MOC, no spread
    delta_pnl = shares * (new_px - exit_px)
    pnl_new = pnl_old + delta_pnl
    orb_split_stats.setdefault(sp, []).append((date_s, pnl_old, pnl_new, delta_pnl))

report_lines.append('## ORB (analysis_results/orb_bplus_book.csv)')
report_lines.append('Scope: pure `eod` exits only (n below); `scale_eod` excluded (no partial-shares '
                     'column to split the leg) — count-only context: %d scale_eod trades not computed.' % len(orb_scale))
for sp, rows in orb_split_stats.items():
    old = np.array([x[1] for x in rows])
    new = np.array([x[2] for x in rows])
    delta = new - old
    dates = [x[0] for x in rows]
    t, ndays = clustered_t(delta, dates)
    m_old = mdd(old)
    m_new = mdd(new)
    report_lines.append(
        f'- **{sp}**: n={len(rows)} time-exits | old net ${old.sum():,.0f} -> new ${new.sum():,.0f} | '
        f'Delta ${delta.sum():,.0f} ({delta.mean():+.1f}/trade, t={t:.2f} n_days={ndays}) | '
        f'MDD old ${m_old:,.0f} -> new ${m_new:,.0f} | stops-in-window: N/A (no stop/shares data, see caveat)')
cd = np.array(orb_close_diff_flags)
report_lines.append(f'- daily-close vs 15:59-bar-close differs >0.5% on {cd.mean()*100:.1f}% of ORB time-exit trades (n={len(cd)})'
                     if len(cd) else '- daily-close vs 15:59 comparison: no data')

# =========================== BF ===========================
bf_rows = list(csv.DictReader(open(ROOT + 'research/bf_stage2_regen7_raw_20260905.csv')))
bf_time = [r for r in bf_rows if str(r.get('exit_time_et', '')).startswith('15:5')]
print(f'BF: {len(bf_rows)} total rows, {len(bf_time)} with exit_time_et in 15:5x '
      f'(exit_reason set: {sorted(set(r["exit_reason"] for r in bf_time))})', flush=True)

report_lines.append('\n## Bull flag (research/bf_stage2_regen7_raw_20260905.csv, regen-7 Stage-2)')
if not bf_time:
    reasons = sorted(set(r['exit_reason'] for r in bf_rows))
    report_lines.append(f'- **n=0 time-exits found** (no row has exit_time_et in 15:5x; exit_reason set '
                         f'in this book = {reasons}). Cell is VACUOUS for BF regen-7 — this honest book '
                         f'has no trades that reach the 15:55 flat; Pass bar trivially holds (Delta=$0, '
                         f'MDD unchanged) but there is nothing to ship.')
else:
    bf_split_stats = {}
    bf_close_diff_flags = []
    bf_stop_fired = 0
    for r in bf_time:
        sym, date_s = r['symbol'], r['date']
        sp = split_of(date_s)
        if sp is None:
            continue
        exit_px = float(r['exit_price'])
        shares_total = float(r['shares'])
        partial_sh = float(r['partial_shares']) if r.get('partial_taken') in ('1', 'True', 'true') and r.get('partial_shares') else 0.0
        exiting_shares = shares_total - partial_sh
        stop_loss = float(r['stop_loss']) if r.get('stop_loss') not in (None, '') else None
        old_exit_time = r['exit_time_et']
        pnl_old = float(r['pnl'])
        dclose = daily_close(sym, date_s)
        if dclose is None:
            continue
        c1559 = bar_close_1559(sym, date_s)
        if c1559 and c1559 != 0:
            bf_close_diff_flags.append(1 if abs(dclose - c1559) / c1559 > 0.005 else 0)
        hs_dollars = exit_px * hs_1555_pct
        new_px = dclose
        fired = False
        if stop_loss is not None:
            bars = bars_between(sym, date_s, old_exit_time, '15:59')
            for ts, o, h, l, c in bars:
                if l <= stop_loss:
                    new_px = o - (o * hs_1555_pct)  # stop-type exit still charged half-spread
                    fired = True
                    bf_stop_fired += 1
                    break
        delta_pnl = exiting_shares * (new_px - exit_px)
        pnl_new = pnl_old + delta_pnl
        bf_split_stats.setdefault(sp, []).append((date_s, pnl_old, pnl_new, delta_pnl))
    for sp, rows in bf_split_stats.items():
        old = np.array([x[1] for x in rows])
        new = np.array([x[2] for x in rows])
        delta = new - old
        dates = [x[0] for x in rows]
        t, ndays = clustered_t(delta, dates)
        report_lines.append(
            f'- **{sp}**: n={len(rows)} time-exits | old net ${old.sum():,.0f} -> new ${new.sum():,.0f} | '
            f'Delta ${delta.sum():,.0f} ({delta.mean():+.1f}/trade, t={t:.2f} n_days={ndays}) | '
            f'MDD old ${mdd(old):,.0f} -> new ${mdd(new):,.0f}')
    report_lines.append(f'- stops fired in the extra 15:55-16:00 window (all splits): {bf_stop_fired}')
    cdb = np.array(bf_close_diff_flags)
    if len(cdb):
        report_lines.append(f'- daily-close vs 15:59-bar-close differs >0.5% on {cdb.mean()*100:.1f}% of BF time-exit trades (n={len(cdb)})')

report_lines.append('\n## Caveat')
report_lines.append('ORB honest book (analysis_results/orb_bplus_book.csv) carries no entry_time/'
                     'exit_time/stop_loss/shares columns — it is a features+pnl summary, not a per-'
                     'minute ledger. shares/exit_price were backed out algebraically from entry_price '
                     'and pnl_pct; the extra-window stop-walk required by the PREREG could NOT be run '
                     'for ORB (declared as scope-limit in PREREG before running) — the ORB Delta above '
                     'is a spread-swap-only estimate (assumes price drifts cleanly to the close with no '
                     'intervening stop), not the full spec. BF got the full bar-walk.')

with open(ROOT + 'research/exec_cost/REPORT.md', 'w') as f:
    f.write('\n'.join(report_lines) + '\n')

print('\n'.join(report_lines))
print('\nDONE', flush=True)
