"""Find the winners (PREREG.md, cells 1,406-1,411): day selection by gapper-universe breadth.

Stage 1 (one DB pass, market-hours gated): per day and minute, over the causal gapper universe, count names with a
bar so far, names whose last close is above their 09:30 open (BR), names whose high of day was set in the last 30
minutes (HH). Writes breadth.parquet (day, m, n_bar, n_above, n_hh, NG, BR, HH).
Stage 2: attach NG / BR(d) / HH(d) at each trade's decision minute d (base book: signal_m; HOD book: entry_m-1),
cut at the TRAIN-H1 median, score kept vs dropped on TRAIN-H2 and VAL (+ the slotted kept book), tercile tables,
and the TRAIN-only winner-day anatomy. Writes BREADTH.md and cells.json.

Usage: python3 research/day_breadth/breadth.py [--smoke-days N] [--now] [--score-only]
"""
import argparse
import json
import sqlite3
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path('/home/ec2-user/onemil')
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'research/hod_consol'))
import run_consol as rc  # noqa: E402
from adversarial_read import stats  # noqa: E402

M0, M1 = rc.W.OPEN_M, rc.W.CLOSE_M
NM = M1 - M0 + 1
HH_WINDOW = 30
BOOKS = {
    'base': (ROOT / 'research/hod_consol/trades/1400_all.csv', 'signal_m', 0),
    'HOD': (ROOT / 'research/hod_exit_lab/b0_trades.csv', 'entry_m', -1),
}
CELLS = [('1406', 'NG', 'base'), ('1407', 'BR', 'base'), ('1408', 'HH', 'base'),
         ('1409', 'NG', 'HOD'), ('1410', 'BR', 'HOD'), ('1411', 'HH', 'HOD')]


def log(msg):
    print(f'[{time.strftime("%H:%M:%S")}] {msg}', flush=True)


def symbol_minute_flags(d):
    """For one symbol-day's RTH bars: (has_bar, above_open, recent_high) boolean arrays over minutes M0..M1."""
    mi = (d.m.values - M0).astype(int)
    first = mi.min()
    o_rows = d[d.m == M0]
    o930 = float(o_rows.o.iloc[0]) if len(o_rows) else float(d.o.iloc[0])
    close = np.full(NM, np.nan)
    close[mi] = d.c.values
    pos = np.where(~np.isnan(close), np.arange(NM), 0)
    close_ff = close[np.maximum.accumulate(pos)]
    high = np.full(NM, -np.inf)
    high[mi] = d.h.values
    H = np.maximum.accumulate(high)
    touch = (high == H) & np.isfinite(high)
    tau = np.maximum.accumulate(np.where(touch, np.arange(NM), -1))
    has = np.arange(NM) >= first
    above = has & (close_ff > o930)
    hh = has & (tau >= 0) & ((np.arange(NM) - tau) <= HH_WINDOW)
    return has, above, hh, len(o_rows) == 0


def build(smoke_days=0):
    """One pass over the TRAIN+VAL universe symbol-days -> breadth.parquet."""
    cands = [(s, d) for s, d in rc.load_candidates() if rc.split_of(d) in ('TRAIN', 'VAL')]
    days = sorted({d for _, d in cands})
    if smoke_days:
        keep = set(days[:smoke_days])
        cands = [(s, d) for s, d in cands if d in keep]
        days = sorted(keep)
    ng = pd.Series([d for _, d in cands]).value_counts().to_dict()
    acc = {d: np.zeros((3, NM), dtype=np.int32) for d in days}
    sip = sqlite3.connect(rc.BARS_SIP_DB, timeout=30)
    rth = sqlite3.connect(rc.BARS_RTH_DB, timeout=30)
    no_bars = no_open = 0
    t0 = time.time()
    for i, (s, d) in enumerate(cands, 1):
        bars, _ = rc.fetch_day_bars_dual(sip, rth, s, d)
        if bars is None or bars.empty:
            no_bars += 1
            continue
        has, above, hh, missing_open = symbol_minute_flags(bars)
        no_open += int(missing_open)
        acc[d][0] += has
        acc[d][1] += above
        acc[d][2] += hh
        if i % 2000 == 0 or i == len(cands):
            log(f'[pass] {i}/{len(cands)} {time.time() - t0:.0f}s no_bars={no_bars} no_0930_bar={no_open}')
    sip.close()
    rth.close()
    rows = []
    for d in days:
        n_bar, n_above, n_hh = acc[d]
        with np.errstate(divide='ignore', invalid='ignore'):
            br, hhs = n_above / n_bar, n_hh / n_bar
        rows.append(pd.DataFrame(dict(day=d, m=np.arange(M0, M1 + 1), n_bar=n_bar, n_above=n_above, n_hh=n_hh,
                                      NG=ng.get(d, 0), BR=br, HH=hhs)))
    out = pd.concat(rows, ignore_index=True)
    out.to_parquet(HERE / 'breadth.parquet', index=False)
    log(f'[build] breadth.parquet: {len(days)} days, {len(cands)} symbol-days, no_bars={no_bars}, '
        f'no_0930_bar={no_open} (first bar open used)')
    return out


def load_book(name, br):
    """Book trades (TRAIN+VAL) with NG/BR/HH attached at the causal decision minute."""
    path, col, shift = BOOKS[name]
    t = pd.read_csv(path, keep_default_na=False, na_values=[''])
    t = t[t.split.isin(['TRAIN', 'VAL'])].copy()
    t['net_R'] = pd.to_numeric(t.net_R, errors='coerce')
    t = t.dropna(subset=['net_R'])
    t['dm'] = (pd.to_numeric(t[col]) + shift).clip(M0, M1).astype(int)
    t = t.merge(br[['day', 'm', 'NG', 'BR', 'HH']], left_on=['day', 'dm'], right_on=['day', 'm'], how='left')
    miss = t.BR.isna().mean()
    if miss > 0:
        log(f'[WARNING] {name}: {miss:.1%} of trades have no breadth value (day outside the universe or no bars)')
    return t.dropna(subset=['BR'])


def cell(t, measure, val_weeks):
    """Pass-bar evaluation for one book x measure with the TRAIN-H1 median cut."""
    h1 = t[(t.split == 'TRAIN') & (t.half == 'H1')]
    cut = float(h1[measure].median())
    res = dict(cut=cut)
    for part, sel in (('H2', (t.split == 'TRAIN') & (t.half == 'H2')), ('VAL', t.split == 'VAL')):
        s = t[sel]
        kept, dropped = s[s[measure] >= cut], s[s[measure] < cut]
        keep_mask = rc.simulate_slots(kept)
        slotted = kept.loc[keep_mask.index[keep_mask]]
        res[part] = dict(kept=stats(kept.net_R, kept.day), dropped=stats(dropped.net_R, dropped.day),
                         slotted=stats(slotted.net_R, slotted.day), n_slotted=len(slotted))
    res['fills_wk'] = res['VAL']['n_slotted'] / val_weeks if val_weeks else 0.0
    k2, kv = res['H2']['kept'], res['VAL']['kept']
    res['pass'] = bool(k2.get('mean', -1) >= 0.10 and kv.get('mean', -1) >= 0.10 and kv.get('t_cluster', 0) >= 2
                       and res['H2']['dropped'].get('mean', 1) < 0 and res['VAL']['dropped'].get('mean', 1) < 0
                       and res['H2']['slotted'].get('mean', -1) > 0 and res['VAL']['slotted'].get('mean', -1) > 0
                       and res['fills_wk'] >= 3)
    edges = h1[measure].quantile([1 / 3, 2 / 3]).values
    terc = {}
    for part, sel in (('H1', (t.split == 'TRAIN') & (t.half == 'H1')), ('H2', (t.split == 'TRAIN') & (t.half == 'H2')),
                      ('VAL', t.split == 'VAL')):
        s = t[sel]
        b = np.digitize(s[measure].values, edges)
        terc[part] = [round(float(s.net_R[b == k].mean()), 3) if (b == k).any() else None for k in range(3)]
    res['terciles'] = terc
    return res


def spy_open_to_1030():
    """SPY return from the 09:30 open to the 10:29 bar close, per day."""
    s = pd.read_parquet(ROOT / 'research/index_orb/cache/SPY_1min.parquet')
    s['day'] = s.timestamp.dt.strftime('%Y-%m-%d')
    s['m'] = s.timestamp.dt.hour * 60 + s.timestamp.dt.minute
    o = s[s.m == 570].set_index('day').open
    c = s[s.m == 629].set_index('day').close
    return (c / o - 1).dropna()


def anatomy(t, br, spy):
    """TRAIN only: top-decile days by summed net R vs the rest, on NG, BR(10:30), HH(10:30), SPY, weekday."""
    tr = t[t.split == 'TRAIN']
    day_R = tr.groupby('day').net_R.sum()
    top = set(day_R[day_R >= day_R.quantile(0.9)].index)
    at1030 = br[br.m == 629].set_index('day')
    rows = []
    for grp, days in (('top 10 % days', [d for d in day_R.index if d in top]),
                      ('other days', [d for d in day_R.index if d not in top])):
        a = at1030.reindex(days)
        wd = pd.to_datetime(pd.Series(days)).dt.day_name().value_counts(normalize=True).round(2).to_dict()
        rows.append(dict(group=grp, n_days=len(days), day_R_mean=round(float(day_R.reindex(days).mean()), 2),
                         NG=round(float(a.NG.mean()), 1), BR_1030=round(float(a.BR.mean()), 3),
                         HH_1030=round(float(a.HH.mean()), 3),
                         SPY_1030_pct=round(float(spy.reindex(days).mean() * 100), 3), weekday=wd))
    share = float(day_R[day_R.index.isin(top)].sum() / day_R.sum()) if day_R.sum() else float('nan')
    return rows, share


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--smoke-days', type=int, default=0)
    ap.add_argument('--now', action='store_true', help='owner override of the market-hours DB blackout')
    ap.add_argument('--score-only', action='store_true')
    a = ap.parse_args()
    if a.score_only:
        br = pd.read_parquet(HERE / 'breadth.parquet')
    else:
        if not a.now and not a.smoke_days:
            rc.wait_for_db_window()
        br = build(a.smoke_days)
    spy = spy_open_to_1030()
    lines = ['# BREADTH.md — find the winners (PREREG.md, cells 1,406–1,411)', '',
             'Kept = measure ≥ TRAIN-H1 median. Stats: trade-weighted mean, t = day-clustered trade-weighted. '
             'SLOTTED = first 12/day, 4 concurrent, applied after the filter.', '']
    results = []
    for book in ('base', 'HOD'):
        t = load_book(book, br)
        val_weeks = t[t.split == 'VAL'].wk.nunique()
        lines += [f'## Book: {book} ({len(t)} trades, TRAIN {int((t.split == "TRAIN").sum())} / '
                  f'VAL {int((t.split == "VAL").sum())})', '',
                  '| cell | measure | cut | H2 kept (t) | H2 dropped | VAL kept (t) | VAL dropped | H2 / VAL slotted '
                  '| fills/wk | terciles H1 / H2 / VAL | PASS |', '|' + '---|' * 11]
        for cid, meas, bk in CELLS:
            if bk != book:
                continue
            r = cell(t, meas, val_weeks)
            r.update(id=cid, measure=meas, book=book)
            results.append(r)
            f = lambda d: f"{d.get('mean', float('nan')):+.3f}"  # noqa: E731
            row = (f"| {cid} | {meas} | {r['cut']:.3f} | {f(r['H2']['kept'])} ({r['H2']['kept'].get('t_cluster', float('nan')):+.2f}) "
                   f"| {f(r['H2']['dropped'])} | {f(r['VAL']['kept'])} ({r['VAL']['kept'].get('t_cluster', float('nan')):+.2f}) "
                   f"| {f(r['VAL']['dropped'])} | {f(r['H2']['slotted'])} / {f(r['VAL']['slotted'])} | {r['fills_wk']:.1f} "
                   f"| {r['terciles']['H1']} / {r['terciles']['H2']} / {r['terciles']['VAL']} | {r['pass']} |")
            lines.append(row)
            print(row, flush=True)
        rows, share = anatomy(t, br, spy)
        lines += ['', f'### Winner-day anatomy, TRAIN only ({book}): top 10 % of days = {share:.0%} of TRAIN R', '',
                  '| group | days | mean day R | NG | BR 10:30 | HH 10:30 | SPY 09:30→10:30 % | weekday mix |',
                  '|---|---|---|---|---|---|---|---|']
        for rw in rows:
            lines.append(f"| {rw['group']} | {rw['n_days']} | {rw['day_R_mean']:+.2f} | {rw['NG']} | {rw['BR_1030']} "
                         f"| {rw['HH_1030']} | {rw['SPY_1030_pct']:+.3f} | {rw['weekday']} |")
        print('\n'.join(lines[-4:]), flush=True)
        lines.append('')
    (HERE / 'BREADTH.md').write_text('\n'.join(lines) + '\n')
    (HERE / 'cells.json').write_text(json.dumps(results, indent=1, default=str))
    log('ALL DONE')


if __name__ == '__main__':
    main()
