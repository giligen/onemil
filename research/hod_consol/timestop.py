"""Cells 1,403-1,405 (PREREG_TIMESTOP.md): base entry + NO-BREAK time stop, from run_consol's cached artifacts.

No database access: signals.parquet (H_t, split, half, wk, sao, ...) and paths.parquet (full-day 1-min bars per
signal) written by run_consol.py. Per cell it prints and writes (TIMESTOP.md, trades/<cell>_all.csv): the
all-signal and SLOTTED books (adversarial_read.stats), slotted halves, ex-top-5 %, fills/week, exit mix, the D1
placebo under the IDENTICAL exit, and the pass-bar verdict; plus the time-to-first-break exhibit.

Usage: python3 research/hod_consol/timestop.py
"""
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run_consol as rc  # noqa: E402
from adversarial_read import stats  # noqa: E402

CELLS = {'1403': 15, '1404': 30, '1405': 60}
EOD = rc.EOD_M


def fill_nobreak(entry, stop, R, rows, H, entry_m, n_min):
    """C1 + no-break stop over bars after the entry bar (time-ordered itertuples with m,o,h,l,c).

    Inside a bar: EOD exit at the open; no-break exit at the open of the first bar after entry_m+n_min when no
    close > H has happened; stop (fill at min(open, stop)); target (+2R). The break flag is set by a bar's CLOSE
    after that bar is processed, so it acts from the next bar (causal)."""
    target = entry + 2.0 * R
    broke = False
    for row in rows:
        if row.m >= EOD:
            return int(row.m), float(row.o), 'eod'
        if not broke and row.m > entry_m + n_min:
            return int(row.m), float(row.o), 'nobreak'
        if row.l <= stop:
            return int(row.m), float(row.o if row.o <= stop else stop), 'stop'
        if row.h >= target:
            return int(row.m), float(target), 'target'
        if row.c > H:
            broke = True
    return None


def walk_ts(sig, idx, n_min):
    """Per-trade net R for the cell; stop-at-open trades booked at -cost (as cells 1,400-1,402)."""
    out = []
    for r in sig.itertuples():
        spread_mean = 2 * rc.PROXY_HALF_SPREAD_PCT * r.entry
        if r.sao:
            _, net_R = rc.cost_net(r.entry, r.entry, r.R, spread_mean)
            exit_m, why = r.entry_m, 'stop_at_open'
        else:
            key = (r.day, r.symbol)
            if key not in idx.index:
                continue
            g = idx.loc[[key]]
            bars = g[(g.m > r.entry_m) & (g.m <= EOD)]
            if bars.empty:
                continue
            res = fill_nobreak(r.entry, r.stop, r.R, bars.itertuples(), r.H_t, r.entry_m, n_min)
            if res is None:
                continue
            exit_m, exit_px, why = res
            _, net_R = rc.cost_net(r.entry, exit_px, r.R, spread_mean)
        out.append(dict(day=r.day, symbol=r.symbol, entry_m=r.entry_m, exit_m=exit_m, why=why,
                        split=r.split, half=r.half, wk=r.wk, net_R=net_R))
    return pd.DataFrame(out)


def d1_ts(va, idx, n_min, rng):
    """D1 placebo under the identical exit: same name-day, random minute, prior-20-bar-low stop, 1.5 % floor."""
    vals = []
    for r in va.itertuples():
        key = (r.day, r.symbol)
        if key not in idx.index:
            continue
        g = idx.loc[[key]]
        draws = []
        for _ in range(rc.N_DRAWS):
            for _try in range(rc.PLACEBO_MAX_REDRAWS):
                m_r = rng.randint(rc.SCAN_START_M, rc.SCAN_END_M)
                prior = g[(g.m >= m_r - rc.BASE_WINDOW) & (g.m < m_r)]
                eb = g[g.m == m_r]
                if len(prior) < rc.BASE_WINDOW or eb.empty:
                    continue
                entry = float(eb.iloc[0].o)
                stop = float(prior.l.min())
                ok, R = rc._floor_ok(entry, stop)
                if not ok:
                    continue
                H = float(g[g.m < m_r].h.max())
                after = g[(g.m > m_r) & (g.m <= EOD)]
                if after.empty:
                    continue
                res = fill_nobreak(entry, stop, R, after.itertuples(), H, m_r, n_min)
                if res is None:
                    continue
                _, net = rc.cost_net(entry, res[1], R, 2 * rc.PROXY_HALF_SPREAD_PCT * entry)
                if not pd.isna(net):
                    draws.append(net)
                break
        if draws:
            vals.append(float(np.mean(draws)))
    return vals


def time_to_break(sig, idx):
    """Minutes from entry to the first 1-min close > H_t (NaN if never), per non-stop-at-open signal."""
    mins = []
    for r in sig[sig.sao == 0].itertuples():
        key = (r.day, r.symbol)
        if key not in idx.index:
            mins.append(np.nan)
            continue
        g = idx.loc[[key]]
        b = g[(g.m > r.entry_m) & (g.m <= EOD) & (g.c > r.H_t)]
        mins.append(float(b.m.iloc[0] - r.entry_m) if len(b) else np.nan)
    return pd.Series(mins)


def fmt(d):
    return (f"n={d.get('n')} mean={d.get('mean', np.nan):+.3f} t={d.get('t_cluster', np.nan):+.2f} "
            f"day_mean={d.get('day_mean', np.nan):+.3f}")


def main():
    sig = pd.read_parquet(HERE / 'signals.parquet')
    sig = sig[sig.split.isin(['TRAIN', 'VAL'])].copy()          # TEST sealed
    paths = pd.read_parquet(HERE / 'paths.parquet')
    idx = paths.sort_values(['day', 'symbol', 'm']).set_index(['day', 'symbol']).sort_index()
    val_weeks = sig[sig.split == 'VAL'].wk.nunique()
    lines = ['# TIMESTOP.md — cells 1,403–1,405 (PREREG_TIMESTOP.md). Primary = SLOTTED book', '']

    ttb = time_to_break(sig, idx)
    br = ttb.dropna()
    lines += ['## Time to first close above the high (breakers only), TRAIN+VAL', '',
              f'breakers {len(br)} of {len(ttb)} ({len(br) / len(ttb):.0%}); '
              + ', '.join(f'within {k} min: {(br <= k).mean():.0%}' for k in (15, 30, 60, 120)), '']
    for n in (15, 30, 60):
        rest = ttb[~(ttb <= n)]
        lines.append(f'- P(eventual break | no break by {n} min) = {rest.notna().mean():.0%} (n={len(rest)})')
    lines += ['', '## Cells', '', '| cell | N | TRAIN slotted | VAL slotted | TRAIN all | VAL all | halves (slotted) '
              '| ex-top-5 % TR/VAL | fills/wk VAL | D1 VAL | exit mix VAL | PASS |', '|' + '---|' * 12]
    print('\n'.join(lines[:8]), flush=True)
    for cid, n in CELLS.items():
        w = walk_ts(sig, idx, n)
        w.assign(date=w.day, pnl_R=w.net_R).to_csv(HERE / 'trades' / f'{cid}_all.csv', index=False)
        res = {}
        for split in ('TRAIN', 'VAL'):
            s = w[w.split == split]
            keep = rc.simulate_slots(s)
            sl = s.loc[keep.index[keep]]
            res[split] = dict(all=stats(s.net_R, s.day), slot=stats(sl.net_R, sl.day), sl=sl)
        sl_tr, sl_va = res['TRAIN']['sl'], res['VAL']['sl']
        h1, h2 = sl_tr[sl_tr.half == 'H1'].net_R.mean(), sl_tr[sl_tr.half == 'H2'].net_R.mean()
        et5_tr, et5_va = rc.ex_top5(sl_tr.net_R), rc.ex_top5(sl_va.net_R)
        fills_wk = len(sl_va) / val_weeks if val_weeks else 0.0
        va_all = w[w.split == 'VAL']
        d1 = d1_ts(sig[sig.split == 'VAL'], idx, n, random.Random(rc.SEED + int(cid)))
        d1_mean = float(np.mean(d1)) if d1 else np.nan
        st, sv = res['TRAIN']['slot'], res['VAL']['slot']
        passed = (st['mean'] >= 0.10 and sv['mean'] >= 0.10 and sv['t_cluster'] >= 2 and h1 > 0 and h2 > 0
                  and et5_tr > 0 and et5_va > 0 and va_all.net_R.mean() - d1_mean >= 0.10 and fills_wk >= 3)
        mix = va_all.why.value_counts(normalize=True).round(2).to_dict()
        row = (f"| {cid} | {n} | {st['mean']:+.3f} ({st['t_cluster']:+.2f}) | {sv['mean']:+.3f} ({sv['t_cluster']:+.2f}) "
               f"| {res['TRAIN']['all']['mean']:+.3f} | {res['VAL']['all']['mean']:+.3f} | {h1:+.3f} / {h2:+.3f} "
               f"| {et5_tr:+.3f} / {et5_va:+.3f} | {fills_wk:.1f} | {d1_mean:+.3f} | {mix} | {passed} |")
        lines.append(row)
        print(row, flush=True)
    (HERE / 'TIMESTOP.md').write_text('\n'.join(lines) + '\n')
    print('ALL DONE', flush=True)


if __name__ == '__main__':
    main()
