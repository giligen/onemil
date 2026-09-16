#!/usr/bin/env python3
"""Corrected P&L: re-book the F6 candidates under each fill correction and cost model.
Also the instrument_id (ticker reuse) and prev_close-definition checks. Read-only."""
import os, sys
import numpy as np, pandas as pd
ROOT = '/home/ec2-user/onemil'; os.chdir(ROOT); sys.path.insert(0, ROOT)
from trading.hod_break import run_book
A = 'research/bf_zero2/audit_data'
R = pd.read_csv(f'{A}/f6_fill_audit.csv', dtype={'day': str, 'symbol': str})
R['split'] = np.where(R.day < '2026-01-01', 'TRAIN', np.where(R.day < '2026-05-31', 'VAL', 'TEST'))
R['split'] = np.where(R.day < '2026-01-01', 'TRAIN', np.where(R.day < '2026-06-01', 'VAL', 'TEST'))
R['wk'] = pd.to_datetime(R.day).dt.to_period('W-FRI').astype(str)
# the scoring population filters (price >= 5, entry <= 14:01, R >= 1%, range-so-far >= 5%)
R = R[(R.entry >= 5) & (R.entry_m <= 841) & (R.r_pct >= 1.0) & (R.range_so_far_incl >= 5)].reset_index(drop=True)
print('population', len(R), R.groupby('split').size().to_dict(), flush=True)
WK = {s: R[R.split == s].wk.nunique() for s in ('TRAIN', 'VAL', 'TEST')}


def book(d, rrcol, whycol, xmcol, rcol, ein, eout, per_day=4, conc=4, tag=''):
    d = d[d[rrcol].notna()].copy()
    rp = d[rcol].clip(lower=0.05)
    d['netR'] = d[rrcol] - 0.5 * ein / 100.0 / rp - np.where(d[whycol] == 'target', 0.0, 0.5 * eout / 100.0 / rp)
    rows = [(r.day, int(r.entry_m), int(getattr(r, xmcol)), r.symbol, r.netR, r.wk, r.split) for r in d.itertuples()]
    t = pd.DataFrame(run_book(rows, per_day, conc), columns=['day', 'em', 'xm', 'symbol', 'net', 'wk', 'split'])
    out = []
    for sp in ('TRAIN', 'VAL', 'TEST'):
        x = t[t.split == sp]
        if not len(x): out.append(f'{sp} n/a'); continue
        w = x.groupby('wk').net.sum().reindex(sorted(R[R.split == sp].wk.unique())).fillna(0)
        tt = x.net.mean() / (x.net.std() / np.sqrt(len(x)))
        out.append(f'{sp} n{len(x):5d} {x.net.mean():+.3f}R {w.mean():+5.1f}/wk t{tt:+5.2f}')
    print(f'{tag:58s} ' + ' | '.join(out), flush=True)
    return t


print('\n=== reproduce the claim (as-is fill, score3 costs, 4/day 4 concurrent)')
book(R, 'rr_asis', 'why_asis', 'xm_asis', 'r_pct', 0, 40, tag='V0 as-is  (claim: +0.130/+0.322/+0.350)')

print('\n=== FILL CORRECTIONS (score3 cost model kept, so only the fill changes)')
book(R, 'rr_v1', 'why_v1', 'xm_v1', 'r_pct_v1', 0, 40, tag='V1 fill floored at the signal bar low')
sub = R[R.crossed_bar0 == 0]
book(sub, 'rr_asis', 'why_asis', 'xm_asis', 'r_pct', 0, 40, tag='V2 drop signals whose level was crossed in bar 0')
book(sub, 'rr_v1', 'why_v1', 'xm_v1', 'r_pct_v1', 0, 40, tag='V1+V2 both')
book(R, 'rr_v3', 'why_v3', 'xm_v3', 'r_pct_v3', 0, 40, tag='V3 pre-registered fill: next bar open, cap +0.6%')

print('\n=== FILL + the pre-registered cost model (half 40 bps in, half out on non-target)')
book(R, 'rr_v1', 'why_v1', 'xm_v1', 'r_pct_v1', 40, 40, tag='V1 + full cost')
book(sub, 'rr_v1', 'why_v1', 'xm_v1', 'r_pct_v1', 40, 40, tag='V1+V2 + full cost')
book(R, 'rr_v3', 'why_v3', 'xm_v3', 'r_pct_v3', 40, 40, tag='V3 + full cost')
book(R, 'rr_v3', 'why_v3', 'xm_v3', 'r_pct_v3', 57, 57, tag='V3 + full cost at the measured mean spread')

print('\n=== the pre-registered book cap (12/day, 4 concurrent)')
book(R, 'rr_asis', 'why_asis', 'xm_asis', 'r_pct', 0, 40, per_day=12, tag='V0 as-is, 12/day')
book(R, 'rr_v1', 'why_v1', 'xm_v1', 'r_pct_v1', 40, 40, per_day=12, tag='V1 + full cost, 12/day')
book(R, 'rr_v3', 'why_v3', 'xm_v3', 'r_pct_v3', 40, 40, per_day=12, tag='V3 + full cost, 12/day')

print('\n=== causal range-so-far (exclude the signal bar own extremes)')
R2 = R[R.range_so_far_excl_entry >= 5]
print('rows kept', len(R2), 'of', len(R), flush=True)
book(R2, 'rr_asis', 'why_asis', 'xm_asis', 'r_pct', 0, 40, tag='V0 with a strictly causal 5% floor')
book(R2[R2.crossed_bar0 == 0], 'rr_v1', 'why_v1', 'xm_v1', 'r_pct_v1', 40, 40, tag='V1+V2+causal floor+full cost')

print('\n=== ticker reuse / instrument_id continuity on the traded symbol-days')
daily = pd.read_parquet('data/research/databento/equs_daily_2025_2026.parquet',
                        columns=['bar_date', 'symbol', 'instrument_id', 'close'])
daily['bar_date'] = daily.bar_date.astype(str).str[:10]
daily = daily[daily.symbol.notna()].sort_values(['symbol', 'bar_date']).reset_index(drop=True)
g = daily.groupby('symbol')
daily['prev_iid'] = g.instrument_id.shift(1)
daily['prev_date'] = g.bar_date.shift(1)
daily['prev_close'] = g.close.shift(1)
k = daily.set_index(['symbol', 'bar_date'])
j = R[['symbol', 'day']].drop_duplicates().set_index(['symbol', 'day']).join(
    k[['prev_iid', 'instrument_id', 'prev_date', 'prev_close']]).reset_index()
j['iid_change'] = (j.prev_iid != j.instrument_id)
j['gap_days'] = (pd.to_datetime(j.day) - pd.to_datetime(j.prev_date)).dt.days
print('traded symbol-days whose prev row has a DIFFERENT instrument_id: %d of %d (%.2f%%)' % (
    j.iid_change.sum(), len(j), 100 * j.iid_change.mean()), flush=True)
print('prev_date gap (calendar days):', j.gap_days.describe(percentiles=[.5, .95, .99]).round(1).to_dict(), flush=True)
print('prev row more than 5 calendar days back: %d (%.2f%%)' % ((j.gap_days > 5).sum(), 100 * (j.gap_days > 5).mean()), flush=True)
print('duplicate (symbol,day) rows in the daily file:', int(daily.duplicated(['symbol', 'bar_date']).sum()), flush=True)
print('DONE', flush=True)
