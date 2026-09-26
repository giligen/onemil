#!/usr/bin/env python3
"""Independent rebuild of the insider-events signals (I1-I4), from PREREG.md prose only.

Written WITHOUT reading cells_form4.py, cell_13d.py, their tests or RESULT_*.md.
Inputs read: PREREG.md, the raw SEC quarterly Form 3/4/5 TSVs under data/extracted/,
sc13d.parquet (subject-company mapped 13D index -- NOT purchases.parquet), and the
multiday Panel/cost/trade machinery (run_final.Panel, run_f2_a1.round_trip_cost_bps /
trade_returns).

Outputs: rebuild_signals.csv, rebuild_trades.csv, data/compare_report.json (independent
check vs the builders' signals_form4.csv / signals_13d.csv / trades_form4.csv / trades_13d.csv).
"""
from __future__ import annotations

import glob
import json
import os
import sys
import time

import numpy as np
import pandas as pd

REPO = '/home/ec2-user/onemil'
sys.path.insert(0, f'{REPO}/research/multiday')
from run_final import Panel                                    # noqa: E402
from run_f2_a1 import round_trip_cost_bps, trade_returns        # noqa: E402

DATA = f'{REPO}/research/insider_events/data'
OUT = f'{REPO}/research/insider_events'

SPLITS = {'TRAIN': ('2016-01-01', '2021-12-31'),
          'VAL': ('2022-01-01', '2023-12-31'),
          'TEST': ('2024-01-01', '2026-09-18')}
HOLD = 20
KEYWORDS = ['CEO', 'CFO', 'PRESIDENT', 'CHAIR', 'DIRECTOR']


def log(m):
    print(f'[{time.strftime("%H:%M:%S")}] {m}', flush=True)


def split_of_dates(dates: pd.Series) -> pd.Series:
    out = pd.Series(index=dates.index, dtype=object)
    for name, (a, b) in SPLITS.items():
        m = (dates >= a) & (dates <= b)
        out[m] = name
    return out


# ---------------------------------------------------------------- Form 4 (I1/I2/I3)


def load_form345() -> pd.DataFrame:
    qdirs = sorted(glob.glob(f'{DATA}/extracted/*'))
    subs, nds, ros = [], [], []
    n_no_aff = 0
    for qd in qdirs:
        q = os.path.basename(qd)
        want = ['ACCESSION_NUMBER', 'FILING_DATE', 'DOCUMENT_TYPE',
                'ISSUERCIK', 'ISSUERTRADINGSYMBOL', 'AFF10B5ONE']
        header = pd.read_csv(f'{qd}/SUBMISSION.tsv', sep='\t', nrows=0).columns
        have = [c for c in want if c in header]
        sub = pd.read_csv(f'{qd}/SUBMISSION.tsv', sep='\t', dtype=str, usecols=have)
        if 'AFF10B5ONE' not in sub.columns:
            sub['AFF10B5ONE'] = np.nan   # SEC added this field only from ~2023q1 on;
            n_no_aff += 1                # pre-2023 quarters cannot be filtered on it (see log)
        sub = sub[sub['DOCUMENT_TYPE'] == '4']
        nd = pd.read_csv(f'{qd}/NONDERIV_TRANS.tsv', sep='\t', dtype=str,
                          usecols=['ACCESSION_NUMBER', 'TRANS_DATE', 'TRANS_CODE',
                                   'TRANS_SHARES', 'TRANS_PRICEPERSHARE',
                                   'TRANS_ACQUIRED_DISP_CD'])
        nd = nd[(nd['TRANS_CODE'] == 'P') & (nd['TRANS_ACQUIRED_DISP_CD'] == 'A')]
        ro = pd.read_csv(f'{qd}/REPORTINGOWNER.tsv', sep='\t', dtype=str,
                          usecols=['ACCESSION_NUMBER', 'RPTOWNERCIK', 'RPTOWNERNAME',
                                   'RPTOWNER_RELATIONSHIP', 'RPTOWNER_TITLE'])
        subs.append(sub); nds.append(nd); ros.append(ro)
        log(f'{q}: sub(form4)={len(sub)} nd(P/A)={len(nd)} ro={len(ro)}')
    sub = pd.concat(subs, ignore_index=True).drop_duplicates('ACCESSION_NUMBER')
    nd = pd.concat(nds, ignore_index=True)
    ro = pd.concat(ros, ignore_index=True)
    log(f'RAW TOTAL: sub(form4)={len(sub)} nd(P/A)={len(nd)} ro={len(ro)}; '
        f'{n_no_aff} quarters had NO AFF10B5ONE column (SEC added it ~2023q1; those '
        f'quarters cannot be filtered for planned 10b5-1 trades -- all pass the filter, logged)')

    df = nd.merge(sub, on='ACCESSION_NUMBER', how='inner')
    n_sub_join = len(df)
    # left-join owner: a Form 4 accession normally carries one reporting owner row; a joint
    # filing with >1 owner would fan out here -- rare, kept as-is (each owner scored separately
    # for I2/I3, which is a per-OWNER signal by definition).
    df = df.merge(ro, on='ACCESSION_NUMBER', how='left')
    log(f'after sub-join {n_sub_join}; after owner-join {len(df)}')

    df['shares'] = pd.to_numeric(df['TRANS_SHARES'], errors='coerce')
    df['price'] = pd.to_numeric(df['TRANS_PRICEPERSHARE'], errors='coerce')
    df = df[np.isfinite(df['price']) & (df['price'] > 0) & np.isfinite(df['shares'])].copy()
    df['value'] = df['shares'] * df['price']

    aff = df['AFF10B5ONE'].astype(str).str.strip().str.lower()
    n_before_aff = len(df)
    df = df[~aff.isin(['1', 'true'])].copy()
    log(f'value>0 rows {n_before_aff}; after AFF10B5ONE!=1 filter {len(df)}')

    df['filing_date'] = pd.to_datetime(df['FILING_DATE'], format='%d-%b-%Y', errors='coerce')
    df['trans_date'] = pd.to_datetime(df['TRANS_DATE'], format='%d-%b-%Y', errors='coerce')
    n_before_date = len(df)
    df = df.dropna(subset=['filing_date']).copy()
    log(f'dropped {n_before_date - len(df)} rows with unparseable FILING_DATE')

    df['symbol'] = df['ISSUERTRADINGSYMBOL'].astype(str).str.strip().str.upper()
    df['owner_cik'] = df['RPTOWNERCIK']
    rel = df['RPTOWNER_RELATIONSHIP'].fillna('')
    title = df['RPTOWNER_TITLE'].fillna('')
    df['is_off_dir'] = rel.str.contains('Officer') | rel.str.contains('Director')
    combined = (title + ' ' + rel).str.upper()
    df['title_kw'] = combined.apply(lambda s: any(k in s for k in KEYWORDS))
    log(f'PURCHASE universe (P/A, form 4, price>0, not-10b5-1): {len(df)} rows, '
        f'{df["symbol"].nunique()} distinct issuer symbols, {df["owner_cik"].nunique()} distinct owners')
    return df.reset_index(drop=True)


def attach_sessions(df: pd.DataFrame, p: Panel, date_col: str) -> pd.DataFrame:
    """anchor_sess = session containing/next-after date_col (for same-day clustering windows);
    sig_sess = first session STRICTLY after date_col (the PREREG entry timing for every cell)."""
    d = df.copy()
    sess = p.sess_np
    dt = d[date_col].values.astype('datetime64[D]').astype('datetime64[ns]')
    d['anchor_sess'] = np.searchsorted(sess, dt, side='left')
    d['sig_sess'] = np.searchsorted(sess, dt, side='right')
    d['si'] = d['symbol'].map(p.sidx)
    return d


def pit_ok(d: pd.DataFrame, p: Panel, sess_col: str) -> pd.Series:
    ok = d['si'].notna() & (d[sess_col] >= 0) & (d[sess_col] < p.n_d)
    d2 = d[ok]
    si = d2['si'].to_numpy(dtype=int)
    ss = d2[sess_col].to_numpy(dtype=int)
    gate = p.elig[si, ss] & p.etb[si] & ~p.taint[si, ss]
    out = pd.Series(False, index=d.index)
    out.loc[d2.index] = gate
    return out


# ---------------------------------------------------------------- I1 cluster


def compute_i1(df: pd.DataFrame, p: Panel) -> pd.DataFrame:
    d = attach_sessions(df, p, 'filing_date')
    d = d[pit_ok(d, p, 'sig_sess')].copy()
    d['si'] = d['si'].astype(int)
    d = d.sort_values(['si', 'anchor_sess']).reset_index(drop=True)
    si_a = d['si'].to_numpy()
    a_a = d['anchor_sess'].to_numpy()
    own_a = d['owner_cik'].fillna('').to_numpy()
    val_a = d['value'].to_numpy()
    n = len(d)
    flags = np.zeros(n, dtype=bool)
    nowners_out = np.zeros(n, dtype=np.int32)
    val_out = np.zeros(n, dtype=np.float64)
    i = 0
    while i < n:
        j = i
        while j < n and si_a[j] == si_a[i]:
            j += 1
        sub_a = a_a[i:j]; sub_o = own_a[i:j]; sub_v = val_a[i:j]
        for k in range(len(sub_a)):
            lo = np.searchsorted(sub_a, sub_a[k] - 9, side='left')
            owners = set(sub_o[lo:k + 1])
            owners.discard('')
            tot = sub_v[lo:k + 1].sum()
            nowners_out[i + k] = len(owners)
            val_out[i + k] = tot
            if len(owners) >= 2 and tot >= 100_000.0:
                flags[i + k] = True
        i = j
    d['n_owners'] = nowners_out
    d['window_value'] = val_out
    out = d[flags].copy()
    out['cell'] = 'I1'
    log(f'I1 CLUSTER: {len(out)} triggering purchase-rows (pre symbol/session dedup)')
    return out


# ---------------------------------------------------------------- I2 officer


def compute_i2(df: pd.DataFrame, p: Panel) -> pd.DataFrame:
    d = df[(df['is_off_dir']) & (df['title_kw']) & (df['value'] >= 50_000.0)].copy()
    d = attach_sessions(d, p, 'filing_date')
    d = d[pit_ok(d, p, 'sig_sess')].copy()
    d['si'] = d['si'].astype(int)
    d['n_owners'] = np.nan
    d['cell'] = 'I2'
    log(f'I2 OFFICER: {len(d)} triggering purchase-rows (pre symbol/session dedup)')
    return d


# ---------------------------------------------------------------- I3 opportunistic


def compute_i3(df: pd.DataFrame, p: Panel) -> pd.DataFrame:
    d = df.dropna(subset=['trans_date']).copy()
    d = d.sort_values(['owner_cik', 'symbol', 'trans_date'])
    prev = d.groupby(['owner_cik', 'symbol'])['trans_date'].shift(1)
    gap = (d['trans_date'] - prev).dt.days
    d['no_recent'] = prev.isna() | (gap > 365)
    d = d[d['no_recent'] & (d['value'] >= 25_000.0)].copy()
    d = attach_sessions(d, p, 'filing_date')
    d = d[pit_ok(d, p, 'sig_sess')].copy()
    d['si'] = d['si'].astype(int)
    d['n_owners'] = np.nan
    d['cell'] = 'I3'
    log(f'I3 OPPORTUNISTIC: {len(d)} triggering purchase-rows (pre symbol/session dedup)')
    return d


# ---------------------------------------------------------------- I4 13D


def compute_i4(p: Panel) -> pd.DataFrame:
    d = pd.read_parquet(f'{DATA}/sc13d.parquet')
    n0 = len(d)
    d = d[d['symbol'].notna()].copy()
    log(f'I4 13D: {n0} initial SC13D filings, {len(d)} mapped to a ticker '
        f'({n0 - len(d)} unmapped, excluded)')
    d['symbol'] = d['symbol'].astype(str).str.strip().str.upper()
    d['filing_date'] = pd.to_datetime(d['filed_date'])
    d = attach_sessions(d, p, 'filing_date')
    d = d[pit_ok(d, p, 'sig_sess')].copy()
    d['si'] = d['si'].astype(int)
    d['value'] = np.nan
    d['n_owners'] = np.nan
    d['cell'] = 'I4'
    log(f'I4 13D: {len(d)} PIT-eligible rows (pre symbol/session dedup)')
    return d


# ---------------------------------------------------------------- dedup + trades


def dedup_positions(d: pd.DataFrame, hold: int) -> pd.DataFrame:
    """One position per (symbol, signal session); an overlapping signal within an active
    hold on the same name extends nothing -- the first signal owns the slot."""
    d = d.sort_values('sig_sess')
    last_exit = {}
    keep = []
    for row in d.itertuples():
        s = row.si; sess = row.sig_sess
        if s in last_exit and sess <= last_exit[s]:
            continue
        keep.append(row.Index)
        last_exit[s] = sess + hold
    out = d.loc[keep].copy()
    log(f'  dedup (symbol,hold-window): {len(d)} -> {len(out)}')
    return out


def make_trades(d: pd.DataFrame, p: Panel, hold: int) -> pd.DataFrame:
    si = d['si'].to_numpy(dtype=int)
    a = d['sig_sess'].to_numpy(dtype=int)
    b = a + hold
    ok = b < p.n_d
    d = d[ok].copy(); si, a, b = si[ok], a[ok], b[ok]
    d['a'] = a; d['b'] = b
    d['gross'] = trade_returns(p, si, a, b)
    adv_in = p.adv[si, a]
    adv_out = p.adv[si, np.minimum(b, p.n_d - 1)]
    d['cost_bps'] = round_trip_cost_bps(adv_in, np.where(np.isfinite(adv_out), adv_out, adv_in))
    d['net'] = d['gross'] - d['cost_bps'] / 1e4
    d['entry_date'] = p.sessions[a]
    d['exit_date'] = p.sessions[b]
    d['signal_date'] = p.sessions[a]  # entry == the signal session close itself (obtainable print)
    d['hold'] = hold
    d['split'] = split_of_dates(d['entry_date'])
    return d


# ---------------------------------------------------------------- compare


def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b) if (a | b) else 1.0


def compare_cell(cell, mine_sig, mine_trd, theirs_sig, theirs_trd):
    key_cols = ['symbol', 'S']
    ms = set(map(tuple, mine_sig[key_cols].to_numpy())) if len(mine_sig) else set()
    ts = set(map(tuple, theirs_sig[key_cols].to_numpy())) if len(theirs_sig) else set()
    jac = jaccard(ms, ts)

    mt = mine_trd.set_index(['symbol', 'S'])
    tt = theirs_trd.drop_duplicates(['symbol', 'S']).set_index(['symbol', 'S'])
    common = mt.index.intersection(tt.index)
    n_common = len(common)
    if n_common:
        diff = (mt.loc[common, 'net'].to_numpy() - tt.loc[common, 'net'].to_numpy())
        match_close = float((np.abs(diff) <= 0.0005).mean())
        med_abs_diff = float(np.median(np.abs(diff)))
    else:
        match_close = float('nan'); med_abs_diff = float('nan')

    def val_mean(df, sess_map=None):
        if 'split' in df.columns:
            sub = df[df['split'] == 'VAL']
        else:
            sub = df
        return float(sub['net'].mean()) if len(sub) else float('nan')

    val_mine = val_mean(mine_trd)
    theirs_trd_dedup = theirs_trd.drop_duplicates(['symbol', 'S'])
    val_theirs = val_mean(theirs_trd_dedup)

    return dict(cell=cell, n_mine=len(ms), n_theirs=len(ts), jaccard=jac,
                n_common_trades=n_common, match_within_0_05pct=match_close,
                median_abs_return_diff=med_abs_diff,
                val_mean_mine=val_mine, val_mean_theirs=val_theirs)


def main():
    t0 = time.time()
    log('loading Panel (run_final)...')
    p = Panel()

    log('loading Form 3/4/5 raw quarterly TSVs...')
    purchases = load_form345()

    i1 = compute_i1(purchases, p)
    i2 = compute_i2(purchases, p)
    i3 = compute_i3(purchases, p)
    i4 = compute_i4(p)

    trades = []
    signals_final = []
    for cell, dcell in [('I1', i1), ('I2', i2), ('I3', i3), ('I4', i4)]:
        ded = dedup_positions(dcell, HOLD)
        trd = make_trades(ded, p, HOLD)
        trd['cell'] = cell
        trd['S'] = trd['sig_sess']
        trades.append(trd)
        sig_out = ded.copy()
        sig_out['S'] = sig_out['sig_sess']
        sig_out['signal_date'] = p.sessions[sig_out['sig_sess'].to_numpy(dtype=int)]
        signals_final.append(sig_out[['cell', 'symbol', 'si', 'S', 'signal_date', 'value', 'n_owners']])

    signals_final = pd.concat(signals_final, ignore_index=True)
    trades_final = pd.concat(trades, ignore_index=True)
    trades_final['S'] = trades_final['sig_sess']

    signals_final.to_csv(f'{OUT}/rebuild_signals.csv', index=False)
    keep_cols = ['cell', 'symbol', 'si', 'S', 'signal_date', 'value', 'n_owners',
                 'a', 'b', 'entry_date', 'exit_date', 'gross', 'cost_bps', 'net', 'hold', 'split']
    trades_final[keep_cols].to_csv(f'{OUT}/rebuild_trades.csv', index=False)
    log(f'wrote rebuild_signals.csv ({len(signals_final)} rows), '
        f'rebuild_trades.csv ({len(trades_final)} rows)')

    log('=== per-cell counts (TEST included but NOT scored below) ===')
    for cell in ['I1', 'I2', 'I3', 'I4']:
        sub = trades_final[trades_final['cell'] == cell]
        for split in ['TRAIN', 'VAL', 'TEST']:
            ss = sub[sub['split'] == split]
            log(f'  {cell} {split}: n={len(ss)} mean_net={ss["net"].mean() if len(ss) else float("nan"):.5f}')

    # -------------------------------------------------- independent-check comparison
    log('loading builder outputs for comparison (signals_form4/13d, trades_form4/13d)...')
    b_sig_f4 = pd.read_csv(f'{OUT}/signals_form4.csv')
    b_trd_f4 = pd.read_csv(f'{OUT}/trades_form4.csv')
    b_trd_f4 = b_trd_f4[(b_trd_f4['hold'] == HOLD) & (b_trd_f4.get('arm', 'all') == 'all')]
    b_sig_13 = pd.read_csv(f'{OUT}/signals_13d.csv')
    b_trd_13 = pd.read_csv(f'{OUT}/trades_13d.csv')
    b_trd_13 = b_trd_13[b_trd_13['hold_label'] == 'primary']

    results = []
    for cell in ['I1', 'I2', 'I3']:
        m_sig = signals_final[signals_final['cell'] == cell]
        m_trd = trades_final[trades_final['cell'] == cell]
        t_sig = b_sig_f4[b_sig_f4['cell'] == cell]
        t_trd = b_trd_f4[b_trd_f4['cell'] == cell]
        results.append(compare_cell(cell, m_sig, m_trd, t_sig, t_trd))

    m_sig = signals_final[signals_final['cell'] == 'I4']
    m_trd = trades_final[trades_final['cell'] == 'I4']
    t_sig = b_sig_13.rename(columns={'S': 'S'})
    t_trd = b_trd_13.rename(columns={'S': 'S'})
    results.append(compare_cell('I4', m_sig, m_trd, t_sig, t_trd))

    with open(f'{DATA}/compare_report.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    for r in results:
        log(f"COMPARE {r['cell']}: n_mine={r['n_mine']} n_theirs={r['n_theirs']} "
            f"jaccard={r['jaccard']:.4f} match<=0.05%={r['match_within_0_05pct']} "
            f"VAL mine={r['val_mean_mine']} theirs={r['val_mean_theirs']}")

    log(f'DONE in {time.time()-t0:.1f}s')


if __name__ == '__main__':
    main()
