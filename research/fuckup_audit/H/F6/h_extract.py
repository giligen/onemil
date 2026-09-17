#!/usr/bin/env python3
"""Stage H / F6 — step 0: extract the F6 rows (and only those) from the two population files.

Two populations are carried side by side for the whole stage:
  P  = `C/pop_c.csv`            the >=5%-range-day universe (Stage B/C).  Its F6 rows ALREADY carry
                                 `range_so_far_pct >= 5` (c0_extract applies the causal floor to every
                                 non-exempt family), so the "no-floor twin" is structurally impossible here.
  Q  = `E/candidates_causal.csv` the causal U1 u U2 universe (Stage E).  NO floor -> the floor's own
                                 contribution is measurable here, and this is the file the METHOD's
                                 reference numbers (+0.090 TRAIN / +0.209 VAL) were computed on.

Kept rows: fam == 'F6', the next-open fill present and >= $5, 570 <= entry_m <= 841, and
(next_r_pct >= 1 OR next_r_pct_m1 >= 1) so the stop-1% variant stays scoreable.  NO range floor is
applied here - it is a filter in the scorer, not in the extract.

Also writes the per-split WEEK LIST of each source file, computed over ALL families' scoreable rows,
because Stage C/E compute weeks-green against that denominator and the parity anchor must match it.

Usage: ulimit -v 1500000; nice -n 10 python3 research/fuckup_audit/H/F6/h_extract.py
"""
import os, time
import numpy as np
import pandas as pd

ROOT = '/home/ec2-user/onemil'
os.chdir(ROOT)
H = 'research/fuckup_audit/H/F6'
RD = dict(keep_default_na=False, na_values=[''])
SRC = {'P': 'research/fuckup_audit/C/pop_c.csv',
       'Q': 'research/fuckup_audit/E/candidates_causal.csv'}
WEEK_COLS = ['day', 'fam', 'range_so_far_pct',
             'next_entry', 'next_entry_m', 'next_r_pct', 'next_r_pct_m1',
             'rest_entry', 'rest_entry_m', 'rest_r_pct', 'rest_r_pct_m1']


def split_of(day):
    return np.where(day < '2026-01-01', 'TRAIN', np.where(day < '2026-06-01', 'VAL', 'TEST'))


def main():
    for tag, src in SRC.items():
        out = f'{H}/f6_{tag}.csv'
        wout = f'{H}/weeks_{tag}.csv'
        if os.path.exists(out):
            os.remove(out)
        t0, n_in, n_out, first = time.time(), 0, 0, True
        wk_rows = []
        for ch in pd.read_csv(src, dtype={'day': str, 'symbol': str, 'fam': str, 'cfg': str},
                              chunksize=200_000, low_memory=True, **RD):
            n_in += len(ch)
            # --- week denominator: score5/e_score's scoreable union over BOTH fills, ALL families
            floor_ok = ch.fam.isin(('F1', 'F2', 'F3', 'F4')) | (ch.range_so_far_pct >= 5)
            anyfill = None
            for t in ('next', 'rest'):
                f = (ch[f'{t}_entry'].notna() & (ch[f'{t}_entry'] >= 5)
                     & (ch[f'{t}_entry_m'] <= 841) & (ch[f'{t}_entry_m'] >= 570)
                     & ((ch[f'{t}_r_pct'] >= 1.0) | (ch[f'{t}_r_pct_m1'] >= 1.0)))
                anyfill = f if anyfill is None else (anyfill | f)
            wk_rows.append(ch.loc[floor_ok & anyfill, ['day']].drop_duplicates())
            # --- the F6 extract
            e, em = ch['next_entry'], ch['next_entry_m']
            keep = (ch.fam == 'F6') & e.notna() & (e >= 5) & (em <= 841) & (em >= 570) \
                & ((ch['next_r_pct'] >= 1.0) | (ch['next_r_pct_m1'] >= 1.0))
            k = ch[keep]
            n_out += len(k)
            k.to_csv(out, index=False, header=first, mode='w' if first else 'a')
            first = False
            print(f'{tag} {n_in:,} -> {n_out:,} | {(time.time()-t0)/60:.1f} min', flush=True)
        d = pd.concat(wk_rows, ignore_index=True).drop_duplicates()
        d['split'] = split_of(d.day.values)
        d['wk'] = pd.to_datetime(d.day).dt.to_period('W-FRI').astype(str)
        w = d[['split', 'wk']].drop_duplicates().sort_values(['split', 'wk'])
        w.to_csv(wout, index=False)
        print(f'{tag} DONE in {n_in:,} F6 out {n_out:,} | weeks ' +
              ' '.join(f'{s}={int((w.split == s).sum())}' for s in ('TRAIN', 'VAL', 'TEST')) +
              f' -> {out}', flush=True)


if __name__ == '__main__':
    main()
