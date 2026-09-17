#!/usr/bin/env python3
"""Stage G verification — recompute N rows of candidates_short.csv from the RAW bars with plain
Python loops, sharing no code with the builder (no numpy vectorisation, no imported detector, its
own bar loader). PLAN §1's "independent reimplementation" at row level: every field that a decision
uses is re-derived and compared exactly.

Usage: python3 research/fuckup_audit/G/verify_rows.py [N] [--src FILE] [--seed S]
"""
import json
import os
import sqlite3
import sys

import pandas as pd

ROOT = '/home/ec2-user/onemil'
os.chdir(ROOT)
G = f'{ROOT}/research/fuckup_audit/G'
STORE = f'{ROOT}/research/fuckup_audit/E/bars_causal'
SIP = f'{ROOT}/research/bf_zero/bars_sip.db'
ATTN = f'{ROOT}/research/lit_review_2026/attention.db'

A = sys.argv[1:]
N = int(A[0]) if A and not A[0].startswith('--') else 5
SRC = A[A.index('--src') + 1] if '--src' in A else f'{G}/candidates_short.csv'
SEED = int(A[A.index('--seed') + 1]) if '--seed' in A else 7
CAP, SLIP, EOD_M, COVER_M, OPEN_M = 0.006, 1.001, 955, 630, 570


def raw_bars(day, sym):
    """The three stores, read one symbol-day at a time, no shared code with the builder."""
    p = f'{STORE}/day={day}/bars.parquet'
    if os.path.exists(p):
        d = pd.read_parquet(p, columns=['symbol', 't', 'o', 'h', 'l', 'c', 'v'])
        d = d[d.symbol == sym]
        if len(d):
            return sorted([(int(r.t), float(r.o), float(r.h), float(r.l), float(r.c), float(r.v))
                           for r in d.itertuples()])
    con = sqlite3.connect(f'file:{SIP}?mode=ro', uri=True)
    rows = con.execute('select t,o,h,l,c,v from bars where day=? and symbol=?', (day, sym)).fetchall()
    con.close()
    if rows:
        out = []
        for t, o, h, l, c, v in rows:
            ts = pd.Timestamp(t).tz_convert('America/New_York')
            out.append((ts.hour * 60 + ts.minute, float(o), float(h), float(l), float(c), float(v)))
        return sorted(out)
    if os.path.exists(ATTN):
        con = sqlite3.connect(f'file:{ATTN}?mode=ro', uri=True)
        rows = con.execute('select m,o,h,l,c,v from bars where day=? and symbol=?',
                           (day, sym)).fetchall()
        con.close()
        if rows:
            return sorted([(int(m), float(o), float(h), float(l), float(c), float(v))
                           for m, o, h, l, c, v in rows])
    return []


def detect(fam, cfg, bars, prev_close, gap_pct):
    """The six short rules, written out as loops."""
    n = len(bars)
    lows = [b[3] for b in bars]
    highs = [b[2] for b in bars]
    closes = [b[4] for b in bars]
    if fam == 'S1':
        if not (gap_pct == gap_pct and gap_pct >= 5.0 and n > 5):
            return None
        lvl, stp, start = min(lows[:5]), max(highs[:5]), 5
    elif fam == 'S2':
        N_ = cfg['N']
        if n <= N_:
            return None
        lvl, stp, start = min(lows[:N_]), max(highs[:N_]), N_
    elif fam == 'S3':
        if not (prev_close and prev_close == prev_close and bars[0][1] > prev_close):
            return None
        lvl, stp, start = prev_close * 0.997, None, 1
    elif fam == 'S4':
        K, X = cfg['K'], cfg['X']
        for i in range(K + 1, n):
            j = i - 1
            lo_k = min(lows[j - K + 1:j + 1])
            hod = max(highs[:j + 1])
            if lo_k >= hod * (1 - X) and lo_k < hod and closes[i] < lo_k:
                if hod <= lo_k:
                    return None
                return i, lo_k, hod
        return None
    elif fam == 'S5':
        i = next((k for k, b in enumerate(bars) if b[0] == OPEN_M + 5), None)
        if i is None or i < 1 or n <= 5:
            return None
        stp = max(highs[:i])
        lvl = bars[i][1]
        return (i, lvl, stp) if stp > lvl else None
    else:
        return None
    for i in range(start, n):
        if lows[i] <= lvl:
            s = stp if stp is not None else max(highs[:i])
            return (i, lvl, s) if s > lvl else None
    return None


def walk(bars, k0, entry, stop, target, horizon_m):
    """eod/cover beats stop beats target; stop fills at max(stop, open)*1.001, target AT target."""
    n = len(bars)
    if k0 >= n:
        return bars[-1][4], 'eod', bars[-1][0], n - 1
    hidx = next((k for k in range(n) if bars[k][0] >= horizon_m), None)
    cand = []
    if hidx is not None:
        cand.append((max(hidx, k0), 0, 'eod'))
    s = next((k for k in range(k0, n) if bars[k][2] >= stop), None)
    if s is not None:
        cand.append((s, 1, 'stop'))
    if target is not None:
        t = next((k for k in range(k0, n) if bars[k][4] <= target), None)
        if t is not None:
            cand.append((t, 2, 'target'))
    if not cand:
        return bars[-1][4], 'eod', bars[-1][0], n - 1
    k, _, why = min(cand)
    px = (bars[k][1] if why == 'eod'
          else (max(stop, bars[k][1]) * SLIP if why == 'stop' else target))
    return px, why, bars[k][0], k


def main():
    d = pd.read_csv(SRC, keep_default_na=False, na_values=[''])
    d = d[d.entry.notna()]
    picks = (d.groupby(['fam', 'cfg'], group_keys=False)
             .apply(lambda g: g.sample(min(len(g), max(1, N // 6 + 1)), random_state=SEED)))
    picks = picks.sample(min(len(picks), max(N, 6)), random_state=SEED)
    bad = 0
    for r in picks.itertuples():
        bars = raw_bars(r.day, r.symbol)
        bars = [b for b in bars if OPEN_M <= b[0] < 960]
        cfg = json.loads(r.cfg)
        res = detect(r.fam, cfg, bars, r.prev_close, r.gap_pct)
        ok = []
        if res is None:
            print(f'MISMATCH {r.day} {r.symbol} {r.fam} {r.cfg}: independent detector found nothing')
            bad += 1
            continue
        i, lvl, stp = res
        ok.append(('sig_m', bars[i][0], int(r.sig_m)))
        ok.append(('level', round(lvl, 6), round(float(r.level), 6)))
        ok.append(('stop', round(stp, 6), round(float(r.stop), 6)))
        fi = i if r.fam == 'S5' else i + 1
        e = bars[fi][1]
        ok.append(('entry', round(e, 6), round(float(r.entry), 6)))
        ok.append(('entry_m', bars[fi][0], int(r.entry_m)))
        R = stp - e
        ok.append(('r_pct', round(R / e * 100, 6), round(float(r.r_pct), 6)))
        px, why, xm, _ = walk(bars, fi + 1, e, stp, None, EOD_M)
        ok.append(('rr_hold', round((e - px) / R, 6), round(float(r.rr_hold), 6)))
        ok.append(('why_hold', why, r.why_hold))
        ok.append(('exit_m_hold', xm, int(r.exit_m_hold)))
        px, why, xm, _ = walk(bars, fi + 1, e, stp, e - 2 * R, EOD_M)
        ok.append(('rr_2r', round((e - px) / R, 6), round(float(r.rr_2r), 6)))
        ok.append(('why_2r', why, r.why_2r))
        px, why, xm, _ = walk(bars, fi + 1, e, stp, None, COVER_M)
        ok.append(('rr_1030', round((e - px) / R, 6), round(float(r.rr_1030), 6)))
        diffs = [(k, a, b) for k, a, b in ok
                 if (a != b and not (isinstance(a, float) and abs(a - b) < 1e-6))]
        tag = 'MATCH' if not diffs else 'MISMATCH'
        bad += bool(diffs)
        print(f'{tag} {r.day} {r.symbol} {r.fam} {r.cfg} | bars {len(bars)} | '
              f'sig_m {bars[i][0]} level {lvl:.4f} stop {stp:.4f} entry {e:.4f} '
              f'rr_hold {(e - px) / R:+.4f}')
        for k, a, b in ok:
            print(f'    {k:14s} independent={a}   file={b}')
        for k, a, b in diffs:
            print(f'    !! {k}: {a} vs {b}')
    print(f'\n{len(picks) - bad}/{len(picks)} rows MATCH')


if __name__ == '__main__':
    main()
