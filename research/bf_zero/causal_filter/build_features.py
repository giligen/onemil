#!/usr/bin/env python3
"""CAUSAL_FILTER step 1 — the causal feature table for the live-config HOD-break population.

Population: research/bf_zero/spec_trades.csv (the 60,461 spec signals re-simulated on the Alpaca
SIP tape, REPORT 6a) restricted to the LIVE config: price >= 20, entry minute <= 840 (+1), early
closes excluded. Test tickers and names absent from `daily_bars` are dropped (PLAN standing rule).

Every feature here is computed from bars/daily rows at or before the signal minute. The COHORT
(which bar store served the symbol-day) is recorded as a DIAGNOSTIC column only; `assert_no_cohort`
in select.py proves it never enters a rule.

Two passes over the days:
  pass V (volume profile) — for every universe symbol-day of a population symbol, the cumulative
     volume at the 9 clock checkpoints, on the SIP-preferred loader. Feeds rv_clock / n_prior.
  pass S (signals)        — per signal bar: bar_vol_x, above_vwap, drive_min, cum volume.

Resumable per day via causal_filter/feat_state.json.
"""
import json, os, sqlite3, sys
import numpy as np
import pandas as pd

ROOT = '/home/ec2-user/onemil'
os.chdir(ROOT)
sys.path.insert(0, ROOT)
os.environ.setdefault('BFZ_SIP_STORE', f'{ROOT}/research/bf_zero/bars_sip.db')
sys.path.insert(0, f'{ROOT}/research/bf_zero')
import build_candidates as B                                        # noqa: E402  (SIP-preferred loader)

D = f'{ROOT}/research/bf_zero/causal_filter'
os.makedirs(D, exist_ok=True)
STATE = f'{D}/feat_state.json'
OUT_S = f'{D}/sig_bars.csv'
OUT_V = f'{D}/vprof_sip.csv'
EARLY_CLOSE = {'2025-07-03', '2025-11-28', '2025-12-24', '2026-07-03'}
VP_MIN = (575, 585, 600, 630, 660, 720, 780, 840, 900)
OPEN_M = 570


def population():
    """The live-config HOD-break signal population, with the membership filters applied."""
    T = pd.read_csv(f'{ROOT}/research/bf_zero/spec_trades.csv', dtype={'symbol': str, 'day': str},
                    keep_default_na=False, na_values=[''])
    n0 = len(T)
    L = T[(T.price >= 20.0) & (T.entry_m <= 841)].copy()
    n1 = len(L)
    L = L[~L.day.isin(EARLY_CLOSE)]
    n2 = len(L)
    from research.scripts.pit_listings import is_test_ticker
    L = L[~L.symbol.map(is_test_ticker)]
    n3 = len(L)
    con = sqlite3.connect(f'file:{ROOT}/data/cache.db?mode=ro', uri=True, timeout=120)
    db_syms = set(pd.read_sql('select distinct symbol from daily_bars', con).symbol.astype(str))
    L = L[L.symbol.isin(db_syms)]
    n4 = len(L)
    con.close()
    print(f'population: spec {n0} -> live-config {n1} -> ex early-close {n2} -> ex test tickers {n3} '
          f'-> in daily_bars {n4}', flush=True)
    return L.reset_index(drop=True)


def main():
    L = population()
    L.to_csv(f'{D}/population.csv', index=False)
    pop_syms = set(L.symbol.unique())
    uni = B.uni[['symbol', 'bar_date']].rename(columns={'bar_date': 'day'})
    uni = uni[uni.symbol.isin(pop_syms) & ~uni.day.isin(EARLY_CLOSE)].drop_duplicates()
    sig_by_day = {d: g for d, g in L.groupby('day')}
    uni_by_day = {d: g.symbol.tolist() for d, g in uni.groupby('day')}
    days = sorted(uni_by_day)
    state = json.load(open(STATE)) if os.path.exists(STATE) else {'done': []}
    done = set(state['done'])
    todo = [d for d in days if d not in done]
    print(f'days {len(days)} todo {len(todo)} | universe symbol-days {len(uni)} | signals {len(L)}', flush=True)
    for n, day in enumerate(todo):
        syms = uni_by_day[day]
        bars = B.load_bars(day, syms)
        vrows, srows = [], []
        sig = sig_by_day.get(day)
        sig_m = dict(zip(sig.symbol, sig.entry_m)) if sig is not None else {}
        for s, gg in bars.items():
            rth = gg[(gg.m >= OPEN_M) & (gg.m < 960)]
            if len(rth) < 10:
                continue
            m = rth.m.values.astype(int)
            v = rth.v.values.astype(float)
            h = rth.h.values.astype(float)
            c = rth.c.values.astype(float)
            o = rth.o.values.astype(float)
            cumv = np.cumsum(v)
            row = {'day': day, 'symbol': s, 'day_vol': float(cumv[-1]),
                   'pm_vol': float(gg[gg.m < OPEN_M].v.sum())}
            for cm in VP_MIN:
                k = np.searchsorted(m, cm, side='right') - 1
                row[f'cv_{cm}'] = float(cumv[k]) if k >= 0 else 0.0
            vrows.append(row)
            em = sig_m.get(s)
            if em is None:
                continue
            idx = np.flatnonzero(m == int(em))
            if not len(idx):
                continue
            i = int(idx[0])
            if i == 0:
                continue
            vwap = np.cumsum(c * v) / np.maximum(cumv, 1)
            o0 = o[0]
            reached = np.flatnonzero(h >= o0 * 1.05)
            vmean = v[:i].mean()
            srows.append({'day': day, 'symbol': s, 'entry_m': int(em),
                          'bar_vol_x': float(v[i] / vmean) if vmean > 0 else np.nan,
                          'vwap_prev': float(vwap[i - 1]),
                          'drive_min': int(m[reached[0]] - OPEN_M) if len(reached) and reached[0] < i else np.nan,
                          'cumv_entry': float(cumv[i]),
                          'open_px': float(o0)})
        if vrows:
            pd.DataFrame(vrows).to_csv(OUT_V, mode='a', header=not os.path.exists(OUT_V), index=False)
        if srows:
            pd.DataFrame(srows).to_csv(OUT_S, mode='a', header=not os.path.exists(OUT_S), index=False)
        state['done'].append(day)
        json.dump(state, open(STATE, 'w'))
        if n % 10 == 0:
            print(f'{n + 1}/{len(todo)} {day} v={len(vrows)} s={len(srows)}', flush=True)
    print('DONE', flush=True)


if __name__ == '__main__':
    main()
