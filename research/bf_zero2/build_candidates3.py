#!/usr/bin/env python3
"""bf_zero2 PASS 1, REBUILT with the live fill convention (2026-09-16, after three audits killed a book that existed
only because of the old fill).

THE SCORER THAT CONSUMES THIS MUST: charge HALF THE SPREAD ON ENTRY AND HALF ON EXIT (the DESIGN.md model, not the
exit-only model score3.py used), price-banded from `spread_pct`; run the book as run_book(rows, 12, 4) - 12 per day,
4 concurrent, as DESIGN.md pre-registered, because a 4-per-day cap is a SELECTION rule not a capacity rule; and apply
the pre-registered gates as written (TRAIN >= +10R/week, VAL >= +7R/week and >= 60% weeks green) with TEST read once.

WHAT CHANGED vs build_candidates2.py — the reason every earlier number is void:
  OLD: a family signalled when a bar's high reached `level x 1.003`, and the trade was filled AT `level x 1.003`, i.e. at
       the instant of the touch. 55% of the resulting trades had the signal bar OPENING above that price, handing the
       book a median 0.58R of entry edge that cannot be obtained.
  NEW: the spec and the shipped engine (`trading/hod_break.py::entry_fill`, `trading/hod_break_engine.py`): a bar's high
       reaching the level is the SIGNAL; the fill is the NEXT bar's OPEN, and only if that open is at or below
       level x (1 + CAP). Otherwise there is no trade — the engine does not chase. Exits then walk from the bar AFTER the
       fill bar (the entry bar is excluded), exactly as the engine does.
  ALSO (audit_data finding 1, the one nobody saw): the old code filled at `level x 1.003` whenever a bar's HIGH reached it,
       WITHOUT requiring the bar's LOW to be at or below that price. 41.2% of the old book's trades were filled BELOW the
       low of the bar that filled them - a price the market never offered - and those trades were 101% of its profit. The
       next-bar-open fill used here is a real printed price by construction, and the cap check voids a stale level.
  ALSO (finding 10): `range_so_far_pct` used the signal bar's own completed high and low; it is now computed on bars
       strictly BEFORE the signal bar.
  ALSO: the cost model is per price band, measured on THIS population (research/bf_zero2/spread_model_first5.csv, fresh
       SIP NBBO in the fill minute of 872 real trades: a median full spread near 1.9% of price at $5-10 and 0.5% at $50+),
       not the 0.40% median from the HOD-break signal study, which sampled a more liquid population.

Everything else (families, configs, universe, causal range-so-far rule, splits) is unchanged so the difference is
attributable to the fill. Output: candidates3.csv with one row per (day, symbol, family, config) that FILLS.
Run detached, ulimit 5.5 GB, nice 10; ~4 hours for 420 days."""
import json, os, sys, time
import numpy as np, pandas as pd
ROOT = '/home/ec2-user/onemil'; os.chdir(ROOT)
os.environ.setdefault('BFZ_SIP_STORE', f'{ROOT}/research/bf_zero/bars_sip.db')
os.environ['BFZ_SLIP'] = '0.0'                     # the level TOUCH is the signal; the fill is handled here, not in the detector
sys.path.insert(0, f'{ROOT}/research/bf_zero')
import build_candidates as B
D = 'research/bf_zero2'
STATE, OUT, MISS = f'{D}/build3_state.json', f'{D}/candidates3.csv', f'{D}/coverage3_missing.csv'
CAP = 0.006                                        # the engine's no-chase cap above the level
OPEN_M, EOD_M = B.OPEN_M, B.EOD_M
SPREAD = [(10, 1.90), (20, 1.20), (50, 0.80), (100, 0.60), (1e18, 0.50)]   # % of price, measured, first five minutes
COLS = ['day', 'symbol', 'fam', 'cfg', 'sig_m', 'entry_m', 'level', 'entry', 'stop', 'r_pct', 'price', 'dist_open_pct',
        'range_so_far_pct', 'rv_adv', 'gap_pct', 'adv20', 'spread_pct', 'rr_hold', 'why_hold', 'exit_m_hold', 'rr_2r', 'why_2r', 'exit_m_2r']
FAMS = ([('F1', dict(P=P)) for P in (0.05, 0.08, 0.12)] + [('F2', dict(P=P)) for P in (0.05, 0.08, 0.12)] +
        [('F3', dict(P=P, M=M)) for P in (0.05, 0.10) for M in (15, 30)] + [('F4', {})] +
        [('F5', dict(K=K, X=X)) for K in (3, 5, 8) for X in (0.02, 0.04, 0.06)] + [('F6', {})] + [('F7', {})] +
        [('F8', dict(N=N)) for N in (5, 15, 30)] + [('F9', dict(G=0.05))])


def spread_pct(p):
    for hi, v in SPREAD:
        if p < hi: return v
    return SPREAD[-1][1]


def walk(o, h, l, c, m, fill_idx, entry, stop, target=None):
    """Exits from the bar AFTER the fill bar, the engine's convention. Stop before target within a bar.
    Target (if given) needs a bar CLOSE at or above it and fills AT the target."""
    for k in range(fill_idx + 1, len(o)):
        if int(m[k]) >= EOD_M: return float(o[k]), 'eod', int(m[k])
        if l[k] <= stop: return float(min(stop, o[k]) * 0.999), 'stop', int(m[k])
        if target is not None and c[k] >= target: return float(target), 'target', int(m[k])
    return float(c[-1]), 'eod', int(m[-1])


def build_day(day, sub):
    bars = B.load_bars(day, sub.symbol.tolist()); rows = []; missing = []
    for r in sub.itertuples():
        gg = bars.get(r.symbol)
        if gg is None: missing.append(r.symbol); continue
        rth = gg[(gg.m >= OPEN_M) & (gg.m < 960)].reset_index(drop=True)
        if len(rth) < 10: missing.append(r.symbol); continue
        o, h, l, c, v = (rth[k].values.astype(float) for k in ('o', 'h', 'l', 'c', 'v')); m = rth.m.values.astype(int)
        o0 = o[0]; cumv = np.cumsum(v); vwap = np.cumsum(c * v) / np.maximum(cumv, 1)
        run_hi = np.maximum.accumulate(h); run_lo = np.minimum.accumulate(l)
        pm = gg[gg.m < OPEN_M]; pmh = float(pm.h.max()) if len(pm) else None
        gap = (o0 / r.prev_close - 1) if r.prev_close == r.prev_close and r.prev_close > 0 else np.nan
        for fam, cfg in FAMS:
            if fam == 'F1': res = B.fam_flag(h, l, cfg['P'], (2, 3, 4, 5, 6), micro=False)
            elif fam == 'F2': res = B.fam_flag(h, l, cfg['P'], (1, 2, 3), micro=True)
            elif fam == 'F3': res = B.fam_drive(h, l, o0, m, cfg['P'], cfg['M'])
            elif fam == 'F4': res = B.fam_vwap(h, l, c, vwap)
            elif fam == 'F5': res = B.fam_hod(h, l, cfg['K'], cfg['X'])
            elif fam == 'F6': res = B.fam_r2g(h, l, o0, r.prev_close if r.prev_close == r.prev_close else None)
            elif fam == 'F7': res = B.fam_level(h, l, pmh, 1, float(l[:1].min())) if pmh else None
            elif fam == 'F8':
                N = cfg['N']; res = B.fam_level(h, l, float(h[:N].max()), N, float(l[:N].min())) if len(h) > N else None
            else: res = B.fam_level(h, l, float(h[:5].max()), 5, float(l[:5].min())) if (len(h) > 5 and gap == gap and gap >= cfg['G']) else None
            if res is None: continue
            i, level, stop, extra = res                      # i = the SIGNAL bar (its high reached the level)
            if i + 1 >= len(o): continue                     # no next bar to fill on
            fill_idx = i + 1; entry = float(o[fill_idx])     # THE LIVE CONVENTION: next bar's open...
            if entry > level * (1 + CAP): continue           # ...and only under the cap; otherwise the engine does not chase
            if stop >= entry: continue
            R = entry - stop
            if R / entry * 100.0 < 1.0: continue
            rr_h, why_h, xm_h = walk(o, h, l, c, m, fill_idx, entry, stop)
            rr_2, why_2, xm_2 = walk(o, h, l, c, m, fill_idx, entry, stop, target=entry + 2 * R)
            rows.append(dict(day=day, symbol=r.symbol, fam=fam, cfg=json.dumps(cfg, sort_keys=True), sig_m=int(m[i]), entry_m=int(m[fill_idx]),
                             level=level, entry=entry, stop=stop, r_pct=R / entry * 100, price=entry,
                             dist_open_pct=(entry / o0 - 1) * 100, range_so_far_pct=(run_hi[i - 1] - run_lo[i - 1]) / o0 * 100 if i >= 1 else 0.0,
                             rv_adv=cumv[i] / r.adv20 if r.adv20 == r.adv20 and r.adv20 > 0 else np.nan,
                             gap_pct=gap * 100 if gap == gap else np.nan, adv20=r.adv20, spread_pct=spread_pct(entry),
                             rr_hold=(rr_h - entry) / R, why_hold=why_h, exit_m_hold=xm_h,
                             rr_2r=(rr_2 - entry) / R, why_2r=why_2, exit_m_2r=xm_2))
    return rows, missing


def main():
    uni = B.uni
    state = json.load(open(STATE)) if os.path.exists(STATE) else {'done': []}
    done = set(state['done']); days = sorted(uni.bar_date.unique())
    if os.environ.get('BFZ_DAYS'): days = days[:int(os.environ['BFZ_DAYS'])]
    todo = [d for d in days if d not in done]
    print(f'pass 3 (live fill) | days {len(days)} | todo {len(todo)} | fams {len(FAMS)} | cap {CAP:.3%}', flush=True)
    t0 = time.time(); n = 0
    for k, day in enumerate(todo):
        rows, miss = build_day(day, uni[uni.bar_date == day])
        if rows: pd.DataFrame(rows).reindex(columns=COLS).to_csv(OUT, mode='a', header=not os.path.exists(OUT), index=False)
        if miss: pd.DataFrame({'symbol': miss, 'bar_date': day}).to_csv(MISS, mode='a', header=not os.path.exists(MISS), index=False)
        n += len(rows); state['done'].append(day); json.dump(state, open(STATE, 'w'))
        if k % 5 == 0 or k == len(todo) - 1:
            el = time.time() - t0
            print(f'{k+1}/{len(todo)} {day} rows+={len(rows)} total {n} | {el/60:.1f} min, {el/(k+1):.1f} s/day', flush=True)
    print('DONE', flush=True)


if __name__ == '__main__':
    main()
