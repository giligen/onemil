#!/usr/bin/env python3
"""No-fill slot recycling — replay of the honest B+ selection with recycling.

Inputs (all produced from the same static-lock dump, identical exit physics):
  ranked_candidates.csv  per-day ranked list (post-filter, pre-slot) + _sized_pnl
  breakout_times.csv     first minute (after 09:30 ET) a candidate's stop-limit
                         trigger was reached; NaN = never
  features CSV           the full morning candidate set per day (catalyst cohorts)
  news CSVs              own-ticker premarket news (catalyst veto)

Step 1 replays the pipeline exactly (top-K + family/super dedup + PDR/G1/RS/
catalyst vetoes, no refill) and MUST reproduce the baseline book to the dollar.
Step 2 adds recycling: a slot whose pick has not triggered by T minutes is
released to the next-ranked candidate that has ALSO not triggered yet (a
candidate that already broke out is a chase — ineligible); the replacement
passes the same vetoes (a vetoed replacement consumes the attempt, slot
stays empty); a released late-trigger pick books $0 (its order was canceled).

Pass rule: research/orb_slot_recycle/DESIGN.md.
"""
import glob
import os
import sys
import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, ROOT)
os.chdir(ROOT)
from study_orb_correlation_filter import symbol_family, symbol_super_group   # noqa: E402
from trading.orb_pdr_veto import pdr_veto_applies                              # noqa: E402
from trading.orb_g1_veto import g1_reject                                       # noqa: E402
from trading.orb_range_size_veto import range_size_veto_applies                 # noqa: E402
from trading.orb_asset_class import DEFAULT_CLASS_MAP, load_class_map, underlying_anchor  # noqa: E402
from trading.orb_catalyst_veto import DEFAULT_MIN_COHORT, anchor_cohort_counts, catalyst_veto_applies  # noqa: E402
from trading.orb_csv import read_orb_csv                                        # noqa: E402
import csv                                                                       # noqa: E402
import yaml                                                                      # noqa: E402

D = 'research/orb_slot_recycle'
FEATURES = 'analysis_results/orb_features_20260905_1940.csv'
N_SLOTS = 3

cfg = yaml.safe_load(open('orb.yaml'))['filter']
PDR_MIN = float(cfg['prev_day_range_veto']['min_prev_day_range_pct'])
G1 = cfg['g1_veto']; RS = cfg['range_size_veto']


def load_catalyst_inputs():
    """anchors (symbol->anchor), cohorts (day->counts), news ((symbol, day)->bool)."""
    F = read_orb_csv(FEATURES); F['date'] = pd.to_datetime(F['date']).dt.strftime('%Y-%m-%d')
    names = {}
    with open(DEFAULT_CLASS_MAP, newline='') as fh:
        for row in csv.DictReader(fh):
            names[row['symbol']] = row.get('name', '')
    cmap = load_class_map()
    anchors = {s: underlying_anchor(s, names.get(s), cmap) for s in set(F['symbol'])}
    cohorts = {day: anchor_cohort_counts(g['symbol'].map(anchors)) for day, g in F.groupby('date')}
    news = {}
    for p in sorted(glob.glob('data/research/orb_news_catalyst_*.csv')):
        for _, x in read_orb_csv(p).iterrows():
            news[(x['symbol'], x['day'])] = (x['n_articles'] or 0) > 0
    return anchors, cohorts, news


def load_inputs():
    r = pd.read_csv(f'{D}/ranked_candidates.csv', low_memory=False)
    r['date'] = pd.to_datetime(r['date']).dt.strftime('%Y-%m-%d')
    bt = pd.read_csv(f'{D}/breakout_times.csv'); bt['date'] = bt['date'].astype(str).str[:10]
    r = r.merge(bt[['symbol', 'date', 'breakout_min']], on=['symbol', 'date'], how='left')
    anchors, cohorts, news = load_catalyst_inputs()
    r['_anchor'] = r['symbol'].map(anchors)
    return r, cohorts, news


def vetoed(row, cohorts, news):
    """The four post-selection vetoes, same helpers as live/pipeline."""
    if pdr_veto_applies(None if pd.isna(row.prev_day_range_pct) else float(row.prev_day_range_pct), PDR_MIN):
        return 'pdr'
    if g1_reject(row.return_volatility_20d, row.prev_day_range_pct,
                 float(G1['return_volatility_20d_min']), float(G1['prev_day_range_pct_min']),
                 short_history_veto=bool(G1.get('short_history_veto', False))) is not None:
        return 'g1'
    if RS.get('enabled') and range_size_veto_applies(row.range_size_pct, float(RS['min_range_size_pct'])):
        return 'rs'
    if catalyst_veto_applies(news.get((row.symbol, row.date)), row._anchor, cohorts.get(row.date, {}), DEFAULT_MIN_COHORT):
        return 'catalyst'
    return None


def replay_day(d, cohorts, news, T=None, max_recycles=0):
    """Return list of (symbol, pnl, filled, recycled) for one day."""
    d = d.sort_values('_rank')
    seen_fam, seen_sup, picks = set(), set(), []
    for row in d.itertuples():
        fam, sup = symbol_family(row.symbol), symbol_super_group(row.symbol)
        if (fam and fam in seen_fam) or (sup and sup in seen_sup):
            continue
        if fam: seen_fam.add(fam)
        if sup: seen_sup.add(sup)
        picks.append(row)
        if len(picks) >= N_SLOTS:
            break
    out = []
    taken = {p.symbol for p in picks}
    held_fam = set(); held_sup = set()
    recycles = 0
    for p in picks:
        v = vetoed(p, cohorts, news)
        if v:
            out.append((p.symbol, 0.0, 0, 0, f'veto:{v}')); continue
        bm = p.breakout_min
        triggered_in_time = (not pd.isna(bm)) and (T is None or bm <= T)
        if triggered_in_time or T is None:
            out.append((p.symbol, float(p._sized_pnl), int(p.entered), 0, 'first'))
            fam, sup = symbol_family(p.symbol), symbol_super_group(p.symbol)
            if fam: held_fam.add(fam)
            if sup: held_sup.add(sup)
            continue
        # released at T: the original books $0 (order canceled at T)
        out.append((p.symbol, 0.0, 0, 0, 'released'))
        if recycles >= max_recycles:
            continue
        recycles += 1
        # next-ranked candidate not yet triggered (breakout_min >= T or never), not taken
        for c in d.itertuples():
            if c.symbol in taken:
                continue
            fam, sup = symbol_family(c.symbol), symbol_super_group(c.symbol)
            if (fam and fam in held_fam) or (sup and sup in held_sup):
                continue
            cbm = c.breakout_min
            if not pd.isna(cbm) and cbm < T:
                continue   # already broke out before the release: a chase, ineligible
            taken.add(c.symbol)
            v = vetoed(c, cohorts, news)
            if v:
                out.append((c.symbol, 0.0, 0, 1, f'recycle-veto:{v}'))
            else:
                out.append((c.symbol, float(c._sized_pnl), int(c.entered), 1, 'recycled'))
                if fam: held_fam.add(fam)
                if sup: held_sup.add(sup)
            break
    return out


def run(r, cohorts, news, T=None, max_recycles=0):
    rows = []
    for day, d in r.groupby('date'):
        for sym, pnl, filled, rec, tag in replay_day(d, cohorts, news, T, max_recycles):
            rows.append(dict(date=day, symbol=sym, pnl=pnl, filled=filled, recycled=rec, tag=tag))
    b = pd.DataFrame(rows); b['mo'] = b.date.str[:7]
    months = sorted(set(r.date.str[:7]))
    m = b.groupby('mo').pnl.sum().reindex(months).fillna(0)
    c = m.cumsum()
    first = b[(b.tag == 'first') & (b.filled == 1)]; rec = b[(b.tag == 'recycled') & (b.filled == 1)]
    era = lambda a, z: float(b[(b.date >= a) & (b.date <= z)].pnl.sum())
    return dict(T=T, max=max_recycles, picks=len(b[b.tag.isin(['first', 'recycled'])]),
                fills=int(b.filled.sum()), fills_mo=round(b.filled.sum() / len(months), 2),
                total=round(m.sum()), mdd=round((c - c.cummax()).min()), red=int((m < 0).sum()),
                worst=round(m.min()), e25H1=round(era('2025-01-01', '2025-06-30')),
                e25H2=round(era('2025-07-01', '2025-12-31')), e2026=round(era('2026-01-01', '2026-12-31')),
                first_fills=len(first), first_mean=round(first.pnl.mean(), 1) if len(first) else np.nan,
                rec_fills=len(rec), rec_mean=round(rec.pnl.mean(), 1) if len(rec) else np.nan,
                rec_vetoed=int((b.tag.str.startswith('recycle-veto')).sum()), released=int((b.tag == 'released').sum())), b


def main():
    r, cohorts, news = load_inputs()
    base, bb = run(r, cohorts, news)
    print('REPLAY baseline:', {k: base[k] for k in ('picks', 'fills', 'total', 'mdd', 'red', 'worst')}, flush=True)
    ref = pd.read_csv(f'{D}/baseline_book.csv')
    print(f"PIPELINE baseline: picks {len(ref)} fills {int(ref.entered.sum())} total {ref._sized_pnl.sum():,.0f}", flush=True)
    if abs(base['total'] - ref._sized_pnl.sum()) > 1 or base['picks'] != len(ref):
        print('FATAL: replay does not reproduce the pipeline — fix before reading any recycling row', flush=True)
        bb.to_csv(f'{D}/replay_baseline_book.csv', index=False)
        return 1
    rows = [base]
    for T in (15, 30, 45):
        for mx in (1, 3):
            s, b = run(r, cohorts, news, T, mx); rows.append(s)
            b.to_csv(f'{D}/recycle_T{T}_max{mx}_book.csv', index=False)
    S = pd.DataFrame(rows)
    def verdict(x):
        if x.T is None or pd.isna(x.T): return ''
        ok = (x.fills_mo >= base['fills_mo'] * 1.2 and x.total >= base['total'] and x.mdd >= base['mdd']
              and (np.isnan(x.rec_mean) or x.rec_mean >= 0.5 * base['first_mean'])
              and all(x[e] >= base[e] - 100 for e in ('e25H1', 'e25H2', 'e2026')))
        return 'KEEP' if ok else 'REJECT'
    S['verdict'] = [verdict(x) for _, x in S.iterrows()]
    S.to_csv(f'{D}/summary.csv', index=False)
    print(S.to_string(index=False), flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
