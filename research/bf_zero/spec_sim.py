#!/usr/bin/env python3
"""bf_zero — PARITY book: run the live spec (`trading/hod_break.py`) over every universe symbol-day
where the study found at least one HOD-consolidation break (F5 K=5 X=0.04, any distance/volume), so
the spec's own first QUALIFYING break is used (it may be later than the study's first break when the
first one fails the causal filters). Then apply the executable book rule (first-come, per-day cap,
concurrency cap) and report per split + week-by-week. This is THE number for the live build.
Also computes the same-clock rv (rv_clock) where a per-symbol baseline exists, for the H-RV report.
"""
import os, sys, json
import numpy as np, pandas as pd
ROOT = '/home/ec2-user/onemil'; os.chdir(ROOT); sys.path.insert(0, ROOT); sys.path.insert(0, f'{ROOT}/research/bf_zero')
import build_candidates as B
from trading.hod_break import HodBreakParams, simulate, detect
D = 'research/bf_zero'; P = HodBreakParams()
STATE, OUT = f'{D}/spec_state.json', f'{D}/spec_trades.csv'

c = pd.read_csv(f'{D}/candidates_full.csv', usecols=['day', 'symbol', 'fam', 'cfg', 'adv20', 'is_wrapper', 'price'], dtype={'symbol': str, 'fam': 'category', 'cfg': 'category'}, keep_default_na=False, na_values=[''])
c = c[(c.fam == 'F5') & (c.cfg == '{"K": 5, "X": 0.04}')].drop(columns=['fam', 'cfg']).drop_duplicates(['day', 'symbol'])
c['adv20'] = pd.to_numeric(c.adv20, errors='coerce'); c['is_wrapper'] = pd.to_numeric(c.is_wrapper, errors='coerce')
u = B.uni[['symbol', 'bar_date', 'high', 'open']].rename(columns={'bar_date': 'day'})
c = c.merge(u, on=['day', 'symbol'], how='left'); c = c[c.high >= c.open * (1 + P.min_dist_open_pct / 100)]   # exact causal superset: a level >= 5% above the open needs day high >= that
print('symbol-days to simulate', len(c), flush=True)
state = json.load(open(STATE)) if os.path.exists(STATE) else {'done': []}; done = set(state['done'])
days = [d for d in sorted(c.day.unique()) if d not in done]
for n, day in enumerate(days):
    sub = c[c.day == day]; bars = B.load_bars(day, sub.symbol.tolist()); rows = []
    for r in sub.itertuples():
        gg = bars.get(r.symbol)
        if gg is None: continue
        rth = gg[(gg.m >= B.OPEN_M) & (gg.m < 960)].reset_index(drop=True)
        if len(rth) < 10: continue
        o, h, l, cl, v = (rth[k].values.astype(float) for k in ('o', 'h', 'l', 'c', 'v')); m = rth.m.values.astype(int)
        t = simulate(o, h, l, cl, v, m, r.adv20, P)
        if t is None: continue
        sig = detect(o, h, l, v, m, r.adv20, P)
        rows.append(dict(day=day, symbol=r.symbol, entry_m=int(m[t.entry_idx]), exit_m=int(m[t.exit_idx]), entry=t.entry, stop=t.stop, target=t.target,
                         r_pct=t.r_per_share / t.entry * 100, rr=t.rr, why=t.reason, level=sig.level, dist_open_pct=sig.dist_open_pct, rv_profile=sig.rv_profile,
                         adv20=r.adv20, is_wrapper=r.is_wrapper, price=t.entry))
    if rows: pd.DataFrame(rows).to_csv(OUT, mode='a', header=not os.path.exists(OUT), index=False)
    state['done'].append(day); json.dump(state, open(STATE, 'w'))
    if n % 20 == 0: print(f'{n + 1}/{len(days)} {day} +{len(rows)}', flush=True)

T = pd.read_csv(OUT, dtype={'symbol': str}, keep_default_na=False)   # the ticker NA is a ticker
T['split'] = np.where(T.day < '2026-01-01', 'TRAIN', np.where(T.day < '2026-06-01', 'VAL', 'TEST')); T['wk'] = pd.to_datetime(T.day).dt.to_period('W-FRI').astype(str)
NW = {s: len(pd.period_range(a, b, freq='W-FRI')) for s, (a, b) in {'TRAIN': ('2025-01-02', '2025-12-31'), 'VAL': ('2026-01-01', '2026-05-31'), 'TEST': ('2026-06-01', '2026-09-11')}.items()}


def run_book(F, n_day, n_conc):
    """the ONE book rule (trading.hod_break.run_book): causal slot freeing, symbol tie-break"""
    from trading.hod_break import run_book as _rb
    idx = [r[0] for r in _rb([(r.day, r.entry_m, r.exit_m, r.symbol, r.Index) for r in F.itertuples()], n_day, n_conc)]
    return F.loc[[t[4] for t in _rb([(r.day, r.entry_m, r.exit_m, r.symbol, r.Index) for r in F.itertuples()], n_day, n_conc)]]


def report(bk, title):
    print(f'\n## {title}')
    for s in ('TRAIN', 'VAL', 'TEST'):
        d = bk[bk.split == s]; nw = NW[s]
        if not len(d): print(f'  {s}: none'); continue
        w = d.groupby('wk').rr.sum(); wsum = float(w.sum()); wk_green = int((w > 0).sum())
        gp = d.rr[d.rr > 0].sum(); gl = -d.rr[d.rr < 0].sum()
        print(f"  {s:5s} trades {len(d):5d} ({len(d) / nw:4.1f}/wk) meanR {d.rr.mean():+.3f} WR {(d.rr > 0).mean() * 100:4.1f} PF {gp / gl if gl else float('inf'):.2f} | weekly R {wsum / nw:+.1f} sd {w.std():.1f} green {wk_green}/{nw} worst {w.min():+.1f} | {d.why.value_counts().to_dict()}")


print(f'\nspec signals (all): {len(T)} | fill model = next-open capped {P.cap:.2%}, target on close, flat {P.flat_minute}')
report(T, 'POPULATION (every spec signal, no caps)')
bk = run_book(T, P.max_per_day, P.max_concurrent); bk.to_csv(f'{D}/spec_book.csv', index=False)
report(bk, f'EXECUTABLE BOOK: first-come {P.max_per_day}/day, {P.max_concurrent} concurrent')
# THE LIVE BOOK: the config's knobs (last entry, per-day cap, concurrency, price floor) applied to the same signals — what the
# engine runs tomorrow, not the study defaults (review A/F, 9/15). Detection above is a superset (last entry 930 ≥ config).
try:
    from config import Config
    LIVE = HodBreakParams(**(Config().hod_break_cfg.get('params') or {})); LIVE_FLOOR = float(Config().hod_break_cfg.get('min_price') or 0)
    TL = T[(T.entry_m <= LIVE.last_entry_minute + 1) & (T.price >= LIVE_FLOOR)]
    bkl = run_book(TL, LIVE.max_per_day, LIVE.max_concurrent); bkl.to_csv(f'{D}/spec_book_live_config.csv', index=False)
    report(bkl, f'LIVE-CONFIG BOOK: last entry {LIVE.last_entry_minute}, price >= {LIVE_FLOOR:.0f}, first-come {LIVE.max_per_day}/day, {LIVE.max_concurrent} concurrent (spread gate NOT modeled here — see capacity_8a.py)')
except Exception as e:
    print(f'live-config book skipped: {e}')
print('\nTEST week-by-week (R, n):'); d = bk[bk.split == 'TEST']; print(d.groupby('wk').rr.agg(['sum', 'count']).round(1).to_string())
print('\nbook by price band:', bk.groupby(pd.cut(bk.price, [1, 2, 5, 10, 20, 50, 1e6]), observed=True).rr.agg(['mean', 'count']).round(3).to_dict('index'))
print('book wrapper vs common:', bk.groupby('is_wrapper').rr.agg(['mean', 'count']).round(3).to_dict('index'))
print('book by entry hour:', bk.groupby(pd.cut(bk.entry_m, [570, 600, 630, 720, 840, 960]), observed=True).rr.agg(['mean', 'count']).round(3).to_dict('index'))
for nd, nc in ((5, 3), (8, 4), (12, 4)):
    report(run_book(T, nd, nc), f'book variant {nd}/day, {nc} concurrent')
print('DONE', flush=True)
