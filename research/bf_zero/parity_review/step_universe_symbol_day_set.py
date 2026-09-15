"""Spec-vs-live SYMBOL-DAY SET test: on N TEST days run `simulate` (the live detector) over EVERY universe symbol-day with
high >= open*1.05 (the spec_sim superset) and compare with spec_trades.csv (which only simulated symbol-days where the
study's F5 detector fired: h >= hod*1.003 and >= 30 RTH bars). Read-only."""
import os, sqlite3, sys
import numpy as np, pandas as pd
os.chdir('/home/ec2-user/onemil'); sys.path.insert(0, '/home/ec2-user/onemil')
from trading.hod_break import HodBreakParams, simulate, detect
D = 'research/bf_zero'; ROOT = '/home/ec2-user/onemil'; OPEN_M = 570
days = sys.argv[1:] or ['2026-08-11', '2026-08-25', '2026-09-03']
P = HodBreakParams()
u = pd.read_csv(f'{D}/universe.csv', dtype={'symbol': str}, keep_default_na=False)
u = u[u.bar_date.isin(days)].copy()
for k in ('open', 'high', 'adv20'): u[k] = pd.to_numeric(u[k], errors='coerce')
u = u[u.high >= u.open * (1 + P.min_dist_open_pct / 100)]
T = pd.read_csv(f'{D}/spec_trades.csv', dtype={'symbol': str}, keep_default_na=False); T = T[T.day.isin(days)]
for k in ('level', 'entry', 'rr', 'entry_m'): T[k] = pd.to_numeric(T[k], errors='coerce')
cache = sqlite3.connect(f'file:{ROOT}/data/cache.db?mode=ro', uri=True, timeout=120)
SIDE = [sqlite3.connect(f'file:{p}?mode=ro', uri=True, timeout=120) for p in (f'{ROOT}/research/ignition_capcheck/topup.db', f'{ROOT}/data/research/databento/pit_bars_1min.db', f'{D}/bars.db') if os.path.exists(p)]


def load_bars(day, syms):
    out = {}
    q = f"select symbol, timestamp as t, open as o, high as h, low as l, close as c, volume as v from intraday_bars_1min where bar_date=? and symbol in ({','.join('?' * len(syms))})"
    for s, gg in pd.read_sql(q, cache, params=[day] + list(syms)).groupby('symbol'): out[s] = gg
    for con in SIDE:
        left = [s for s in syms if s not in out]
        if not left: break
        t = pd.read_sql("select symbol, t, o, h, l, c, v from bars where day=?", con, params=[day])
        for s, gg in t[t.symbol.isin(left)].groupby('symbol'): out[s] = gg
    res = {}
    for s, gg in out.items():
        ts = pd.to_datetime(gg.t, utc=True).dt.tz_convert('America/New_York')
        gg = gg.assign(m=(ts.dt.hour * 60 + ts.dt.minute).values).sort_values('m').drop_duplicates('m')
        res[s] = gg[['m', 'o', 'h', 'l', 'c', 'v']].reset_index(drop=True)
    return res


tot_live = tot_spec = tot_extra = tot_missing = 0; extra_rows = []
for day in days:
    sub = u[u.bar_date == day]; bars = load_bars(day, sub.symbol.tolist()); live = {}
    nobars = 0; short = 0
    for r in sub.itertuples():
        gg = bars.get(r.symbol)
        if gg is None: nobars += 1; continue
        rth = gg[(gg.m >= OPEN_M) & (gg.m < 960)].reset_index(drop=True)
        if len(rth) < 30: short += 1
        if len(rth) < 7: continue
        o, h, l, cl, v = (rth[k].values.astype(float) for k in ('o', 'h', 'l', 'c', 'v')); m = rth.m.values.astype(int)
        t = simulate(o, h, l, cl, v, m, r.adv20, P)
        if t is None: continue
        sig = detect(o, h, l, v, m, r.adv20, P)
        live[r.symbol] = dict(level=sig.level, entry=t.entry, rr=t.rr, entry_m=int(m[t.entry_idx]), nbars=len(rth), break_h=float(h[sig.bar_idx]))
    spec = {r.symbol: r for r in T[T.day == day].itertuples()}
    extra = sorted(set(live) - set(spec)); missing = sorted(set(spec) - set(live))
    same = [s for s in live if s in spec and abs(live[s]['entry'] - spec[s].entry) < 1e-6 and abs(live[s]['rr'] - spec[s].rr) < 1e-6]
    e20 = [s for s in extra if live[s]['level'] >= 20]
    print(f"{day}: universe symbol-days {len(sub)} (no bars {nobars}, <30 bars {short}) | live-spec signals {len(live)} | spec_trades rows {len(spec)} | identical {len(same)} | EXTRA in live {len(extra)} (level>=20: {len(e20)}, meanR {np.mean([live[s]['rr'] for s in e20]) if e20 else float('nan'):+.3f}) | in spec not live {len(missing)} {missing[:8]}")
    for s in extra:
        d_ = live[s]; extra_rows.append(dict(day=day, symbol=s, **d_, touch_bps=(d_['break_h'] / d_['level'] - 1) * 1e4))
    tot_live += len(live); tot_spec += len(spec); tot_extra += len(extra); tot_missing += len(missing)
E = pd.DataFrame(extra_rows)
if len(E):
    print('EXTRA (live-only) signals:', len(E), '| break bar high within 30 bps of level:', int((E.touch_bps < 30).sum()), '| <30 RTH bars:', int((E.nbars < 30).sum()), '| level>=20:', int((E.level >= 20).sum()), 'meanR', round(E.rr[E.level >= 20].mean(), 3), 'n_all meanR', round(E.rr.mean(), 3))
    E.to_csv('/tmp/claude-1000/-home-ec2-user-onemil/257c3e2d-cf38-45d5-94e7-4877f8170f44/scratchpad/live_only_signals.csv', index=False)
print('TOTAL live', tot_live, 'spec', tot_spec, 'extra', tot_extra, 'missing', tot_missing)
