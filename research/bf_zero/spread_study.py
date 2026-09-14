#!/usr/bin/env python3
"""HOD-break spread study (pre-registered 2026-09-14, owner: "use historical data for the spread selection").

Question: does the quoted spread at signal time, as a fraction of the trade's risk R, predict the spec
trade's outcome — and where should the cost gate `max_spread_frac_r` sit?

Data: a random sample of the spec's own signals (`spec_trades.csv`, 38,953 rows with entry minute, entry,
stop, R outcome under the exact live fill rule) across TRAIN 2025 / VAL Jan–May 26 / TEST Jun–Sep 26.
For each sampled signal the NBBO quotes in the SIGNAL minute (the closed break bar) are fetched from
Alpaca's historical quotes API (SIP): spread = median(ask − bid) over that minute, at the last quote
before the next bar's open — i.e. what the engine sees when it decides. spread_frac_r = spread / R.

Rule (fixed before running): quintiles of spread_frac_r cut on TRAIN only; report mean R, WR and
trades/week per bucket per split; a gate at a bucket edge is adopted only if the excluded buckets are
worse than the kept ones on TRAIN AND VAL, and TEST (read once) agrees in sign. Also reported: the
"after-cost" mean R = R − spread_frac_r (entry at the ask, stop at the bid ≈ one spread per losing trade;
a deliberately pessimistic charge) so the gate can be judged net of the cost it is meant to avoid.

Usage: python3 research/bf_zero/spread_study.py [N_SAMPLE=7000] [SEED=7]   → spread_study.csv + spread_study.md
Cost: free (Alpaca subscription); ~200 requests/min → ~35 min for 7,000 signals. Never selects on TEST.
"""
import os, sys, time
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo
import numpy as np, pandas as pd

ROOT = '/home/ec2-user/onemil'; os.chdir(ROOT); sys.path.insert(0, ROOT)
from config import Config                                     # noqa: E402
from alpaca.data.historical import StockHistoricalDataClient  # noqa: E402
from alpaca.data.requests import StockQuotesRequest           # noqa: E402
from alpaca.data.enums import DataFeed                        # noqa: E402

D = 'research/bf_zero'; ET = ZoneInfo('America/New_York')
N = int(sys.argv[1]) if len(sys.argv) > 1 else 7000; SEED = int(sys.argv[2]) if len(sys.argv) > 2 else 7
OUT = f'{D}/spread_study.csv'

t = pd.read_csv(f'{D}/spec_trades.csv', dtype={'symbol': str})
t['split'] = np.where(t.day < '2026-01-01', 'TRAIN', np.where(t.day < '2026-06-01', 'VAL', 'TEST'))
t = t[t.price >= 5]                                          # the live book's price floor
sample = t.groupby('split', group_keys=False).apply(lambda g: g.sample(min(len(g), N // 3), random_state=SEED)).reset_index(drop=True)
print(f'signals {len(t):,} (price >= $5) | sampled {len(sample):,} by split {sample.split.value_counts().to_dict()}', flush=True)
done = pd.read_csv(OUT, dtype={'symbol': str}) if os.path.exists(OUT) else pd.DataFrame(columns=['day', 'symbol'])
done_keys = set(zip(done.day, done.symbol))
cfg = Config(); client = StockHistoricalDataClient(cfg.alpaca_api_key, cfg.alpaca_api_secret)
rows = []; t0 = time.time(); n_ok = 0; n_fail = 0
for i, r in enumerate(sample.itertuples()):
    if (r.day, r.symbol) in done_keys: continue
    sig_min = int(r.entry_m) - 1                                   # the closed break bar = the minute before the entry (next-open) bar
    start = datetime.strptime(r.day, '%Y-%m-%d').replace(tzinfo=ET) + timedelta(minutes=sig_min); end = start + timedelta(minutes=1)
    try:
        q = client.get_stock_quotes(StockQuotesRequest(symbol_or_symbols=r.symbol, start=start.astimezone(timezone.utc), end=end.astimezone(timezone.utc), feed=DataFeed.SIP, limit=2000))
        qs = q.data.get(r.symbol, []) if hasattr(q, 'data') else q.get(r.symbol, [])
        sp = [(float(x.ask_price) - float(x.bid_price)) for x in qs if float(x.ask_price) > 0 and float(x.bid_price) > 0 and float(x.ask_price) >= float(x.bid_price)]
        if not sp: n_fail += 1; rows.append(dict(day=r.day, symbol=r.symbol, split=r.split, n_quotes=0)); continue
        last = qs[-1]; rows.append(dict(day=r.day, symbol=r.symbol, split=r.split, n_quotes=len(sp), spread_med=float(np.median(sp)), spread_last=float(last.ask_price) - float(last.bid_price),
                                        ask_last=float(last.ask_price), bid_last=float(last.bid_price), entry=r.entry, stop=r.stop, r_pct=r.r_pct, rr=r.rr, why=r.why, price=r.price, rv_profile=r.rv_profile))
        n_ok += 1
    except Exception as e:
        n_fail += 1; rows.append(dict(day=r.day, symbol=r.symbol, split=r.split, n_quotes=-1, err=str(e)[:80]))
        if 'too many' in str(e).lower() or '429' in str(e): time.sleep(20)
    time.sleep(float(os.environ.get("SPREAD_SLEEP", "0.4")))
    if len(rows) >= 200:
        pd.DataFrame(rows).to_csv(OUT, mode='a', header=not os.path.exists(OUT), index=False); rows = []
    if i % 200 == 0: print(f'{i}/{len(sample)} ok {n_ok} fail {n_fail} | {(time.time() - t0) / 60:.1f} min', flush=True)
if rows: pd.DataFrame(rows).to_csv(OUT, mode='a', header=not os.path.exists(OUT), index=False)

# ---------------- scoring ----------------
d = pd.read_csv(OUT, dtype={'symbol': str}); d = d[d.n_quotes > 0].copy()
d['R'] = d.entry - d.stop; d['spread_frac_r'] = d.spread_last / d.R; d['spread_bps'] = d.spread_last / d.entry * 1e4
d['rr_after_cost'] = d.rr - d.spread_frac_r                    # pessimistic: one full spread charged to every trade
edges = d[d.split == 'TRAIN'].spread_frac_r.quantile([.2, .4, .6, .8]).values
d['q'] = np.searchsorted(edges, d.spread_frac_r) + 1
lines = [f'# HOD-break spread study — {len(d):,} signals with quotes ({d.split.value_counts().to_dict()}); TRAIN quintile edges of spread/R: {np.round(edges, 3).tolist()}', '']
for s in ('TRAIN', 'VAL', 'TEST'):
    x = d[d.split == s]
    g = x.groupby('q').agg(n=('rr', 'size'), meanR=('rr', 'mean'), meanR_after_cost=('rr_after_cost', 'mean'), WR=('rr', lambda r: (r > 0).mean() * 100), spread_bps=('spread_bps', 'median'), sfr=('spread_frac_r', 'median')).round(3)
    lines += [f'## {s}', g.to_markdown(), '']
for frac in (0.10, 0.15, 0.20, 0.30):
    lines.append(f'gate spread/R <= {frac:.0%}: ' + ' | '.join(f"{s}: keep {int((d[(d.split == s)].spread_frac_r <= frac).sum())}/{int((d.split == s).sum())} meanR kept {d[(d.split == s) & (d.spread_frac_r <= frac)].rr.mean():+.3f} dropped {d[(d.split == s) & (d.spread_frac_r > frac)].rr.mean():+.3f} | after-cost kept {d[(d.split == s) & (d.spread_frac_r <= frac)].rr_after_cost.mean():+.3f}" for s in ('TRAIN', 'VAL', 'TEST')))
lines += ['', 'by price band (all splits): ' + d.groupby(pd.cut(d.price, [5, 10, 20, 50, 1e6]), observed=True).agg(n=('rr', 'size'), spread_bps=('spread_bps', 'median'), sfr=('spread_frac_r', 'median'), meanR=('rr', 'mean'), after=('rr_after_cost', 'mean')).round(3).to_string().replace('\n', '\n  ')]
open(f'{D}/spread_study.md', 'w').write('\n'.join(lines)); print('\n'.join(lines)); print('DONE', flush=True)
