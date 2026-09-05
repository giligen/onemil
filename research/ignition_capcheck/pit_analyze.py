"""Ignition capcheck — survivorship delta (2026-09-05).

Adds the point-in-time trades (trades_PIT.csv: the candidate symbol-days the
cache-based universe never saw, bars from Databento) to the baseline book
(trades_all_annotated.csv) and reports what the missing cohort would have
done: P&L, WR, monsters, per era and per missing-class bucket, on the river
(ALL) and the complex-confirmed (CC) proxy book. Anchor cohorts are
recomputed on the UNION (a new symbol can confirm an old one).
"""
import csv, sys
import pandas as pd
sys.path.insert(0, '/home/ec2-user/onemil')
from trading.orb_csv import read_orb_csv  # noqa: E402  ticker-safe reader (NA/NAN/NULL are symbols)
from trading.orb_asset_class import load_class_map, underlying_anchor
sys.stdout.reconfigure(line_buffering=True)
ROOT = '/home/ec2-user/onemil'; D = f'{ROOT}/research/ignition_capcheck'
base = read_orb_csv(f'{D}/trades_all_annotated.csv', dtype={'symbol': str}, keep_default_na=False)
base = base.drop_duplicates(['day', 'symbol']).copy(); base['src'] = 'base'
pit = read_orb_csv(f'{D}/trades_PIT.csv', dtype={'symbol': str}, keep_default_na=False); pit['src'] = 'pit'
for df in (base, pit):
    for c in df.columns:
        if c not in ('day', 'symbol', 'reason', 'src', 'anchor', 'era'):
            df[c] = pd.to_numeric(df[c], errors='coerce')
cls = read_orb_csv(f'{ROOT}/data/research/databento/pit_missing_classified.csv')
cls = cls[cls.book == 'ignition'].drop_duplicates(['symbol', 'bar_date']).set_index(['symbol', 'bar_date'])['cls']
bucket = {'unknown_to_alpaca': 'survivorship', 'inactive_delisted': 'survivorship', 'active_tradable_common_ABSENT': 'ticker_reuse/vendor',
          'active_filtered_by_is_common_stock': 'wrapper_policy', 'active_not_tradable': 'correct_exclusion', 'test_ticker': 'correct_exclusion'}
pit['bucket'] = [bucket.get(cls.get((s, d)), 'unclassified') for s, d in zip(pit.symbol, pit.day)]
names = {r['symbol']: r['name'] for r in csv.DictReader(open(f'{ROOT}/data/research/orb_asset_class_map_20260711.csv', newline=''))}
alp = read_orb_csv(f'{ROOT}/data/research/databento/alpaca_assets_all_20260905.csv', keep_default_na=False).drop_duplicates('symbol')
names = {**dict(zip(alp.symbol, alp.name)), **names}
cmap = load_class_map()
allt = pd.concat([base.drop(columns=[c for c in ('anchor', 'coh', 'complex_conf', 'era', 'ym', 'monster2', 'monster3') if c in base.columns]), pit], ignore_index=True)
allt['anchor'] = allt.symbol.map({s: underlying_anchor(s, names.get(s), cmap) for s in allt.symbol.unique()})
sz = allt[allt.anchor.notna()].groupby(['day', 'anchor']).size().rename('coh')
allt = allt.merge(sz, left_on=['day', 'anchor'], right_index=True, how='left')
allt['cc'] = allt.coh.fillna(0) >= 2
allt['era'] = pd.cut(pd.to_datetime(allt.day).astype('int64'), bins=[0, pd.Timestamp('2025-07-01').value, pd.Timestamp('2026-01-01').value, pd.Timestamp('2100-01-01').value], labels=['25H1', '25H2', '2026'])
allt['m2'] = allt.rr >= 2.0
b, p = allt[allt.src == 'base'], allt[allt.src == 'pit']
def ledger(df, label):
    print(f"{label:34s} n={len(df):5d}  pnl=${df.pnl.sum():+12,.0f}  WR={((df.pnl > 0).mean() * 100 if len(df) else 0):5.1f}%  monsters>=2R={int(df.m2.sum()):3d}  meanR={df.rr.mean() if len(df) else 0:+.3f}")
print("=== RIVER (ALL, no catalyst gate) ===")
ledger(b, 'baseline'); ledger(p, 'PIT additions'); ledger(pd.concat([b, p]), 'union')
print("=== CC proxy book (anchor cohort >= 2, recomputed on union) ===")
ledger(b[b.cc], 'baseline CC'); ledger(p[p.cc], 'PIT additions CC'); ledger(pd.concat([b, p])[lambda d: d.cc], 'union CC')
print(f"baseline CC trades whose cc flag CHANGED by adding PIT symbols: {int((b.cc != base.set_index(['day','symbol']).loc[list(zip(b.day, b.symbol)), 'complex_conf'].astype(bool).values).sum())}")
print("\n=== PIT additions by bucket ===")
print(p.groupby('bucket').agg(n=('pnl', 'size'), pnl=('pnl', 'sum'), wr=('pnl', lambda s: (s > 0).mean() * 100), m2=('m2', 'sum'), cc_n=('cc', 'sum'), cc_pnl=('pnl', lambda s: s[p.loc[s.index, 'cc']].sum())).round(1).to_string())
print("\n=== PIT additions by era (river / CC) ===")
print(p.groupby('era', observed=True).agg(n=('pnl', 'size'), pnl=('pnl', 'sum'), cc_n=('cc', 'sum'), cc_pnl=('pnl', lambda s: s[p.loc[s.index, 'cc']].sum())).round(0).to_string())
print("\n=== PIT additions by month (river) ===")
print(p.groupby(p.day.str[:7]).pnl.agg(['size', 'sum']).round(0).to_string())
print("\n=== top |pnl| PIT trades ===")
print(p.reindex(p.pnl.abs().sort_values(ascending=False).index).head(12)[['day', 'symbol', 'bucket', 'rr', 'pnl', 'pos', 'cc', 'anchor']].to_string(index=False))
print(f"\nPIT anchor resolved: {int(p.anchor.notna().sum())}/{len(p)}")
allt.to_csv(f'{D}/trades_union_pit_annotated.csv', index=False)
