"""ORB — the 2x-wrapper universe rule, decided by evidence (2026-09-05).

Background: `AlpacaClient._is_common_stock` (commit 83472a4, 2026-04-04, a
bull-flag rule) drops leveraged/inverse wrappers from the nightly asset list
that refreshes daily_bars — the table ORB (live AND BT) seeds its universe
from. Wrappers listed BEFORE April 2026 sit in daily_bars from earlier cache
builds; wrappers listed AFTER never enter. The book therefore runs an
accidental rule: "old wrappers in, new wrappers out".

Decision rule (fixed BEFORE the numbers): pick IN or OUT by sized total
P&L, max drawdown and all three eras (25H1 / 25H2 / 2026) positive; if the
two are within noise (|ΔP&L| < $5K and MDD within $2K) IN wins, because the
catalyst-confirmation design (stock + its wrappers = anchor cohort) assumes
wrappers exist. The accident (as-is) is not a candidate.

Variants, same pipeline, same config, only the features rows differ:
  asis          production entered-inclusive features (the accident)
  wrappers_in   asis + post-April wrapper rows from the point-in-time
                features (pit_missing_classified: wrapper_policy)
  wrappers_out  asis minus every wrapper (trading.orb_asset_class)
  pit_full      asis + ALL point-in-time rows (survivorship answer)

Usage: python3 wrapper_rule_test.py PIT_FEATURES.csv [PROD_FEATURES.csv]
"""
import csv
import glob
import os
import subprocess
import sys

import pandas as pd

ROOT = '/home/ec2-user/onemil'
sys.path.insert(0, ROOT)
from trading.orb_csv import read_orb_csv  # noqa: E402  ticker-safe reader (NA/NAN/NULL are symbols)
from data_sources.alpaca_client import AlpacaClient  # noqa: E402

D = f'{ROOT}/research/orb_entered_inclusive'
OUT = f'{D}/wrapper_rule'
PIPELINE = f'{ROOT}/study_orb_pipeline_static_lock.py'


def latest_prod_features() -> str:
    """Most recent production features CSV (corrmatrix sidecar excluded)."""
    return sorted(p for p in glob.glob(f'{ROOT}/analysis_results/orb_features_*.csv')
                  if 'corrmatrix' not in p)[-1]


def wrapper_symbols(symbols) -> set:
    """Symbols the ACTUAL universe rule (`AlpacaClient._is_common_stock`, the
    filter on the nightly asset list) would drop — so OUT = that rule applied
    uniformly to history, IN = that rule removed. The lev-family sets from
    orb_correlation are added (they are wrappers by curation, whatever the
    name says). Symbols with no known name are kept (cannot be judged)."""
    names = {r['symbol']: r['name'] for r in csv.DictReader(
        open(f'{ROOT}/data/research/orb_asset_class_map_20260711.csv', newline=''))}
    alp = f'{ROOT}/data/research/databento/alpaca_assets_all_20260905.csv'
    if os.path.exists(alp):
        a = read_orb_csv(alp, keep_default_na=False).drop_duplicates('symbol')
        names = {**dict(zip(a['symbol'], a['name'])), **names}
    from trading.orb_asset_class import _lev_family_symbols
    fam = _lev_family_symbols()
    return {s for s in symbols
            if s in fam or (names.get(s) and not AlpacaClient._is_common_stock(s, names[s]))}


def run_variant(name: str, feats: pd.DataFrame) -> str:
    """Write the variant's features CSV, run the pipeline on it, return the book path."""
    os.makedirs(OUT, exist_ok=True)
    fcsv, book, monthly = f'{OUT}/features_{name}.csv', f'{OUT}/book_{name}.csv', f'{OUT}/monthly_{name}.csv'
    feats.to_csv(fcsv, index=False)
    env = dict(os.environ, ORB_BT_FEATURES_CSV=fcsv, ORB_BT_BOOK_OUT=book, ORB_BT_MONTHLY_OUT=monthly)
    with open(f'{ROOT}/logs/orb_pipeline_wrapper_{name}_20260905.log', 'w') as log:
        rc = subprocess.call([sys.executable, PIPELINE], cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
    print(f"[{name}] features {len(feats):,} rows -> pipeline exit {rc} -> {book}", flush=True)
    if rc != 0:
        sys.exit(rc)
    return book


def book_stats(path: str, label: str) -> dict:
    """Sized P&L, MDD, per-era P&L, negative months, wrapper share of the book."""
    b = read_orb_csv(path)
    b['date'] = pd.to_datetime(b['date']).dt.strftime('%Y-%m-%d')
    pnl = b['_sized_pnl'] if '_sized_pnl' in b.columns else b['pnl']
    daily = b.assign(p=pnl).groupby('date')['p'].sum().sort_index()
    cum = daily.cumsum()
    era = pd.cut(pd.to_datetime(b['date']).astype('int64'),
                 bins=[0, pd.Timestamp('2025-07-01').value, pd.Timestamp('2026-01-01').value, pd.Timestamp('2100-01-01').value],
                 labels=['25H1', '25H2', '2026'])
    e = b.assign(p=pnl, era=era).groupby('era', observed=True)['p'].sum()
    monthly = b.assign(p=pnl).groupby(b['date'].str[:7])['p'].sum()
    ent = b[b['entered'] == 1] if 'entered' in b.columns else b
    wr = wrapper_symbols(set(b['symbol']))
    return {'variant': label, 'picks': len(b), 'fills': len(ent), 'pnl': round(pnl.sum()),
            'mdd': round(float((cum - cum.cummax()).min())),
            '25H1': round(e.get('25H1', 0)), '25H2': round(e.get('25H2', 0)), '2026': round(e.get('2026', 0)),
            'neg_months': int((monthly < 0).sum()), 'worst_month': round(monthly.min()),
            'wrapper_picks': int(b['symbol'].isin(wr).sum())}


def decide(s_in: dict, s_out: dict) -> str:
    """The pre-committed rule."""
    def ok(s):
        return all(s[k] > 0 for k in ('25H1', '25H2', '2026'))
    if abs(s_in['pnl'] - s_out['pnl']) < 5000 and abs(s_in['mdd'] - s_out['mdd']) < 2000:
        return 'IN (within noise -> design wins)'
    score_in = (ok(s_in), s_in['pnl'] + s_in['mdd'])      # mdd is negative: less DD = higher
    score_out = (ok(s_out), s_out['pnl'] + s_out['mdd'])
    return 'IN' if score_in >= score_out else 'OUT'


def main() -> None:
    pit_csv = sys.argv[1]
    prod_csv = sys.argv[2] if len(sys.argv) > 2 else latest_prod_features()
    prod, pit = read_orb_csv(prod_csv), read_orb_csv(pit_csv)
    for df in (prod, pit):
        df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')
    cls = read_orb_csv(f'{ROOT}/data/research/databento/pit_missing_classified.csv')
    cls = cls[cls['book'] == 'orb'].drop_duplicates(['symbol', 'bar_date']).set_index(['symbol', 'bar_date'])['cls']
    pit['cls'] = [cls.get((s, d), 'unclassified') for s, d in zip(pit['symbol'], pit['date'])]
    key_prod = set(zip(prod['symbol'], prod['date']))
    pit_new = pit[[k not in key_prod for k in zip(pit['symbol'], pit['date'])]]
    print(f"prod {prod_csv}: {len(prod):,} | PIT new keys {len(pit_new):,} by class: "
          f"{pit_new['cls'].value_counts().to_dict()}", flush=True)
    wr_prod = wrapper_symbols(set(prod['symbol']))
    print(f"wrappers in the production features: {len(wr_prod)} symbols, "
          f"{int(prod['symbol'].isin(wr_prod).sum()):,} rows", flush=True)

    new_wr = pit_new[pit_new['cls'] == 'active_filtered_by_is_common_stock'].drop(columns=['cls'])
    variants = {
        'asis': prod,
        'wrappers_in': pd.concat([prod, new_wr], ignore_index=True),
        'wrappers_out': prod[~prod['symbol'].isin(wr_prod)],
        'pit_full': pd.concat([prod, pit_new.drop(columns=['cls'])], ignore_index=True),
    }
    stats = [book_stats(run_variant(n, f), n) for n, f in variants.items()]
    tab = pd.DataFrame(stats)
    print("\n" + tab.to_string(index=False))
    s = {r['variant']: r for r in stats}
    print(f"\nDECISION (pre-committed rule): wrappers {decide(s['wrappers_in'], s['wrappers_out'])}")
    print(f"survivorship (pit_full - asis): P&L {s['pit_full']['pnl'] - s['asis']['pnl']:+,}  MDD {s['pit_full']['mdd'] - s['asis']['mdd']:+,}")
    tab.to_csv(f'{OUT}/summary.csv', index=False)


if __name__ == '__main__':
    main()
