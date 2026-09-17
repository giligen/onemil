"""TASK A — retro-apply the shipped B+ veto stack to the live ORB trades.

Read-only. Writes only under research/fuckup_audit/D1_orb/.

For every live ORB trade (research/fuckup_audit/live_trades_dump.csv,
strategy='orb') decide, from the entered-inclusive features CSV row for that
(symbol, date) — falling back to cache.db daily_bars/intraday bars where the
row is missing — which of today's B+ rules would have blocked it:

    composite threshold  (orb.yaml filter.threshold, frozen B+ fit)
    Q1 filter            (orb.yaml quintile_cutoffs, frozen B+ fit)
    PDR veto             prev_day_range_pct <= 11.0
    G1 veto              rv20 >= 7.106 AND pdr >= 9.226 (fail-open rv20 NaN/0)
    G1 short-history     rv20 == 0.0 marker (currently OFF in orb.yaml)
    range-size veto      range_size_pct <= 2.221
    catalyst veto        own-ticker PM news OR anchor cohort >= 2

Every rule is evaluated through the SHARED production helper (parity by
construction) — no reimplementation of a threshold here.
"""
from __future__ import annotations

import json
import os
import sqlite3
import sys

import pandas as pd
import yaml

sys.path.insert(0, '/home/ec2-user/onemil')
os.chdir('/home/ec2-user/onemil')

from trading.orb_csv import read_orb_csv
from study_orb_filter import FILTER_FEATURES, composite_score
from study_orb_sizing import assign_quintile
from trading.orb_pdr_veto import pdr_veto_applies, compute_prev_day_range_pct
from trading.orb_g1_veto import g1_reject
from trading.orb_range_size_veto import range_size_veto_applies
from trading.orb_catalyst_veto import (anchor_cohort_counts,
                                       catalyst_veto_applies)
from trading.orb_asset_class import (DEFAULT_CLASS_MAP, underlying_anchor,
                                     load_class_map)

OUT = 'research/fuckup_audit/D1_orb'
FEATURES = 'analysis_results/orb_features_20260916_2053.csv'
DUMP = 'research/fuckup_audit/live_trades_dump.csv'
BPLUS_GO = '2026-08-17'          # B+ live GO (orb.yaml strategy.enabled true)
BPLUS_PARAMS = '2026-08-15'      # frozen z-params/cutoffs shipped


def log(*a):
    print(*a)
    sys.stdout.flush()


# ---------------------------------------------------------------- config
cfg = yaml.safe_load(open('orb.yaml'))
filt = cfg['filter']
THRESHOLD = float(filt['threshold'])
CUTOFFS = [float(x) for x in cfg['quintile_cutoffs']]
PDR_MIN = float(filt['prev_day_range_veto']['min_prev_day_range_pct'])
G1 = filt['g1_veto']
G1_RV = float(G1['return_volatility_20d_min'])
G1_PDR = float(G1['prev_day_range_pct_min'])
G1_SHORT_SHIPPED = bool(G1.get('short_history_veto', False))
RS_MIN = float(filt['range_size_veto']['min_range_size_pct'])
MIN_COHORT = int(filt['catalyst_veto']['min_cohort'])
ZPARAMS = {f: {'mean': float(filt['features'][f]['mean']),
               'std': float(filt['features'][f]['std']),
               'sign': int(filt['features'][f]['sign'])}
           for f, _ in FILTER_FEATURES}
log(f"B+ config: threshold={THRESHOLD} cutoffs={CUTOFFS} pdr<={PDR_MIN} "
    f"g1(rv>={G1_RV},pdr>={G1_PDR},short_veto={G1_SHORT_SHIPPED}) "
    f"range_size<={RS_MIN} cohort>={MIN_COHORT}")

# ---------------------------------------------------------------- live
live = pd.read_csv(DUMP, keep_default_na=False)
orb = live[live['strategy'] == 'orb'].copy()
orb['pnl'] = pd.to_numeric(orb['pnl'], errors='coerce')
orb['R'] = pd.to_numeric(orb['R'], errors='coerce')
orb['era'] = orb['trade_date'].apply(lambda d: 'B+' if d >= BPLUS_GO else 'pre-B+')


def pd_get(s, k, default=None):
    try:
        d = json.loads(s) if s else {}
    except Exception:
        return default
    return d.get(k, default)


for k in ('composite_score', 'quintile', 'range_high', 'range_low',
          'range_size', 'adaptive_mult', 'anchor', 'anchor_cohort',
          'has_news', 'pm_mult'):
    orb['live_' + k] = orb['pattern_data'].apply(lambda s, k=k: pd_get(s, k))
log(f"live ORB trades: {len(orb)}  pre-B+={int((orb['era']=='pre-B+').sum())} "
    f"B+={int((orb['era']=='B+').sum())}  "
    f"realized ${orb['pnl'].sum():,.0f}")

# ---------------------------------------------------------------- features
feat = read_orb_csv(FEATURES)
feat['date'] = pd.to_datetime(feat['date']).dt.strftime('%Y-%m-%d')
log(f"features rows: {len(feat)}  window {feat['date'].min()}..{feat['date'].max()}")

# composite / quintile on the WHOLE candidate set, frozen B+ params
need = [f for f, _ in FILTER_FEATURES]
feat_ok = feat.dropna(subset=need).copy()
feat_ok['_composite'] = composite_score(feat_ok, ZPARAMS)
feat_ok['_quintile'] = assign_quintile(feat_ok['_composite'], CUTOFFS)
fmap = {(r['symbol'], r['date']): r for _, r in feat_ok.iterrows()}

# ---------------------------------------------------------------- news/anchor
names = {}
try:
    import csv as _csv
    with open(DEFAULT_CLASS_MAP, newline='') as fh:
        for row in _csv.DictReader(fh):
            names[row['symbol']] = row.get('name', '')
except Exception as e:
    log(f"class-map names unavailable: {e}")
cmap = load_class_map()
anchors = {s: underlying_anchor(s, names.get(s), cmap) for s in set(feat['symbol']) | set(orb['symbol'])}

raw_news = {}
import glob
for p in sorted(glob.glob('data/research/orb_news_catalyst_*.csv')):
    nw = read_orb_csv(p)
    for _, r in nw.iterrows():
        try:
            n = int(r['n_articles'] or 0)
        except Exception:
            n = 0
        raw_news[(r['symbol'], str(r['day']))] = n > 0
log(f"news map: {len(raw_news)} symbol-days")

# cohort counts per day over the FULL candidate universe (as the pipeline does)
feat_a = feat.assign(_a=feat['symbol'].map(anchors))
cohorts = {d: anchor_cohort_counts(g['_a']) for d, g in feat_a.groupby('date')}

# ---------------------------------------------------------------- fallback
con = sqlite3.connect('file:data/cache.db?mode=ro', uri=True)


def daily_prev(sym, day):
    q = ("SELECT bar_date, high, low, close FROM daily_bars WHERE symbol=? "
         "AND bar_date < ? ORDER BY bar_date DESC LIMIT 25")
    d = pd.read_sql(q, con, params=(sym, day))
    return d


def fallback_features(sym, day):
    """prev_day_range_pct, return_volatility_20d, range_size_pct from cache."""
    out = {'prev_day_range_pct': None, 'return_volatility_20d': None,
           'range_size_pct': None, 'src': []}
    d = daily_prev(sym, day)
    if len(d):
        r0 = d.iloc[0]
        out['prev_day_range_pct'] = compute_prev_day_range_pct(
            r0['high'], r0['low'], r0['close'])
        out['src'].append('daily_bars')
        closes = d['close'].astype(float).to_numpy()[::-1]   # chronological
        closes = closes[-21:]
        if len(closes) >= 5:
            import numpy as np
            rets = np.diff(closes) / closes[:-1]
            out['return_volatility_20d'] = float(np.std(rets, ddof=0) * 100)
        else:
            out['return_volatility_20d'] = 0.0   # short-history marker
    return out


rows = []
for _, t in orb.iterrows():
    sym, day = t['symbol'], t['trade_date']
    f = fmap.get((sym, day))
    rec = {
        'symbol': sym, 'date': day, 'era': t['era'],
        'pnl': t['pnl'], 'R': t['R'], 'exit_reason': t['exit_reason'],
        'shares': t['shares'], 'total_risk': t['total_risk'],
        'live_composite': t['live_composite_score'],
        'live_quintile': t['live_quintile'],
        'in_features': f is not None,
    }
    if f is not None:
        rec.update({
            'src': 'features',
            'composite': float(f['_composite']),
            'quintile': f['_quintile'],
            'prev_day_range_pct': float(f['prev_day_range_pct']),
            'return_volatility_20d': float(f['return_volatility_20d']),
            'range_size_pct': float(f['range_size_pct']),
            'entered': int(f['entered']) if 'entered' in f else 1,
        })
    else:
        fb = fallback_features(sym, day)
        rs = None
        rh, rl = t['live_range_high'], t['live_range_low']
        rec.update({
            'src': 'cache:' + ('+'.join(fb['src']) or 'none'),
            'composite': None, 'quintile': None,
            'prev_day_range_pct': fb['prev_day_range_pct'],
            'return_volatility_20d': fb['return_volatility_20d'],
            'range_size_pct': rs,
            'entered': 1,
        })
    a = anchors.get(sym)
    rec['anchor'] = a
    rec['cohort'] = cohorts.get(day, {}).get(a, 0) if a else 0
    rec['has_news'] = raw_news.get((sym, day))
    rows.append(rec)

res = pd.DataFrame(rows)
con.close()

# ---------------------------------------------------------------- vetoes
def veto_flags(r):
    f = {}
    f['thr'] = (r['composite'] is not None and not pd.isna(r['composite'])
                and r['composite'] < THRESHOLD)
    f['q1'] = (r['quintile'] == 'Q1')
    f['pdr'] = pdr_veto_applies(
        None if r['prev_day_range_pct'] is None or pd.isna(r['prev_day_range_pct'])
        else float(r['prev_day_range_pct']), PDR_MIN)
    f['g1'] = g1_reject(r['return_volatility_20d'], r['prev_day_range_pct'],
                        G1_RV, G1_PDR, short_history_veto=False) is not None
    f['g1_short'] = (r['return_volatility_20d'] is not None
                     and not pd.isna(r['return_volatility_20d'])
                     and float(r['return_volatility_20d']) == 0.0)
    f['rs'] = range_size_veto_applies(r['range_size_pct'], RS_MIN)
    f['cat'] = catalyst_veto_applies(r['has_news'], r['anchor'],
                                     cohorts.get(r['date'], {}), MIN_COHORT)
    return f


for k in ('thr', 'q1', 'pdr', 'g1', 'g1_short', 'rs', 'cat'):
    res['v_' + k] = [veto_flags(r)[k] for _, r in res.iterrows()]
SHIPPED = ['v_thr', 'v_q1', 'v_pdr', 'v_g1', 'v_rs', 'v_cat']
res['blocked'] = res[SHIPPED].any(axis=1)
res['blocked_short'] = res['blocked'] | res['v_g1_short']
res['block_reasons'] = res.apply(
    lambda r: '+'.join(k[2:] for k in SHIPPED if r[k]) or '', axis=1)
res['month'] = res['date'].str[:7]
res.to_csv(f'{OUT}/taskA_live_retro_veto.csv', index=False)

# ---------------------------------------------------------------- report
def blk(df, label):
    n = len(df)
    if n == 0:
        return f"{label:28s} {0:4d}   —"
    return (f"{label:28s} {n:4d}  ${df['pnl'].sum():>9,.0f}  "
            f"{df['R'].sum():>7.2f}R  mean {df['R'].mean():>6.3f}R  "
            f"WR {100*(df['pnl']>0).mean():>5.1f}%")


log("\n" + "=" * 78)
log("TASK A — B+ veto stack retro-applied to live ORB trades")
log("=" * 78)
for era in ('pre-B+', 'B+', 'ALL'):
    sub = res if era == 'ALL' else res[res['era'] == era]
    log(f"\n[{era}] n={len(sub)} realized ${sub['pnl'].sum():,.0f} "
        f"{sub['R'].sum():.2f}R  WR {100*(sub['pnl']>0).mean():.1f}%")
    log("  " + blk(sub, 'ALL'))
    for k, name in (('v_thr', 'composite < threshold'), ('v_q1', 'Q1 filter'),
                    ('v_pdr', f'PDR <= {PDR_MIN}'), ('v_g1', 'G1 fingerprint'),
                    ('v_rs', f'range-size <= {RS_MIN}'),
                    ('v_cat', 'catalyst required'),
                    ('v_g1_short', 'G1 short-history (OFF)')):
        log("  " + blk(sub[sub[k]], name))
    log("  " + blk(sub[sub['blocked']], 'ANY shipped veto -> BLOCKED'))
    log("  " + blk(sub[~sub['blocked']], 'SURVIVES B+ -> KEPT'))

log("\n--- surviving book by month (all eras) ---")
surv = res[~res['blocked']]
mt = res.groupby('month').agg(n=('pnl', 'size'), pnl=('pnl', 'sum'))
ms = surv.groupby('month').agg(n=('pnl', 'size'), pnl=('pnl', 'sum'),
                               R=('R', 'sum'), wr=('pnl', lambda s: 100*(s > 0).mean()))
log(pd.concat([mt.add_prefix('live_'), ms.add_prefix('kept_')], axis=1).fillna(0).to_string())

log("\n--- stops only (exit_reason contains 'stop') ---")
st = res[res['exit_reason'].str.contains('stop', case=False, na=False)]
log("  " + blk(st, 'all live stops'))
log("  " + blk(st[st['blocked']], 'blocked by B+'))
log("  " + blk(st[~st['blocked']], 'kept by B+'))
log(f"  share of live stop $ blocked: "
    f"{100*st.loc[st['blocked'],'pnl'].sum()/st['pnl'].sum():.1f}%")

log("\n--- reconstruction coverage ---")
log(f"  features row found: {int(res['in_features'].sum())}/{len(res)}")
log(res['src'].value_counts().to_string())
for c in ('prev_day_range_pct', 'return_volatility_20d', 'range_size_pct',
          'composite'):
    miss = res[c].isna().sum()
    log(f"  {c:26s} missing {miss}")
log(f"  has_news known: {int(res['has_news'].notna().sum())}/{len(res)} "
    f"(unknown fails open)")
log(f"  exit_reason mix: {res['exit_reason'].value_counts().to_dict()}")

# live vs B+ composite drift
d = res[res['composite'].notna() & res['live_composite'].notna()].copy()
if len(d):
    d['drift'] = d['composite'].astype(float) - d['live_composite'].astype(float)
    log(f"\n  composite drift (B+ frozen fit vs the score live recorded): "
        f"n={len(d)} mean {d['drift'].mean():+.4f} "
        f"median {d['drift'].median():+.4f} "
        f"|drift|>0.05 on {int((d['drift'].abs()>0.05).sum())} trades")
    log(f"  quintile disagreement: "
        f"{int((d['quintile'] != d['live_quintile']).sum())}/{len(d)}")
log(f"\nWrote {OUT}/taskA_live_retro_veto.csv")
