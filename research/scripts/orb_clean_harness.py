#!/usr/bin/env python3
"""ORB clean re-derivation harness (2026-08-14, post-DST-bug rebuild).

Context: study_orb.py's session-open DST bug (fixed in 3fab1f9) voided all
prior ORB research. This harness re-derives the strategy on clean data:

  1. Rebuilds per-trade "physics" (opening range, breakout entry, post-entry
     bars) directly from cached 1-min bars with an EXACT 9:30 ET anchor —
     rows whose CSV entry trigger disagrees with the reconstructed trigger
     by >10bps are DROPPED as contaminated (belt and suspenders; on a
     regenerated features CSV the drop count should be ~0).
  2. Simulates a family of exit policies (shipped static-lock, profit
     targets, partial-scale, time-boxed) from the same physics.
  3. Applies the selection/sizing/veto stack (composite threshold, quintile
     keep-set, top-N/day + family dedup, PDR veto, catalyst veto, PM mult)
     with every knob parameterized.
  4. Reports owner-bar metrics: monthly P&L, day/trade WR, MDD, top-5 tail
     share, era consistency, positive-month rate.

Physics is pickled per (features_csv, ranges) to the scratchpad so repeated
sweeps don't re-walk cache.db.

Usage:
    python3 research/scripts/orb_clean_harness.py --ranges edt --suite exits
    python3 research/scripts/orb_clean_harness.py --ranges all --suite baseline \
        --features-csv analysis_results/orb_features_<new>.csv

Research-only: writes nothing outside analysis dumps dir / scratchpad.
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import os
import pickle
import sys
from dataclasses import dataclass, field
from datetime import time as dtime, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, REPO)
os.chdir(REPO)

from persistence.database import Database
from study_orb import _bars_to_df
from study_orb_filter import FILTER_FEATURES, fit_z_params, composite_score
from study_orb_sizing import fit_quintile_cutoffs, assign_quintile
from study_orb_correlation_filter import symbol_family, symbol_super_group
from trading.orb_touchgo_filter import (
    TouchgoConfig, evaluate_rule_m, evaluate_rule_d, find_breakout_bar_ts,
)
from trading.orb_pdr_veto import pdr_veto_applies
from trading.orb_pm_mult import pm_size_multiplier

SCRATCH = os.environ.get(
    'ORB_HARNESS_SCRATCH',
    '/tmp/claude-1000/-home-ec2-user-onemil/257c3e2d-cf38-45d5-94e7-4877f8170f44/scratchpad')

EDT_RANGES = [('2025-03-10', '2025-10-31'), ('2026-03-09', '2026-08-13')]
ENTRY_SLIP_BPS = 30.0   # stop-limit band: entry = range_high * (1 + 30bps)
EXIT_SLIP_BPS = 10.0
Q_ORDER = {'Q4': 0, 'Q5': 1, 'Q3': 2, 'Q2': 3, 'Q1': 4}
ERAS = [('2025H1', '2025-01-01', '2025-06-30'),
        ('2025H2', '2025-07-01', '2025-12-31'),
        ('2026', '2026-01-01', '2026-12-31')]


# ---------------------------------------------------------------------------
# Physics: reconstruct range/entry/post-entry bars with correct 9:30 ET anchor
# ---------------------------------------------------------------------------

def _et_minutes(ts: pd.Series) -> np.ndarray:
    et = ts.dt.tz_convert('America/New_York')
    return (et.dt.hour * 60 + et.dt.minute).to_numpy()


def build_physics(df: pd.DataFrame, verbose: bool = True) -> Dict:
    """Per (symbol, date): reconstructed entry price, range, post-entry OHLC.

    Returns dict key -> dict(entry, rh, rl, rsize, etm/o/h/l/c arrays where
    index 0 is the market breakout bar) plus a `drops` counter dict.
    """
    pairs = list(df[['symbol', 'date']].drop_duplicates().apply(
        lambda r: (r['symbol'], r['date'].strftime('%Y-%m-%d')), axis=1))
    if verbose:
        print(f"[physics] loading bars for {len(pairs)} pairs from cache.db ...")
        sys.stdout.flush()
    db = Database(db_path='data/cache.db')
    raw = db.get_intraday_bars_bulk(pairs)
    db.close()

    csv_entry = {(r['symbol'], r['date'].strftime('%Y-%m-%d')): float(r['entry_price'])
                 for _, r in df.iterrows()}
    phys: Dict = {}
    drops = {'no_bars': 0, 'no_930': 0, 'short_range': 0, 'no_breakout': 0,
             'trigger_mismatch': 0}
    for i, (key, bars_raw) in enumerate(raw.items()):
        if verbose and i and i % 1000 == 0:
            print(f"[physics] {i}/{len(raw)} ...")
            sys.stdout.flush()
        bars = _bars_to_df(bars_raw)
        if bars.empty:
            drops['no_bars'] += 1
            continue
        etm = _et_minutes(bars['timestamp'])
        open_idx = np.where(etm == 570)[0]          # 9:30 ET exactly
        if len(open_idx) == 0:
            drops['no_930'] += 1
            continue
        open_ts = bars['timestamp'].iloc[open_idx[0]]
        range_end = open_ts + timedelta(minutes=5)
        rmask = (bars['timestamp'] >= open_ts) & (bars['timestamp'] < range_end)
        rbars = bars[rmask]
        if len(rbars) < 5:
            drops['short_range'] += 1
            continue
        rh = float(rbars['high'].max()); rl = float(rbars['low'].min())
        search = bars[(bars['timestamp'] >= range_end) &
                      (bars['timestamp'] < range_end + timedelta(minutes=60))]
        entry_ts = find_breakout_bar_ts(search, rh)
        if entry_ts is None:
            drops['no_breakout'] += 1
            continue
        entry = rh * (1 + ENTRY_SLIP_BPS / 10000)
        ce = csv_entry.get(key)
        if ce is not None and abs(ce - entry) / entry > 0.001:
            drops['trigger_mismatch'] += 1          # contaminated CSV row
            continue
        post = bars[bars['timestamp'] >= entry_ts].reset_index(drop=True)
        petm = _et_minutes(post['timestamp'])
        keep = petm <= 959                          # cap at 15:59 ET
        phys[key] = {
            'entry': entry, 'rh': rh, 'rl': rl, 'rsize': rh - rl,
            'etm': petm[keep],
            'o': post['open'].to_numpy(float)[keep],
            'h': post['high'].to_numpy(float)[keep],
            'l': post['low'].to_numpy(float)[keep],
            'c': post['close'].to_numpy(float)[keep],
        }
    if verbose:
        print(f"[physics] built {len(phys)} tradeable pairs; drops={drops}")
        sys.stdout.flush()
    phys['__drops__'] = drops
    return phys


def load_or_build_physics(df: pd.DataFrame, cache_tag: str) -> Dict:
    os.makedirs(SCRATCH, exist_ok=True)
    path = os.path.join(SCRATCH, f'orb_physics_{cache_tag}.pkl')
    if os.path.exists(path):
        print(f"[physics] cache hit: {path}")
        with open(path, 'rb') as fh:
            return pickle.load(fh)
    phys = build_physics(df)
    with open(path, 'wb') as fh:
        pickle.dump(phys, fh)
    print(f"[physics] cached -> {path}")
    return phys


# ---------------------------------------------------------------------------
# Exit simulators (all take the physics record; index 0 = breakout bar)
# ---------------------------------------------------------------------------

def _fc_minutes(hhmm: str) -> int:
    h, m = (int(x) for x in hhmm.split(':'))
    return h * 60 + m


@dataclass
class ExitCfg:
    mode: str = 'static_lock'          # static_lock | target | partial
    trigger_R: float = 1.75            # static_lock arm level
    stop_R: float = 0.5                # static_lock locked stop
    target_R: float = 2.0              # target mode
    p1_R: float = 1.0                  # partial mode: first target
    p1_frac: float = 0.5               # partial mode: fraction sold at p1
    rem_lock: bool = True              # partial: remainder runs static lock
    force_close: str = '15:45'
    touchgo: Optional[TouchgoConfig] = None   # None = off


def simulate_exit(p: Dict, x: ExitCfg) -> Optional[Tuple[float, str]]:
    """Return (per-share blended exit price incl. slip, reason) or None."""
    ep, rl, R = p['entry'], p['rl'], p['rsize']
    fc = _fc_minutes(x.force_close)
    n = int(np.searchsorted(p['etm'], fc, side='right'))
    if n == 0:
        return None
    o, h, l, c = p['o'][:n], p['h'][:n], p['l'][:n], p['c'][:n]
    slip = 1 - EXIT_SLIP_BPS / 10000

    if x.touchgo is not None:
        fire, ex = evaluate_rule_m(float(o[0]), float(h[0]), float(l[0]),
                                   float(c[0]), x.touchgo)
        if fire and ex is not None:
            return ex * slip, 'tag_bb'
        if n >= 2:
            fire, ex = evaluate_rule_d(ep, float(l[1]), R, x.touchgo)
            if fire and ex is not None:
                return ex * slip, 'tag_b1'

    if x.mode == 'static_lock':
        trig = ep + x.trigger_R * R
        lock = ep + x.stop_R * R
        stop = rl
        armed = False
        for i in range(1, n):
            if not armed and h[i] >= trig:
                armed = True
                stop = max(stop, lock)
            if l[i] <= stop:
                return stop * slip, ('lock' if armed else 'stop')
        return float(c[n - 1]) * slip, 'eod'

    if x.mode == 'target':
        tgt = ep + x.target_R * R
        for i in range(1, n):
            if l[i] <= rl:                       # pessimistic: stop first
                return rl * slip, 'stop'
            if h[i] >= tgt:
                return tgt * slip, 'target'
        return float(c[n - 1]) * slip, 'eod'

    if x.mode == 'partial':
        tgt = ep + x.p1_R * R
        trig = ep + x.trigger_R * R
        lock = ep + x.stop_R * R
        stop = rl
        realized = 0.0
        rem = 1.0
        armed = False
        for i in range(1, n):
            if rem == 1.0:
                if l[i] <= stop:                 # pessimistic: stop first
                    return stop * slip, 'stop'
                if h[i] >= tgt:
                    realized = x.p1_frac * tgt * slip
                    rem = 1.0 - x.p1_frac
                    stop = max(stop, ep)         # breakeven on remainder
                    continue
            else:
                if x.rem_lock and not armed and h[i] >= trig:
                    armed = True
                    stop = max(stop, lock)
                if l[i] <= stop:
                    return realized + rem * stop * slip, 'p1_stop'
        last = float(c[n - 1]) * slip
        return realized + rem * last, ('p1_eod' if rem < 1.0 else 'eod')

    raise ValueError(f"unknown exit mode {x.mode}")


# ---------------------------------------------------------------------------
# Selection / sizing / veto stack
# ---------------------------------------------------------------------------

@dataclass
class BookCfg:
    name: str = 'shipped'
    exit: ExitCfg = field(default_factory=ExitCfg)
    fit: str = 'yaml'                  # yaml (frozen live) | refit | null
    train: Tuple[str, str] = ('2025-01-01', '2025-06-30')
    threshold: object = 0.0            # float, or 'qNN' = NNth pct of TRAIN composite
    keep_quintiles: Tuple[str, ...] = ('Q2', 'Q3', 'Q4', 'Q5')   # shipped: skip Q1
    n_per_day: int = 4
    account: float = 100_000.0
    risk: float = 3000.0
    min_stop_pct: float = 1.0
    mults: str = 'yaml'                # yaml | uniform
    ranking: str = 'q4_first'          # q4_first (shipped) | composite
    pm_mult: bool = True
    pdr_min: Optional[float] = 8.0     # None = veto off
    catalyst: bool = True


def _yaml_fit() -> Tuple[Dict, List[float], Dict[str, float]]:
    import yaml
    cfg = yaml.safe_load(open('orb.yaml'))
    feats = cfg['filter']['features']
    params = {f: {'mean': float(feats[f]['mean']), 'std': float(feats[f]['std']),
                  'sign': int(feats[f]['sign'])}
              for f, _s in FILTER_FEATURES}
    cutoffs = [float(x) for x in cfg['quintile_cutoffs']]
    mults = {q: float(v) for q, v in cfg['adaptive_mults'].items()}
    return params, cutoffs, mults


_NEWS_CACHE: Dict = {}


def _load_news_and_anchors(universe_syms) -> Tuple[Dict, Dict, Dict]:
    """(raw_news {(sym,day):bool}, eff_news {(sym,day):bool}, anchors {sym:a})."""
    if 'done' in _NEWS_CACHE:
        return (_NEWS_CACHE['raw'], _NEWS_CACHE['eff'], _NEWS_CACHE['anchors'])
    from trading.orb_asset_class import (
        DEFAULT_CLASS_MAP, effective_has_news, load_class_map, underlying_anchor)
    import csv as _csv
    cmap = load_class_map()
    names = {}
    try:
        with open(DEFAULT_CLASS_MAP, newline='') as fh:
            for row in _csv.DictReader(fh):
                names[row['symbol']] = row.get('name', '')
    except Exception as e:
        print(f"[news] class-map names unavailable ({e})")
    raw, eff = {}, {}
    for pth in sorted(glob.glob('data/research/orb_news_catalyst_*.csv')):
        for _, r in pd.read_csv(pth).iterrows():
            hn = (r['n_articles'] or 0) > 0
            raw[(r['symbol'], r['day'])] = hn
            eff[(r['symbol'], r['day'])] = effective_has_news(
                hn, cmap.get(r['symbol'], 'unknown'))
    anchors = {s: underlying_anchor(s, names.get(s), cmap) for s in universe_syms}
    _NEWS_CACHE.update(done=True, raw=raw, eff=eff, anchors=anchors)
    return raw, eff, anchors


_PM_CACHE: Dict = {}


def _load_pm_map() -> Dict:
    if 'map' in _PM_CACHE:
        return _PM_CACHE['map']
    paths = sorted(glob.glob('data/research/orb_premarket_dollar_vol_*.csv'))
    pm = {}
    if paths:
        d = pd.concat([pd.read_csv(x) for x in paths], ignore_index=True)
        d = d.dropna(subset=['pm_dollar_vol']).drop_duplicates(
            subset=['symbol', 'day'], keep='last')
        pm = {(r['symbol'], r['day']): r['pm_dollar_vol'] for _, r in d.iterrows()}
    _PM_CACHE['map'] = pm
    return pm


def run_book(df: pd.DataFrame, phys: Dict, cfg: BookCfg,
             sim_cache: Dict) -> pd.DataFrame:
    """Full pipeline for one config. Returns selected trades with _sized_pnl."""
    d = df.copy()

    # --- exit simulation (cached per ExitCfg repr) ---
    xkey = repr(cfg.exit)
    if xkey not in sim_cache:
        exits = {}
        for key, p in phys.items():
            if key == '__drops__':
                continue
            r = simulate_exit(p, cfg.exit)
            if r is not None:
                exits[key] = r
        sim_cache[xkey] = exits
    exits = sim_cache[xkey]
    d['_key'] = list(zip(d['symbol'], d['date'].dt.strftime('%Y-%m-%d')))
    d = d[d['_key'].isin(exits.keys())].copy()
    d['_exit_price'] = d['_key'].map(lambda k: exits[k][0])
    d['_exit_reason'] = d['_key'].map(lambda k: exits[k][1])
    d['_entry'] = d['_key'].map(lambda k: phys[k]['entry'])

    # --- composite + quintiles ---
    thresh = cfg.threshold
    if cfg.fit == 'yaml':
        params, cutoffs, ymults = _yaml_fit()
        d['_composite'] = composite_score(d, params)
        if isinstance(thresh, str):
            tr = d[(d['date'] >= cfg.train[0]) & (d['date'] <= cfg.train[1])]
            thresh = float(np.percentile(tr['_composite'], float(thresh[1:])))
    elif cfg.fit == 'null':
        # Null hypothesis: random scores (seeded) — does the composite layer
        # beat random selection at all?
        rng = np.random.RandomState(1234)
        d['_composite'] = rng.random(len(d))
        q = float(thresh[1:]) if isinstance(thresh, str) else 50.0
        thresh = float(np.percentile(d['_composite'], q))
        cutoffs = list(np.percentile(
            d.loc[d['_composite'] >= thresh, '_composite'], [20, 40, 60, 80]))
    else:
        tr = d[(d['date'] >= cfg.train[0]) & (d['date'] <= cfg.train[1])]
        if len(tr) < 50:
            raise SystemExit(f"refit train window too small ({len(tr)} rows)")
        params = fit_z_params(tr, FILTER_FEATURES)
        d['_composite'] = composite_score(d, params)
        comp_tr = composite_score(tr, params)
        if isinstance(thresh, str):
            thresh = float(np.percentile(comp_tr, float(thresh[1:])))
        cutoffs = fit_quintile_cutoffs(comp_tr[comp_tr >= thresh])
    kept = d[d['_composite'] >= thresh].copy()
    kept['_quintile'] = assign_quintile(kept['_composite'], cutoffs)
    kept = kept[kept['_quintile'].isin(cfg.keep_quintiles)].copy()

    mults = ({q: 1.0 for q in 'Q1 Q2 Q3 Q4 Q5'.split()} if cfg.mults == 'uniform'
             else _yaml_fit()[2])

    # --- top-N + family dedup per day ---
    rows = []
    for day, dg in kept.groupby('date'):
        g = dg.copy()
        if cfg.ranking == 'composite':
            g = g.sort_values('_composite', ascending=False)
        else:
            g['_q_rank'] = g['_quintile'].map(Q_ORDER)
            g = g.sort_values(['_q_rank', '_composite'], ascending=[True, False])
        seen_f, seen_s = set(), set()
        picks = []
        for _, r in g.iterrows():
            fam = symbol_family(r['symbol']); sup = symbol_super_group(r['symbol'])
            if fam and fam in seen_f:
                continue
            if sup and sup in seen_s:
                continue
            if fam:
                seen_f.add(fam)
            if sup:
                seen_s.add(sup)
            picks.append(r)
            if len(picks) >= cfg.n_per_day:
                break
        rows.extend(picks)
    if not rows:
        return pd.DataFrame()
    sel = pd.DataFrame(rows)

    # --- sizing ---
    cap = cfg.account / cfg.n_per_day
    stop_pct = sel['range_size_pct'].clip(lower=cfg.min_stop_pct)
    position = (cfg.risk / (stop_pct / 100.0)).clip(upper=cap)
    sel['_shares'] = np.floor(position / sel['_entry']).astype(int).clip(lower=0)
    sel = sel[sel['_shares'] > 0].copy()
    sel['_mult'] = sel['_quintile'].map(mults)
    sel['_day'] = sel['date'].dt.strftime('%Y-%m-%d')

    # --- PM mult (news-gated x2.0, shipped semantics) ---
    if cfg.pm_mult:
        pm_map = _load_pm_map()
        _raw, eff_news, _anch = _load_news_and_anchors(set(df['symbol']))
        from trading.orb_pm_mult import (
            DEFAULT_HIGH_CUT_USD, DEFAULT_HIGH_MULT, DEFAULT_HIGH_MULT_NEWS)
        sel['_pm_mult'] = [
            pm_size_multiplier(pm_map.get((s, dy)), DEFAULT_HIGH_CUT_USD,
                               DEFAULT_HIGH_MULT,
                               has_news=eff_news.get((s, dy)),
                               high_mult_news=DEFAULT_HIGH_MULT_NEWS,
                               news_gate=True)
            for s, dy in zip(sel['symbol'], sel['_day'])]
    else:
        sel['_pm_mult'] = 1.0

    sel['_sized_pnl'] = ((sel['_exit_price'] - sel['_entry']) * sel['_shares']
                         * sel['_mult'] * sel['_pm_mult'])

    # --- PDR veto (post-selection, no refill) ---
    if cfg.pdr_min is not None:
        mask = sel['prev_day_range_pct'].apply(
            lambda v: pdr_veto_applies(None if pd.isna(v) else float(v),
                                       cfg.pdr_min))
        sel = sel[~mask].copy()

    # --- catalyst veto (post-selection, no refill) ---
    if cfg.catalyst:
        from trading.orb_catalyst_veto import (
            DEFAULT_MIN_COHORT, anchor_cohort_counts, catalyst_veto_applies)
        raw_news, _eff, anchors = _load_news_and_anchors(set(df['symbol']))
        da = df.assign(_a=df['symbol'].map(anchors),
                       _day=df['date'].dt.strftime('%Y-%m-%d'))
        cohorts = {day: anchor_cohort_counts(g['_a'])
                   for day, g in da.groupby('_day')}
        cv = [catalyst_veto_applies(raw_news.get((s, dy)), anchors.get(s),
                                    cohorts.get(dy, {}), DEFAULT_MIN_COHORT)
              for s, dy in zip(sel['symbol'], sel['_day'])]
        sel = sel[~pd.Series(cv, index=sel.index)].copy()

    return sel


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def book_metrics(sel: pd.DataFrame, name: str) -> Dict:
    if sel.empty:
        return {'name': name, 'trades': 0, 'total': 0.0}
    daily = sel.groupby('date')['_sized_pnl'].sum().sort_index()
    cum = daily.cumsum()
    mdd = float((cum - cum.cummax()).min())
    monthly = sel.groupby(sel['date'].dt.to_period('M'))['_sized_pnl'].sum()
    total = float(sel['_sized_pnl'].sum())
    top5 = float(sel['_sized_pnl'].nlargest(5).sum())
    eras = {}
    for ename, s, e in ERAS:
        sub = sel[(sel['date'] >= s) & (sel['date'] <= e)]
        if len(sub):
            eras[ename] = float(sub['_sized_pnl'].sum())
    return {
        'name': name,
        'trades': len(sel),
        'total': total,
        'trade_wr': float((sel['_sized_pnl'] > 0).mean() * 100),
        'day_wr': float((daily > 0).mean() * 100),
        'mdd': mdd,
        'months': len(monthly),
        'pos_months': int((monthly > 0).sum()),
        'worst_month': float(monthly.min()),
        'best_month': float(monthly.max()),
        'median_month': float(monthly.median()),
        'top5_share': (top5 / total * 100) if total > 0 else float('nan'),
        'eras': eras,
        'monthly': monthly,
    }


def print_metrics(m: Dict, monthly: bool = False):
    if m['trades'] == 0:
        print(f"  {m['name']}: NO TRADES")
        return
    era_s = '  '.join(f"{k}:{v:+,.0f}" for k, v in m['eras'].items())
    print(f"  {m['name']:38s} total ${m['total']:+10,.0f}  n={m['trades']:4d} "
          f"tWR {m['trade_wr']:4.1f}%  dWR {m['day_wr']:4.1f}%  "
          f"MDD ${m['mdd']:+9,.0f}  months+ {m['pos_months']}/{m['months']} "
          f"worst_m ${m['worst_month']:+8,.0f}  top5 {m['top5_share']:5.1f}%  [{era_s}]")
    if monthly:
        print(m['monthly'].round(0).to_string())


# ---------------------------------------------------------------------------
# Suites
# ---------------------------------------------------------------------------

def _tg(on: bool = True, m_thresh: float = 0.5, d_R: float = 0.75) -> Optional[TouchgoConfig]:
    if not on:
        return None
    return TouchgoConfig(rule_m_threshold=m_thresh, rule_d_revert_R=d_R)


def suite_configs(suite: str, fit: str, train: Tuple[str, str]) -> List[BookCfg]:
    SL = lambda **kw: ExitCfg(mode='static_lock', touchgo=_tg(), **kw)
    base = dict(fit=fit, train=train)
    if suite == 'baseline':
        return [BookCfg(name='shipped(yaml-fit,$100K)', exit=SL(), **base)]
    if suite == 'exits':
        out = [BookCfg(name='EXIT static_lock (shipped)', exit=SL(), **base)]
        for tR in (1.0, 1.5, 2.0, 3.0):
            out.append(BookCfg(name=f'EXIT target +{tR}R',
                               exit=ExitCfg(mode='target', target_R=tR,
                                            touchgo=_tg()), **base))
        out.append(BookCfg(name='EXIT partial 50%@1R->BE+lock',
                           exit=ExitCfg(mode='partial', p1_R=1.0,
                                        touchgo=_tg()), **base))
        out.append(BookCfg(name='EXIT partial 50%@1.5R->BE+lock',
                           exit=ExitCfg(mode='partial', p1_R=1.5,
                                        touchgo=_tg()), **base))
        out.append(BookCfg(name='EXIT SL timebox 11:30',
                           exit=SL(force_close='11:30'), **base))
        out.append(BookCfg(name='EXIT SL timebox 13:00',
                           exit=SL(force_close='13:00'), **base))
        out.append(BookCfg(name='EXIT SL no-touchgo',
                           exit=ExitCfg(mode='static_lock', touchgo=None), **base))
        return out
    if suite == 'entries':
        b = dict(exit=SL(), **base)
        out = []
        for th in (0.0, 0.25, 0.5):
            out.append(BookCfg(name=f'ENTRY thresh {th}', threshold=th, **b))
        out.append(BookCfg(name='ENTRY Q4Q5 only',
                           keep_quintiles=('Q4', 'Q5'), **b))
        out.append(BookCfg(name='ENTRY Q3Q4Q5',
                           keep_quintiles=('Q3', 'Q4', 'Q5'), **b))
        out.append(BookCfg(name='ENTRY N=2/day', n_per_day=2, **b))
        out.append(BookCfg(name='VETO no-PDR', pdr_min=None, **b))
        out.append(BookCfg(name='VETO no-catalyst', catalyst=False, **b))
        out.append(BookCfg(name='VETO none', pdr_min=None, catalyst=False, **b))
        out.append(BookCfg(name='SIZING uniform-mults', mults='uniform', **b))
        out.append(BookCfg(name='SIZING no-PM-mult', pm_mult=False, **b))
        out.append(BookCfg(name='SIZING flat(uniform,noPM)', mults='uniform',
                           pm_mult=False, **b))
        return out
    if suite == 'small':
        # $10K-account candidates: N=2 concurrent, $250-500 risk, no PM mult,
        # uniform mults. Explore exit x quality x catalyst-veto interactions.
        sm = dict(account=10_000.0, n_per_day=2, risk=375.0, mults='uniform',
                  pm_mult=False, **base)
        exits = {
            'SL': SL(),
            'tgt1.5R': ExitCfg(mode='target', target_R=1.5, touchgo=_tg()),
            'tgt2R': ExitCfg(mode='target', target_R=2.0, touchgo=_tg()),
            'part1.5R': ExitCfg(mode='partial', p1_R=1.5, touchgo=_tg()),
            'SL-11:30': SL(force_close='11:30'),
        }
        quals = {
            'th.25': dict(threshold=0.25),
            'Q4Q5': dict(keep_quintiles=('Q4', 'Q5')),
        }
        out = []
        for xn, x in exits.items():
            for qn, q in quals.items():
                for cv in (True, False):
                    out.append(BookCfg(
                        name=f'10K {xn} {qn} cv={"Y" if cv else "N"}',
                        exit=x, catalyst=cv, **q, **sm))
        return out
    if suite == 'wr':
        # WR-oriented quick-target exits on the $10K shape
        sm = dict(account=10_000.0, n_per_day=2, risk=375.0, mults='uniform',
                  pm_mult=False, threshold=0.25, **base)
        out = []
        for tR in (0.75, 1.0, 1.25):
            for cv in (True, False):
                out.append(BookCfg(
                    name=f'10K tgt{tR}R th.25 cv={"Y" if cv else "N"}',
                    exit=ExitCfg(mode='target', target_R=tR, touchgo=_tg()),
                    catalyst=cv, **sm))
        return out
    if suite == 'pdr':
        # PDR threshold re-sweep on clean data ($10K shape, catalyst off)
        out = []
        for pdr in (None, 4.0, 6.0, 8.0, 10.0, 12.0):
            out.append(BookCfg(
                name=f'10K tgt1.5R th.25 cvN pdr={pdr}',
                exit=ExitCfg(mode='target', target_R=1.5, touchgo=_tg()),
                threshold=0.25, catalyst=False, pdr_min=pdr,
                account=10_000.0, n_per_day=2, risk=375.0, mults='uniform',
                pm_mult=False, **base))
        return out
    if suite == 'final':
        # Phase B candidate set. Run with --fit refit --train <clean TRAIN>
        # for fit-honest numbers, or --fit yaml for shipped-frozen params.
        sm = dict(account=10_000.0, n_per_day=2, risk=375.0, mults='uniform',
                  pm_mult=False, threshold=0.25, **base)
        out = [
            BookCfg(name='BASE shipped $100K', exit=SL(), **base),
            BookCfg(name='C1 tgt1.5R pdr10 cvN',
                    exit=ExitCfg(mode='target', target_R=1.5, touchgo=_tg()),
                    pdr_min=10.0, catalyst=False, **sm),
            BookCfg(name='C2 tgt2R pdr10 cvN',
                    exit=ExitCfg(mode='target', target_R=2.0, touchgo=_tg()),
                    pdr_min=10.0, catalyst=False, **sm),
            BookCfg(name='C3 part1.5R pdr10 cvN',
                    exit=ExitCfg(mode='partial', p1_R=1.5, touchgo=_tg()),
                    pdr_min=10.0, catalyst=False, **sm),
            BookCfg(name='C4 SL pdr10 cvY', exit=SL(), pdr_min=10.0,
                    catalyst=True, **sm),
            BookCfg(name='C5 SL-11:30 pdr10 cvY', exit=SL(force_close='11:30'),
                    pdr_min=10.0, catalyst=True, **sm),
        ]
        return out
    if suite == 'small_risk':
        # risk sensitivity on a fixed shape
        out = []
        for risk in (250.0, 375.0, 500.0):
            out.append(BookCfg(
                name=f'10K tgt1.5R th.25 cvN risk{int(risk)}',
                exit=ExitCfg(mode='target', target_R=1.5, touchgo=_tg()),
                threshold=0.25, catalyst=False, account=10_000.0,
                n_per_day=2, risk=risk, mults='uniform', pm_mult=False,
                **base))
        return out
    raise SystemExit(f"unknown suite {suite}")


# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--features-csv', default=None)
    ap.add_argument('--ranges', default='edt', help='edt | all | s1:e1,s2:e2')
    ap.add_argument('--suite', default='baseline')
    ap.add_argument('--fit', default='yaml', choices=['yaml', 'refit'])
    ap.add_argument('--train', default='2025-01-01:2025-06-30')
    ap.add_argument('--monthly', action='store_true')
    ap.add_argument('--dump-dir', default=None,
                    help='dump per-config selected trades CSVs here')
    args = ap.parse_args()

    csv = args.features_csv
    if csv is None:
        cands = sorted(glob.glob('analysis_results/orb_features_*.csv'))
        csv = [p for p in cands if 'corrmatrix' not in p][-1]
    print(f"features CSV: {csv}")
    df = pd.read_csv(csv)
    needed = [f for f, _ in FILTER_FEATURES]
    df = df.dropna(subset=needed + ['pnl', 'date', 'range_size_pct', 'entry_price'])
    df['date'] = pd.to_datetime(df['date'])

    if args.ranges == 'edt':
        ranges = EDT_RANGES
    elif args.ranges == 'all':
        ranges = [(str(df['date'].min().date()), str(df['date'].max().date()))]
    else:
        ranges = [tuple(r.split(':')) for r in args.ranges.split(',')]
    mask = pd.Series(False, index=df.index)
    for s, e in ranges:
        mask |= (df['date'] >= s) & (df['date'] <= e)
    df = df[mask].copy()
    print(f"rows after range filter: {len(df)}  ({ranges})")

    tag = hashlib.md5((os.path.basename(csv) + repr(ranges)).encode()).hexdigest()[:10]
    phys = load_or_build_physics(df, tag)
    train = tuple(args.train.split(':'))

    sim_cache: Dict = {}
    results = []
    for cfg in suite_configs(args.suite, args.fit, train):
        sel = run_book(df, phys, cfg, sim_cache)
        m = book_metrics(sel, cfg.name)
        results.append((cfg, sel, m))
        print_metrics(m, monthly=args.monthly)
        sys.stdout.flush()
        if args.dump_dir and not sel.empty:
            os.makedirs(args.dump_dir, exist_ok=True)
            safe = ''.join(ch if ch.isalnum() else '_' for ch in cfg.name)
            sel.to_csv(os.path.join(args.dump_dir, f'{safe}.csv'), index=False)
    return results


if __name__ == '__main__':
    main()
