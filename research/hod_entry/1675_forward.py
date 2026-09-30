#!/usr/bin/env python3
"""Cell 1,675 -- short the HOD-break failure, sealed forward test (PREREG_1675.md, FROZEN).

Stage `train`: re-fits the FF10-at-k=1 models from the EXISTING 1669_per_fill.csv (1669 never
persisted them) -- same HistGradientBoostingClassifier(max_iter=200), same RNG_SEED=1669, same
feature columns (trading.hod_failure_features.ordered_feature_list) -- and saves both to
research/hod_entry/models/.

Stage `score`: scores the sealed forward population (research/hod_entry/forward_2026q3/
causal_arming_causal.csv, status=='fill', built by build_fwd_population.py -- the SAME cell-1,438
rule, unchanged, sliced to 2026-06-01..2026-09-04 only -- 2026-09-05..2026-09-26 is NOT covered,
universe.csv/nbbo.csv both stop 2026-09-04) with both saved models, applies the short rule, and
writes the week table + per-short ledger + RESULT_1675.md.

F11-F15: computed by IMPORTING research/hod_entry/1667_sweep.py's own compute_intraday_features
(same level-bar match, same DST-aware minute_of_day/et_offset_minutes, same bars_sip.db read-only
source) -- not reimplemented, so this is the same code path, not a lookalike. FEATURES_CSV on the
imported module is monkeypatched to forward_2026q3/1667_features_fwd.csv so the original
1667_features.csv is never touched. F8 (needed only for F15) is not available for the forward
population -- F15 is NaN wherever F8 is NaN, same graceful-degradation convention the function
already uses (counted, not silently dropped).

Reversal variant convention (PREREG read 3, no PREREG-given cost split for the cut leg, so this is
a disclosed choice): reversal = cut the long at the short's own entry price (open of bar fill+2,
COVER_BPS-costed marketable sell) + the short trade itself, both expressed in DOLLARS and divided
by the LONG's own R unit (fill-stop) to land "in the long's units" as the PREREG asks; baseline =
the long's own recorded net_R from the causal_arming population.

Pessimistic gap pricing (read 4): a short's stop/target resolution is repriced at the resolving
bar's OPEN whenever that bar GAPPED past the level (open already through it) rather than merely
touching it intrabar -- matches the project's obtainability rule (CLAUDE.md 1b): a touch is not a
fill, a resting order fills at the bar's open under a gap.
"""
import argparse
import importlib.util
import json
import os
import sqlite3
import sys
import time

import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import HistGradientBoostingClassifier

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)

import causal_arming as ca  # noqa: E402
import sip_rebuild as sr  # noqa: E402
from trading import hod_failure_features as hf  # noqa: E402

MODELS_DIR = os.path.join(HERE, 'models')
FWD_DIR = os.path.join(HERE, 'forward_2026q3')
PER_FILL_1669 = os.path.join(HERE, '1669_per_fill.csv')
RNG_SEED = 1669
ENTRY_BPS, COVER_BPS, EOD_COVER_BPS = 0.0007, 0.0006, 0.0011
GO_CAP_PER_DAY = 12
GO_THRESH = 0.6
os.makedirs(MODELS_DIR, exist_ok=True)

log = sr.log


# --------------------------------------------------------------------------------------- train
def stage_train():
    per_fill = pd.read_csv(PER_FILL_1669, dtype={'date': str, 'symbol': str}, low_memory=False)
    comp = per_fill['k1_computable'].astype(str).isin(['True', 'true', '1', '1.0'])
    feat_cols = hf.ordered_feature_list(per_fill.columns)
    log(f'[train] {len(feat_cols)} feature columns for FF10 k=1')
    saved = {}
    for half, fname in (('TRAIN-H2', 'ff10_k1_trainh2'), ('VAL', 'ff10_k1_val')):
        sub = per_fill[(per_fill.split == half) & comp & per_fill['FF10'].notna()].copy()
        X = sub[feat_cols].astype(float)
        y = sub['FF10'].astype(int)
        model = HistGradientBoostingClassifier(max_iter=200, random_state=RNG_SEED)
        model.fit(X, y)
        path = os.path.join(MODELS_DIR, f'{fname}.joblib')
        joblib.dump(model, path)
        saved[half] = dict(path=path, n=int(len(y)), pos_rate=float(y.mean()))
        log(f'[train] {half}: n={len(y)} pos_rate={y.mean():.3f} -> {path}')
    with open(os.path.join(MODELS_DIR, 'ff10_k1_features.json'), 'w') as fh:
        json.dump({'label': 'FF10', 'k': 1, 'feature_cols': feat_cols, 'rng_seed': RNG_SEED,
                    'model_class': 'HistGradientBoostingClassifier(max_iter=200)',
                    'trained': saved}, fh, indent=2)
    log(f'[train] DONE -> {MODELS_DIR}')
    return feat_cols, saved


# --------------------------------------------------------------------------------------- F11-F15
def load_f1667():
    spec = importlib.util.spec_from_file_location('f1667', os.path.join(HERE, '1667_sweep.py'))
    f1667 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(f1667)
    f1667.FEATURES_CSV = os.path.join(FWD_DIR, '1667_features_fwd.csv')  # never touch the original
    return f1667


def compute_f11_f15(fills):
    """fills needs: date(day), symbol, fill_min, level, atr14, n_cross. Returns fills + F11-F15 +
    level_bar_found, via 1667_sweep.compute_intraday_features (same code path, bars_sip.db RO)."""
    f1667 = load_f1667()
    df = fills.rename(columns={'day': 'date'}).copy()
    if 'F8' not in df.columns:
        df['F8'] = np.nan
    log(f'[f11-15] scoring {len(df)} forward fills via 1667_sweep.compute_intraday_features '
        f'(bars_sip.db read-only, FEATURES_CSV redirected to {f1667.FEATURES_CSV})')
    out = f1667.compute_intraday_features(df, resume=False)
    cov = out['level_bar_found'].mean()
    log(f'[f11-15] DONE level-bar coverage {cov:.1%} ({int(out.level_bar_found.sum())}/{len(out)})')
    return out.rename(columns={'date': 'day'})


# --------------------------------------------------------------------------------------- atr14
def atr14_causal(con, symbol, day, cache={}):
    """Standard 14-session ATR (simple mean of True Range), strictly before `day`."""
    key = symbol
    if key not in cache:
        cache[key] = pd.read_sql(
            "select bar_date, high, low, close from daily_bars where symbol=? and bar_date<? order by bar_date desc limit 30",
            con, params=[symbol, day])
    d = cache[key]
    if len(d) < 5:
        return None
    d = d.sort_values('bar_date')
    prev_close = d['close'].shift(1)
    tr = np.maximum(d['high'] - d['low'], np.maximum((d['high'] - prev_close).abs(), (d['low'] - prev_close).abs()))
    tr = tr.dropna().tail(14)
    return float(tr.mean()) if len(tr) else None


# --------------------------------------------------------------------------------------- score
def load_models():
    models = {h: joblib.load(os.path.join(MODELS_DIR, f'ff10_k1_{h}.joblib')) for h in ('trainh2', 'val')}
    feat_cols = json.load(open(os.path.join(MODELS_DIR, 'ff10_k1_features.json')))['feature_cols']
    return models, feat_cols


def etb_flags(symbols):
    """Alpaca shortable/easy_to_borrow flags via the project's own data_sources/alpaca_client.py
    AlpacaClient.get_shortability -- the same rail trading/hod_failure_short.py uses, fail-closed
    per symbol (False/False on error, never fabricated). Uses the LIVE (ALPACA_API_KEY /
    ALPACA_PAPER=false) credentials, not the paper-only ALPACA_HOD_* pair, since ETB status
    differs between paper and live inventories and the book this feeds is the live account.
    {} on any auth/config failure (logged ERROR) -- ETB share then reads as unavailable, which
    fails the go/no-go safely rather than fabricating a share."""
    out = {}
    try:
        from dotenv import load_dotenv
        load_dotenv(os.path.join(ROOT, '.env'))
        sys.path.insert(0, ROOT)
        from data_sources.alpaca_client import AlpacaClient
        key = os.environ.get('ALPACA_API_KEY')
        sec = os.environ.get('ALPACA_API_SECRET')
        paper = os.environ.get('ALPACA_PAPER', 'true').lower() == 'true'
        if not key or not sec:
            log('[etb] ERROR: no ALPACA_API_KEY/ALPACA_API_SECRET in environment/.env -- ETB flags unavailable')
            return out
        client = AlpacaClient(key, sec, paper=paper)
        log(f'[etb] AlpacaClient ready (paper={paper}); checking {len(symbols)} symbols')
        for i, s in enumerate(symbols):
            flags = client.get_shortability(s)
            out[s] = (flags['shortable'], flags['easy_to_borrow'])
            if (i + 1) % 50 == 0:
                log(f'[etb] {i + 1}/{len(symbols)} assets checked')
                time.sleep(0.3)
        log(f'[etb] DONE {len(out)}/{len(symbols)} assets resolved, '
            f'{sum(v[1] for v in out.values())} easy-to-borrow')
    except Exception as e:  # noqa: BLE001
        log(f'[etb] ERROR: Alpaca client unavailable ({type(e).__name__}: {e}) -- ETB flags unavailable')
    return out


def simulate_short(bars, i0, fill, stop, level, atr14, arm_extra, models, feat_cols):
    """One fill's short-rule read: FF10-k1 scores from both models, and (if fired) the short's own
    trade result at level pricing AND at pessimistic gap pricing, plus the reversal-variant R."""
    R = fill - stop
    if R <= 0:
        return None
    target_long = fill + sr.TARGET_R * R
    break_idx = hf.find_break_bar(bars, bars['minarr'][i0], level)
    break_bar_v = bars['v'][break_idx] if break_idx is not None else np.nan
    atr14_pct = (atr14 / fill * 100) if atr14 else np.nan
    arm_ctx = dict(r_pct=(fill - stop) / fill * 100, atr14_pct=atr14_pct, minutes_since_open=bars['minarr'][i0] - 570)
    arm_ctx.update(arm_extra)  # F11-F15
    feats, computable = hf.k1_features(bars, i0, fill, stop, target_long, level, atr14, break_bar_v,
                                        arm_ctx, spy_bars=None, fill_min=bars['minarr'][i0])
    out = dict(computable=computable)
    if not computable:
        return out
    row = pd.DataFrame([{c: feats.get(c, np.nan) for c in feat_cols}]).astype(float)
    out['p_trainh2'] = float(models['trainh2'].predict_proba(row)[:, 1][0])
    out['p_val'] = float(models['val'].predict_proba(row)[:, 1][0])

    n = len(bars['o'])
    i_entry = i0 + 2                       # open of bar fill+2
    out['fireable'] = i_entry < n
    if not out['fireable']:
        return out
    short_entry = float(bars['o'][i_entry])
    day_high_so_far = float(bars['h'][:i_entry + 1].max())
    short_stop = day_high_so_far + 0.01
    short_target = stop                     # the long's stop level, 1R in the long's units
    r_unit = short_stop - short_entry
    out.update(short_entry=short_entry, short_stop=short_stop, short_target=short_target,
               r_unit=r_unit, entry_min=int(bars['minarr'][i_entry]))
    if r_unit <= 0:
        out['fireable'] = False
        return out

    exit_px = why = None
    eod = gapped = False
    for j in range(i_entry, n):
        if bars['minarr'][j] >= hf.EOD_M:
            exit_px, why, eod = float(bars['c'][j - 1] if j > i_entry else short_entry), 'eod', True
            break
        if bars['h'][j] >= short_stop:
            why = 'stop'
            gapped = bars['o'][j] >= short_stop
            exit_px = float(bars['o'][j]) if gapped else short_stop
            break
        if bars['l'][j] <= short_target:
            why = 'target'
            gapped = bars['o'][j] <= short_target
            exit_px = float(bars['o'][j]) if gapped else short_target
            break
    if exit_px is None:
        exit_px, why, eod = float(bars['c'][-1]), 'eod_noclose', True
    cover_bps = EOD_COVER_BPS if eod else COVER_BPS
    gross = (short_entry - exit_px) / r_unit
    cost = (ENTRY_BPS * short_entry + cover_bps * exit_px) / r_unit
    net_R = gross - cost
    # pessimistic companion: level pricing regardless of gap (the optimistic number), reported
    # alongside so read 4's split is exact (gapped_through=True rows already ARE the pessimistic
    # price above; for non-gapped rows the two are identical).
    if why in ('stop', 'target') and gapped:
        level_px = short_stop if why == 'stop' else short_target
        gross_opt = (short_entry - level_px) / r_unit
        net_R_optimistic = gross_opt - (ENTRY_BPS * short_entry + cover_bps * level_px) / r_unit
    else:
        net_R_optimistic = net_R
    out.update(exit_price=exit_px, why=why, gross_R=gross, cost_R=cost, net_R=net_R,
               gapped_through=bool(gapped), net_R_optimistic=net_R_optimistic, net_R_pessimistic=net_R)

    # reversal variant: cut the long at short_entry (COVER_BPS marketable sell) + the short trade,
    # both in dollars, divided by the LONG's own R unit -- "in the long's units".
    cut_long_dollars = (short_entry - fill) - COVER_BPS * short_entry
    short_dollars = net_R * r_unit
    out['reversal_R_long_units'] = (cut_long_dollars + short_dollars) / R
    return out


def stage_score(workers):
    models, feat_cols = load_models()
    fwd_csv = os.path.join(FWD_DIR, 'causal_arming_causal.csv')
    if not os.path.exists(fwd_csv):
        log(f'[score] ERROR: {fwd_csv} missing -- population stage not done. STOPPING.')
        return None
    pop = pd.read_csv(fwd_csv, dtype={'day': str, 'symbol': str}, keep_default_na=False, na_values=[''])
    fills = pop[pop.status == 'fill'].reset_index(drop=True)
    log(f'[score] {len(fills)} candidate fills from {fwd_csv}')

    con = sqlite3.connect(sr.CACHE_DB_URI, uri=True, timeout=120)
    fills['atr14'] = [atr14_causal(con, r.symbol, r.day) for r in fills.itertuples()]
    log(f'[score] atr14 computed, {fills.atr14.notna().mean():.1%} coverage')

    f11_15 = compute_f11_f15(fills[['day', 'symbol', 'fill_min', 'level', 'atr14', 'n_cross']])
    fills = fills.merge(f11_15[['day', 'symbol', 'F11', 'F12', 'F13', 'F14', 'F15', 'level_bar_found']],
                          on=['day', 'symbol'], how='left')

    sipcon = sqlite3.connect(ca.BARS_SIP_URI, uri=True, timeout=120)
    rows = []
    n_ok = n_nobar = n_notcomp = 0
    for day, sub in fills.groupby('day'):
        bars_by_sym = ca.load_day_bars(con, day, sub.symbol.tolist(), sipcon)
        for r in sub.itertuples():
            bdf = bars_by_sym.get(r.symbol)
            if bdf is None or len(bdf) < 3:
                n_nobar += 1
                continue
            # ca.load_day_bars returns a DataFrame with column 'm'; hod_failure_features'
            # pure functions take the project's numpy-array bar-dict convention ('minarr').
            bars = dict(minarr=bdf['m'].to_numpy(), o=bdf['o'].to_numpy(), h=bdf['h'].to_numpy(),
                         l=bdf['l'].to_numpy(), c=bdf['c'].to_numpy(), v=bdf['v'].to_numpy())
            i0 = hf.find_fill_index(bars, r.fill_min)
            if i0 is None:
                n_nobar += 1
                continue
            arm_extra = dict(F11=r.F11, F12=r.F12, F13=r.F13, F14=r.F14, F15=r.F15)
            res = simulate_short(bars, i0, r.fill, r.stop, r.level, r.atr14, arm_extra, models, feat_cols)
            if res is None or not res.get('computable'):
                n_notcomp += 1
                continue
            n_ok += 1
            rows.append({**res, 'day': day, 'symbol': r.symbol, 'wk': r.wk, 'fill_min': r.fill_min,
                          'long_fill': r.fill, 'long_stop': r.stop, 'long_net_R': r.net_R, 'atr14': r.atr14,
                          'level_bar_found': r.level_bar_found})
        log(f'[score] day {day} done | scored so far {n_ok} | no-bar {n_nobar} | not-computable {n_notcomp}')
    con.close(); sipcon.close()
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(HERE, '1675_per_short.csv'), index=False)
    log(f'[score] DONE scored={n_ok} nobar={n_nobar} notcomputable={n_notcomp} -> 1675_per_short.csv')
    return out


# --------------------------------------------------------------------------------------- reads
def _fired(df, model_col, thresh=GO_THRESH):
    f = df[(df.computable) & (df.fireable == True) & (df[model_col] >= thresh)].copy()  # noqa: E712
    f['cap_rank'] = f.sort_values('entry_min').groupby('day').cumcount() + 1
    f['under_cap'] = f.cap_rank <= GO_CAP_PER_DAY
    return f


def week_clustered_t(fired):
    """t of the week-mean net_R, clustered by week (weeks as the iid unit)."""
    wk_means = fired.groupby('wk').net_R.mean()
    n = len(wk_means)
    if n < 2 or wk_means.std(ddof=1) == 0:
        return np.nan, n
    return float(wk_means.mean() / (wk_means.std(ddof=1) / np.sqrt(n))), n


def build_week_table(fired, etb_map):
    fired = fired.copy()
    fired['is_etb'] = fired.symbol.map(lambda s: etb_map.get(s, (None, None))[1]) if etb_map else np.nan
    rows = []
    for wk, g in fired.groupby('wk'):
        rows.append(dict(
            wk=wk, shorts=len(g),
            etb_share=float(pd.Series(g.is_etb).mean()) if etb_map else np.nan,
            hit_rate=float((g.net_R > 0).mean()), mean_net_R=float(g.net_R.mean()), sum_R=float(g.net_R.sum()),
            dollars_at_150=float(g.net_R.sum() * 150),
            worst_day=float(g.groupby('day').net_R.sum().min()),
            max_concurrent=int(g.groupby('day').symbol.count().max()),
            shorts_under_cap=int(g.under_cap.sum())))
    return fired, pd.DataFrame(rows).sort_values('wk').reset_index(drop=True)


def verdict_line(fired, etb_map, pess=False):
    """Frozen rule, evaluated on one (model, ETB on/off) cell."""
    col = 'net_R_pessimistic' if pess else 'net_R'
    f = fired if not etb_map else fired[fired.symbol.map(lambda s: bool(etb_map.get(s, (False, False))[1]))]
    if not len(f):
        return dict(n=0, mean_R=np.nan, t=np.nan, n_weeks=0, green_weeks=0, worst_week=np.nan,
                     etb_share=np.nan, shorts_per_wk=np.nan, pess_mean=np.nan, go=False)
    wk = f.groupby('wk')[col].sum()
    t, n_weeks = week_clustered_t(f.rename(columns={col: 'net_R'}))
    green = int((wk > 0).sum())
    etb_share = float(fired.symbol.map(lambda s: bool(etb_map.get(s, (False, False))[1])).mean()) if etb_map else np.nan
    shorts_per_wk = f.groupby('wk').size().mean()
    pess_mean = f.net_R_pessimistic.mean()
    d = dict(n=len(f), mean_R=float(f[col].mean()), t=t, n_weeks=n_weeks, green_weeks=green,
              worst_week=float(wk.min()), etb_share=etb_share, shorts_per_wk=float(shorts_per_wk),
              pess_mean=float(pess_mean))
    d['go'] = bool(d['mean_R'] >= 0.05 and (not np.isnan(t) and t >= 2.0) and green >= max(1, int(0.615 * n_weeks))
                    and d['worst_week'] >= -3 and (np.isnan(etb_share) or etb_share >= 0.70)
                    and shorts_per_wk >= 3 and pess_mean >= 0.03)
    return d


def write_result(scored, etb_map, window_note):
    lines = ['# RESULT_1675 -- short the HOD-break failure, sealed forward test', '',
              f'**Population window: {window_note}** -- universe.csv and causal_filter/nbbo.csv both end '
              '2026-09-04; their builder script could not be located within budget, so 2026-09-05..09-26 is '
              'NOT covered. Same population definition (cell 1,438), not a substitute.', '']
    if scored is None or not len(scored):
        lines += ['No scored fills -- population/scoring did not complete. GO/NO-GO: **NO-GO** (no data). '
                   'Mechanics stay OFF.']
        open(os.path.join(HERE, 'RESULT_1675.md'), 'w').write('\n'.join(lines))
        return

    verdicts = {}
    for label, col in (('TRAIN-H2 model', 'p_trainh2'), ('VAL model', 'p_val')):
        fired = _fired(scored, col)
        fired, wt = build_week_table(fired, etb_map)
        lines.append(f'## {label} -- week table (P>=0.6), n={len(fired)} fired shorts, {len(wt)} weeks')
        lines.append(wt.to_markdown(index=False) if len(wt) else '(no shorts fired)')
        lines.append('')
        for etb_on in (False, True):
            v = verdict_line(fired, etb_map if etb_on else None)
            verdicts[(label, etb_on)] = v
            tag = 'WITH ETB filter' if etb_on else 'no ETB filter'
            lines.append(f'* {label}, {tag}: n={v["n"]}, mean net R={v["mean_R"]:+.4f}, week-t={v["t"]:.2f} '
                          f'(n_weeks={v["n_weeks"]}), green {v["green_weeks"]}/{v["n_weeks"]}, '
                          f'worst week={v["worst_week"]:+.2f}R, ETB share={v["etb_share"]}, '
                          f'shorts/wk under cap={v["shorts_per_wk"]:.2f}, pessimistic-gap mean={v["pess_mean"]:+.4f} '
                          f'-> **{"GO" if v["go"] else "NO-GO"}**')
        # cap split
        under, over = fired[fired.under_cap], fired[~fired.under_cap]
        lines.append(f'* Cap split (first {GO_CAP_PER_DAY}/day): under-cap n={len(under)} mean R='
                      f'{under.net_R.mean():+.4f} | over-cap n={len(over)} mean R='
                      f'{(over.net_R.mean() if len(over) else float("nan")):+.4f}')
        # stop-bucket / time-of-day
        for lo, hi, tag in ((1.5, 3.0, '1.5-3%'), (3.0, 1e9, '>=3%')):
            sub = fired[(fired.long_fill > 0) & (((fired.long_fill - fired.long_stop) / fired.long_fill * 100).between(lo, hi, inclusive='left' if hi < 1e9 else 'both'))]
            lines.append(f'* Stop bucket {tag}: n={len(sub)} mean R={(sub.net_R.mean() if len(sub) else float("nan")):+.4f}')
        gapped = fired.gapped_through.mean() if len(fired) else float('nan')
        lines.append(f'* Gapped-through share of resolving exits: {gapped:.1%}; pessimistic-gap pooled mean = '
                      f'{fired.net_R_pessimistic.mean():+.4f} vs optimistic {fired.net_R_optimistic.mean():+.4f}')
        rev = fired.reversal_R_long_units.mean() if len(fired) else float('nan')
        hold = fired.long_net_R.mean() if len(fired) else float('nan')
        lines.append(f'* Reversal variant (cut+short, long units): mean={rev:+.4f} vs holding the long mean='
                      f'{hold:+.4f} (n={len(fired)})')
        lines.append('')

    both_go = all(verdicts[(lab, True)]['go'] for lab in ('TRAIN-H2 model', 'VAL model'))
    lines.append(f'## GO / NO-GO (frozen rule, BOTH models + ETB filter required): '
                  f'{"**GO**" if both_go else "**NO-GO**"}')
    with open(os.path.join(HERE, 'RESULT_1675.md'), 'w') as fh:
        fh.write('\n'.join(lines))
    log(f'[result] RESULT_1675.md written, verdict GO={both_go}')
    return verdicts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage', choices=['train', 'score', 'reads', 'all'], default='all')
    ap.add_argument('--workers', type=int, default=8)
    a = ap.parse_args()
    if a.stage in ('train', 'all'):
        stage_train()
    scored = None
    if a.stage in ('score', 'all'):
        scored = stage_score(a.workers)
    if a.stage == 'reads':
        scored = pd.read_csv(os.path.join(HERE, '1675_per_short.csv'), dtype={'day': str, 'symbol': str})
    if a.stage in ('score', 'reads', 'all'):
        if scored is not None and len(scored):
            etb_map = etb_flags(sorted(scored.symbol.unique().tolist()))
        else:
            etb_map = {}
        write_result(scored, etb_map, '2026-06-01..2026-09-04 (universe.csv/nbbo.csv end 09-04; 09-05..09-26 not covered)')
    return 0


if __name__ == '__main__':
    sys.exit(main())
