"""Independent rebuild of PREREG_1548.md — extension anatomy + the sweep entry on predicted extenders.

Built from the prose ONLY (never opened cell_1548.py / test_cell_1548.py / cell_1548_fills.csv /
RESULT_1548.md). Reuses the repo's day-clustered-t / ex-top-5% / winner-cap / weeks-spanned helpers
from cell_1445.py (named in the task) and the ET-minute conversion pattern used across this
directory's other rebuilds (rebuild_1491.py). Everything else — the join, the drawdown/touch walk,
the sweep-limit fill rule, the cost model, the null — is built fresh from PREREG_1548.md's text.

Usage: python3 rebuild_1548.py [--smoke N]
"""
import argparse
import sqlite3
import sys
import time
from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

sys.path.insert(0, '.')
from cell_1445 import day_clustered_t, ex_top5_mean, winner_capped_mean, weeks_spanned, WINNER_CAP_R  # noqa: E402

HERE = '.'
PRED_CSV = f'{HERE}/model_1478_L3_v2_predictions.csv'
CAUSAL_CSV = f'{HERE}/causal_arming_causal.csv'
FEATURES_A = f'{HERE}/features_1478_A.csv'
BARS_DB = f'{HERE}/bars_fills_1478.db'

ET = ZoneInfo('America/New_York')
OPEN_M, EOD_M = 570, 955          # 09:30, 15:55 ET -- sip_rebuild.py convention (confirmed independently
                                  # in cell_1430.py / cell_1479.py / entry_replay.py / cell_1493.py)
TOUCH_MULT = 1.05                 # the +5% extension label
W_CAP = 120.0                      # cap on W per the prose
D_FLOOR = 0.20                    # floor on d (% of level) per the prose

# cost constants from the prose (research/hod_entry/cell_1478.py SLIP_STOP_BPS; EOD-at-the-bid given directly)
SLIP_STOP_BPS = {'TRAIN': 0.88 * 2.9 + 0.12 * 94.0, 'VAL': 0.88 * 3.2 + 0.12 * 76.0}
EOD_BID_BPS = {'TRAIN': 11.5, 'VAL': 9.7}

SEED = 1548
N_DRAWS = 1000


def log(msg):
    print(f'[{time.strftime("%H:%M:%S")}] {msg}', flush=True)


def et_minute(iso_ts):
    """UTC ISO string -> ET minutes-since-midnight (float)."""
    dt = datetime.fromisoformat(iso_ts).astimezone(ET)
    return dt.hour * 60 + dt.minute + dt.second / 60.0


# --------------------------------------------------------------------------------------------
# Load & join
# --------------------------------------------------------------------------------------------

def load_population():
    """Kept population (hgb_kept_L3 == True) on TRAIN(-H2)/VAL, joined to the base-fill fields."""
    pred = pd.read_csv(PRED_CSV)
    pred = pred[pred.split.isin(['TRAIN', 'VAL'])].copy()
    kept = pred[pred.hgb_kept_L3 == True].reset_index(drop=True)          # noqa: E712
    dropped = pred[pred.hgb_kept_L3 == False].reset_index(drop=True)      # noqa: E712
    log(f'load_population: kept {len(kept)} ({kept.split.value_counts().to_dict()}), '
        f'dropped {len(dropped)} ({dropped.split.value_counts().to_dict()})')

    causal = pd.read_csv(CAUSAL_CSV, low_memory=False)
    fills = causal[causal.status == 'fill'].reset_index(drop=True)
    base_cols = ['day', 'symbol', 'fill_min', 'fill', 'stop', 'level', 'why', 'net_R']
    fills = fills[base_cols].rename(columns={'stop': 'consol_low', 'why': 'base_why', 'net_R': 'base_net_R_causal'})
    dup = fills.groupby(['day', 'symbol']).size()
    assert dup.max() == 1, 'causal_arming_causal.csv fill rows are not unique on (day, symbol)'

    feat = pd.read_csv(FEATURES_A, usecols=['day', 'symbol', 'fill_min', 'half_entry', 'store_served_1438'])

    def join(df, name):
        m = df.merge(fills, on=['day', 'symbol'], how='left', suffixes=('', '_causal'))
        n_na = int(m.consol_low.isna().sum())
        if n_na:
            log(f'join({name}): WARNING {n_na}/{len(m)} rows had no causal fill match -- dropped '
                f'(base fields unknown, cannot walk)')
            m = m.dropna(subset=['consol_low']).reset_index(drop=True)
        fm_mismatch = (m.fill_min - m.fill_min_causal).abs().gt(1e-6).sum()
        if fm_mismatch:
            log(f'join({name}): WARNING {fm_mismatch} rows have fill_min mismatch between prediction '
                f'and causal files (join key day+symbol ambiguous?) -- keeping causal fill_min as truth')
        m = m.merge(feat, on=['day', 'symbol'], how='left', suffixes=('', '_feat'))
        n_na2 = int(m.half_entry.isna().sum())
        if n_na2:
            log(f'join({name}): WARNING {n_na2} rows had no features_1478_A half_entry match -- '
                f'dropped (cost unknown)')
            m = m.dropna(subset=['half_entry']).reset_index(drop=True)
        # real-SIP flag per the prose: store_served_1438 == 0 (cross-check the two sources agree)
        mismatch_ssv = (m.store_served_1438 != m.store_served_1438_feat).sum()
        if mismatch_ssv:
            log(f'join({name}): WARNING {mismatch_ssv} rows disagree on store_served_1438 between '
                f'predictions and features_1478_A -- using features_1478_A (the prose\'s source)')
        m['real_sip'] = m.store_served_1438_feat == 0
        m['R_dollars'] = m.fill - m.consol_low
        bad_r = (m.R_dollars <= 0).sum()
        if bad_r:
            log(f'join({name}): WARNING {bad_r} rows have R_dollars <= 0 (fill <= consol_low) -- '
                f'dropped (undefined R)')
            m = m[m.R_dollars > 0].reset_index(drop=True)
        return m

    kept = join(kept, 'kept')
    dropped = join(dropped, 'dropped')
    return kept, dropped


# --------------------------------------------------------------------------------------------
# Bars
# --------------------------------------------------------------------------------------------

def load_bars(symdays):
    con = sqlite3.connect(BARS_DB)
    con.execute('CREATE TEMP TABLE need (symbol TEXT, day TEXT)')
    con.executemany('INSERT INTO need VALUES (?,?)', list(symdays))
    q = ('SELECT b.symbol, b.day, b.t, b.o, b.h, b.l, b.c FROM bars b '
         'JOIN need n ON b.symbol = n.symbol AND b.day = n.day')
    df = pd.read_sql_query(q, con)
    con.close()
    df['m'] = df['t'].map(et_minute)
    df = df[(df.m >= OPEN_M) & (df.m <= EOD_M)].copy()          # RTH only, and not past the 15:55 force-flat
    df = df.sort_values(['symbol', 'day', 'm'], kind='stable')
    groups = {k: g[['m', 'o', 'h', 'l', 'c']].to_numpy(float)
              for k, g in df.groupby(['symbol', 'day'], sort=False)}
    log(f'load_bars: {len(df)} RTH bar rows for {len(symdays)} (symbol,day) pairs, {len(groups)} with bars')
    return groups


# --------------------------------------------------------------------------------------------
# Part A: anatomy walk (drawdown to touch / to 15:55, touch minutes, breach, stopped-before-touch)
# --------------------------------------------------------------------------------------------

def anatomy_walk(bars_arr, fill_min, level, consol_low):
    """From the fill bar (m = floor(fill_min)) forward through 15:55: track running min low (drawdown
    below `level`) and the first bar whose high >= level*1.05 (the touch). Returns a dict with
    min_low, touch_m (or nan), breached_consol (bool over the pre-touch/whole window)."""
    if bars_arr is None or len(bars_arr) == 0:
        return None
    fb = np.floor(fill_min)
    sub = bars_arr[bars_arr[:, 0] >= fb]
    if len(sub) == 0:
        return None
    target = level * TOUCH_MULT
    touch_m = np.nan
    min_low = sub[0, 3]
    breached_before_touch = False
    for i in range(len(sub)):
        m, o, h, l, c = sub[i]
        min_low = min(min_low, l)
        if l <= consol_low:
            breached_before_touch = True
        if h >= target:
            touch_m = m
            break
    return {'fill_bar_m': fb, 'min_low': min_low, 'touch_m': touch_m,
            'breached_before_touch': breached_before_touch, 'last_m': sub[-1, 0]}


def build_part_a(kept, bars):
    """Per (split x real-SIP-scope) anatomy table for extenders (L3==1) and non-extenders (L3==0)."""
    rows = []
    walk_cache = {}
    for r in kept.itertuples():
        key = (r.symbol, r.day)
        w = anatomy_walk(bars.get(key), r.fill_min, r.level, r.consol_low)
        walk_cache[(r.day, r.symbol)] = w
    recs = []
    for r in kept.itertuples():
        w = walk_cache[(r.day, r.symbol)]
        if w is None:
            continue
        is_ext = bool(r.L3 == 1)
        dd_pct = (r.level - w['min_low']) / r.level * 100.0
        touched = not np.isnan(w['touch_m'])
        minutes_to_touch = (w['touch_m'] - w['fill_bar_m']) if touched else np.nan
        recs.append(dict(day=r.day, symbol=r.symbol, split=r.split, real_sip=r.real_sip,
                          is_extender=is_ext, dd_pct=dd_pct, touched=touched,
                          minutes_to_touch=minutes_to_touch, breached_consol=w['breached_before_touch'],
                          base_why=r.base_why))
    return pd.DataFrame(recs)


def quantiles(s, qs=(0.10, 0.25, 0.50, 0.75, 0.90)):
    s = pd.Series(s).dropna()
    if not len(s):
        return {q: np.nan for q in qs}
    return {q: float(s.quantile(q)) for q in qs}


def summarize_anatomy(df):
    """Build the Part A table: rows = (split, scope, group); columns = the anatomy stats."""
    out = []
    for split in ['TRAIN', 'VAL']:
        for scope, mask_fn in [('all', lambda d: d), ('real_sip', lambda d: d[d.real_sip])]:
            d = mask_fn(df[df.split == split])
            for is_ext, label in [(True, 'extender'), (False, 'non_extender')]:
                g = d[d.is_extender == is_ext]
                if not len(g):
                    continue
                dd_q = quantiles(g.dd_pct)
                mt_q = quantiles(g.minutes_to_touch) if is_ext else {q: np.nan for q in (.1, .25, .5, .75, .9)}
                exit_mix = g.base_why.value_counts(normalize=True).round(4).to_dict()
                row = dict(split=split, scope=scope, group=label, n=len(g),
                           dd_p10=dd_q[0.10], dd_p25=dd_q[0.25], dd_p50=dd_q[0.50],
                           dd_p75=dd_q[0.75], dd_p90=dd_q[0.90],
                           mt_p10=mt_q[0.10], mt_p25=mt_q[0.25], mt_p50=mt_q[0.50],
                           mt_p75=mt_q[0.75], mt_p90=mt_q[0.90],
                           share_breached_consol=float(g.breached_consol.mean()) if is_ext else float(g.breached_consol.mean()),
                           share_stopped_before_touch=(float((g.base_why == 'stop').mean()) if is_ext else np.nan),
                           exit_mix=exit_mix)
                out.append(row)
    return pd.DataFrame(out)


# --------------------------------------------------------------------------------------------
# Part B: the two cells
# --------------------------------------------------------------------------------------------

def compute_dsw(anatomy_df):
    """d, s, W from TRAIN-H2 (split=='TRAIN'), ALL rows (not real-SIP-only), extenders only."""
    ext = anatomy_df[(anatomy_df.split == 'TRAIN') & (anatomy_df.scope == 'all') & (anatomy_df.is_extender)] \
        if 'scope' in anatomy_df.columns else None
    return ext


def sweep_fill(bars_arr, fill_min, limit_px, w_minutes):
    """1,548: BUY LIMIT at `limit_px` resting from the fill bar for `w_minutes`. Fills at limit_px on
    the first bar (>= fill bar, inclusive) whose low is strictly below the limit."""
    if bars_arr is None or len(bars_arr) == 0:
        return None
    fb = np.floor(fill_min)
    sub = bars_arr[(bars_arr[:, 0] >= fb) & (bars_arr[:, 0] <= fb + w_minutes)]
    for i in range(len(sub)):
        m, o, h, l, c = sub[i]
        if l < limit_px:
            return float(m), float(limit_px)
    return None


def walk_from(bars_arr, entry_m, stop_px, target_px):
    """Post-entry path physics, entry bar inclusive: gap-through at the open, stop first on a bar
    touching both, 15:55 (EOD_M) forces flat at that bar's open."""
    sub = bars_arr[bars_arr[:, 0] >= entry_m]
    for i in range(len(sub)):
        m, o, h, l, c = sub[i]
        if m >= EOD_M:
            return float(m), float(o), 'eod'
        if l <= stop_px:
            px = o if o <= stop_px else stop_px
            return float(m), float(px), 'stop'
        if h >= target_px:
            return float(m), float(target_px), 'target'
    last = sub[-1]
    return float(last[0]), float(last[4]), 'eod_fallback'


def cost_and_net(entry_px, exit_px, exit_why, R_dollars, split, entry_cost_dollars):
    exit_bps = SLIP_STOP_BPS[split] if exit_why in ('stop', 'stop_bar') else \
        (EOD_BID_BPS[split] if exit_why in ('eod', 'eod_fallback') else 0.0)
    exit_cost = exit_px * exit_bps / 1e4
    raw_R = (exit_px - entry_px) / R_dollars
    cost_R = (entry_cost_dollars + exit_cost) / R_dollars
    net_R = raw_R - cost_R
    net_pct = (exit_px - entry_px - entry_cost_dollars - exit_cost) / entry_px * 100.0
    return raw_R, cost_R, net_R, net_pct


def run_cells(kept, bars, d, s, W):
    """Returns a fills-level DataFrame with both cells stacked. d, s are in % of level (as returned by
    the anatomy quantiles) -- converted to fractions here for the price arithmetic."""
    stop_off = 1 - (s / 100.0)
    target_mult = TOUCH_MULT
    limit_off = 1 - (d / 100.0)
    recs = []
    for r in kept.itertuples():
        arr = bars.get((r.symbol, r.day))
        stop_px = r.level * stop_off
        target_px = r.level * target_mult
        # 1,548 SWEEP
        limit_px = r.level * limit_off
        f = sweep_fill(arr, r.fill_min, limit_px, W)
        if f is None:
            recs.append(dict(day=r.day, symbol=r.symbol, split=r.split, cell='1548', filled=False,
                              entry=np.nan, exit_m=np.nan, exit_px=np.nan, why=np.nan,
                              net_R=np.nan, net_pct=np.nan, real_sip=r.real_sip, is_extender=r.L3 == 1))
        else:
            fill_m, entry_px = f
            R_d = entry_px - stop_px
            if arr is None or R_d <= 0:
                recs.append(dict(day=r.day, symbol=r.symbol, split=r.split, cell='1548', filled=False,
                                  entry=np.nan, exit_m=np.nan, exit_px=np.nan, why='bad_R',
                                  net_R=np.nan, net_pct=np.nan, real_sip=r.real_sip, is_extender=r.L3 == 1))
            else:
                exit_m, exit_px, why = walk_from(arr, fill_m, stop_px, target_px)
                _, _, net_R, net_pct = cost_and_net(entry_px, exit_px, why, R_d, r.split, 0.0)
                recs.append(dict(day=r.day, symbol=r.symbol, split=r.split, cell='1548', filled=True,
                                  entry=entry_px, exit_m=exit_m, exit_px=exit_px, why=why,
                                  net_R=net_R, net_pct=net_pct, real_sip=r.real_sip, is_extender=r.L3 == 1))
        # 1,549 WIDE (base entry, always "filled" -- it already happened)
        entry_px = r.fill
        R_d = entry_px - stop_px
        if arr is None or R_d <= 0:
            recs.append(dict(day=r.day, symbol=r.symbol, split=r.split, cell='1549', filled=False,
                              entry=entry_px, exit_m=np.nan, exit_px=np.nan, why='bad_R',
                              net_R=np.nan, net_pct=np.nan, real_sip=r.real_sip, is_extender=r.L3 == 1))
        else:
            fb = np.floor(r.fill_min)
            exit_m, exit_px, why = walk_from(arr, fb, stop_px, target_px)
            _, _, net_R, net_pct = cost_and_net(entry_px, exit_px, why, R_d, r.split, r.half_entry)
            recs.append(dict(day=r.day, symbol=r.symbol, split=r.split, cell='1549', filled=True,
                              entry=entry_px, exit_m=exit_m, exit_px=exit_px, why=why,
                              net_R=net_R, net_pct=net_pct, real_sip=r.real_sip, is_extender=r.L3 == 1))
    return pd.DataFrame(recs)


def count_matched_null(pool_outcome_R, pool_day, n_filled, seed, n_draws=N_DRAWS):
    rng = np.random.default_rng(seed)
    pool = np.asarray(pool_outcome_R)
    if len(pool) == 0 or n_filled == 0:
        return np.array([])
    idx = np.arange(len(pool))
    means = np.empty(n_draws)
    for i in range(n_draws):
        draw = rng.choice(idx, size=n_filled, replace=True)
        means[i] = pool[draw].mean()
    return means


def cell_stats(fills_df, cell, split, kept_full, dropped_fills_df=None):
    d = fills_df[(fills_df.cell == cell) & (fills_df.split == split)]
    total_pop = (kept_full.split == split).sum()
    filled = d[d.filled]
    n_fill = len(filled)
    fill_share = n_fill / max(len(d), 1)
    stats = dict(cell=cell, split=split, n_pop=len(d), n_fill=n_fill, fill_share=fill_share)
    if n_fill == 0:
        stats.update(mean_R=np.nan, t=np.nan, ex_top5=np.nan, winner_capped=np.nan,
                     fills_per_week=np.nan, real_sip_mean=np.nan, real_sip_t=np.nan,
                     mean_pct=np.nan)
        return stats, filled
    stats['mean_R'] = float(filled.net_R.mean())
    stats['mean_pct'] = float(filled.net_pct.mean())
    stats['t'] = day_clustered_t(filled.net_R, filled.day)
    stats['ex_top5'] = ex_top5_mean(filled.net_R)
    stats['winner_capped'] = winner_capped_mean(filled.net_R)
    stats['fills_per_week'] = n_fill / weeks_spanned(filled.day)
    rs = filled[filled.real_sip]
    stats['real_sip_mean'] = float(rs.net_R.mean()) if len(rs) else np.nan
    stats['real_sip_t'] = day_clustered_t(rs.net_R, rs.day) if len(rs) > 1 else np.nan
    stats['real_sip_n'] = len(rs)
    stats['exit_mix'] = filled.why.value_counts(normalize=True).round(4).to_dict()
    return stats, filled


# --------------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--smoke', type=int, default=0, help='limit to N kept rows for a smoke run')
    args = ap.parse_args()

    kept, dropped = load_population()
    if args.smoke:
        kept = kept.sample(n=min(args.smoke, len(kept)), random_state=SEED).reset_index(drop=True)
        log(f'SMOKE: subsampled kept to {len(kept)} rows')

    symdays = set(zip(kept.symbol, kept.day)) | set(zip(dropped.symbol, dropped.day))
    bars = load_bars(symdays)

    log('Part A: anatomy walk (kept population)')
    anatomy_raw = build_part_a(kept, bars)
    anatomy_table = summarize_anatomy(anatomy_raw)

    train_ext = anatomy_raw[(anatomy_raw.split == 'TRAIN') & (anatomy_raw.is_extender)]
    d = max(float(train_ext.dd_pct.quantile(0.50)), D_FLOOR)
    s = float(train_ext.dd_pct.quantile(0.75))
    W = min(float(train_ext.minutes_to_touch.quantile(0.75)), W_CAP)
    log(f'Part B params from TRAIN-H2 kept extenders (n={len(train_ext)}): d={d:.4f}% s={s:.4f}% W={W:.1f}min')

    log('Part B: running the two cells on the kept population')
    fills_kept = run_cells(kept, bars, d, s, W)

    log('Part B calibration: running the two cells on the DROPPED (non-kept) fills')
    symdays_dropped = set(zip(dropped.symbol, dropped.day))
    bars_dropped = {k: v for k, v in bars.items() if k in symdays_dropped} if not args.smoke else \
        load_bars(symdays_dropped)
    fills_dropped = run_cells(dropped, bars_dropped, d, s, W)

    results = []
    for cell in ['1548', '1549']:
        for split in ['TRAIN', 'VAL']:
            st, filled = cell_stats(fills_kept, cell, split, kept)
            st['population'] = 'kept'
            results.append(st)
            if len(filled):
                pool_days = set(filled.day)
                pool = kept[(kept.split == split) & (kept.day.isin(pool_days))].outcome_R.dropna().to_numpy()
                null_means = count_matched_null(pool, None, len(filled), SEED + hash(cell) % 1000)
                pct = float((null_means <= st['mean_R']).mean() * 100) if len(null_means) else np.nan
                st['null_percentile'] = pct
            else:
                st['null_percentile'] = np.nan
            std, _ = cell_stats(fills_dropped, cell, split, dropped)
            std['population'] = 'dropped'
            results.append(std)

    results_df = pd.DataFrame(results)

    fills_kept['population'] = 'kept'
    fills_dropped['population'] = 'dropped'
    all_fills = pd.concat([fills_kept, fills_dropped], ignore_index=True)
    out_cols = ['day', 'symbol', 'split', 'cell', 'filled', 'entry', 'exit_m', 'exit_px', 'why', 'net_R', 'net_pct']
    all_fills[out_cols].to_csv(f'{HERE}/rebuild_1548_fills.csv', index=False)
    log(f'wrote rebuild_1548_fills.csv ({len(all_fills)} rows)')

    return dict(d=d, s=s, W=W, anatomy=anatomy_table, results=results_df, n_kept=len(kept), n_dropped=len(dropped))


if __name__ == '__main__':
    out = main()
    print('\n=== d,s,W ===')
    print(out['d'], out['s'], out['W'])
    print('\n=== Part A (anatomy) ===')
    with pd.option_context('display.max_columns', None, 'display.width', 200):
        print(out['anatomy'][['split', 'scope', 'group', 'n', 'dd_p50', 'dd_p75', 'mt_p50', 'mt_p75',
                               'share_breached_consol', 'share_stopped_before_touch']])
    print('\n=== Part B (cells) ===')
    with pd.option_context('display.max_columns', None, 'display.width', 200):
        print(out['results'][['cell', 'split', 'population', 'n_pop', 'n_fill', 'fill_share', 'mean_R',
                               'mean_pct', 't', 'ex_top5', 'winner_capped', 'fills_per_week',
                               'real_sip_mean', 'real_sip_t', 'real_sip_n', 'null_percentile']])
