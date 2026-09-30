#!/usr/bin/env python3
"""Cell 1,679: the ORB give-back -- exit rules on REAL ledgers, R-floored.

PREREG: research/orb_exit/PREREG_1679.md (FROZEN 2026-09-30 16:10 UTC).

Why: cell 1,678 Part 3 applied HOD-trained post-entry exit models to ORB's
book and every cell "passed" -- flagged NOT a claim, because the book CSV
has no real entry/exit times (both were reconstructed by touch-search) and
ORB's own R can be tiny (R-multiples explode). The one robust number from
that read: 21% of ORB fills reach +1R and then close <=0 (give-back). This
cell re-reads exit rules on ledgers with REAL times: L1 (live trades.db,
exact entry/exit timestamps and shares) and L2 (the BT book, with the entry
MINUTE reconstructed by the BT's OWN rule -- find_breakout_bar_ts -- and the
reconstruction validated against L1's overlap).

Reuses (imported via importlib, unchanged, per project convention -- module
names start with a digit so cannot be `import`ed normally):
  research/hod_entry/1678_remaining_r.py (f1678) -- build_shape_at_k,
    featset_cols, scan_rule (rule X/X+ firing logic), stats_block, decompose
    (give-back-saved / continuation-forgone), DECISION_KS, K_GRID, MODELS_DIR,
    ORB_LIVE_RISK ($375), and (via f1678.f1668/.f1670) the DST-aware ET-minute
    <-> UTC conversion, BarStore, find_fill_index, walk_k, dR_cut, path_features.
  research/hod_entry/1669_fast_failure.py is NOT imported separately -- its
    DST-aware conversion and day-clustered stats are the SAME primitives
    1678 already re-exports (f1668.minute_of_day/et_offset_minutes,
    f1676.day_clustered_t via f1678.stats_block) -- importing it again would
    just re-run the same module under a second name.

The BT entry rule for the L2 reconstruction (grepped, matches live per
trading/orb_touchgo_filter.find_breakout_bar_ts's own docstring: "Sharing
this single function keeps BT and live identical by construction"):
  opening range = the 5 RTH minutes 09:30-09:34 ET (minarr 570-574);
  range closes at 09:35 ET (minarr 575); entry bar = first bar in
  [09:35, 10:35) ET whose HIGH > range_high (strict); entry price = the
  book's own entry_price (not re-derived); stop = the book's range_low.
Live/BT exit precedence (study_orb_pipeline_static_lock.simulate_static_lock):
  per post-entry bar, arm the lock when THIS bar's high >= entry+1.75R, then
  (same bar) exit if THIS bar's low <= the (possibly just-armed) stop; EOD
  truncates at 15:45 ET (minarr 945); every exit carries -10bps slippage.
  Mechanical rules (a)-(c)/(e) below mirror this exact precedence with their
  own trigger/stop levels; rule (f) and read 2's X/X+ instead OVERLAY a
  single fixed/model checkpoint on the ACTUAL exit (fire-or-defer), matching
  1678's own scan_rule design.

R floor (memory "R must exceed the spread"): fills with R_unit < 0.5% of
entry price are EXCLUDED from R-unit (multiple) reads and reported
separately; dollar reads use the full population (a $ amount does not
explode the way entry/R_unit does when R_unit -> 0).

Usage:
    python3 research/orb_exit/1679_orb_exit.py
"""
import importlib.util
import json
import logging
import os
import sqlite3
import sys

import joblib
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)

LOG_FILE = os.path.join(HERE, '1679_orb_exit.log')
READS_CSV = os.path.join(HERE, '1679_reads.csv')
PERFILL_CSV = os.path.join(HERE, '1679_per_fill.csv')
RESULT_MD = os.path.join(HERE, 'RESULT_1679.md')

TRADES_DB = os.path.join(ROOT, 'data', 'trades.db')
ORB_BOOK_CSV = os.path.join(ROOT, 'analysis_results', 'orb_bplus_book.csv')

logger = logging.getLogger('1679')


def setup_logging():
    logger.setLevel(logging.INFO)
    if logger.handlers:
        return
    fh = logging.FileHandler(LOG_FILE)
    fh.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(message)s'))
    logger.addHandler(fh)
    sh = logging.StreamHandler()
    sh.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(message)s'))
    logger.addHandler(sh)


def _load_module(name, fname, root=ROOT):
    spec = importlib.util.spec_from_file_location(name, os.path.join(root, fname))
    mod = importlib.util.module_from_spec(spec)
    old_argv = sys.argv
    sys.argv = [sys.argv[0]]
    spec.loader.exec_module(mod)
    sys.argv = old_argv
    return mod


# ---------------------------------------------------------------------------
# Constants -- ORB's own, NOT the HOD population's (f1668.EOD_M=955/15:55 is
# the HOD book's EOD; ORB's live force-close is 15:45 per orb.yaml, read-only).
# ---------------------------------------------------------------------------
RANGE_LO_M, RANGE_HI_M, RANGE_END_M, SEARCH_END_M = 570.0, 574.0, 575.0, 635.0
ORB_EOD_M = 945.0          # 15:45 ET -- orb.yaml exit.force_close_time_et
LOCK_TRIGGER_R_LIVE = 1.75  # orb.yaml exit.lock_arm_at_r
LOCK_STOP_R_LIVE = 0.5      # orb.yaml exit.lock_stop_r
EXIT_SLIP_BPS = 10.0        # study_orb_pipeline_static_lock.EXIT_SLIP_BPS
R_FLOOR_PCT = 0.005         # "R must exceed the spread" memory
L2_RISK = 375.0             # PREREG: L2 dollar reads at $375 risk
C020 = 0.20                 # PREREG: rule X(c=0.20) only -- no c sweep (no tuning)
L1_START, L1_END = '2026-05-19', '2026-09-23'


# ---------------------------------------------------------------------------
# Ledger loaders
# ---------------------------------------------------------------------------

def find_range_and_breakout(bars):
    """Opening range (09:30-09:34 ET) + the market breakout bar (first bar in
    [09:35,10:35) ET whose high > range_high), matching find_breakout_bar_ts
    / study_orb_pipeline_static_lock.py exactly."""
    m = bars['minarr']
    rmask = (m >= RANGE_LO_M) & (m <= RANGE_HI_M)
    if rmask.sum() == 0:
        return None
    range_high = float(bars['h'][rmask].max())
    range_low = float(bars['l'][rmask].min())
    smask = np.where((m >= RANGE_END_M) & (m < SEARCH_END_M))[0]
    for j in smask:
        if bars['h'][j] > range_high:
            return dict(i0=int(j), range_high=range_high, range_low=range_low)
    return None


L1_CATEGORY = {
    'force_close': 'force_close',
    'stop_loss': 'stop', 'stop_loss_market_fallback': 'stop', 'lock_stop': 'stop',
    'tag_bb': 'target',
}
L2_CATEGORY = {
    'eod': 'force_close', 'scale_eod': 'force_close',
    'stop': 'stop', 'lock': 'stop', 'scale_lock': 'stop',
    'tag_bb': 'target', 'tag_b1': 'target',
}


def load_l1(store, f1668):
    """L1 LIVE: data/trades.db strategy='orb', account live (NULL), filled
    and exited, 2026-05-19..09-23. Exact entry/exit times and prices, actual
    shares -- no reconstruction needed except locating the bar indices."""
    con = sqlite3.connect(f'file:{TRADES_DB}?mode=ro', uri=True)
    q = """SELECT id, trade_date, symbol, fill_price, filled_at, exit_price,
                  exited_at, pnl, shares, stop_loss_price, exit_reason, exit_branch
           FROM trades WHERE strategy='orb' AND account IS NULL
             AND fill_price IS NOT NULL AND exit_price IS NOT NULL
             AND trade_date BETWEEN ? AND ?"""
    df = pd.read_sql(q, con, params=(L1_START, L1_END))
    con.close()
    logger.info('L1: %d candidate rows from trades.db', len(df))
    fills, n_no_bars, n_no_i0, n_bad_R, n_no_exit, n_other_reason = [], 0, 0, 0, 0, 0
    for r in df.itertuples():
        bars = store.day_bars(r.symbol, r.trade_date)
        if bars is None or len(bars['o']) < 5:
            n_no_bars += 1
            logger.warning('L1 %s %s: no/short bars in bars_sip.db -- EXCLUDED', r.symbol, r.trade_date)
            continue
        fill_min = f1668.minute_of_day(r.filled_at, r.trade_date)
        i0 = f1668.find_fill_index(bars, fill_min)
        if i0 is None or i0 + 1 >= len(bars['o']):
            n_no_i0 += 1
            continue
        entry, stop = float(r.fill_price), float(r.stop_loss_price)
        R_unit = entry - stop
        if not (R_unit > 0):
            n_bad_R += 1
            logger.warning('L1 %s %s: R_unit<=0 (entry=%.4f stop=%.4f) -- EXCLUDED', r.symbol, r.trade_date, entry, stop)
            continue
        exit_min = f1668.minute_of_day(r.exited_at, r.trade_date)
        exit_idx = f1668.find_fill_index(bars, exit_min)
        if exit_idx is None or exit_idx <= i0:
            n_no_exit += 1
            logger.warning('L1 %s %s: exit bar <= entry bar -- EXCLUDED', r.symbol, r.trade_date)
            continue
        cat = L1_CATEGORY.get(r.exit_reason)
        if cat is None:
            n_other_reason += 1
            logger.warning('L1 %s %s: unmapped exit_reason=%s exit_branch=%s -- categorized other', r.symbol, r.trade_date, r.exit_reason, r.exit_branch)
            cat = 'other'
        actual_R = (float(r.exit_price) - entry) / R_unit
        fills.append(dict(
            ledger='L1', fill_id=f'{r.symbol}_{r.trade_date}_{r.id}', date=r.trade_date, symbol=r.symbol,
            entry=entry, stop=stop, R_unit=R_unit, R_pct=R_unit / entry, i0=i0, exit_idx=exit_idx,
            actual_R=actual_R, actual_dollar=float(r.pnl), shares=float(r.shares), category=cat,
            exit_reason=r.exit_reason, half='whole'))
    n_total = len(df)
    n_ok = len(fills)
    logger.info('L1 coverage: %d/%d usable (%.1f%%) | no_bars=%d no_i0=%d bad_R=%d no_exit=%d other_reason=%d',
                n_ok, n_total, 100.0 * n_ok / max(n_total, 1), n_no_bars, n_no_i0, n_bad_R, n_no_exit, n_other_reason)
    return fills, dict(n_total=n_total, n_ok=n_ok, n_no_bars=n_no_bars, n_no_i0=n_no_i0, n_bad_R=n_bad_R, n_no_exit=n_no_exit)


def load_l2(store):
    """L2 BT: analysis_results/orb_bplus_book.csv, entered==1. Entry price
    and pnl_pct from the book; entry MINUTE, range_low (stop) and exit bar
    are RECONSTRUCTED from bars_sip.db by the BT's own rule (see module
    docstring). 2025 vs 2026 halves per PREREG."""
    orb_csv = _load_module('orb_csv_1679', 'trading/orb_csv.py')
    df = orb_csv.read_orb_csv(ORB_BOOK_CSV)
    ent = df[df['entered'] == 1].copy()
    ent['date'] = ent['date'].astype(str)
    logger.info('L2: %d entered rows from orb_bplus_book.csv', len(ent))
    fills = []
    n_no_bars = n_no_range = n_bad_R = n_no_exit = n_other_reason = 0
    for r in ent.itertuples():
        bars = store.day_bars(r.symbol, r.date)
        if bars is None or len(bars['o']) < 5:
            n_no_bars += 1
            logger.warning('L2 %s %s: no/short bars in bars_sip.db -- EXCLUDED', r.symbol, r.date)
            continue
        rec = find_range_and_breakout(bars)
        if rec is None:
            n_no_range += 1
            logger.warning('L2 %s %s: no 5-min range or no breakout bar found -- EXCLUDED', r.symbol, r.date)
            continue
        i0 = rec['i0']
        entry, stop = float(r.entry_price), rec['range_low']
        R_unit = entry - stop
        if not (R_unit > 0):
            n_bad_R += 1
            logger.warning('L2 %s %s: R_unit<=0 -- EXCLUDED', r.symbol, r.date)
            continue
        exit_price = entry * (1 + float(r.pnl_pct) / 100.0)
        cat = L2_CATEGORY.get(r.exit_reason)
        if cat is None:
            n_other_reason += 1
            logger.warning('L2 %s %s: unmapped exit_reason=%s -- categorized other', r.symbol, r.date, r.exit_reason)
            cat = 'other'
        exit_idx = reconstruct_l2_exit_idx(bars, i0, exit_price, r.exit_reason)
        if exit_idx is None or exit_idx <= i0:
            n_no_exit += 1
            logger.warning('L2 %s %s: exit bar <= entry bar -- EXCLUDED', r.symbol, r.date)
            continue
        actual_R = (exit_price - entry) / R_unit
        half = '2025' if r.date < '2026-01-01' else '2026'
        fills.append(dict(
            ledger='L2', fill_id=f'{r.symbol}_{r.date}', date=r.date, symbol=r.symbol,
            entry=entry, stop=stop, R_unit=R_unit, R_pct=R_unit / entry, i0=i0, exit_idx=exit_idx,
            actual_R=actual_R, actual_dollar=actual_R * L2_RISK, shares=L2_RISK / R_unit, category=cat,
            exit_reason=r.exit_reason, half=half))
    n_total = len(ent)
    n_ok = len(fills)
    logger.info('L2 coverage: %d/%d usable (%.1f%%) | no_bars=%d no_range_or_breakout=%d bad_R=%d no_exit=%d other_reason=%d',
                n_ok, n_total, 100.0 * n_ok / max(n_total, 1), n_no_bars, n_no_range, n_bad_R, n_no_exit, n_other_reason)
    return fills, dict(n_total=n_total, n_ok=n_ok, n_no_bars=n_no_bars, n_no_range=n_no_range, n_bad_R=n_bad_R, n_no_exit=n_no_exit)


def reconstruct_l2_exit_idx(bars, i0, exit_price, exit_reason, eod_m=ORB_EOD_M):
    n = len(bars['o'])
    if i0 + 1 >= n:
        return None
    up = exit_reason in ('tag_bb', 'tag_b1')
    down = exit_reason in ('stop', 'lock', 'scale_lock')
    is_eod = exit_reason in ('eod', 'scale_eod')
    for j in range(i0 + 1, n):
        if bars['minarr'][j] >= eod_m:
            return j
        if not is_eod:
            if up and bars['h'][j] >= exit_price:
                return j
            if down and bars['l'][j] <= exit_price:
                return j
    return n - 1


def reconstruction_agreement(store, l1_fills, l2_fills):
    """Validate the L2 entry-minute reconstruction against L1's REAL
    filled_at on overlapping (symbol,date) pairs: apply the SAME breakout
    search to L1's own bars and diff vs L1's actual entry minute."""
    l2_pairs = {(f['symbol'], f['date']) for f in l2_fills}
    deltas = []
    for f in l1_fills:
        key = (f['symbol'], f['date'])
        if key not in l2_pairs:
            continue
        bars = store.day_bars(f['symbol'], f['date'])
        rec = find_range_and_breakout(bars) if bars is not None else None
        if rec is None:
            continue
        recon_min = float(bars['minarr'][rec['i0']])
        actual_min = float(bars['minarr'][f['i0']])
        deltas.append(abs(recon_min - actual_min))
    if not deltas:
        logger.warning('reconstruction_agreement: n_overlap=0 -- no (symbol,date) shared by L1 and L2 entered rows')
        return dict(n_overlap=0, median_abs_delta_min=np.nan, mean_abs_delta_min=np.nan)
    d = np.array(deltas)
    out = dict(n_overlap=len(d), median_abs_delta_min=float(np.median(d)), mean_abs_delta_min=float(d.mean()),
               share_exact=float((d == 0).mean()), share_le1=float((d <= 1).mean()))
    logger.info('reconstruction agreement: n=%d median|delta|=%.2f min mean|delta|=%.2f min share_exact=%.2f share<=1min=%.2f',
                out['n_overlap'], out['median_abs_delta_min'], out['mean_abs_delta_min'], out['share_exact'], out['share_le1'])
    return out


# ---------------------------------------------------------------------------
# Read 1: give-back anatomy
# ---------------------------------------------------------------------------

def giveback_anatomy(fills, store):
    rows = []
    for f in fills:
        bars = store.day_bars(f['symbol'], f['date'])
        i0, exit_idx = f['i0'], f['exit_idx']
        seg_h = bars['h'][i0 + 1:exit_idx + 1]
        seg_m = bars['minarr'][i0 + 1:exit_idx + 1]
        if len(seg_h) == 0:
            mfe_R, peak_min = f['actual_R'], bars['minarr'][exit_idx]
        else:
            peak_pos = int(np.argmax(seg_h))
            mfe_R = (float(seg_h[peak_pos]) - f['entry']) / f['R_unit']
            mfe_R = max(mfe_R, f['actual_R'])
            peak_min = float(seg_m[peak_pos])
        rows.append(dict(fill_id=f['fill_id'], ledger=f['ledger'], half=f['half'], date=f['date'],
                          R_pct=f['R_pct'], actual_R=f['actual_R'], mfe_R=mfe_R,
                          minutes_peak_to_exit=float(bars['minarr'][exit_idx]) - peak_min,
                          category=f['category']))
    return pd.DataFrame(rows)


def anatomy_summary(anat_df, floor_ok_mask):
    """Give-back anatomy on the R-floored population: share ever reaching
    +0.5/+1/+1.5R that closes <=0, mean R given back (mirrors 1678's own
    giveback_share/r_given_back definition exactly, applied per threshold),
    minutes peak->exit, and the force_close/stop/target close-by mix (on the
    FULL population -- the close-by mix is not an R-multiple, no floor)."""
    sub = anat_df[floor_ok_mask]
    out = {}
    for thr in (0.5, 1.0, 1.5):
        ever = sub['mfe_R'] >= thr
        le0 = sub['actual_R'] <= 0.0
        out[f'giveback_share_{thr}R'] = float((ever & le0).mean()) if len(sub) else np.nan
        out[f'giveback_n_ever_{thr}R'] = int(ever.sum())
        out[f'r_given_back_{thr}R'] = float((sub.loc[ever, 'mfe_R'] - sub.loc[ever, 'actual_R']).mean()) if ever.any() else np.nan
        out[f'minutes_peak_to_exit_{thr}R'] = float(sub.loc[ever, 'minutes_peak_to_exit'].mean()) if ever.any() else np.nan
    out['n_floor_ok'] = int(len(sub))
    cat_counts = anat_df['category'].value_counts().to_dict()
    out['closed_by'] = cat_counts
    return out


# ---------------------------------------------------------------------------
# Read 2: HOD-trained exit models (1,678), unchanged
# ---------------------------------------------------------------------------

def build_perfillk(fills, store, f1668, f1670, f1678):
    rows = []
    for f in fills:
        bars = store.day_bars(f['symbol'], f['date'])
        i0, entry, stop, R_unit = f['i0'], f['entry'], f['stop'], f['R_unit']
        mean_vol = bars['v'][:i0 + 1].mean() if i0 >= 1 else np.nan
        for k in f1678.DECISION_KS:
            w = f1668.walk_k(bars, i0, k, stop, entry + 999 * R_unit)
            if w is None:
                continue
            open_k = (i0 + k) < f['exit_idx']
            pf = f1670.path_features(bars, i0, i0 + k, entry, R_unit, np.nan)
            shp = f1678.build_shape_at_k(bars, i0, k, entry, R_unit, np.nan, mean_vol)
            next_open = w['next_open']
            gross = ((next_open - entry) / R_unit) if next_open is not None else np.nan
            cost = (f1668.ENTRY_BPS * entry + f1668.CUT_BPS * next_open) / R_unit if next_open is not None else np.nan
            dR_full = (gross - cost - f['actual_R']) if next_open is not None else np.nan
            row = dict(fill_id=f['fill_id'], date=f['date'], symbol=f['symbol'], half=f['half'], k=k,
                       open_k=open_k, mtm_R=pf['mtm_R'], mfe_R=pf['mfe_R'], mae_R=pf['mae_R'],
                       min_since_new_high=pf['min_since_new_high'], bars_since_higher_low=pf['bars_since_higher_low'],
                       level_retouched=pf['level_retouched'], dR_full=dR_full,
                       cost_R=(f1668.CUT_BPS * next_open / R_unit) if next_open is not None else np.nan)
            row.update(shp)
            rows.append(row)
    return pd.DataFrame(rows)


def load_pred_store(f1678):
    pred_store = {}
    for fs in ('path', 'path_shape'):
        for k in f1678.K_GRID:
            for scoring_dir in ('TRAIN-H2->VAL', 'VAL->TRAIN-H2'):
                tag = scoring_dir.replace('->', '_to_').replace('-', '')
                fp = os.path.join(f1678.MODELS_DIR, f'1678_{fs}_remaining_R_k{k}_{tag}.joblib')
                if os.path.exists(fp):
                    pred_store[(fs, k, scoring_dir)] = joblib.load(fp)
    logger.info('Read2: loaded %d persisted 1,678 models', len(pred_store))
    return pred_store


def run_read2(perfillk, dates_by_fill, floor_ok_by_fill, pred_store, f1678, label):
    """rule X(c=0.20) and X+ from minute 15, both featsets, both scoring-
    direction models, UNCHANGED -- reused verbatim via f1678.scan_rule."""
    if perfillk.empty:
        return pd.DataFrame()
    rows = []
    for featset in ('path', 'path_shape'):
        cols = f1678.featset_cols(featset)
        for scoring_dir in ('TRAIN-H2->VAL', 'VAL->TRAIN-H2'):
            piv = {}
            for field in ('open_k', 'mtm_R', 'dR_full', 'cost_R'):
                piv[field] = perfillk.pivot_table(index='fill_id', columns='k', values=field, aggfunc='first')
            pred_wide = {}
            for k in f1678.DECISION_KS:
                model = pred_store.get((featset, k, scoring_dir))
                if model is None:
                    continue
                sub = perfillk[perfillk.k == k].set_index('fill_id')
                X = sub[cols].apply(pd.to_numeric, errors='coerce').values
                pred_wide[k] = pd.Series(model.predict(X), index=sub.index)
            piv['pred'] = pd.DataFrame(pred_wide)
            idx = piv['open_k'].index
            floor_ok = idx.map(lambda fid: floor_ok_by_fill.get(fid, False))
            dates_of = pd.Series([dates_by_fill[fid] for fid in idx], index=idx)
            for variant in ('X', 'X+'):
                ever_pool, fired, realized, checkpoints = f1678.scan_rule(piv, dates_of, C020, variant)
                fired_floor = fired & pd.Series(floor_ok, index=idx)
                n_pool, n_fired = int(ever_pool.sum()), int(fired_floor.sum())
                dR_fired = realized[fired_floor]
                dates_fired = dates_of.reindex(dR_fired.index)
                st = f1678.stats_block(dR_fired.values, dates_fired.values)
                cost0 = piv['cost_R'][checkpoints[0]].reindex(dR_fired.index).fillna(0)
                dec = f1678.decompose(dR_fired, cost0)
                dollar = st['mean_dR'] * L2_RISK if label == 'L2' and pd.notna(st['mean_dR']) else np.nan
                rows.append(dict(read='2', rule=f'{variant}_c020', featset=featset, scoring=scoring_dir,
                                  ledger=label, n_pool=n_pool, n_fired=n_fired,
                                  share_fired=(n_fired / n_pool if n_pool else np.nan),
                                  **st, **dec, dollar_effect_mean=dollar))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Read 3 / 4: pre-declared mechanical alternatives + give-back-saved /
# continuation-forgone decomposition (f1677.decompose, reused via f1678).
# ---------------------------------------------------------------------------

def _lock_walk(bars, i0, entry, stop_orig, R_unit, trigger_r, lock_stop_r, eod_m=ORB_EOD_M, slip_bps=EXIT_SLIP_BPS):
    """(a)/(b)/(e): arm on this bar's high >= entry+trigger_r*R, then (same
    bar) exit if this bar's low <= the (possibly just-armed) stop -- the
    EXACT precedence of study_orb_pipeline_static_lock.simulate_static_lock's
    'static lock loop'. EOD (>=eod_m) truncates first, same convention."""
    n = len(bars['o'])
    if i0 + 1 >= n:
        return None
    trigger_lvl = entry + trigger_r * R_unit
    lock_stop = entry + lock_stop_r * R_unit
    stop_price = stop_orig
    armed = False
    slip = slip_bps / 10000.0
    for j in range(i0 + 1, n):
        if bars['minarr'][j] >= eod_m:
            return float(bars['c'][j]) * (1 - slip)
        bh, bl = float(bars['h'][j]), float(bars['l'][j])
        if not armed and bh >= trigger_lvl:
            armed = True
            stop_price = max(stop_price, lock_stop)
        if bl <= stop_price:
            return stop_price * (1 - slip)
    return float(bars['c'][n - 1]) * (1 - slip)


def _trail_walk(bars, i0, entry, stop_orig, R_unit, trail_r=1.0, eod_m=ORB_EOD_M, slip_bps=EXIT_SLIP_BPS):
    """(c): stop trails at running-MFE - trail_r*R (never below the original
    stop), same same-bar update-then-check precedence as _lock_walk."""
    n = len(bars['o'])
    if i0 + 1 >= n:
        return None
    run_high = float(bars['h'][i0])
    stop_price = stop_orig
    slip = slip_bps / 10000.0
    for j in range(i0 + 1, n):
        if bars['minarr'][j] >= eod_m:
            return float(bars['c'][j]) * (1 - slip)
        bh, bl = float(bars['h'][j]), float(bars['l'][j])
        run_high = max(run_high, bh)
        stop_price = max(stop_price, run_high - trail_r * R_unit)
        if bl <= stop_price:
            return stop_price * (1 - slip)
    return float(bars['c'][n - 1]) * (1 - slip)


def mechanical_rules(f, bars, f1668, f1670):
    """Rules (a)-(f) from PREREG_1679 Reads section 3, each paired vs the
    ledger's own actual exit. (a)-(c)/(e) fully replace the post-entry exit
    mechanism (independent of touchgo Rule M/D -- disclosed in RESULT.md);
    (d) blends a +1R partial with an (e)-style runner; (f) overlays a single
    fixed checkpoint on the ACTUAL exit (fire-or-defer, like read 2's X/X+)."""
    i0, entry, stop, R_unit, actual_R = f['i0'], f['entry'], f['stop'], f['R_unit'], f['actual_R']
    slip = EXIT_SLIP_BPS / 10000.0
    out = {}

    px = _lock_walk(bars, i0, entry, stop, R_unit, 1.0, 0.5)
    out['lock_1R_0.5R'] = (px - entry) / R_unit if px is not None else np.nan

    px = _lock_walk(bars, i0, entry, stop, R_unit, 1.0, 0.0)
    out['lock_1R_BE'] = (px - entry) / R_unit if px is not None else np.nan

    px = _trail_walk(bars, i0, entry, stop, R_unit, 1.0)
    out['trail_MFE_1R'] = (px - entry) / R_unit if px is not None else np.nan

    px_e = _lock_walk(bars, i0, entry, stop, R_unit, LOCK_TRIGGER_R_LIVE, LOCK_STOP_R_LIVE)
    out['live_lock_ref'] = (px_e - entry) / R_unit if px_e is not None else np.nan

    # (d) 50% out at +1R (touch, with slip), rest rides the live-lock rule
    # (its OWN independent walk over the full path -- a separate share lot).
    n = len(bars['o'])
    touch1r_j = None
    for j in range(i0 + 1, n):
        if bars['minarr'][j] >= ORB_EOD_M:
            break
        if float(bars['h'][j]) >= entry + 1.0 * R_unit:
            touch1r_j = j
            break
    if px_e is None:
        out['scale50_1R_plus_live'] = np.nan
    elif touch1r_j is None:
        out['scale50_1R_plus_live'] = (px_e - entry) / R_unit  # never reached +1R -> identical to (e)
    else:
        leg1_px = (entry + 1.0 * R_unit) * (1 - slip)
        leg1_R = (leg1_px - entry) / R_unit
        leg2_R = (px_e - entry) / R_unit
        out['scale50_1R_plus_live'] = 0.5 * leg1_R + 0.5 * leg2_R

    # (f) time stop at 60 min if <+0.5R -- fire-or-defer overlay on the
    # ACTUAL exit, via f1668.dR_cut (reused, not reimplemented).
    k = 60
    if i0 + k < f['exit_idx'] and i0 + k < n:
        w = f1668.walk_k(bars, i0, k, stop, entry + 999 * R_unit)
        pf = f1670.path_features(bars, i0, i0 + k, entry, R_unit, np.nan)
        if w is not None and w['next_open'] is not None and pf['mtm_R'] < 0.5:
            out['time_stop_60m_fired'] = True
            out['time_stop_60m'] = f1668.dR_cut(entry, stop, actual_R, w['next_open'])
        else:
            out['time_stop_60m_fired'] = False
            out['time_stop_60m'] = np.nan
    else:
        out['time_stop_60m_fired'] = False
        out['time_stop_60m'] = np.nan
    return out


def run_read3(fills, store, f1668, f1670, f1678, label):
    per_fill_rows = []
    for f in fills:
        bars = store.day_bars(f['symbol'], f['date'])
        mech = mechanical_rules(f, bars, f1668, f1670)
        row = dict(f)
        for rule_name, new_R in mech.items():
            row[rule_name] = new_R
        per_fill_rows.append(row)
    pf = pd.DataFrame(per_fill_rows)

    FULL_RULES = ['lock_1R_0.5R', 'lock_1R_BE', 'trail_MFE_1R', 'scale50_1R_plus_live', 'live_lock_ref']
    reads = []
    for rule in FULL_RULES:
        sub = pf.dropna(subset=[rule])
        dR = sub[rule] - sub['actual_R']
        st = f1678.stats_block(dR.values, sub['date'].values)
        dec = f1678.decompose(pd.Series(dR.values), pd.Series(np.zeros(len(dR))))
        if label == 'L2':
            dollar = st['mean_dR'] * L2_RISK if pd.notna(st['mean_dR']) else np.nan
        else:
            dollar = float((dR.values * sub['R_unit'].values * sub['shares'].values).mean()) if len(sub) else np.nan
        reads.append(dict(read='3', rule=rule, ledger=label, n_pool=len(sub), n_fired=len(sub),
                           share_fired=1.0, **st, **dec, dollar_effect_mean=dollar))

    fired = pf[pf['time_stop_60m_fired']]
    dR = fired['time_stop_60m']
    st = f1678.stats_block(dR.values, fired['date'].values)
    dec = f1678.decompose(pd.Series(dR.values), pd.Series(np.zeros(len(dR))))
    dollar = (st['mean_dR'] * L2_RISK if label == 'L2' else
              float((dR.values * fired['R_unit'].values * fired['shares'].values).mean()) if len(fired) else np.nan) \
        if pd.notna(st['mean_dR']) else np.nan
    reads.append(dict(read='3', rule='time_stop_60m', ledger=label, n_pool=len(pf), n_fired=len(fired),
                       share_fired=(len(fired) / len(pf) if len(pf) else np.nan), **st, **dec, dollar_effect_mean=dollar))
    return pd.DataFrame(reads), pf


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    setup_logging()
    logger.info('=== cell 1,679: ORB give-back exit rules -- starting ===')
    f1668 = _load_module('f1668_1679', 'research/hod_entry/1668_failure.py')
    st = os.statvfs('/')
    free_gb = st.f_bavail * st.f_frsize / (1024 ** 3)
    logger.info('disk free on /: %.1f GB', free_gb)
    if free_gb < 5.0:
        logger.error('disk free %.1f GB < 5.0 GB floor -- aborting', free_gb)
        sys.exit(1)
    f1670 = _load_module('f1670_1679', 'research/hod_entry/1670_timing_map.py')
    f1678 = _load_module('f1678_1679', 'research/hod_entry/1678_remaining_r.py')

    store = f1668.BarStore(f1668.BARS_DB)
    l1_fills, l1_cov = load_l1(store, f1668)
    l2_fills, l2_cov = load_l2(store)

    agree = reconstruction_agreement(store, l1_fills, l2_fills)

    anat_l1 = giveback_anatomy(l1_fills, store)
    anat_l2 = giveback_anatomy(l2_fills, store)
    floor_l1 = anat_l1['R_pct'] >= R_FLOOR_PCT
    floor_l2 = anat_l2['R_pct'] >= R_FLOOR_PCT
    n_floor_fail_l1 = int((~floor_l1).sum())
    n_floor_fail_l2 = int((~floor_l2).sum())
    logger.info('R floor: L1 %d/%d fail (<0.5%% of entry), L2 %d/%d fail',
                n_floor_fail_l1, len(anat_l1), n_floor_fail_l2, len(anat_l2))

    anatomy = {
        'L1_whole': anatomy_summary(anat_l1, floor_l1),
        'L2_whole': anatomy_summary(anat_l2, floor_l2),
        'L2_2025': anatomy_summary(anat_l2[anat_l2.half == '2025'], floor_l2[anat_l2.half == '2025']),
        'L2_2026': anatomy_summary(anat_l2[anat_l2.half == '2026'], floor_l2[anat_l2.half == '2026']),
    }
    logger.info('anatomy: %s', json.dumps({k: {kk: vv for kk, vv in v.items() if kk != 'closed_by'} for k, v in anatomy.items()}, default=str))

    floor_ok_l1 = {f['fill_id']: (f['R_pct'] >= R_FLOOR_PCT) for f in l1_fills}
    floor_ok_l2 = {f['fill_id']: (f['R_pct'] >= R_FLOOR_PCT) for f in l2_fills}
    dates_l1 = {f['fill_id']: f['date'] for f in l1_fills}
    dates_l2 = {f['fill_id']: f['date'] for f in l2_fills}

    pred_store = load_pred_store(f1678)
    perfillk_l1 = build_perfillk(l1_fills, store, f1668, f1670, f1678)
    perfillk_l2 = build_perfillk(l2_fills, store, f1668, f1670, f1678)
    read2_l1 = run_read2(perfillk_l1, dates_l1, floor_ok_l1, pred_store, f1678, 'L1')
    read2_l2_whole = run_read2(perfillk_l2, dates_l2, floor_ok_l2, pred_store, f1678, 'L2')
    pk25 = perfillk_l2[perfillk_l2.half == '2025']
    pk26 = perfillk_l2[perfillk_l2.half == '2026']
    d25 = {k: v for k, v in dates_l2.items() if v < '2026-01-01'}
    d26 = {k: v for k, v in dates_l2.items() if v >= '2026-01-01'}
    read2_l2_25 = run_read2(pk25, d25, floor_ok_l2, pred_store, f1678, 'L2')
    read2_l2_26 = run_read2(pk26, d26, floor_ok_l2, pred_store, f1678, 'L2')
    for d, tag in ((read2_l2_25, '2025'), (read2_l2_26, '2026'), (read2_l2_whole, 'whole')):
        if len(d):
            d['half'] = tag
    read2_l1['half'] = 'whole'
    read2 = pd.concat([read2_l1, read2_l2_25, read2_l2_26, read2_l2_whole], ignore_index=True, sort=False)
    logger.info('Read2 done: %d rows', len(read2))

    l1_floor_fills = [f for f in l1_fills if floor_ok_l1[f['fill_id']]]
    l2_floor_fills = [f for f in l2_fills if floor_ok_l2[f['fill_id']]]
    read3_l1, pf3_l1 = run_read3(l1_floor_fills, store, f1668, f1670, f1678, 'L1')
    read3_l2_all, pf3_l2 = run_read3(l2_floor_fills, store, f1668, f1670, f1678, 'L2')
    pf3_l2 = pf3_l2.merge(pd.DataFrame([{'fill_id': f['fill_id'], 'half': f['half']} for f in l2_floor_fills]),
                           on='fill_id', how='left', suffixes=('', '_h'))
    read3_l2_25, _ = run_read3([f for f in l2_floor_fills if f['half'] == '2025'], store, f1668, f1670, f1678, 'L2')
    read3_l2_26, _ = run_read3([f for f in l2_floor_fills if f['half'] == '2026'], store, f1668, f1670, f1678, 'L2')
    read3_l1['half'] = 'whole'
    read3_l2_all['half'] = 'whole'
    read3_l2_25['half'] = '2025'
    read3_l2_26['half'] = '2026'
    read3 = pd.concat([read3_l1, read3_l2_25, read3_l2_26, read3_l2_all], ignore_index=True, sort=False)
    logger.info('Read3 done: %d rows', len(read3))

    reads_all = pd.concat([read2, read3], ignore_index=True, sort=False)
    reads_all.to_csv(READS_CSV, index=False)

    pf_l1_out = pd.DataFrame(l1_fills)[['ledger', 'fill_id', 'date', 'symbol', 'entry', 'stop', 'R_unit', 'R_pct',
                                         'actual_R', 'actual_dollar', 'shares', 'category', 'exit_reason', 'half']]
    pf_l2_out = pd.DataFrame(l2_fills)[['ledger', 'fill_id', 'date', 'symbol', 'entry', 'stop', 'R_unit', 'R_pct',
                                         'actual_R', 'actual_dollar', 'shares', 'category', 'exit_reason', 'half']]
    pf_rules = pd.concat([pf3_l1, pf3_l2], ignore_index=True, sort=False)
    rule_cols = ['fill_id', 'lock_1R_0.5R', 'lock_1R_BE', 'trail_MFE_1R', 'scale50_1R_plus_live', 'live_lock_ref',
                 'time_stop_60m_fired', 'time_stop_60m']
    per_fill = pd.concat([pf_l1_out, pf_l2_out], ignore_index=True, sort=False).merge(
        pf_rules[rule_cols], on='fill_id', how='left')
    per_fill.to_csv(PERFILL_CSV, index=False)

    store.close()
    logger.info('=== DONE: reads=%s per_fill=%s ===', READS_CSV, PERFILL_CSV)

    summary = dict(l1_cov=l1_cov, l2_cov=l2_cov, agreement=agree, anatomy=anatomy,
                   n_floor_fail_l1=n_floor_fail_l1, n_floor_fail_l2=n_floor_fail_l2,
                   n_l1_total=len(anat_l1), n_l2_total=len(anat_l2))
    with open(os.path.join(HERE, '1679_summary.json'), 'w') as fh:
        json.dump(summary, fh, indent=2, default=str)
    return summary


if __name__ == '__main__':
    main()
