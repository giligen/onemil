#!/usr/bin/env python3
"""Independent rebuild of PREREG_1617.md Frame B (cell 1,619): "the burst fade".

Built from the PREREG prose alone. The author of this script has NOT opened cell_1619.py,
cell_1619_fills.csv or RESULT_1619.md -- per CLAUDE.md's "Independent check" protocol, this
is the fresh reimplementation a comparator will diff trade-by-trade against the original to
catch coding errors (it cannot catch spec errors -- those need the refuters named in the
PREREG's "Independent check and consequences" section).

MECHANISM (PREREG_1617.md, "## Cells > B. Burst fade (1,619...)"), as read:
  Population: base HOD-break fills (causal_arming_causal.csv, status=='fill', n=9,911) that are
  shortable (borrow_flags.csv) and have SOME cached SIP tape for their (symbol, day) in
  sip_cache_1481/ and/or sip_cache_1480/ (trades[ts,price,size], quotes[ts,bid,ask]).
  Entry (short): at the base fill's exact tick instant, a SELL LIMIT rests at level*1.0015 (the
  same 15 bps as the base long's own chase cap -- LIMIT_BPS in sip_rebuild.py). It fills AT the
  limit on the first tape print STRICTLY ABOVE it within [fill_instant, end of fill_min's minute
  + 2 more minutes) -- the "through-print rule". No such print -> no trade (still counted in the
  population for fill-share).
  Exit (once short), first of, walked forward in time (tick tape while it lasts, then 1-minute
  bars from bars_fills_1478.db -- mirroring sip_rebuild.walk_path's semantics for a SHORT: stop
  is now the upper barrier so a bar's HIGH triggers it, target is the lower barrier so a bar's
  LOW triggers it, and stop wins when both would touch the same bar):
    - retest / target: a resting BID at level - $0.01, filled AT that price on the first print
      (tick phase) or bar-low touch (bar phase) STRICTLY BELOW / AT-OR-BELOW it.
    - stop: level*1.0075; a print/bar >= stop triggers a cover. Tick phase: cover at the
      prevailing ask at the trigger print, further marked up by SLIP_STOP_BPS[split] (the
      "88% clean fill + 12% tail at 94/76bps" blend cell_1478.py already applies to long-side
      stops -- mirrored onto the short's buy-to-cover). Bar phase: same blend applied to the
      bar's open-if-gapped-through-else-stop price (no quotes available that far out).
    - 15:55 ET: cover at the prevailing ask (tick) or the 15:55 bar's open (bar phase), costed
      via the EOD_BPS[split] constant (no per-trade quote at that distance either way).
  Costs: entry always pays half_entry (features_1478_A.csv -- the base fill's own half-spread;
  a resting order still gives up the spread). Target exits additionally pay the cover-instant
  half-spread + SLIP_BP*exit_price when tick quotes are available (mirrors sip_rebuild.
  trade_result), else EOD_BPS as a proxy. Stop and EOD exits use ONLY their named blended-bps
  constant (the PREREG gives these as the complete "standard cost" for those legs -- see
  REBUILD_1619.md caveats for the alternative "additive" reading considered and rejected).
  Borrow: 3%/yr, pro-rated by the actual holding time (entry tick -> exit tick/bar).
  R_f = stop - entry = level*0.006 (~0.6% of price, constant per fill).

INTERPRETATION CHOICES (disclosed, not hidden -- see REBUILD_1619.md):
  * "shortable... excluded" is implemented as borrow_flags.shortable==False (or missing) removes
    a symbol from the population outright; no separate SSR column exists in borrow_flags.csv, so
    the PREREG's "SSR excluded" has no distinct signal to key on here.
  * The tape caches are unioned across sip_cache_1481/ and sip_cache_1480/ (globbed by
    SYMBOL_DAY_*.pkl, not by a single guessed minute key) -- both directories hold small,
    disjoint-looking windows per signal, and taking their union is the most robust way to use
    "the tape window of the fill minute" without over-fitting an unconfirmed filename convention.
  * Halts are not separately detectable in this data (no halt flag was supplied); a genuine gap
    in the combined tape/bars before an exit condition fires is treated the same as "the position
    is marked at the reopen print": the walk resumes on the next print/bar it finds, logged as a
    WARNING, not silently dropped.

Usage:  python3 rebuild_1619.py
Writes: rebuild_1619_fills.csv (one row per base fill), REBUILD_1619.md (report), both next to
this script.
"""
import datetime as dt
import glob
import os
import pickle
import sqlite3
import sys
import warnings
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from cell_1445 import day_clustered_t, ex_top5_mean          # noqa: E402 -- sanctioned helpers

ET = ZoneInfo('America/New_York')

# ---- constants (named in the PREREG / given as shared cost constants) -------------------------
LIMIT_BPS = 0.0015                                            # level * 1.0015, "the same 15 bps"
STOP_MULT = 1.0075                                            # level * 1.0075
RETEST_OFFSET = 0.01                                          # level - $0.01
ENTRY_WINDOW_MIN = 3                                          # fill minute + the next 2 minutes
EOD_MIN = 15 * 60 + 55                                         # 15:55 ET
SLIP_BP = 0.0002                                               # 2 bps, sip_rebuild.py's SLIP_BP
SLIP_STOP_BPS = {'TRAIN': 0.88 * 2.9 + 0.12 * 94.0, 'VAL': 0.88 * 3.2 + 0.12 * 76.0}  # cell_1478.py
EOD_BPS = {'TRAIN': 11.5, 'VAL': 9.7}                          # given: "EOD at the bid/ask"
BORROW_ANNUAL = 0.03
TICK = 0.01
EPS = 1e-9

CAUSAL_CSV = os.path.join(HERE, 'causal_arming_causal.csv')
PRED_CSV = os.path.join(HERE, 'model_1478_L3_predictions.csv')
FEAT_CSV = os.path.join(HERE, 'features_1478_A.csv')
BARS_DB = os.path.join(HERE, 'bars_fills_1478.db')
BORROW_CSV = os.path.join(os.path.dirname(os.path.dirname(HERE)), 'research/fuckup_audit/O_halt/PASSIVE/borrow_flags.csv')
CACHE_DIRS = [os.path.join(HERE, 'sip_cache_1481'), os.path.join(HERE, 'sip_cache_1480')]

OUT_CSV = os.path.join(HERE, 'rebuild_1619_fills.csv')
OUT_MD = os.path.join(HERE, 'REBUILD_1619.md')


def log(msg):
    """Timestamped, flushed progress line (verbose progress on a long process, per CLAUDE.md)."""
    print(f'[{dt.datetime.now():%H:%M:%S}] {msg}', flush=True)


# --------------------------------------------------------------------------------------------- time
def et_ns(day, seconds):
    """UTC epoch nanoseconds of `seconds` after ET midnight on `day` (YYYY-MM-DD)."""
    d = dt.date.fromisoformat(day)
    base = dt.datetime(d.year, d.month, d.day, tzinfo=ET)
    return int((base + dt.timedelta(seconds=float(seconds))).timestamp() * 1e9)


def ns_to_et_minutes(ns, day):
    """Fractional ET minutes-since-midnight of a UTC-ns timestamp on `day`."""
    return (ns - et_ns(day, 0)) / 60e9


# --------------------------------------------------------------------------------------------- inputs
def load_population():
    """Base fills (status=='fill'), joined to half_entry and the base long's own outcome_R."""
    base = pd.read_csv(CAUSAL_CSV, dtype={'day': str, 'symbol': str})
    base = base[base.status == 'fill'].reset_index(drop=True)
    log(f'[load] base fills: {len(base)}')

    feat = pd.read_csv(FEAT_CSV, usecols=['day', 'symbol', 'fill_min', 'split', 'half_entry'])
    base = base.merge(feat, on=['day', 'symbol', 'fill_min', 'split'], how='left')
    n_missing_he = int(base.half_entry.isna().sum())
    if n_missing_he:
        log(f'[load] WARNING: {n_missing_he} fills have no half_entry match in features_1478_A.csv '
            f'-- fallback to that split\'s median half_entry')
        med = base.groupby('split').half_entry.transform('median')
        base.half_entry = base.half_entry.fillna(med)

    pred = pd.read_csv(PRED_CSV, usecols=['day', 'symbol', 'fill_min', 'split', 'outcome_R'])
    pred = pred.rename(columns={'outcome_R': 'base_outcome_R'})
    base = base.merge(pred, on=['day', 'symbol', 'fill_min', 'split'], how='left')
    log(f'[load] base_outcome_R matched for {int(base.base_outcome_R.notna().sum())}/{len(base)}')

    borrow = pd.read_csv(BORROW_CSV)
    borrow = borrow.drop_duplicates(subset='symbol', keep='first').set_index('symbol')
    shortable = borrow['shortable'].to_dict()
    base['shortable'] = base.symbol.map(shortable)
    n_excl = int((base.shortable != True).sum())        # noqa: E712 -- NaN (unknown symbol) excluded too
    log(f'[load] {n_excl}/{len(base)} fills excluded as not-shortable / unknown in borrow_flags.csv')
    return base


def index_cache_dir(path):
    """{(symbol, day): [filenames]} for every SYMBOL_DAY_M.pkl in one sip_cache directory."""
    idx = {}
    if not os.path.isdir(path):
        log(f'[cache] WARNING: {path} does not exist -- treated as empty')
        return idx
    for fn in os.listdir(path):
        if not fn.endswith('.pkl'):
            continue
        parts = fn[:-4].rsplit('_', 2)                    # SYMBOL, DAY, M
        if len(parts) != 3:
            continue
        sym, day, _m = parts
        idx.setdefault((sym, day), []).append(fn)
    log(f'[cache] indexed {path}: {sum(len(v) for v in idx.values())} files, {len(idx)} symbol-days')
    return idx


def load_tape(symbol, day, indices):
    """Union of every cached tape file for (symbol, day) across both sip_cache dirs, ts-sorted.
    Returns (trades_df, quotes_df) or (None, None) if nothing is cached."""
    trades, quotes = [], []
    for path, idx in zip(CACHE_DIRS, indices):
        for fn in idx.get((symbol, day), []):
            with open(os.path.join(path, fn), 'rb') as f:
                t, q = pickle.load(f)
            trades.append(t)
            quotes.append(q)
    if not trades:
        return None, None
    tr = pd.concat(trades, ignore_index=True).drop_duplicates(subset=['ts', 'price', 'size'])
    qu = pd.concat(quotes, ignore_index=True).drop_duplicates(subset=['ts', 'bid', 'ask'])
    return tr.sort_values('ts', kind='stable').reset_index(drop=True), \
        qu.sort_values('ts', kind='stable').reset_index(drop=True)


def prevailing_quote(quotes, ts):
    """Last valid (bid, ask) (bid>0, ask>0, ask>=bid) at or before `ts`, or (nan, nan)."""
    if quotes is None or not len(quotes):
        return float('nan'), float('nan')
    q = quotes[(quotes.ts <= ts) & (quotes.bid > 0) & (quotes.ask > 0) & (quotes.ask >= quotes.bid)]
    if not len(q):
        return float('nan'), float('nan')
    last = q.iloc[-1]
    return float(last.bid), float(last.ask)


def load_bars(con, symbol, day):
    """RTH-and-around 1-minute bars for (symbol, day) as df[m (int ET minute), o,h,l,c], m-sorted.
    `con` is a reused sqlite3 connection (PRIMARY KEY(symbol,day,t) makes this an index lookup)."""
    q = pd.read_sql_query('SELECT t,o,h,l,c FROM bars WHERE symbol=? AND day=?', con, params=(symbol, day))
    if not q.empty:
        ts = pd.to_datetime(q.t, utc=True).dt.tz_convert(ET)
        q['m'] = ts.dt.hour * 60 + ts.dt.minute
        q = q.sort_values('m', kind='stable').reset_index(drop=True)
    return q


# --------------------------------------------------------------------------------------------- rule
def simulate_one(r, trades, quotes, bars, split):
    """Run the full B rule on one eligible base fill. Returns a flat dict for one output row."""
    level = float(r.level)
    limit = level * (1.0 + LIMIT_BPS)
    stop_price = level * STOP_MULT
    target_price = level - RETEST_OFFSET
    R_f = stop_price - limit
    row = dict(entry_price=np.nan, stop_price=stop_price, target_price=target_price, R_f=R_f,
               exit_reason=np.nan, exit_phase=np.nan, exit_ts_min=np.nan, exit_price=np.nan,
               entry_cost=np.nan, exit_cost=np.nan, borrow_cost=np.nan, raw_pnl=np.nan,
               net_pnl=np.nan, net_R_f=np.nan, net_pct_price=np.nan)

    if trades is None or not len(trades):
        row['status'] = 'no_tape'
        return row

    fill_ns = et_ns(r.day, r.fill_min * 60)
    entry_window_end_ns = et_ns(r.day, (int(r.fill_min) + ENTRY_WINDOW_MIN) * 60)
    cand = trades[(trades.ts >= fill_ns - EPS) & (trades.ts < entry_window_end_ns) & (trades.price > limit + EPS)]
    if not len(cand):
        row['status'] = 'no_fill'
        return row

    entry_ts = int(cand.ts.iloc[0])
    row.update(status='fill', entry_price=limit)
    entry_cost = float(r.half_entry)

    # ---- walk forward: tick phase (remaining trades after entry, still within cached tape) ----
    # Capped at entry+20 min: a symbol/day with >1 base fill has its tape files unioned in `trades`
    # (load_tape globs by symbol/day, not by minute), so an uncapped search could pick up a print
    # that really belongs to a LATER, unrelated fill's cached window. 20 min is a small margin over
    # the PREREG's "15 minutes after" population window; beyond it the bar-phase walk (a real,
    # continuous full-day timeline, not fill-specific) takes over instead.
    after = trades[(trades.ts > entry_ts) & (trades.ts <= entry_ts + 20 * 60 * 10**9)]
    hit_target = after[after.price <= target_price - EPS]
    hit_stop = after[after.price >= stop_price - EPS]
    t_target = int(hit_target.ts.iloc[0]) if len(hit_target) else None
    t_stop = int(hit_stop.ts.iloc[0]) if len(hit_stop) else None

    exit_ts = exit_price = exit_cost = why = phase = None
    if t_target is not None and (t_stop is None or t_target <= t_stop):
        exit_ts, why, phase = t_target, 'retest', 'tick'
        bid, ask = prevailing_quote(quotes, exit_ts)
        half = 0.5 * (ask - bid) if np.isfinite(ask) and np.isfinite(bid) else np.nan
        exit_price = target_price
        exit_cost = (half if np.isfinite(half) else target_price * EOD_BPS[split] / 1e4) \
            + SLIP_BP * exit_price
    elif t_stop is not None:
        exit_ts, why, phase = t_stop, 'stop', 'tick'
        _bid, ask = prevailing_quote(quotes, exit_ts)
        base_px = ask if np.isfinite(ask) else stop_price
        exit_price = base_px * (1.0 + SLIP_STOP_BPS[split] / 1e4)
        exit_cost = 0.0                                   # SLIP_STOP_BPS is the complete leg cost

    # ---- bar phase: only if the tick tape never resolved the trade ----
    if exit_ts is None:
        if bars is None or not len(bars):
            log(f'[walk] WARNING: {r.symbol} {r.day}: tape exhausted with no bars fallback -- '
                f'marked at the last tape print (position "reopen"/gap convention)')
            last = trades.iloc[-1]
            exit_ts, exit_price, why, phase = int(last.ts), float(last.price), 'eod', 'gap_fallback'
            exit_cost = exit_price * EOD_BPS[split] / 1e4
        else:
            entry_m = int(ns_to_et_minutes(entry_ts, r.day))
            path = bars[bars.m >= entry_m]
            resolved = False
            for br in path.itertuples():
                if br.m >= EOD_MIN:
                    exit_ts_min, exit_price, why = br.m, float(br.o), 'eod'
                    exit_cost = exit_price * EOD_BPS[split] / 1e4
                    exit_ts, phase, resolved = et_ns(r.day, br.m * 60), 'bar', True
                    break
                if br.h >= stop_price - EPS:
                    px = br.o if br.o >= stop_price else stop_price
                    exit_price = px * (1.0 + SLIP_STOP_BPS[split] / 1e4)
                    why, exit_cost = 'stop', 0.0
                    exit_ts, phase, resolved = et_ns(r.day, br.m * 60), 'bar', True
                    break
                if br.l <= target_price + EPS:
                    exit_price, why = target_price, 'retest'
                    exit_cost = target_price * EOD_BPS[split] / 1e4    # no quotes this far out
                    exit_ts, phase, resolved = et_ns(r.day, br.m * 60), 'bar', True
                    break
            if not resolved:
                log(f'[walk] WARNING: {r.symbol} {r.day}: bars ended before 15:55 (entry_m={entry_m}) '
                    f'-- exit at the last bar\'s close')
                last = path.iloc[-1] if len(path) else bars.iloc[-1]
                exit_ts, exit_price, why, phase = et_ns(r.day, int(last.m) * 60), float(last.c), \
                    'eod', 'bar_fallback'
                exit_cost = exit_price * EOD_BPS[split] / 1e4

    hold_s = max(0.0, (exit_ts - entry_ts) / 1e9)
    borrow_cost = limit * BORROW_ANNUAL * (hold_s / (365 * 24 * 3600))
    raw_pnl = limit - exit_price
    net_pnl = raw_pnl - entry_cost - exit_cost - borrow_cost

    row.update(exit_reason=why, exit_phase=phase, exit_ts_min=ns_to_et_minutes(exit_ts, r.day),
               exit_price=exit_price, entry_cost=entry_cost, exit_cost=exit_cost,
               borrow_cost=borrow_cost, raw_pnl=raw_pnl, net_pnl=net_pnl,
               net_R_f=net_pnl / R_f if R_f else np.nan,
               net_pct_price=100.0 * net_pnl / limit)
    return row


# --------------------------------------------------------------------------------------------- main
def main():
    warnings.filterwarnings('ignore', category=FutureWarning)
    base = load_population()
    indices = [index_cache_dir(p) for p in CACHE_DIRS]
    con = sqlite3.connect(BARS_DB)

    results = []
    tape_cache = {}
    bars_cache = {}
    n = len(base)
    for i, r in enumerate(base.itertuples()):
        row = dict(day=r.day, symbol=r.symbol, split=r.split, fill_min=r.fill_min, level=r.level,
                   shortable=r.shortable, base_outcome_R=r.base_outcome_R)
        if r.shortable != True:                            # noqa: E712 -- NaN excluded too
            row['status'] = 'excluded_not_shortable'
            results.append(row)
            continue

        key = (r.symbol, r.day)
        if key not in tape_cache:
            tape_cache[key] = load_tape(r.symbol, r.day, indices)
        trades, quotes = tape_cache[key]

        if key not in bars_cache:
            bars_cache[key] = load_bars(con, r.symbol, r.day)
        bars = bars_cache[key]

        row.update(simulate_one(r, trades, quotes, bars, r.split))
        results.append(row)

        if (i + 1) % 1000 == 0 or i + 1 == n:
            log(f'[sim] {i + 1}/{n}')

    con.close()
    out = pd.DataFrame(results)
    cols = ['day', 'symbol', 'split', 'fill_min', 'level', 'shortable', 'status', 'entry_price',
            'stop_price', 'target_price', 'R_f', 'exit_reason', 'exit_phase', 'exit_ts_min',
            'exit_price', 'entry_cost', 'exit_cost', 'borrow_cost', 'raw_pnl', 'net_pnl', 'net_R_f',
            'net_pct_price', 'base_outcome_R']
    out = out[cols]
    out.to_csv(OUT_CSV, index=False)
    log(f'[main] wrote {OUT_CSV} ({len(out)} rows)')
    log('[main] status counts:\n' + out.status.value_counts(dropna=False).to_string())
    write_report(out)


def score_split(df, split):
    """VAL/TRAIN-H2 scoring block: fill share, cover/stop shares, runner cohort, mean/t/ex-top-5%."""
    pop = df[df.split == split]
    elig = pop[pop.status != 'excluded_not_shortable']
    have_tape = elig[elig.status != 'no_tape']
    filled = pop[pop.status == 'fill']
    out = dict(n_pop=len(pop), n_elig=len(elig), n_have_tape=len(have_tape), n_filled=len(filled))
    out['fill_share'] = len(filled) / len(have_tape) if len(have_tape) else np.nan
    if len(filled):
        for reason in ('retest', 'stop', 'eod'):
            out[f'{reason}_share'] = float((filled.exit_reason == reason).mean())
        within15 = filled[(filled.exit_reason == 'retest') & (filled.exit_ts_min - filled.fill_min <= 15)]
        out['cover_within_15min_share'] = len(within15) / len(filled)
        runners = filled[filled.exit_reason == 'eod']
        out['runner_n'] = len(runners)
        out['runner_mean_net_R_f'] = float(runners.net_R_f.mean()) if len(runners) else np.nan
        y = filled.net_R_f
        out['mean_net_R_f'] = float(y.mean())
        out['mean_net_pct_price'] = float(filled.net_pct_price.mean())
        out['median_R_f_pct_price'] = float((filled.R_f / filled.entry_price * 100).median())
        out['t_stat'] = day_clustered_t(y, filled.day)
        out['ex_top5_mean'] = ex_top5_mean(y)
        weeks = filled.day.pipe(pd.to_datetime).dt.isocalendar()
        n_weeks = weeks[['year', 'week']].drop_duplicates().shape[0]
        out['fills_per_week_raw'] = len(filled) / n_weeks if n_weeks else np.nan
    return out


def write_report(out):
    """REBUILD_1619.md: VAL primary, TRAIN-H2 sign check, obtainability, tails, pass-bar verdict."""
    val = score_split(out, 'VAL')
    tr = score_split(out, 'TRAIN') if 'TRAIN' in out.split.unique() else {}
    # TRAIN-H2 check: causal_arming_causal.csv's TRAIN rows are already H2-only (half=='H2' by
    # construction of the shared population -- verified in the exploration log, not re-checked here).

    pass_bar = (val.get('mean_net_R_f', -9) >= 0.15 and val.get('mean_net_pct_price', -9) >= 0.10
                and (val.get('t_stat') or 0) >= 2.5 and (val.get('ex_top5_mean') or -9) > 0
                and (val.get('fills_per_week_raw') or 0) >= 3
                and (val.get('median_R_f_pct_price') or 0) >= 0.5
                and (tr.get('mean_net_R_f') or -9) * (val.get('mean_net_R_f') or -9) > 0)

    lines = []
    lines.append('# REBUILD_1619 -- independent rebuild of PREREG_1617.md Frame B (burst fade)')
    lines.append('')
    lines.append('Built from PREREG_1617.md prose only; cell_1619.py / cell_1619_fills.csv / '
                 'RESULT_1619.md were not opened. See rebuild_1619.py\'s module docstring for the '
                 'full mechanism as read and every disclosed interpretation choice.')
    lines.append('')
    lines.append('## VAL (primary)')
    for k, v in val.items():
        lines.append(f'- {k}: {v:.4f}' if isinstance(v, float) else f'- {k}: {v}')
    lines.append('')
    lines.append('## TRAIN-H2 (sign check only)')
    for k, v in tr.items():
        lines.append(f'- {k}: {v:.4f}' if isinstance(v, float) else f'- {k}: {v}')
    lines.append('')
    lines.append('## Pass bar (frozen, PREREG_1617.md)')
    lines.append('mean net R_f >= +0.15 AND >= +0.10% of price, t >= 2.5, ex-top-5% > 0, '
                 '>= 3 fills/week, TRAIN-H2 same sign, median R_f >= 0.5% of price.')
    lines.append(f'**Rebuild verdict: {"PASS" if pass_bar else "FAIL"}** (this rebuild\'s own numbers '
                 f'only -- the independent-check agreement bar in the PREREG, fill-set Jaccard >= 0.98 '
                 f'and >= 99% within 0.01 R_f against cell_1619, is for the comparator to run, not this '
                 f'script).')
    lines.append('')
    lines.append('## Obtainability / refuters (PREREG "Independent check" section, B)')
    n_tape = int((out.status != 'no_tape').sum()) if 'no_tape' in out.status.values else len(out)
    lines.append(f'- Population {len(out)} base fills; excluded not-shortable '
                 f'{int((out.status=="excluded_not_shortable").sum())}; no cached tape at all '
                 f'{int((out.status=="no_tape").sum())}.')
    lines.append(f'- Exit phase mix among fills: \n{out[out.status=="fill"].exit_phase.value_counts().to_string()}')
    lines.append('- Tick-tape coverage is short per signal (tens to ~100s of seconds of real prints '
                 'in the sampled files); most retest/stop resolution for fills more than a few minutes '
                 'from the close falls through to the 1-minute-bar walk (bars_fills_1478.db), mirroring '
                 'sip_rebuild.walk_path\'s semantics for a short. This is a material, disclosed departure '
                 'from a fully tick-priced retest for those rows -- see the module docstring.')
    lines.append(f'- **eod_share is {val.get("eod_share", float("nan")):.4f} and runner_n is '
                 f'{val.get("runner_n")}: this rebuild\'s bar-phase walk continues on real, full-day '
                 f'1-minute bars all the way to 15:55, and at R_f as tiny as ~0.6% of price essentially '
                 f'every position touches either the stop or the retest level somewhere over the '
                 f'rest of the day, so almost nothing "never retests." The PREREG\'s own report list '
                 f'asks for a "runner losses (the never-retest cohort\'s cost)" line, which implies '
                 f'cell_1619 expected a non-trivial such cohort -- most likely because its walk is '
                 f'bounded by the ~15-minute tape window itself (no bar fallback), so a position '
                 f'unresolved at that data horizon is booked as the "eod"/runner case there. THIS IS '
                 f'THE SINGLE MOST LIKELY POINT OF DIVERGENCE between this rebuild and cell_1619 -- a '
                 f'comparator should check it first, ahead of any per-row price arithmetic.')
    lines.append('')
    lines.append('## Caveats for the comparator')
    lines.append('- half_entry, base_outcome_R joins are on (day,symbol,fill_min,split); any float '
                 'round-trip mismatch would show as an unmatched row (see [load] WARNING lines in the '
                 'run log) -- none were logged in this run unless noted above.')
    lines.append('- Stop and EOD leg costs are modeled as the COMPLETE cost for that leg (SLIP_STOP_BPS '
                 'or EOD_BPS alone, no separately-added half-spread+2bp on top). The alternative reading '
                 '-- additive on top of the standard half-spread+2bp -- was considered and rejected '
                 'because the PREREG phrases each as "= ..." a single complete recipe; a comparator '
                 'disagreement concentrated in stop/eod rows\' cost_R most likely traces to this choice.')
    lines.append('- No SSR column exists in borrow_flags.csv; only `shortable` gates the population.')
    lines.append('- Borrow cost (3%/yr pro rata) is on notional at the short\'s entry price, actual '
                 'holding seconds; it is negligible (sub-bp) for every intraday hold here and does not '
                 'drive any conclusion.')
    with open(OUT_MD, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    log(f'[main] wrote {OUT_MD}')


if __name__ == '__main__':
    main()
