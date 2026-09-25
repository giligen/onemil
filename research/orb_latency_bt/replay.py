#!/usr/bin/env python3
"""ORB order-latency replay on Databento tick data. Cell 1,426.

Implements research/orb_latency_bt/PREREG.md exactly (frozen 2026-09-25 14:55 UTC).
Question: how much of the ORB backtest edge survives if the resting buy-stop is
placed d seconds late (d in DELAY_GRID), replayed against the real XNAS tape
instead of the BT's flat 30bps entry-slip model?

Entry rule (extracted from trading/orb_engine.py and trading/orb_planner.py, cited
by line number so a reviewer can check this against the live code without rerunning
anything):
  * range_high = the 5-min opening-range high (study_orb_features.py:262).
  * trigger (stop price) = range_high, unmodified.
      trading/orb_engine.py:3408  `stop_trigger = round(plan.range_high, 2)`
      comment: "BT parity: stop triggers at range_high".
  * BT entry_price = range_high * (1 + entry_slip_bps/10000), entry_slip_bps=30
      (orb.yaml.template:76).
      trading/orb_planner.py:151  `entry_price = range_high * (1.0 + self.entry_slip_bps / 10000.0)`
  * limit_price (submitted to the broker) = round(plan.entry_price, 2)
      trading/orb_engine.py:3439  `limit_price = round(plan.entry_price, 2)`
    i.e. limit == the book's own `entry_price` column, and because both the BT
    entry-slip and the live stop-limit buffer are 30bps, trigger = entry_price / 1.003.
  * Chase guard (trading/orb_engine.py:598-608, 3438-3474; shared trading/buy_stop_guard.py):
    default `_buy_stop_guard_cfg = {"enabled": True, "rebump_buffer": 0.02}` (line 608),
    not overridden in orb.yaml.template. PREREG's replay rule (frozen, reproduced here
    verbatim, not re-derived):
      t* = first XNAS trade print >= trigger in [09:35:00, 09:40:00) ET.
        no t*            -> unfilled (P&L 0)
        t* >= T0          -> fill at ask(t*) [strictly prior mbp-1 record] if ask <= limit,
                             else SKIPPED
        t* <  T0          -> fill at ask(T0) if ask(T0) + rebump_buffer <= limit,
                             else SKIPPED (never chased)
      T0 = 09:35:00 ET + d.

P&L: BT `_sized_pnl` + shares * (BT entry_price - replay fill_price); shares is
recovered from the book (shares = _sized_pnl / pnl, both already in the CSV — pnl is
the BT's own per-share, unsized P&L). Skipped/unfilled fills contribute 0. R = P&L / 375.

Usage:
    python3 replay.py population                 # build research/orb_latency_bt/population.csv
    python3 replay.py cost-gate [--sample 30]     # PREREG cost gate, STOP if projected > $15
    python3 replay.py fetch [--threads 8]         # Databento fetch -> raw/*.parquet
    python3 replay.py run                         # replay all delays, write results.csv
    python3 replay.py report                      # write REPORT.md from results.csv
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from dotenv import load_dotenv

ROOT = '/home/ec2-user/onemil'
sys.path.insert(0, ROOT)
os.chdir(ROOT)
load_dotenv(f'{ROOT}/.env')

OUT_DIR = f'{ROOT}/research/orb_latency_bt'
RAW_DIR = f'{OUT_DIR}/raw'
POP_CSV = f'{OUT_DIR}/population.csv'
MANIFEST = f'{OUT_DIR}/manifest.csv'
COST_GATE_JSON = f'{OUT_DIR}/cost_gate_result.json'
RESULTS_CSV = f'{OUT_DIR}/results.csv'
REPORT_MD = f'{OUT_DIR}/REPORT.md'

DATASET = 'XNAS.ITCH'
BUDGET = 15.0
ET = 'America/New_York'

ENTRY_SLIP_BPS = 30.0
REBUMP_BUFFER = 0.02
DELAY_GRID = [0, 1, 2, 3, 5, 10, 15, 20, 30, 45, 60, 90]
R_DENOM = 375.0

# window: 09:34:50 .. 09:40:00 ET, expressed in seconds-since-09:35:00 (can be negative)
WIN_START_S = -10.0
WIN_END_S = 300.0

BOOKS = [
    ('research/orb_2023/book_1418_liveexit.csv',),
    ('research/orb_2024/book_1415_liveexit.csv',),
    ('research/thermo/book_2025_26.csv',),
]


def log(msg: str) -> None:
    """Verbose, timestamped progress line to stdout (flushed immediately)."""
    print(f'[{datetime.now(timezone.utc).isoformat(timespec="seconds")}] {msg}', flush=True)


# --------------------------------------------------------------- population
def bucket_period(date_str: str) -> str:
    """Assign a PREREG report period to a fill date (ISO 'YYYY-MM-DD')."""
    if date_str < '2025-01-01':
        return '2023-24'
    if date_str < '2025-07-01':
        return '2025H1'
    return '2025H2-2026-09'


def build_population() -> pd.DataFrame:
    """Load the three live-config books, keep entered==1 fills, derive the
    tick-replay inputs (trigger, limit, shares) and write population.csv.

    entered==0 rows are counted (logged) but not written to population.csv —
    PREREG: "counted but not replayed (a resting order that never triggered
    stays unfilled)"; they need no tick fetch since the BT itself never fired.
    """
    frames = []
    n_not_entered = 0
    for (path,) in BOOKS:
        df = pd.read_csv(path, keep_default_na=False, na_values=[''])
        n_not_entered += int((df['entered'] == 0).sum())
        df = df[df['entered'] == 1].copy()
        df['source_book'] = path
        frames.append(df)
    pop = pd.concat(frames, ignore_index=True)
    before = len(pop)
    pop = pop.drop_duplicates(subset=['symbol', 'date'], keep='first')
    if len(pop) != before:
        log(f'WARNING dropped {before - len(pop)} duplicate (symbol,date) rows across books')

    pop['period'] = pop['date'].apply(bucket_period)
    pop['limit'] = pop['entry_price'].round(2)
    pop['trigger'] = (pop['entry_price'] / (1.0 + ENTRY_SLIP_BPS / 10000.0)).round(4)

    bad_pnl = pop['pnl'].abs() < 1e-6
    pop['shares'] = np.where(bad_pnl, np.nan, pop['_sized_pnl'] / pop['pnl'])
    n_bad = int(bad_pnl.sum())
    if n_bad:
        log(f'WARNING {n_bad} fills have |pnl| < 1e-6 (flat exit) — shares undetermined, '
            f'replay P&L adjustment will fall back to 0 for these (no chase-price info to size)')
    pop.loc[bad_pnl, 'shares'] = 0.0

    pop.to_csv(POP_CSV, index=False)
    log(f'population: {len(pop)} entered==1 fills ({n_not_entered} entered==0 counted, not replayed), '
        f'periods: {pop["period"].value_counts().to_dict()}')
    return pop


# --------------------------------------------------------------- databento
def get_client():
    """Databento historical client from DATABENTO_API_KEY (.env, python-dotenv)."""
    import databento as db
    key = os.environ.get('DATABENTO_API_KEY') or os.environ.get('DATABENTO_KEY')
    assert key, 'no DATABENTO_API_KEY in .env — fetch cannot run (reported, not silently skipped)'
    return db.Historical(key)


def window_utc(date_str: str):
    """(start_utc_iso, end_utc_iso) for 09:34:50-09:40:00 ET on date_str."""
    start = pd.Timestamp(f'{date_str} 09:34:50', tz=ET).tz_convert('UTC')
    end = pd.Timestamp(f'{date_str} 09:40:00', tz=ET).tz_convert('UTC')
    return start.isoformat(), end.isoformat()


def cmd_cost_gate(args):
    """PREREG cost gate: metadata.get_cost on `sample` random fills (both schemas),
    extrapolate to the full population, STOP if projected > $15."""
    pop = pd.read_csv(POP_CSV) if os.path.exists(POP_CSV) else build_population()
    client = get_client()
    rng = np.random.default_rng(42)
    n = min(args.sample, len(pop))
    idx = rng.choice(len(pop), size=n, replace=False)
    sample = pop.iloc[idx]
    total = 0.0
    n_ok = 0
    n_unresolved = 0
    for i, r in enumerate(sample.itertuples(), 1):
        s, e = window_utc(r.date)
        try:
            c1 = float(client.metadata.get_cost(dataset=DATASET, schema='mbp-1', symbols=[r.symbol],
                                                 stype_in='raw_symbol', start=s, end=e))
            c2 = float(client.metadata.get_cost(dataset=DATASET, schema='trades', symbols=[r.symbol],
                                                 stype_in='raw_symbol', start=s, end=e))
            total += c1 + c2
            n_ok += 1
        except Exception as ex:
            n_unresolved += 1
            log(f'  WARNING {r.symbol} {r.date} unresolved for cost pricing, excluded from '
                f'the average ({type(ex).__name__}): {ex}')
        if i % 10 == 0 or i == n:
            log(f'  priced {i}/{n} ok={n_ok} unresolved={n_unresolved} running=${total:.4f}')
    if n_ok == 0:
        log('STOP: every sampled fill was unresolved, cannot extrapolate a cost')
        with open(COST_GATE_JSON, 'w') as f:
            json.dump(dict(sampled=n, verdict='STOP', reason='all_unresolved'), f, indent=2)
        return 2
    avg = total / n_ok
    projected = avg * len(pop)
    verdict = 'PASS' if projected <= BUDGET else 'STOP'
    log(f'COST_GATE sampled={n} ok={n_ok} unresolved={n_unresolved} avg_per_fill=${avg:.6f} '
        f'total_fills={len(pop)} projected=${projected:.4f} budget=${BUDGET} verdict={verdict}')
    with open(COST_GATE_JSON, 'w') as f:
        json.dump(dict(sampled=n, n_ok=n_ok, n_unresolved=n_unresolved, avg_per_fill=avg,
                        total_fills=len(pop), projected=projected,
                        budget=BUDGET, verdict=verdict,
                        ts=datetime.now(timezone.utc).isoformat()), f, indent=2)
    return 0 if verdict == 'PASS' else 2


def fetch_one(client, symbol, start, end, schema):
    for attempt in range(4):
        try:
            data = client.timeseries.get_range(dataset=DATASET, schema=schema, symbols=[symbol],
                                                start=start, end=end, stype_in='raw_symbol')
            return data.to_df().reset_index()
        except Exception as ex:
            log(f'  WARN {symbol} {schema} attempt {attempt + 1}: {type(ex).__name__}: {ex}')
            if attempt == 3:
                raise
            time.sleep(3 * (attempt + 1))


def fetch_fill(client, row):
    """Fetch trades + mbp-1 for one (symbol, date) window -> raw/{date}__{symbol}.parquet."""
    start, end = window_utc(row.date)
    frames = []
    for schema in ('trades', 'mbp-1'):
        try:
            df = fetch_one(client, row.symbol, start, end, schema)
        except Exception as ex:
            log(f'  ERROR {row.date} {row.symbol} {schema} giving up: {ex}')
            continue
        if df is not None and len(df):
            keep = [c for c in ('ts_event', 'price', 'size', 'bid_px_00', 'ask_px_00') if c in df.columns]
            df = df[keep].copy()
            df['schema'] = schema
            frames.append(df)
    n = 0
    if frames:
        out = pd.concat(frames, ignore_index=True)
        out.to_parquet(f'{RAW_DIR}/{row.date}__{row.symbol}.parquet', index=False, compression='zstd')
        n = len(out)
    return row.date, row.symbol, n


def load_manifest():
    if os.path.exists(MANIFEST):
        return pd.read_csv(MANIFEST)
    return pd.DataFrame(columns=['unit', 'status', 'n_rows', 'ts_utc'])


def append_manifest(unit, status, n_rows):
    header = not os.path.exists(MANIFEST)
    with open(MANIFEST, 'a') as f:
        if header:
            f.write('unit,status,n_rows,ts_utc\n')
        f.write(f'{unit},{status},{n_rows},{datetime.now(timezone.utc).isoformat()}\n')


def cmd_fetch(args):
    """Fetch trades+mbp-1 for every population fill, threaded, resumable via manifest.csv."""
    gate = json.load(open(COST_GATE_JSON)) if os.path.exists(COST_GATE_JSON) else None
    if not gate or gate.get('verdict') != 'PASS':
        log('REFUSING to fetch: cost gate has not PASSed (run cost-gate first)')
        return 2
    pop = pd.read_csv(POP_CSV)
    manifest = load_manifest()
    done = set(manifest[manifest.status == 'done']['unit']) if len(manifest) else set()
    pending = [r for r in pop.itertuples()
               if f'{r.date}__{r.symbol}' not in done
               or not os.path.exists(f'{RAW_DIR}/{r.date}__{r.symbol}.parquet')]
    log(f'fetch plan: {len(pop)} fills total, {len(pending)} pending, threads={args.threads}')
    client = get_client()
    completed = len(pop) - len(pending)
    with ThreadPoolExecutor(max_workers=args.threads) as ex:
        futs = {ex.submit(fetch_fill, client, r): f'{r.date}__{r.symbol}' for r in pending}
        for fut in as_completed(futs):
            unit = futs[fut]
            try:
                d, sym, n = fut.result()
                append_manifest(unit, 'done', n)
                completed += 1
                if completed % 25 == 0 or completed == len(pop):
                    log(f'FILL_DONE {unit} rows={n} ({completed}/{len(pop)})')
            except Exception as ex:
                append_manifest(unit, 'error', 0)
                log(f'FILL_ERROR {unit}: {type(ex).__name__}: {ex}')
    log('FETCH_COMPLETE')
    return 0


# --------------------------------------------------------------- replay core
def to_sec_since_0935(ts_event_ns: np.ndarray, date_str: str) -> np.ndarray:
    """Convert ts_event (UTC ns epoch) to seconds since 09:35:00 ET on date_str."""
    anchor = pd.Timestamp(f'{date_str} 09:35:00', tz=ET).tz_convert('UTC').value  # ns
    return (ts_event_ns.astype('int64') - anchor) / 1e9


def find_t_star(trades: pd.DataFrame, trigger: float):
    """First XNAS trade print with price >= trigger inside [09:35:00, 09:40:00) ET
    (WIN_START_S..WIN_END_S convention here is applied by the caller pre-filtering
    to [0, 300) for this specific function — see replay_one). Returns (t_sec, None) or (None, None)."""
    hit = trades[trades['price'] >= trigger]
    if hit.empty:
        return None
    return float(hit['t_sec'].iloc[0])


def ask_at(mbp: pd.DataFrame, t: float):
    """Ask price from the strictly-prior mbp-1 record (last record with t_sec < t)."""
    prior = mbp[mbp['t_sec'] < t]
    if prior.empty:
        return None
    ask = prior['ask_px_00'].iloc[-1]
    if pd.isna(ask) or ask <= 0:
        return None
    return float(ask)


def replay_one(trades: pd.DataFrame, mbp: pd.DataFrame, trigger: float, limit: float,
                delay_s: float, rebump_buffer: float = REBUMP_BUFFER) -> dict:
    """Replay one fill at one delay `d`. trades/mbp must already carry a 't_sec'
    column (seconds since 09:35:00 ET) and be sorted by it.

    Returns dict(status, fill_price, t_star) where status is one of:
      'unfilled_no_tstar'  - no XNAS print ever reached trigger in the window
      'unfilled_no_quote'  - t* found but no prior two-sided quote available
      'filled'
      'skipped_guard'      - ask (+ rebump_buffer if t*<T0) > limit
    """
    signal = trades[(trades['t_sec'] >= 0.0) & (trades['t_sec'] < WIN_END_S)]
    t_star = find_t_star(signal, trigger)
    if t_star is None:
        return dict(status='unfilled_no_tstar', fill_price=None, t_star=None)
    T0 = float(delay_s)
    if t_star >= T0:
        ask = ask_at(mbp, t_star)
        if ask is None:
            return dict(status='unfilled_no_quote', fill_price=None, t_star=t_star)
        if ask <= limit:
            return dict(status='filled', fill_price=ask, t_star=t_star)
        return dict(status='skipped_guard', fill_price=None, t_star=t_star)
    else:
        ask = ask_at(mbp, T0)
        if ask is None:
            return dict(status='unfilled_no_quote', fill_price=None, t_star=t_star)
        if ask + rebump_buffer <= limit:
            return dict(status='filled', fill_price=ask, t_star=t_star)
        return dict(status='skipped_guard', fill_price=None, t_star=t_star)


AVAIL_CSV = f'{OUT_DIR}/availability.csv'


def cmd_run(args):
    """Replay every population fill at every delay in DELAY_GRID -> results.csv.
    Also writes availability.csv (one row/fill: has_trade, has_twosided_quote,
    usable — PREREG availability rail) and captures d=0's t_star for the two
    report lenses (XNAS-trigger-bias, gap-through) that only need d=0."""
    pop = pd.read_csv(POP_CSV)
    rows = []
    avail_rows = []
    n_missing_data = 0
    for i, r in enumerate(pop.to_dict('records'), 1):
        r = argparse.Namespace(**r)
        path = f'{RAW_DIR}/{r.date}__{r.symbol}.parquet'
        if not os.path.exists(path):
            n_missing_data += 1
            for d in DELAY_GRID:
                rows.append(dict(symbol=r.symbol, date=r.date, period=r.period, delay_s=d,
                                  status='missing_tick_data', fill_price=np.nan, t_star=np.nan,
                                  pnl_replay=np.nan))
            avail_rows.append(dict(symbol=r.symbol, date=r.date, period=r.period,
                                    has_trade=False, has_twosided_quote=False, usable=False,
                                    win_flag=r.win, pnl_bt=r.pnl))
            continue
        raw = pd.read_parquet(path)
        trades = raw[raw['schema'] == 'trades'].copy()
        mbp = raw[raw['schema'] == 'mbp-1'].copy()
        trades['t_sec'] = to_sec_since_0935(trades['ts_event'].values, r.date)
        mbp['t_sec'] = to_sec_since_0935(mbp['ts_event'].values, r.date)
        trades = trades.sort_values('t_sec')
        mbp = mbp.sort_values('t_sec')
        win_trades = trades[(trades['t_sec'] >= WIN_START_S) & (trades['t_sec'] < WIN_END_S)]
        win_mbp = mbp[(mbp['t_sec'] >= WIN_START_S) & (mbp['t_sec'] < WIN_END_S)]
        has_trade = len(win_trades) >= 1
        has_2sided = bool(((win_mbp.get('bid_px_00') > 0) & (win_mbp.get('ask_px_00') > 0)).any()) \
            if 'bid_px_00' in win_mbp.columns and 'ask_px_00' in win_mbp.columns else False
        avail_rows.append(dict(symbol=r.symbol, date=r.date, period=r.period,
                                has_trade=has_trade, has_twosided_quote=has_2sided,
                                usable=has_trade and has_2sided, win_flag=r.win, pnl_bt=r.pnl))
        for d in DELAY_GRID:
            res = replay_one(trades, mbp, r.trigger, r.limit, d)
            if res['status'] == 'filled':
                shares = 0.0 if pd.isna(r.shares) else r.shares
                pnl = r._sized_pnl + shares * (r.entry_price - res['fill_price'])
            else:
                pnl = 0.0
            rows.append(dict(symbol=r.symbol, date=r.date, period=r.period, delay_s=d,
                              status=res['status'], fill_price=res['fill_price'],
                              t_star=res['t_star'], pnl_replay=pnl))
        if i % 50 == 0 or i == len(pop):
            log(f'REPLAYED {i}/{len(pop)} fills')
    if n_missing_data:
        log(f'WARNING {n_missing_data}/{len(pop)} fills have no tick parquet (fetch incomplete or errored)')
    out = pd.DataFrame(rows)
    out.to_csv(RESULTS_CSV, index=False)
    avail = pd.DataFrame(avail_rows)
    avail.to_csv(AVAIL_CSV, index=False)
    usable_frac = avail['usable'].mean() if len(avail) else 0.0
    winner_miss = 1 - avail.loc[avail.win_flag == 1, 'usable'].mean() if (avail.win_flag == 1).any() else np.nan
    loser_miss = 1 - avail.loc[avail.win_flag == 0, 'usable'].mean() if (avail.win_flag == 0).any() else np.nan
    log(f'RESULTS_WRITTEN {RESULTS_CSV} rows={len(out)}; AVAILABILITY usable={usable_frac:.1%} '
        f'winner_missing={winner_miss:.1%} loser_missing={loser_miss:.1%}')
    return 0


def day_clustered_t(df: pd.DataFrame, value_col: str = 'pnl_replay') -> float:
    """Day-clustered t-stat for the mean of value_col (CR1-style: cluster by
    `date`, se = sqrt(sum_over_days((sum of within-day deviations)^2)) / n)."""
    if df.empty:
        return np.nan
    x = df[value_col].fillna(0.0)
    n = len(x)
    mean = x.mean()
    dev = x - mean
    cluster_sums = dev.groupby(df['date']).sum()
    var = (cluster_sums ** 2).sum() / (n ** 2)
    se = np.sqrt(var)
    return float(mean / se) if se > 0 else np.nan


def ex_top5_mean_r(df: pd.DataFrame) -> float:
    """Mean R after dropping the top 5% of fills by pnl_replay (ceil, by count)."""
    if df.empty:
        return np.nan
    k = int(np.ceil(0.05 * len(df)))
    if k == 0:
        return float((df['pnl_replay'].fillna(0.0) / R_DENOM).mean())
    trimmed = df.sort_values('pnl_replay', ascending=False).iloc[k:]
    return float((trimmed['pnl_replay'].fillna(0.0) / R_DENOM).mean())


def period_delay_table(df: pd.DataFrame) -> pd.DataFrame:
    """One row per delay: n_filled, n_skipped_guard, total_usd, mean_R, day_t, ex_top5_R."""
    rows = []
    for d in DELAY_GRID:
        sub = df[df.delay_s == d]
        n_filled = int((sub.status == 'filled').sum())
        n_skipped = int((sub.status == 'skipped_guard').sum())
        total_usd = float(sub['pnl_replay'].fillna(0.0).sum())
        mean_r = float((sub['pnl_replay'].fillna(0.0) / R_DENOM).mean()) if len(sub) else np.nan
        t = day_clustered_t(sub)
        ex5 = ex_top5_mean_r(sub)
        rows.append(dict(delay_s=d, n=len(sub), n_filled=n_filled, n_skipped_guard=n_skipped,
                          total_usd=total_usd, mean_R=mean_r, day_t=t, ex_top5pct_R=ex5))
    return pd.DataFrame(rows)


def df_to_md(df: pd.DataFrame, float_cols=()) -> str:
    cols = list(df.columns)
    lines = ['| ' + ' | '.join(cols) + ' |', '|' + '|'.join(['---'] * len(cols)) + '|']
    for _, r in df.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            if c in float_cols and pd.notna(v):
                cells.append(f'{v:.3f}')
            elif isinstance(v, float) and pd.notna(v):
                cells.append(f'{v:.3f}')
            else:
                cells.append(str(v))
        lines.append('| ' + ' | '.join(cells) + ' |')
    return '\n'.join(lines)


def cmd_report(args):
    """Build research/orb_latency_bt/REPORT.md: tables, curve, lenses, decision number."""
    results = pd.read_csv(RESULTS_CSV)
    pop = pd.read_csv(POP_CSV)
    avail = pd.read_csv(AVAIL_CSV) if os.path.exists(AVAIL_CSV) else pd.DataFrame()
    results = results[results.status != 'missing_tick_data'].copy()
    n_missing = (len(pd.read_csv(RESULTS_CSV)) - len(results)) // len(DELAY_GRID)  # rows -> fills

    is_oos = results.period.isin(['2023-24', '2025H2-2026-09'])
    oos = results[is_oos]
    results['date_dt'] = pd.to_datetime(results['date'])
    may_sep = results[(results.date_dt >= '2026-05-01') & (results.date_dt <= '2026-09-30')]

    tbl_2023_24 = period_delay_table(results[results.period == '2023-24'])
    tbl_2025h1 = period_delay_table(results[results.period == '2025H1'])
    tbl_2025h2 = period_delay_table(results[results.period == '2025H2-2026-09'])
    tbl_oos = period_delay_table(oos)
    tbl_may = period_delay_table(may_sep)

    # decision number: largest d with oos mean_R >= 0 AND may-sep total_usd >= 0
    qualifying = []
    for d in DELAY_GRID:
        oos_r = tbl_oos.loc[tbl_oos.delay_s == d, 'mean_R'].iloc[0]
        may_usd = tbl_may.loc[tbl_may.delay_s == d, 'total_usd'].iloc[0]
        if pd.notna(oos_r) and oos_r >= 0 and pd.notna(may_usd) and may_usd >= 0:
            qualifying.append(d)
    decision_d = max(qualifying) if qualifying else None
    eng_target = (decision_d - 2) if decision_d is not None else None

    # availability rail
    usable_frac = float(avail['usable'].mean()) if len(avail) else float('nan')
    winner_miss = 1 - avail.loc[avail.win_flag == 1, 'usable'].mean() if len(avail) and (avail.win_flag == 1).any() else np.nan
    loser_miss = 1 - avail.loc[avail.win_flag == 0, 'usable'].mean() if len(avail) and (avail.win_flag == 0).any() else np.nan
    gap_pp = abs(winner_miss - loser_miss) * 100 if pd.notna(winner_miss) and pd.notna(loser_miss) else np.nan
    rail_pass = (usable_frac >= 0.80) and (pd.isna(gap_pp) or gap_pp <= 5.0)

    # lens (i): d=0 fill price vs BT's own entry_price (30bps model)
    d0 = results[results.delay_s == 0].merge(pop[['symbol', 'date', 'entry_price']], on=['symbol', 'date'])
    d0_filled = d0[d0.status == 'filled']
    if len(d0_filled):
        d0_filled = d0_filled.copy()
        d0_filled['agrees'] = (d0_filled['fill_price'] - d0_filled['entry_price']).abs() <= 0.01
        lens1_frac = float(d0_filled['agrees'].mean())
        lens1_mean_diff = float((d0_filled['fill_price'] - d0_filled['entry_price']).mean())
    else:
        lens1_frac, lens1_mean_diff = np.nan, np.nan

    # lens (ii): XNAS-only trigger bias — share of d=0 t_star > 2s after 09:35:00
    d0_has_tstar = d0[d0.t_star.notna()]
    lens2_frac = float((d0_has_tstar['t_star'] > 2.0).mean()) if len(d0_has_tstar) else np.nan

    # lens (iii): gap-through fills, |t_star| <= 1s of 09:35:00
    gap_through = d0_has_tstar[d0_has_tstar['t_star'].abs() <= 1.0]
    gap_n = len(gap_through)
    gap_mean_r = float((gap_through['pnl_replay'].fillna(0.0) / R_DENOM).mean()) if gap_n else np.nan

    with open(REPORT_MD, 'w') as f:
        f.write('# REPORT — ORB latency replay (cell 1,426)\n\n')
        f.write('Report-only cell; decision number pre-committed in PREREG.md. Frozen population, delay grid, '
                'entry rule and exit rule — see PREREG.md for the exact text.\n\n')
        f.write('## Coverage\n')
        f.write(f'- population (entered==1) fills: {len(pop)}; entered==0 counted-not-replayed: '
                f'{259}\n')
        f.write(f'- missing tick parquet (fetch gap): {n_missing}\n')
        f.write(f'- availability rail: usable={usable_frac:.1%} (pass bar >=80%), '
                f'winner/loser missingness gap={gap_pp:.1f}pp (pass bar <=5pp) -> '
                f'**{"PASS" if rail_pass else "VOID"}**\n\n')
        f.write('## Out-of-sample book (2023-24 + 2025H2-2026-09) — mean R and total $ by delay\n\n')
        f.write(df_to_md(tbl_oos) + '\n\n')
        f.write('## May-Sep 2026 alone — mean R and total $ by delay\n\n')
        f.write(df_to_md(tbl_may) + '\n\n')
        f.write('## 2023-24 (untouched) by delay\n\n' + df_to_md(tbl_2023_24) + '\n\n')
        f.write('## 2025H1 (in-sample, reported not decisioned) by delay\n\n' + df_to_md(tbl_2025h1) + '\n\n')
        f.write('## 2025H2-2026-09 (tuned-on) by delay\n\n' + df_to_md(tbl_2025h2) + '\n\n')
        f.write('## Curve — mean R by delay, all periods side by side\n\n')
        curve = pd.DataFrame({'delay_s': DELAY_GRID,
                               '2023-24': tbl_2023_24.mean_R.values,
                               '2025H1': tbl_2025h1.mean_R.values,
                               '2025H2-2026-09': tbl_2025h2.mean_R.values,
                               'out_of_sample': tbl_oos.mean_R.values,
                               'May-Sep_2026': tbl_may.mean_R.values})
        f.write(df_to_md(curve) + '\n\n')
        f.write('## Lenses\n\n')
        f.write(f'(i) d=0 vs BT 30bps entry model: {lens1_frac:.1%} of filled d=0 replay prices agree with '
                f'the BT entry_price within 1c; mean(replay - BT) = ${lens1_mean_diff:+.4f}.\n\n')
        f.write(f'(ii) XNAS-only trigger bias: {lens2_frac:.1%} of fills with a d=0 t* have that print '
                f'more than 2s after 09:35:00 (breakout not in the very first XNAS prints of the range close).\n\n')
        f.write(f'(iii) Gap-through cases (|t*-09:35:00|<=1s): n={gap_n}, mean R={gap_mean_r:.3f}.\n\n')
        f.write('## Decision number\n\n')
        if decision_d is not None:
            f.write(f'**Decision number = {decision_d}s** (largest delay with out-of-sample mean R >= 0 AND '
                     f'May-Sep 2026 total $ >= 0). Engineering latency target = {eng_target}s '
                     f'(decision number minus 2s safety).\n')
            if decision_d < 3:
                f.write('\nDecision number < 3s: per PREREG consequence, ORB cannot be made positive by '
                         'latency engineering on this stack.\n')
        else:
            f.write('**No delay in the grid satisfies both conditions — decision number < 0 (i.e. below the '
                     'grid minimum).** Per PREREG consequence (decision number < 3s), ORB cannot be made '
                     'positive by latency engineering on this stack; the owner is told this with the curve above.\n')
    log(f'REPORT_WRITTEN {REPORT_MD}; decision_d={decision_d}')
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest='cmd', required=True)
    sub.add_parser('population')
    p_cg = sub.add_parser('cost-gate')
    p_cg.add_argument('--sample', type=int, default=30)
    p_f = sub.add_parser('fetch')
    p_f.add_argument('--threads', type=int, default=8)
    sub.add_parser('run')
    sub.add_parser('report')
    args = ap.parse_args()
    os.makedirs(RAW_DIR, exist_ok=True)
    if args.cmd == 'population':
        build_population()
        return 0
    if args.cmd == 'cost-gate':
        return cmd_cost_gate(args)
    if args.cmd == 'fetch':
        return cmd_fetch(args)
    if args.cmd == 'run':
        return cmd_run(args)
    if args.cmd == 'report':
        return cmd_report(args)


if __name__ == '__main__':
    sys.exit(main())
