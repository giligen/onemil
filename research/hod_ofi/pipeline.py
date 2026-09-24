#!/usr/bin/env python3
"""HOD-break order-flow imbalance study. PREREG_OFI.md cells 1,393-1,395.
See SPEC.md for the two data-collection clarifications this implements:
  1. Causal window end = first trade print >= break level inside minute entry_m
     (fallback: start of entry_m + 60s, row flagged).
  2. Fetch window = union over all possible 'end' values of [end-5min, end+1min]
     = [entry_m*60 - 300, entry_m*60 + 120] (7 min) for signal windows so 'end'
     can be located from the SAME fetch (no extra API call); placebo windows have
     no location step so they fetch exactly [m_p*60 - 300, m_p*60] (5 min).
     Feature computation only ever looks at [end-5min, end), matching "nothing
     after the break" in PREREG_OFI.md.

Usage:
    python3 pipeline.py cost-gate [--sample 40]
    python3 pipeline.py fetch [--limit N_DAYS]
    python3 pipeline.py features
    python3 pipeline.py score
"""
from __future__ import annotations

import argparse
import glob
import hashlib
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

OUT_DIR = f'{ROOT}/research/hod_ofi'
RAW_DIR = f'{OUT_DIR}/raw'
MANIFEST = f'{OUT_DIR}/manifest.csv'
COST_GATE_JSON = f'{OUT_DIR}/cost_gate_result.json'
FEATURES_OUT = f'{OUT_DIR}/window_features.csv'
REPORT_MD = f'{OUT_DIR}/REPORT.md'

FEATURES_CSV = f'{ROOT}/research/bf_zero/causal_filter/features.csv'
B0_TRADES_CSV = f'{ROOT}/research/hod_exit_lab/b0_trades.csv'

DATASET = 'XNAS.ITCH'
BUDGET = 110.0
ET = 'America/New_York'

SIGNAL_PAD_BEFORE_S = 300   # end can be as early as S -> feat_start = S-300
SIGNAL_PAD_AFTER_S = 120    # end can be as late as S+60 (fallback) -> +60 tail = S+120
PLACEBO_LOOKBACK_S = 300
PLACEBO_LO_M, PLACEBO_HI_M = 600, 840   # 10:00-14:00 ET, minutes since midnight

BLOCK_START_HHMM, BLOCK_END_HHMM = 1325, 2005   # UTC blackout, same convention as run_consol.py


def log(msg):
    print(f"[{datetime.now(timezone.utc).strftime('%H:%M:%S')}] {msg}", flush=True)


# ---------------------------------------------------------------- population
def load_population() -> pd.DataFrame:
    """TRAIN+VAL rows only (TEST sealed, never fetched)."""
    df = pd.read_csv(FEATURES_CSV, usecols=['day', 'symbol', 'entry_m', 'entry', 'split', 'half'])
    pop = df[df['split'].isin(['TRAIN', 'VAL'])].reset_index(drop=True)
    return pop


def placebo_minute(day: str, symbol: str, entry_m: int) -> int:
    """Deterministic seeded random minute in 10:00-14:00 ET, != entry_m +/- 15."""
    key = f'{day}|{symbol}|{entry_m}'
    seed = int(hashlib.sha256(key.encode()).hexdigest()[:8], 16)
    rng = np.random.default_rng(seed)
    for _ in range(200):
        m = int(rng.integers(PLACEBO_LO_M, PLACEBO_HI_M))
        if abs(m - entry_m) > 15:
            return m
    raise RuntimeError(f'could not place placebo minute for {key}')


def build_windows(pop: pd.DataFrame) -> pd.DataFrame:
    """One signal + one placebo window per population row."""
    rows = []
    for r in pop.itertuples():
        s = r.entry_m * 60
        rows.append(dict(day=r.day, symbol=r.symbol, kind='signal', entry_m=r.entry_m,
                          entry_px=r.entry, split=r.split, half=r.half,
                          fetch_start_s=s - SIGNAL_PAD_BEFORE_S, fetch_end_s=s + SIGNAL_PAD_AFTER_S))
        mp = placebo_minute(r.day, r.symbol, r.entry_m)
        ps = mp * 60
        rows.append(dict(day=r.day, symbol=r.symbol, kind='placebo', entry_m=mp,
                          entry_px=np.nan, split=r.split, half=r.half,
                          fetch_start_s=ps - PLACEBO_LOOKBACK_S, fetch_end_s=ps))
    w = pd.DataFrame(rows)
    bad = w['fetch_start_s'] < 0
    if bad.any():
        log(f'WARNING dropping {bad.sum()} windows with fetch_start before midnight ET (pre-market entry_m)')
        w = w[~bad].reset_index(drop=True)
    return w


def et_seconds_to_utc(day: str, sec: float) -> str:
    sec = int(round(sec))
    h, rem = divmod(sec, 3600)
    m, s = divmod(rem, 60)
    ts = pd.Timestamp(f'{day} {h:02d}:{m:02d}:{s:02d}', tz=ET)
    return ts.tz_convert('UTC').strftime('%Y-%m-%dT%H:%M:%S')


def get_client():
    import databento as db
    key = os.environ.get('DATABENTO_API_KEY') or os.environ.get('DATABENTO_KEY')
    assert key, 'no DATABENTO_API_KEY in .env'
    return db.Historical(key)


# ---------------------------------------------------------------- cost gate
def cmd_cost_gate(args):
    pop = load_population()
    windows = build_windows(pop)
    log(f'population={len(pop)} rows (TRAIN+VAL), windows={len(windows)} (signal+placebo)')
    client = get_client()
    rng = np.random.default_rng(42)
    n = min(args.sample, len(windows))
    idx = rng.choice(len(windows), size=n, replace=False)
    sample = windows.iloc[idx]
    total = 0.0
    for i, r in enumerate(sample.itertuples(), 1):
        start = et_seconds_to_utc(r.day, r.fetch_start_s)
        end = et_seconds_to_utc(r.day, r.fetch_end_s)
        c1 = float(client.metadata.get_cost(dataset=DATASET, schema='mbp-1', symbols=[r.symbol],
                                             stype_in='raw_symbol', start=start, end=end))
        c2 = float(client.metadata.get_cost(dataset=DATASET, schema='trades', symbols=[r.symbol],
                                             stype_in='raw_symbol', start=start, end=end))
        total += c1 + c2
        if i % 10 == 0:
            log(f'  priced {i}/{n} running=${total:.4f}')
    avg = total / n
    projected = avg * len(windows)
    verdict = 'PASS' if projected <= BUDGET else 'STOP'
    log(f'COST_GATE sampled={n} avg_per_window=${avg:.6f} total_windows={len(windows)} '
        f'projected=${projected:.2f} budget=${BUDGET} verdict={verdict}')
    with open(COST_GATE_JSON, 'w') as f:
        json.dump(dict(sampled=n, avg_per_window=avg, total_windows=len(windows),
                        projected=projected, budget=BUDGET, verdict=verdict,
                        ts=datetime.now(timezone.utc).isoformat()), f, indent=2)
    if verdict == 'STOP':
        log('STOP: projected cost exceeds budget, report without fetching (per SPEC step 2)')
        return 2
    return 0


# ---------------------------------------------------------------- fetch
def blackout_wait():
    while True:
        now = datetime.now(timezone.utc)
        hhmm = now.hour * 100 + now.minute
        if BLOCK_START_HHMM <= hhmm < BLOCK_END_HHMM:
            secs = (20 * 3600 + 6 * 60) - (now.hour * 3600 + now.minute * 60 + now.second)
            secs = max(secs, 30)
            log(f'BLACKOUT ({hhmm} UTC in [1325,2005)): sleeping {secs}s')
            time.sleep(min(secs, 1800))
            continue
        return


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
            time.sleep(5 * (attempt + 1))


CHUNK_SIZE = 25   # windows per schedulable unit -- keeps one heavy day from hogging a thread


def fetch_chunk(day, chunk_id, chunk_windows, client):
    """Fetch one chunk (<=CHUNK_SIZE windows, both schemas) -> raw/{day}__{chunk_id}.parquet."""
    blackout_wait()
    frames = []
    for r in chunk_windows.itertuples():
        start = et_seconds_to_utc(day, r.fetch_start_s)
        end = et_seconds_to_utc(day, r.fetch_end_s)
        for schema in ('mbp-1', 'trades'):
            try:
                df = fetch_one(client, r.symbol, start, end, schema)
            except Exception as ex:
                log(f'  ERROR {day} {r.symbol} {r.kind} {schema} giving up: {ex}')
                continue
            if df is not None and len(df):
                keep = [c for c in ('ts_event', 'price', 'size', 'bid_px_00', 'ask_px_00',
                                     'bid_sz_00', 'ask_sz_00') if c in df.columns]
                df = df[keep].copy()
                df['symbol_win'] = r.symbol
                df['kind'] = r.kind
                df['entry_m'] = r.entry_m
                df['schema'] = schema
                frames.append(df)
    n = 0
    if frames:
        out = pd.concat(frames, ignore_index=True)
        out.to_parquet(f'{RAW_DIR}/{day}__{chunk_id:04d}.parquet', index=False, compression='zstd')
        n = len(out)
    return day, chunk_id, n


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


def build_chunks(windows: pd.DataFrame, limit_days=0):
    """[(day, chunk_id, chunk_df), ...]; chunk_id restarts at 0 per day."""
    days = sorted(windows['day'].unique())
    if limit_days:
        days = days[:limit_days]
    chunks = []
    for d in days:
        dw = windows[windows.day == d].reset_index(drop=True)
        for i in range(0, len(dw), CHUNK_SIZE):
            chunks.append((d, i // CHUNK_SIZE, dw.iloc[i:i + CHUNK_SIZE]))
    return days, chunks


def cmd_fetch(args):
    gate = json.load(open(COST_GATE_JSON)) if os.path.exists(COST_GATE_JSON) else None
    if not gate or gate.get('verdict') != 'PASS':
        log('REFUSING to fetch: cost gate has not PASSed (run cost-gate first)')
        return 2
    pop = load_population()
    windows = build_windows(pop)
    days, chunks = build_chunks(windows, args.limit)
    manifest = load_manifest()
    done_units = set(manifest[manifest.status == 'done']['unit']) if len(manifest) else set()
    pending = [(d, cid, cw) for d, cid, cw in chunks
               if f'{d}__{cid:04d}' not in done_units
               or not os.path.exists(f'{RAW_DIR}/{d}__{cid:04d}.parquet')]
    log(f'fetch plan: {len(days)} days, {len(chunks)} chunks total, {len(pending)} pending '
        f'(chunk={CHUNK_SIZE} windows, resumable manifest={MANIFEST})')
    client = get_client()
    completed = len(chunks) - len(pending)
    with ThreadPoolExecutor(max_workers=int(os.environ.get('OFI_THREADS', '6'))) as ex:  # main session 9/24: env-tunable
        futs = {ex.submit(fetch_chunk, d, cid, cw, client): f'{d}__{cid:04d}' for d, cid, cw in pending}
        for fut in as_completed(futs):
            unit = futs[fut]
            try:
                day, cid, n = fut.result()
                append_manifest(unit, 'done', n)
                completed += 1
                if completed % 20 == 0 or completed == len(chunks):
                    log(f'CHUNK_DONE {unit} rows={n} ({completed}/{len(chunks)})')
            except Exception as ex:
                append_manifest(unit, 'error', 0)
                log(f'CHUNK_ERROR {unit}: {type(ex).__name__}: {ex}')
    log('FETCH_COMPLETE')
    return 0


# ---------------------------------------------------------------- features
def to_et_sec(ts: pd.Series) -> pd.Series:
    t = pd.to_datetime(ts, utc=True).dt.tz_convert(ET)
    return t.dt.hour * 3600 + t.dt.minute * 60 + t.dt.second + t.dt.microsecond / 1e6


def ofi_updates(mbp: pd.DataFrame) -> np.ndarray:
    """Cont-Kukanov-Stoikov L1 OFI per update; +ve = buy pressure."""
    b, a = mbp['bid_px_00'].to_numpy(), mbp['ask_px_00'].to_numpy()
    bs, as_ = mbp['bid_sz_00'].to_numpy(float), mbp['ask_sz_00'].to_numpy(float)
    b_p, a_p = np.roll(b, 1), np.roll(a, 1)
    bs_p, as_p = np.roll(bs, 1), np.roll(as_, 1)
    bid_e = np.where(b > b_p, bs, np.where(b == b_p, bs - bs_p, -bs_p))
    ask_e = np.where(a > a_p, -as_p, np.where(a == a_p, as_ - as_p, as_p))
    e = bid_e + ask_e
    if len(e):
        e[0] = 0.0
    return e


def compute_window_features(mbp: pd.DataFrame, trades: pd.DataFrame, kind: str, entry_m: int,
                             entry_px: float) -> dict:
    """Everything from [end-5min, end); nothing after (PREREG). end located per SPEC clarification 1."""
    S = entry_m * 60
    locate_fallback = False
    if kind == 'signal':
        cand = trades[(trades.sec >= S) & (trades.sec < S + 60) & (trades.price >= entry_px)]
        if len(cand):
            end = float(cand.sort_values('sec').iloc[0].sec)
        else:
            end = float(S + 60)
            locate_fallback = True
    else:
        end = float(S)  # placebo: end = m_p's start, no search

    feat_start = end - 300
    # Narrow to only the columns each schema owns: the raw parquet is a union-schema concat
    # (mbp-1 rows carry NaN price/size, trades rows carry NaN bid/ask), so an un-narrowed
    # merge_asof below would collide on bid_px_00/ask_px_00 and get silently suffixed.
    mbp_w = mbp[(mbp.sec >= feat_start) & (mbp.sec < end)].sort_values('sec')[
        ['sec', 'bid_px_00', 'ask_px_00', 'bid_sz_00', 'ask_sz_00']]
    trades_w = trades[(trades.sec >= feat_start) & (trades.sec < end)].sort_values('sec')[
        ['sec', 'price', 'size']]

    n_win_s = 300
    coverage_frac = mbp_w['sec'].astype(int).nunique() / n_win_s if len(mbp_w) else 0.0

    if len(mbp_w) >= 2:
        e = ofi_updates(mbp_w)
        depth = ((mbp_w['bid_sz_00'] + mbp_w['ask_sz_00']) / 2).to_numpy(float)
        mean_depth = np.nanmean(depth) if len(depth) else np.nan
        ofi_5 = float(np.nansum(e) / mean_depth) if mean_depth else np.nan
        last60 = mbp_w['sec'].to_numpy() >= (end - 60)
        ofi_1 = float(np.nansum(e[last60]) / mean_depth) if mean_depth and last60.any() else np.nan
    else:
        ofi_5 = ofi_1 = np.nan

    if len(trades_w) and len(mbp_w):
        m = pd.merge_asof(trades_w, mbp_w[['sec', 'bid_px_00', 'ask_px_00']], on='sec', direction='backward')
        mid = (m['bid_px_00'] + m['ask_px_00']) / 2
        sign = np.select([m['price'] > mid, m['price'] < mid], [1, -1], default=0)
        tot = m['size'].sum()
        tsi_5 = float((sign * m['size']).sum() / tot) if tot else np.nan
    else:
        tsi_5 = np.nan

    last_q = mbp[mbp.sec <= end].sort_values('sec').tail(1)
    if len(last_q):
        bid, ask = float(last_q['bid_px_00'].iloc[0]), float(last_q['ask_px_00'].iloc[0])
        mid = (bid + ask) / 2
        spread_bps = (ask - bid) / mid * 1e4 if mid else np.nan
    else:
        spread_bps = np.nan

    return dict(OFI_5=ofi_5, OFI_1=ofi_1, TSI_5=tsi_5, spread_bps_at_break=spread_bps,
                coverage_frac=coverage_frac, locate_fallback=locate_fallback, end_sec=end)


def cmd_features(args):
    pop = load_population()
    windows = build_windows(pop)
    out_rows = []
    for day, day_windows in windows.groupby('day'):
        paths = sorted(glob.glob(f'{RAW_DIR}/{day}__*.parquet')) or \
            ([f'{RAW_DIR}/{day}.parquet'] if os.path.exists(f'{RAW_DIR}/{day}.parquet') else [])
        if not paths:
            for r in day_windows.itertuples():
                out_rows.append(dict(day=day, symbol=r.symbol, kind=r.kind, entry_m=r.entry_m,
                                      split=r.split, half=r.half, OFI_5=np.nan, OFI_1=np.nan,
                                      TSI_5=np.nan, spread_bps_at_break=np.nan, coverage_frac=0.0,
                                      locate_fallback=None, missing_raw=True))
            continue
        raw = pd.concat([pd.read_parquet(p) for p in paths], ignore_index=True)
        raw['sec'] = to_et_sec(raw['ts_event'])
        for r in day_windows.itertuples():
            sub = raw[(raw.symbol_win == r.symbol) & (raw.kind == r.kind) & (raw.entry_m == r.entry_m)]
            mbp = sub[sub.schema == 'mbp-1']
            trades = sub[sub.schema == 'trades']
            feat = compute_window_features(mbp, trades, r.kind, r.entry_m, r.entry_px)
            feat.update(day=day, symbol=r.symbol, kind=r.kind, entry_m=r.entry_m,
                        split=r.split, half=r.half, missing_raw=(len(mbp) == 0 and len(trades) == 0))
            out_rows.append(feat)
        log(f'features {day}: {len(day_windows)} windows')
    pd.DataFrame(out_rows).to_csv(FEATURES_OUT, index=False)
    log(f'FEATURES_COMPLETE rows={len(out_rows)} -> {FEATURES_OUT}')


# ---------------------------------------------------------------- score
def cmd_score(args):
    sys.path.insert(0, f'{ROOT}/research/hod_consol')
    from adversarial_read import stats  # noqa: E402
    from run_consol import simulate_slots  # noqa: E402

    feat = pd.read_csv(FEATURES_OUT)
    b0 = pd.read_csv(B0_TRADES_CSV)
    sig = feat[feat.kind == 'signal'].merge(
        b0[['day', 'symbol', 'entry_m', 'exit_m', 'net_R', 'wk']], on=['day', 'symbol', 'entry_m'], how='inner')
    plc = feat[feat.kind == 'placebo']

    usable = sig['coverage_frac'] >= 0.60
    winners = sig['net_R'] > 0
    miss_w = (~usable[winners]).mean() if winners.any() else np.nan
    miss_l = (~usable[~winners]).mean() if (~winners).any() else np.nan
    gap_pp = abs(miss_w - miss_l) * 100 if pd.notna(miss_w) and pd.notna(miss_l) else np.nan
    avail_status = 'OK' if (usable.mean() >= 0.80 and gap_pp <= 5) else 'VOID'
    log(f'AVAILABILITY coverage_usable={usable.mean():.3f} miss_winner={miss_w:.3f} '
        f'miss_loser={miss_l:.3f} gap_pp={gap_pp:.2f} status={avail_status}')

    sig_u = sig[usable].copy()
    cells = {'F1_OFI5': 'OFI_5', 'F2_OFI1': 'OFI_1', 'F3_TSI5': 'TSI_5'}
    report_lines = [f'# HOD-OFI REPORT\ncoverage_usable={usable.mean():.3f} avail={avail_status} '
                     f'gap_pp={gap_pp:.2f} n_signal={len(sig)} n_usable={len(sig_u)}\n']
    summary = []
    for cell, col in cells.items():
        tr_h1 = sig_u[(sig_u.split == 'TRAIN') & (sig_u.half == 'H1')]
        cut = tr_h1[col].median()
        sig_u['_keep'] = sig_u[col] >= cut

        def cohort_stat(df, keep_val):
            sub = df[df['_keep'] == keep_val]
            return sub, stats(sub['net_R'], sub['day']) if len(sub) >= 3 else dict(n=len(sub), mean=np.nan, t_cluster=np.nan)

        rows = {}
        for split_name, mask in [('TRAIN-H2', (sig_u.split == 'TRAIN') & (sig_u.half == 'H2')),
                                  ('VAL', sig_u.split == 'VAL')]:
            d = sig_u[mask]
            keep_d, keep_s = cohort_stat(d, True)
            drop_d, drop_s = cohort_stat(d, False)
            lift = (keep_s.get('mean', np.nan) or np.nan) - (drop_s.get('mean', np.nan) or np.nan)
            se_k = (keep_s['mean'] / keep_s['t_cluster']) if keep_s.get('t_cluster') else np.nan
            se_d = (drop_s['mean'] / drop_s['t_cluster']) if drop_s.get('t_cluster') else np.nan
            t_lift = lift / np.sqrt(se_k ** 2 + se_d ** 2) if pd.notna(se_k) and pd.notna(se_d) else np.nan
            rho, _ = (np.nan, np.nan)
            try:
                from scipy.stats import spearmanr
                rho, _ = spearmanr(d[col], d['net_R'])
            except Exception:
                pass
            rows[split_name] = dict(n=len(d), keep_n=len(keep_d), drop_n=len(drop_d), lift=lift,
                                     t_lift=t_lift, spearman=rho)
        keep_all = sig_u[sig_u['_keep']][['day', 'entry_m', 'exit_m']].dropna()
        n_weeks = sig_u['wk'].nunique() if 'wk' in sig_u else np.nan
        fills_wk = np.nan
        if len(keep_all) and pd.notna(n_weeks) and n_weeks:
            keep_trades = sig_u[sig_u['_keep']].dropna(subset=['exit_m'])
            kept_slots = simulate_slots(keep_trades.rename(columns={'entry_m': 'entry_m', 'exit_m': 'exit_m'}))
            fills_wk = kept_slots.sum() / n_weeks
        val = sig_u[sig_u.split == 'VAL']
        val_sorted = val[val['_keep']].sort_values('net_R', ascending=False)
        trim_n = max(1, int(round(0.05 * len(val_sorted))))
        ex_top5_mean = val_sorted.iloc[trim_n:]['net_R'].mean() if len(val_sorted) > trim_n else np.nan
        drop_val_mean = val[~val['_keep']]['net_R'].mean() if len(val[~val['_keep']]) else np.nan
        ex_top5_lift = ex_top5_mean - drop_val_mean if pd.notna(ex_top5_mean) and pd.notna(drop_val_mean) else np.nan

        rho_pl, _ = (np.nan, np.nan)
        try:
            from scipy.stats import spearmanr
            plc_j = plc.merge(sig[['day', 'symbol', 'net_R']], on=['day', 'symbol'], how='inner')
            rho_pl, _ = spearmanr(plc_j[col], plc_j['net_R']) if len(plc_j) > 5 else (np.nan, np.nan)
        except Exception:
            pass

        pass_bar = (rows['TRAIN-H2']['lift'] >= 0.10 and rows['VAL']['lift'] >= 0.10 and
                    rows['VAL']['t_lift'] >= 2 and fills_wk >= 3 and
                    rows['TRAIN-H2']['spearman'] >= 0.6 and rows['VAL']['spearman'] >= 0.6 and
                    pd.notna(ex_top5_lift) and ex_top5_lift >= 0)
        verdict = 'PASS' if pass_bar else 'FAIL'
        log(f'{cell} cut={cut:.4f} TRAINH2_lift={rows["TRAIN-H2"]["lift"]:.3f} VAL_lift={rows["VAL"]["lift"]:.3f} '
            f'VAL_t={rows["VAL"]["t_lift"]:.2f} fills/wk={fills_wk:.2f} rho_TH2={rows["TRAIN-H2"]["spearman"]:.2f} '
            f'rho_VAL={rows["VAL"]["spearman"]:.2f} placebo_rho={rho_pl:.2f} ex_top5_lift={ex_top5_lift:.3f} '
            f'verdict={verdict}')
        summary.append(dict(cell=cell, cut=cut, **rows, fills_wk=fills_wk, placebo_rho=rho_pl,
                             ex_top5_lift=ex_top5_lift, verdict=verdict))
        report_lines.append(f'## {cell} ({col})\ncut(TRAIN-H1 median)={cut:.4f}\n'
                             f'TRAIN-H2: n={rows["TRAIN-H2"]["n"]} lift={rows["TRAIN-H2"]["lift"]:.3f} '
                             f'spearman={rows["TRAIN-H2"]["spearman"]:.3f}\n'
                             f'VAL: n={rows["VAL"]["n"]} lift={rows["VAL"]["lift"]:.3f} '
                             f't={rows["VAL"]["t_lift"]:.2f} spearman={rows["VAL"]["spearman"]:.3f}\n'
                             f'fills/week={fills_wk:.2f} placebo_spearman={rho_pl:.3f} '
                             f'ex_top5_lift={ex_top5_lift:.3f}\nVERDICT={verdict}\n')

    with open(REPORT_MD, 'w') as f:
        f.write('\n'.join(report_lines))
    with open(f'{OUT_DIR}/score_summary.json', 'w') as f:
        json.dump(summary, f, indent=2, default=float)
    log(f'SCORE_COMPLETE -> {REPORT_MD}')


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd', required=True)
    p1 = sub.add_parser('cost-gate'); p1.add_argument('--sample', type=int, default=40)
    p2 = sub.add_parser('fetch'); p2.add_argument('--limit', type=int, default=0)
    sub.add_parser('features')
    sub.add_parser('score')
    args = ap.parse_args()
    fn = {'cost-gate': cmd_cost_gate, 'fetch': cmd_fetch, 'features': cmd_features,
          'score': cmd_score}[args.cmd]
    return fn(args) or 0


if __name__ == '__main__':
    raise SystemExit(main())
