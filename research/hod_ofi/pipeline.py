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
                          sig_entry_m=r.entry_m,
                          entry_px=r.entry, split=r.split, half=r.half,
                          fetch_start_s=s - SIGNAL_PAD_BEFORE_S, fetch_end_s=s + SIGNAL_PAD_AFTER_S))
        mp = placebo_minute(r.day, r.symbol, r.entry_m)
        ps = mp * 60
        # sig_entry_m ties this placebo back to the ONE signal it was drawn for --
        # a name-day can have several signals, so (day, symbol) alone is ambiguous.
        rows.append(dict(day=r.day, symbol=r.symbol, kind='placebo', entry_m=mp,
                          sig_entry_m=r.entry_m,
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


def _coverage_frac(mbp: pd.DataFrame, feat_start: float, end: float) -> float:
    """Share of the 300 window seconds s in [end-300, end) at whose end (instant
    min(s+1, end)) the prevailing mbp-1 record -- the last record at or before that
    instant, including records fetched before feat_start -- is two-sided (bid>0,
    ask>0, ask>=bid). No prior record at all -> that second is unquoted."""
    avail = mbp[mbp['sec'] <= end].sort_values('sec', kind='mergesort')
    if len(avail) == 0:
        return 0.0
    secs = avail['sec'].to_numpy(dtype=float)
    bid = avail['bid_px_00'].to_numpy(dtype=float)
    ask = avail['ask_px_00'].to_numpy(dtype=float)
    s_grid = feat_start + np.arange(300)
    instants = np.minimum(s_grid + 1, end)
    idx = np.searchsorted(secs, instants, side='right') - 1
    quoted = idx >= 0
    idx_c = np.clip(idx, 0, len(secs) - 1)
    b = np.where(quoted, bid[idx_c], np.nan)
    a = np.where(quoted, ask[idx_c], np.nan)
    two_sided = quoted & (b > 0) & (a > 0) & (a >= b)
    return float(two_sided.mean())


def ofi_updates(mbp: pd.DataFrame) -> np.ndarray:
    """Cont-Kukanov-Stoikov L1 OFI per update; +ve = buy pressure."""
    b, a = mbp['bid_px_00'].to_numpy(), mbp['ask_px_00'].to_numpy()
    bs, as_ = mbp['bid_sz_00'].to_numpy(float), mbp['ask_sz_00'].to_numpy(float)
    b_p, a_p = np.roll(b, 1), np.roll(a, 1)
    bs_p, as_p = np.roll(bs, 1), np.roll(as_, 1)
    bid_e = np.where(b > b_p, bs, np.where(b == b_p, bs - bs_p, -bs_p))
    ask_e = np.where(a > a_p, as_p, np.where(a == a_p, as_p - as_, -as_))
    e = bid_e + ask_e
    if len(e):
        e[0] = 0.0
    return e


def compute_window_features(mbp: pd.DataFrame, trades: pd.DataFrame, kind: str, entry_m: int,
                             entry_px: float) -> dict:
    """Everything from [end-5min, end); nothing after (PREREG). end located per SPEC clarification 1."""
    S = entry_m * 60
    # Decision instant = S for BOTH kinds (defect 7). The HOD-break book enters at the OPEN of bar entry_m after the
    # break bar (entry_m - 1) CLOSES (research/bf_zero/spread_study.py:48); the PREREG window is "before the break
    # bar closes". Run 1 ended at the first print >= entry inside minute entry_m -- up to 60 s after the decision.
    end = float(S)
    locate_fallback = False
    if kind == 'signal':
        # Diagnostic only, never moves `end`: no XNAS print at/through the entry price inside the entry minute.
        locate_fallback = not bool(((trades.sec >= S) & (trades.sec < S + 60) & (trades.price >= entry_px)).any())

    feat_start = end - 300
    # Narrow to only the columns each schema owns: the raw parquet is a union-schema concat
    # (mbp-1 rows carry NaN price/size, trades rows carry NaN bid/ask), so an un-narrowed
    # merge_asof below would collide on bid_px_00/ask_px_00 and get silently suffixed.
    mbp_w = mbp[(mbp.sec >= feat_start) & (mbp.sec < end)].sort_values('sec', kind='mergesort')[
        ['sec', 'bid_px_00', 'ask_px_00', 'bid_sz_00', 'ask_sz_00']]
    trades_w = trades[(trades.sec >= feat_start) & (trades.sec < end)].sort_values('sec', kind='mergesort')[
        ['sec', 'price', 'size']]

    coverage_frac = _coverage_frac(mbp, feat_start, end)

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
        m = pd.merge_asof(trades_w, mbp_w[['sec', 'bid_px_00', 'ask_px_00']], on='sec', direction='backward',
                           allow_exact_matches=False)
        mid = (m['bid_px_00'] + m['ask_px_00']) / 2
        sign = np.select([m['price'] > mid, m['price'] < mid], [1, -1], default=0)
        tot = m['size'].sum()
        tsi_5 = float((sign * m['size']).sum() / tot) if tot else np.nan
    else:
        tsi_5 = np.nan

    last_q = mbp[mbp.sec <= end].sort_values('sec', kind='mergesort').tail(1)
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
    days_arg = (getattr(args, 'days', '') or '').strip()
    if days_arg:
        days_filter = [d.strip() for d in days_arg.split(',') if d.strip()]
        windows = windows[windows['day'].isin(days_filter)].reset_index(drop=True)
        log(f'features: SMOKE limited to days={days_filter} ({len(windows)} windows)')
    out_path = (getattr(args, 'out', '') or '').strip() or FEATURES_OUT
    out_rows = []
    for day, day_windows in windows.groupby('day'):
        paths = sorted(glob.glob(f'{RAW_DIR}/{day}__*.parquet')) or \
            ([f'{RAW_DIR}/{day}.parquet'] if os.path.exists(f'{RAW_DIR}/{day}.parquet') else [])
        if not paths:
            for r in day_windows.itertuples():
                out_rows.append(dict(day=day, symbol=r.symbol, kind=r.kind, entry_m=r.entry_m,
                                      sig_entry_m=r.sig_entry_m,
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
            feat.update(day=day, symbol=r.symbol, kind=r.kind, entry_m=r.entry_m, sig_entry_m=r.sig_entry_m,
                        split=r.split, half=r.half, missing_raw=(len(mbp) == 0 and len(trades) == 0))
            out_rows.append(feat)
        log(f'features {day}: {len(day_windows)} windows')
    pd.DataFrame(out_rows).to_csv(out_path, index=False)
    log(f'FEATURES_COMPLETE rows={len(out_rows)} -> {out_path}')


# ---------------------------------------------------------------- score
def _keep_lift_ols(d: pd.DataFrame) -> dict:
    """net_R ~ 1 + keep by OLS; lift == keep coefficient (== keep_mean - drop_mean).
    t_lift = day-clustered t (statsmodels cov_type='cluster', groups=day); t_iid beside it."""
    import statsmodels.api as sm
    d = d.dropna(subset=['net_R', '_keep', 'day'])
    if len(d) < 3 or d['_keep'].nunique() < 2:
        return dict(n=len(d), lift=np.nan, t_lift=np.nan, t_iid=np.nan)
    X = sm.add_constant(d['_keep'].astype(float), has_constant='add')
    y = d['net_R'].astype(float)
    m_cluster = sm.OLS(y, X).fit(cov_type='cluster', cov_kwds={'groups': d['day']})
    m_iid = sm.OLS(y, X).fit()
    return dict(n=len(d), lift=float(m_cluster.params['_keep']),
                t_lift=float(m_cluster.tvalues['_keep']), t_iid=float(m_iid.tvalues['_keep']))


def _decile_table(d: pd.DataFrame, col: str) -> tuple:
    """Decile-mean-net-R table (qcut of `col` within this holdout, duplicates='drop') and the
    pass-bar Spearman: decile index (1..k) vs the decile's mean net R -- NOT the trade-level rho."""
    dd = d.dropna(subset=[col, 'net_R']).copy()
    if len(dd) < 10:
        return pd.DataFrame(columns=['decile', 'n', 'mean_net_R']), np.nan
    dd['decile'] = pd.qcut(dd[col], 10, labels=False, duplicates='drop') + 1
    tbl = dd.groupby('decile').agg(n=('net_R', 'size'), mean_net_R=('net_R', 'mean')).reset_index()
    rho = tbl['decile'].corr(tbl['mean_net_R'], method='spearman') if len(tbl) >= 3 else np.nan
    return tbl, rho


def _corrections_section() -> str:
    """Verbatim defects-1-6 section from SPEC_FIX.md, for the REPORT.md header."""
    spec = open(f'{OUT_DIR}/SPEC_FIX.md').read()
    return spec[spec.index('## Defects'):spec.index('## Deliverables')].strip()


def cmd_score(args):
    from scipy.stats import spearmanr
    sys.path.insert(0, f'{ROOT}/research/hod_consol')
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
    fallback_share = sig['locate_fallback'].mean()
    log(f'AVAILABILITY coverage_usable={usable.mean():.3f} miss_winner={miss_w:.3f} '
        f'miss_loser={miss_l:.3f} gap_pp={gap_pp:.2f} status={avail_status} fallback_share={fallback_share:.3f}')

    sig_u = sig[usable].copy()
    cells = {'F1_OFI5': 'OFI_5', 'F2_OFI1': 'OFI_1', 'F3_TSI5': 'TSI_5'}
    report_lines = [f'# HOD-OFI REPORT\ncoverage_usable={usable.mean():.3f} avail={avail_status} '
                     f'gap_pp={gap_pp:.2f} fallback_share={fallback_share:.3f} '
                     f'n_signal={len(sig)} n_usable={len(sig_u)}\n',
                     '## Corrections vs run 1\n' + _corrections_section() + '\n']
    summary = []
    for cell, col in cells.items():
        d_cell = sig_u.dropna(subset=[col]).copy()  # defect 6: drop NaN feature rows before cut/rank/Spearman
        tr_h1 = d_cell[(d_cell.split == 'TRAIN') & (d_cell.half == 'H1')]
        cut = tr_h1[col].median()
        d_cell['_keep'] = d_cell[col] >= cut

        rows, deciles = {}, {}
        for split_name, mask in [('TRAIN-H2', (d_cell.split == 'TRAIN') & (d_cell.half == 'H2')),
                                  ('VAL', d_cell.split == 'VAL')]:
            d = d_cell[mask]
            stat = _keep_lift_ols(d)
            dec_tbl, dec_rho = _decile_table(d, col)
            trade_rho, _ = spearmanr(d[col], d['net_R']) if len(d) > 5 else (np.nan, np.nan)
            rows[split_name] = dict(n=stat['n'], keep_n=int(d['_keep'].sum()), drop_n=int((~d['_keep']).sum()),
                                     lift=stat['lift'], t_lift=stat['t_lift'], t_iid=stat['t_iid'],
                                     spearman=dec_rho, trade_rho=trade_rho)
            deciles[split_name] = dec_tbl
        n_weeks = d_cell['wk'].nunique() if 'wk' in d_cell else np.nan
        fills_wk = np.nan
        if pd.notna(n_weeks) and n_weeks:
            keep_trades = d_cell[d_cell['_keep']].dropna(subset=['exit_m'])
            if len(keep_trades):
                fills_wk = simulate_slots(keep_trades).sum() / n_weeks
        val = d_cell[d_cell.split == 'VAL']
        val_sorted = val[val['_keep']].sort_values('net_R', ascending=False, kind='mergesort')
        trim_n = max(1, int(round(0.05 * len(val_sorted))))
        ex_top5_mean = val_sorted.iloc[trim_n:]['net_R'].mean() if len(val_sorted) > trim_n else np.nan
        drop_val_mean = val[~val['_keep']]['net_R'].mean() if len(val[~val['_keep']]) else np.nan
        ex_top5_lift = ex_top5_mean - drop_val_mean if pd.notna(ex_top5_mean) and pd.notna(drop_val_mean) else np.nan

        # placebo: cut at ITS OWN TRAIN-H1 median; lift = kept-dropped mean net R of the
        # matched signal's trade (joined on day, symbol, sig_entry_m), TRAIN-H2 & VAL only.
        plc_c = plc.dropna(subset=[col]).copy()
        plc_tr_h1 = plc_c[(plc_c.split == 'TRAIN') & (plc_c.half == 'H1')]
        plc_cut = plc_tr_h1[col].median()
        plc_c['_keep'] = plc_c[col] >= plc_cut
        plc_j = plc_c.merge(sig[['day', 'symbol', 'entry_m', 'net_R']].rename(columns={'entry_m': 'sig_entry_m'}),
                             on=['day', 'symbol', 'sig_entry_m'], how='inner')
        rho_pl, _ = spearmanr(plc_j[col], plc_j['net_R']) if len(plc_j) > 5 else (np.nan, np.nan)
        placebo_lift = {}
        for split_name, mask in [('TRAIN-H2', (plc_j.split == 'TRAIN') & (plc_j.half == 'H2')),
                                  ('VAL', plc_j.split == 'VAL')]:
            dd = plc_j[mask]
            k = dd[dd['_keep']]['net_R'].mean() if dd['_keep'].any() else np.nan
            dr = dd[~dd['_keep']]['net_R'].mean() if (~dd['_keep']).any() else np.nan
            placebo_lift[split_name] = (k - dr) if pd.notna(k) and pd.notna(dr) else np.nan

        pass_bar = (rows['TRAIN-H2']['lift'] >= 0.10 and rows['VAL']['lift'] >= 0.10 and
                    rows['VAL']['t_lift'] >= 2 and pd.notna(fills_wk) and fills_wk >= 3 and
                    pd.notna(rows['TRAIN-H2']['spearman']) and rows['TRAIN-H2']['spearman'] >= 0.6 and
                    pd.notna(rows['VAL']['spearman']) and rows['VAL']['spearman'] >= 0.6 and
                    pd.notna(ex_top5_lift) and ex_top5_lift >= 0 and
                    pd.notna(placebo_lift.get('VAL')) and placebo_lift['VAL'] < 0.10)
        verdict = 'PASS' if pass_bar else 'FAIL'
        log(f'{cell} cut={cut:.4f} TRAINH2_lift={rows["TRAIN-H2"]["lift"]:.3f} VAL_lift={rows["VAL"]["lift"]:.3f} '
            f'VAL_t_cluster={rows["VAL"]["t_lift"]:.2f} VAL_t_iid={rows["VAL"]["t_iid"]:.2f} fills/wk={fills_wk:.2f} '
            f'decile_rho_TH2={rows["TRAIN-H2"]["spearman"]:.2f} decile_rho_VAL={rows["VAL"]["spearman"]:.2f} '
            f'placebo_rho={rho_pl:.2f} placebo_VAL_lift={placebo_lift.get("VAL", np.nan):.3f} '
            f'ex_top5_lift={ex_top5_lift:.3f} verdict={verdict}')
        summary.append(dict(cell=cell, cut=cut, **rows, fills_wk=fills_wk, placebo_rho=rho_pl,
                             placebo_lift=placebo_lift, ex_top5_lift=ex_top5_lift, verdict=verdict))
        report_lines.append(
            f'## {cell} ({col})\ncut(TRAIN-H1 median)={cut:.4f}  placebo_cut(own TRAIN-H1 median)={plc_cut:.4f}\n'
            f'TRAIN-H2: n={rows["TRAIN-H2"]["n"]} lift={rows["TRAIN-H2"]["lift"]:.3f} '
            f'decile_spearman={rows["TRAIN-H2"]["spearman"]:.3f} trade_spearman(info)={rows["TRAIN-H2"]["trade_rho"]:.3f}\n'
            f'VAL: n={rows["VAL"]["n"]} lift={rows["VAL"]["lift"]:.3f} t_cluster={rows["VAL"]["t_lift"]:.2f} '
            f't_iid={rows["VAL"]["t_iid"]:.2f} decile_spearman={rows["VAL"]["spearman"]:.3f} '
            f'trade_spearman(info)={rows["VAL"]["trade_rho"]:.3f}\n'
            f'fills/week={fills_wk:.2f} placebo_trade_spearman={rho_pl:.3f} '
            f'placebo_lift TRAIN-H2={placebo_lift.get("TRAIN-H2", np.nan):.3f} VAL={placebo_lift.get("VAL", np.nan):.3f} '
            f'(PASS additionally needs placebo VAL lift < 0.10)\n'
            f'ex_top5_lift={ex_top5_lift:.3f}\nVERDICT={verdict}\n')
        for split_name in ('TRAIN-H2', 'VAL'):
            report_lines.append(f'### {cell} decile table -- {split_name} (n, mean net R)\n' +
                                 deciles[split_name].to_string(index=False) + '\n')

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
    p3 = sub.add_parser('features')
    p3.add_argument('--days', type=str, default='', help='comma-separated days, limits the run (smoke test)')
    p3.add_argument('--out', type=str, default='', help='override output CSV path (smoke test)')
    sub.add_parser('score')
    args = ap.parse_args()
    fn = {'cost-gate': cmd_cost_gate, 'fetch': cmd_fetch, 'features': cmd_features,
          'score': cmd_score}[args.cmd]
    return fn(args) or 0


if __name__ == '__main__':
    raise SystemExit(main())
