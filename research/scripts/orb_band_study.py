"""ORB universe-widening study: does $3-60 beat $3-30? (2026-07-05, W8 follow-up)

Owner directive: keep the system SIMPLE. The preferred ship is B1 — widen
the existing universe's max open price from $30 to $60 and let the new
candidates flow through the UNCHANGED frozen pipeline (one config knob,
same account, same slots, same fit). B2 (band-own fit) is computed only
as a reference to show whether separateness would buy anything.

Variants (all use static_lock_1R + touchgo + PDR veto, the shipped stack):
  BASE : existing $3-30 candidates, frozen H1-2025 fit  (known ~$210K)
  B1   : $3-30 + $30-60 candidates, SAME frozen fit + cutoffs + mults
         (fits computed on the $3-30 TRAIN only — exactly what production
         would do: its params don't change when the universe widens)
  B2   : $30-60 candidates alone, band-own H1-2025 fit (reference)

Pre-declared ship bar for B1 (set BEFORE looking at results):
  - B1 minus BASE positive in BOTH 2025H2 and 2026 eras
  - combined OOS lift >= +$15K on the $100K model
  - B1 MDD <= $25K
  - new-band daily P&L correlation with base ORB <= 0.5 (else it's the
    same bet levered, not new edge)

Data hygiene: gap capped at 50% in the new band (a $30+ stock gapping
>50% is a split artifact / halt situation until proven otherwise).

Usage:
  PYTHONPATH=/home/ec2-user/onemil python3 research/scripts/orb_band_study.py
Artifacts:
  /tmp/orb_band_30_60_candidates.csv   (new-band candidate resims)
  /tmp/orb_band_study_report.txt       (full report, also printed)
"""
from __future__ import annotations

import os
import sqlite3
import sys
from datetime import datetime, date, timedelta, timezone

import numpy as np
import pandas as pd

ROOT = '/home/ec2-user/onemil'
sys.path.insert(0, ROOT)
sys.path.insert(0, '/tmp')

from dotenv import load_dotenv
load_dotenv(os.path.join(ROOT, '.env'))

from persistence.database import Database
from study_orb_filter import FILTER_FEATURES, fit_z_params, composite_score
from study_orb_sizing import (
    FILTER_THRESHOLD, fit_quintile_cutoffs, assign_quintile, ADAPTIVE_MULT_MIN,
)
from trading.orb_correlation import symbol_family, symbol_super_group
from trading.orb_touchgo_filter import (
    evaluate_rule_m, evaluate_rule_d, load_touchgo_config,
)

BAND_MIN = float(os.environ.get('ORB_BAND_MIN', 30.0))
BAND_MAX = float(os.environ.get('ORB_BAND_MAX', 60.0))
GAP_MIN_PCT = 5.0
GAP_CAP_PCT = float(os.environ.get('ORB_BAND_GAP_CAP', 50.0))
PREV_VOL_MIN = 500_000
START, END = '2025-01-01', '2026-07-02'

ACCOUNT = 100_000.0
N_SLOTS = 4
RISK = 3000.0
MIN_STOP_PCT = 1.0
OLD_POS = 50_000.0
Q_CAPS = {'Q1': 3.0, 'Q2': 3.0, 'Q3': 3.0, 'Q4': 3.0, 'Q5': 1.5}
Q_ORDER = {'Q4': 0, 'Q5': 1, 'Q3': 2, 'Q2': 3, 'Q1': 4}
LOCK_TRIGGER_R = 1.75
LOCK_STOP_R = 0.5
ENTRY_SLIP_BPS = float(os.environ.get('ORB_BAND_ENTRY_SLIP_BPS', 30.0))
EXIT_SLIP_BPS = float(os.environ.get('ORB_BAND_EXIT_SLIP_BPS', 10.0))
PDR_VETO_THR = 8.0
TOUCHGO_CFG = load_touchgo_config({})

REPORT_LINES = []


def log(msg=''):
    print(msg, flush=True)
    REPORT_LINES.append(str(msg))


def _safe_div(a, b):
    return a / b if b else 0.0


# ---------------------------------------------------------------------------
# 1. Discovery — mirror the live snapshot-universe criteria, band-shifted
# ---------------------------------------------------------------------------

def discover_candidates():
    conn = sqlite3.connect(os.path.join(ROOT, 'data', 'cache.db'), timeout=30)
    q = f"""
    WITH d AS (
      SELECT symbol, bar_date, open, high, low, close, volume,
             LAG(close)  OVER (PARTITION BY symbol ORDER BY bar_date) pc,
             LAG(volume) OVER (PARTITION BY symbol ORDER BY bar_date) pv,
             LAG(high)   OVER (PARTITION BY symbol ORDER BY bar_date) ph,
             LAG(low)    OVER (PARTITION BY symbol ORDER BY bar_date) pl
      FROM daily_bars
      WHERE bar_date >= date('{START}', '-45 days') AND bar_date <= '{END}'
    )
    SELECT symbol, bar_date, open, pc AS prev_close, pv AS prev_volume,
           ph AS prev_high, pl AS prev_low
    FROM d
    WHERE bar_date >= '{START}'
      AND pc > 0 AND pv >= {PREV_VOL_MIN}
      AND open > {BAND_MIN} AND open <= {BAND_MAX}
      AND (open - pc) / pc * 100 >= {GAP_MIN_PCT}
      AND (open - pc) / pc * 100 <= {GAP_CAP_PCT}
    ORDER BY bar_date, symbol
    """
    df = pd.read_sql(q, conn)
    conn.close()
    log(f"discovery: {len(df)} candidate symbol-days in ({BAND_MIN},{BAND_MAX}] "
        f"gap {GAP_MIN_PCT}-{GAP_CAP_PCT}% pv>={PREV_VOL_MIN}")
    return df


def load_daily_context():
    """symbol -> daily bars frame (for 20d features)."""
    conn = sqlite3.connect(os.path.join(ROOT, 'data', 'cache.db'), timeout=30)
    df = pd.read_sql(
        f"SELECT symbol, bar_date, open, high, low, close, volume FROM daily_bars "
        f"WHERE bar_date >= date('{START}','-45 days') AND bar_date <= '{END}'",
        conn)
    conn.close()
    df['bar_date'] = pd.to_datetime(df['bar_date'])
    return {s: g.sort_values('bar_date').reset_index(drop=True)
            for s, g in df.groupby('symbol')}


# ---------------------------------------------------------------------------
# 2. Bars — cache-first, additive fetch for missing
# ---------------------------------------------------------------------------

def load_bars(pairs, db):
    raw = db.get_intraday_bars_bulk(pairs)
    missing = [k for k in pairs if k not in raw or not raw.get(k)]
    log(f"bars: {len(pairs) - len(missing)}/{len(pairs)} cached; fetching {len(missing)}")
    if missing:
        from data_sources.alpaca_client import AlpacaClient
        client = AlpacaClient(os.getenv('ALPACA_API_KEY'), os.getenv('ALPACA_API_SECRET'))
        fetched = 0
        for i, (sym, ds) in enumerate(missing):
            try:
                td = date.fromisoformat(ds)
                # ET offset via zoneinfo (DST-correct)
                from zoneinfo import ZoneInfo
                et = ZoneInfo('America/New_York')
                mo = datetime(td.year, td.month, td.day, 9, 30, tzinfo=et).astimezone(timezone.utc)
                mc = datetime(td.year, td.month, td.day, 16, 0, tzinfo=et).astimezone(timezone.utc)
                bf = client.get_historical_1min_bars(sym, mo, mc)
                if bf is not None and not bf.empty:
                    recs = bf.to_dict('records')
                    db.save_intraday_bars(sym, ds, recs)
                    raw[(sym, ds)] = recs
                    fetched += 1
            except Exception as e:
                if i < 5:
                    log(f"  fetch {sym} {ds} failed: {e}")
            if (i + 1) % 200 == 0:
                log(f"  ... {i + 1}/{len(missing)} fetched={fetched}")
        log(f"bars: fetched {fetched}/{len(missing)} missing")
    return raw


def to_df(bars):
    if not bars:
        return pd.DataFrame()
    df = pd.DataFrame(bars)
    df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
    return df.sort_values('timestamp').reset_index(drop=True)


# ---------------------------------------------------------------------------
# 3. Per-candidate: range, entry, static-lock resim, features
#    (definitions mirror study_orb_features.py / dump_orb_candidates.py)
# ---------------------------------------------------------------------------

def session_open_ts(bars_df):
    et = bars_df['timestamp'].dt.tz_convert('America/New_York')
    m = bars_df[et.dt.time == pd.Timestamp('09:30').time()]
    return m.iloc[0]['timestamp'] if len(m) else None


def simulate_static_lock(bars, entry_price, range_high, range_low, entry_time):
    range_size = range_high - range_low
    trigger_lvl = entry_price + LOCK_TRIGGER_R * range_size
    lock_stop = entry_price + LOCK_STOP_R * range_size
    stop_price = range_low
    armed = False
    post = bars[bars['timestamp'] >= entry_time].reset_index(drop=True)
    if len(post) >= 1:
        eb = post.iloc[0]
        fire_m, exit_m = evaluate_rule_m(
            float(eb['open']), float(eb['high']), float(eb['low']),
            float(eb['close']), TOUCHGO_CFG)
        if fire_m and exit_m is not None:
            return exit_m * (1 - EXIT_SLIP_BPS / 10000), 'tag_bb'
    if len(post) >= 2:
        b1 = post.iloc[1]
        fire_d, exit_d = evaluate_rule_d(
            entry_price, float(b1['low']), range_size, TOUCHGO_CFG)
        if fire_d and exit_d is not None:
            return exit_d * (1 - EXIT_SLIP_BPS / 10000), 'tag_b1'
    for _, row in post.iloc[1:].iterrows():
        bh, bl = float(row['high']), float(row['low'])
        if not armed and bh >= trigger_lvl:
            armed = True
            stop_price = max(stop_price, lock_stop)
        if bl <= stop_price:
            return stop_price * (1 - EXIT_SLIP_BPS / 10000), 'lock' if armed else 'stop'
    return float(post.iloc[-1]['close']) * (1 - EXIT_SLIP_BPS / 10000), 'eod'


def build_candidate(sym, ds, row, bars_df, sym_daily):
    if bars_df.empty:
        return None
    ots = session_open_ts(bars_df)
    if ots is None:
        return None
    rmask = (bars_df['timestamp'] >= ots) & (bars_df['timestamp'] < ots + timedelta(minutes=5))
    rb = bars_df.loc[rmask].reset_index(drop=True)
    if len(rb) < 5:
        return None
    range_high = float(rb['high'].max())
    range_low = float(rb['low'].min())
    open_p = float(rb['open'].iloc[0])
    close_p = float(rb['close'].iloc[-1])
    range_size = range_high - range_low
    if range_size <= 0 or open_p <= 0:
        return None

    # Entry: first bar 9:35-10:35 whose high crosses range_high
    win = bars_df[(bars_df['timestamp'] >= ots + timedelta(minutes=5)) &
                  (bars_df['timestamp'] < ots + timedelta(minutes=65))]
    hit = win[win['high'] > range_high]
    if hit.empty:
        return None
    entry_time = hit.iloc[0]['timestamp']
    entry_price = range_high * (1 + ENTRY_SLIP_BPS / 10000)
    exit_price, exit_reason = simulate_static_lock(
        bars_df, entry_price, range_high, range_low, entry_time)
    shares = OLD_POS / entry_price
    pnl = (exit_price - entry_price) * shares

    # Features (BT-parity definitions)
    prev_close = float(row['prev_close'])
    prev_high = float(row['prev_high'] or 0)
    prev_low = float(row['prev_low'] or 0)
    feat = {
        'gap_pct': _safe_div(open_p - prev_close, prev_close) * 100,
        'range_size_pct': _safe_div(range_size, open_p) * 100,
        'range_total_volume': float(rb['volume'].sum()),
        'range_close_position': _safe_div(close_p - range_low, range_size)
                                if range_size > 0 else 0.5,
        'prev_day_range_pct': _safe_div(prev_high - prev_low, prev_close) * 100,
        'prev_day_close_position': _safe_div(prev_close - prev_low, prev_high - prev_low)
                                   if prev_high > prev_low else 0.5,
    }
    br = (rb['high'] - rb['low']) / rb['close'].replace(0, np.nan)
    feat['range_avg_bar_range_pct'] = float(br.mean(skipna=True) * 100) \
        if not br.isna().all() else 0.0
    # 20d context strictly before this date
    dtx = pd.Timestamp(ds)
    prior = sym_daily[sym_daily['bar_date'] < dtx].tail(20)
    if len(prior) < 5:
        return None
    high_20d = float(prior['high'].max())
    feat['price_vs_20d_high_pct'] = _safe_div(open_p - high_20d, high_20d) * 100

    return dict(symbol=sym, date=ds, entry_price=round(entry_price, 4),
                exit_price=round(exit_price, 4), exit_reason=exit_reason,
                pnl=round(pnl, 2), range_size_pct=feat['range_size_pct'],
                **{k: v for k, v in feat.items() if k != 'range_size_pct'})


# ---------------------------------------------------------------------------
# 4. Pipeline (same machinery as orb_replica / production BT)
# ---------------------------------------------------------------------------

def run_pipeline(df, fit_df=None, pdr_veto=True, label=''):
    """fit_df: rows used for z/quintile/mult fits (defaults to df's TRAIN).
    Fits ALWAYS restricted to 2025-H1 of fit_df — frozen-fit semantics."""
    df = df.copy()
    stop = df['range_size_pct'].clip(lower=MIN_STOP_PCT)
    df['_rp_position'] = (RISK / (stop / 100.0)).clip(upper=ACCOUNT / N_SLOTS)
    df['_rp_pnl'] = df['pnl'] * df['_rp_position'] / OLD_POS

    src = fit_df if fit_df is not None else df
    src = src.copy()
    sstop = src['range_size_pct'].clip(lower=MIN_STOP_PCT)
    src['_rp_position'] = (RISK / (sstop / 100.0)).clip(upper=ACCOUNT / N_SLOTS)
    src['_rp_pnl'] = src['pnl'] * src['_rp_position'] / OLD_POS
    train = src[(src['date'] >= '2025-01-01') & (src['date'] <= '2025-06-30')]
    params = fit_z_params(train, FILTER_FEATURES)
    df['_composite'] = composite_score(df, params)
    train = train.copy()
    train['_composite'] = composite_score(train, params)
    tk = train[train['_composite'] >= FILTER_THRESHOLD].copy()
    cutoffs = fit_quintile_cutoffs(tk['_composite'])
    tk['_quintile'] = assign_quintile(tk['_composite'], cutoffs)
    avg = float(tk['_rp_pnl'].mean())
    mults = {q: max(ADAPTIVE_MULT_MIN,
                    min(Q_CAPS[q], float(tk[tk['_quintile'] == q]['_rp_pnl'].mean()) / avg))
             for q in Q_CAPS}

    kept = df[df['_composite'] >= FILTER_THRESHOLD].copy()
    kept['_quintile'] = assign_quintile(kept['_composite'], cutoffs)
    kept = kept[kept['_quintile'] != 'Q1']
    rows = []
    for day, dg in kept.groupby('date'):
        d = dg.copy()
        d['_q_rank'] = d['_quintile'].map(Q_ORDER)
        d = d.sort_values(['_q_rank', '_composite'], ascending=[True, False])
        sf, ss, today = set(), set(), []
        for _, r in d.iterrows():
            f = symbol_family(r['symbol'])
            sgr = symbol_super_group(r['symbol'])
            if f and f in sf:
                continue
            if sgr and sgr in ss:
                continue
            if f:
                sf.add(f)
            if sgr:
                ss.add(sgr)
            today.append(r)
            if len(today) >= N_SLOTS:
                break
        rows.extend(today)
    sel = pd.DataFrame(rows)
    if sel.empty:
        log(f"{label}: EMPTY selection")
        return sel
    sel['_sized_pnl'] = sel.apply(lambda r: r['_rp_pnl'] * mults[r['_quintile']], axis=1)
    if pdr_veto:
        sel = sel[~(sel['prev_day_range_pct'] <= PDR_VETO_THR)].copy()
    sel['date'] = pd.to_datetime(sel['date'])
    return sel


def summarize(sel, label):
    if sel.empty:
        log(f"{label:<28} EMPTY")
        return None
    tot = sel['_sized_pnl'].sum()
    h1 = sel[sel['date'] < '2025-07-01']['_sized_pnl'].sum()
    h2 = sel[(sel['date'] >= '2025-07-01') & (sel['date'] < '2026-01-01')]['_sized_pnl'].sum()
    y26 = sel[sel['date'] >= '2026-01-01']['_sized_pnl'].sum()
    d = sel.groupby('date')['_sized_pnl'].sum().sort_index()
    cum = d.cumsum()
    mdd = (cum - cum.cummax()).min()
    top5 = sel.nlargest(5, '_sized_pnl')['_sized_pnl'].sum()
    wr = (sel['_sized_pnl'] > 0).mean() * 100
    log(f"{label:<28} n={len(sel):>4} TOT ${tot:>+10,.0f}  25H1 ${h1:>+8,.0f}  "
        f"25H2 ${h2:>+8,.0f}  2026 ${y26:>+8,.0f}  MDD ${mdd:>+8,.0f}  "
        f"WR {wr:4.1f}%  top5 {top5 / tot * 100 if tot else 0:5.0f}%")
    return d


def main():
    t0 = datetime.now()
    cand_rows = discover_candidates()
    daily_ctx = load_daily_context()
    db = Database(db_path=os.path.join(ROOT, 'data', 'cache.db'))
    pairs = [(r['symbol'], r['bar_date']) for _, r in cand_rows.iterrows()]
    bars_raw = load_bars(pairs, db)

    built = []
    for _, r in cand_rows.iterrows():
        sym, ds = r['symbol'], r['bar_date']
        sd = daily_ctx.get(sym)
        if sd is None:
            continue
        try:
            c = build_candidate(sym, ds, r, to_df(bars_raw.get((sym, ds))), sd)
            if c:
                built.append(c)
        except Exception:
            continue
    band = pd.DataFrame(built)
    tag = f"{BAND_MIN:g}_{BAND_MAX:g}_slip{ENTRY_SLIP_BPS:g}"
    band.to_csv(f'/tmp/orb_band_{tag}_candidates.csv', index=False)
    log(f"built {len(band)} band candidates with entries "
        f"({len(band) / max(len(cand_rows), 1) * 100:.0f}% of discovered broke out)")
    log()

    # Existing $3-30 candidates (same resim conventions)
    old = pd.read_csv('/tmp/orb_candidates_resim.csv')
    needed = [f for f, _ in FILTER_FEATURES] + ['pnl', 'date', 'range_size_pct',
                                                'prev_day_range_pct', 'symbol']
    old = old.dropna(subset=[c for c in needed if c in old.columns])
    old['date'] = old['date'].astype(str).str[:10]
    band['date'] = band['date'].astype(str).str[:10]

    log("=" * 110)
    base_sel = run_pipeline(old, label='BASE')
    base_d = summarize(base_sel, 'BASE $3-30 (frozen, veto)')

    merged = pd.concat([old[[c for c in old.columns if c in set(band.columns) | {'pnl'}]],
                        band], ignore_index=True)
    # B1: fits from the $3-30 TRAIN ONLY (production frozen-fit semantics)
    b1_sel = run_pipeline(merged, fit_df=old, label='B1')
    b1_d = summarize(b1_sel, 'B1 $3-60 (FROZEN fit, veto)')

    # B2 reference: band alone, band-own fit
    if len(band) > 100:
        b2_sel = run_pipeline(band, label='B2')
        summarize(b2_sel, 'B2 $30-60 alone (own fit)')
        b2n = run_pipeline(band, pdr_veto=False, label='B2nv')
        summarize(b2n, 'B2 $30-60 alone (NO veto)')

    # B1 no-veto variant (veto transfer check on merged)
    b1nv = run_pipeline(merged, fit_df=old, pdr_veto=False, label='B1nv')
    summarize(b1nv, 'B1 $3-60 (frozen, NO veto)')

    log()
    if base_d is not None and b1_d is not None:
        # New-band contribution inside B1
        band_keys = set(zip(band['symbol'], band['date']))
        b1_sel['_is_band'] = [
            (s, d.strftime('%Y-%m-%d')) in band_keys
            for s, d in zip(b1_sel['symbol'], b1_sel['date'])]
        nb = b1_sel[b1_sel['_is_band']]
        log(f"B1 new-band picks: {len(nb)} trades, ${nb['_sized_pnl'].sum():+,.0f} "
            f"(displacing small-caps on shared slots)")
        for era, lo, hi in [('25H1', '2025-01-01', '2025-07-01'),
                            ('25H2', '2025-07-01', '2026-01-01'),
                            ('2026', '2026-01-01', '2027-01-01')]:
            e = nb[(nb['date'] >= lo) & (nb['date'] < hi)]['_sized_pnl'].sum()
            log(f"  band-picks {era}: ${e:+,.0f}")
        # Correlation of band-day pnl with base
        nb_d = nb.groupby('date')['_sized_pnl'].sum()
        joint = pd.concat([base_d.rename('base'), nb_d.rename('band')], axis=1).fillna(0)
        if len(joint) > 20 and joint['band'].std() > 0:
            log(f"daily corr(base, band-picks): {joint['base'].corr(joint['band']):.2f}")
        delta_d = (pd.concat([b1_d.rename('b1'), base_d.rename('base')], axis=1)
                   .fillna(0))
        delta = delta_d['b1'] - delta_d['base']
        log(f"\nB1 - BASE total: ${delta.sum():+,.0f}")
        for era, lo, hi in [('25H1', '2025-01-01', '2025-07-01'),
                            ('25H2', '2025-07-01', '2026-01-01'),
                            ('2026', '2026-01-01', '2027-01-01')]:
            e = delta[(delta.index >= lo) & (delta.index < hi)].sum()
            log(f"  B1-BASE {era}: ${e:+,.0f}")

    log(f"\nruntime: {(datetime.now() - t0).total_seconds() / 60:.1f} min")
    with open(f'/tmp/orb_band_study_report_{tag}.txt', 'w') as f:
        f.write('\n'.join(REPORT_LINES))


if __name__ == '__main__':
    main()
