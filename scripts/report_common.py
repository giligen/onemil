"""Shared helpers for the daily green-check + weekly report Telegrams.

The daily check computes the day's OPERATIONAL-GREEN verdict — the unit the
2026-07-06 ramp policy advances on — and persists a streak file that
`orb_ramp_check.py` and the weekly report read. The weekly report is the
owner's decision document (P&L, gate progress, edge capture, flags).

All loaders are top-level functions so tests can monkeypatch them.
"""
from __future__ import annotations

import glob
import json
import os
import sqlite3
import subprocess
import sys
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from trading.exit_reasons import is_known, needs_reconcile  # noqa: E402

STREAK_PATH = ROOT / 'logs' / 'green_streak.json'
SELECTION_AUDIT = ROOT / 'logs' / 'orb_selection_audit.jsonl'
STAGE_RISK_USD = 1000.0      # Stage 0 — keep in sync with orb.yaml sizing
MODEL_RISK_USD = 3000.0      # BT $100K-model risk (study convention)
STAGE_SCALE = STAGE_RISK_USD / MODEL_RISK_USD
RAMP_START = '2026-05-18'
ADVANCE_LOSS_FLOOR = -7500.0   # Stage 0: -1 x (daily limit 1500 x 5)
GREEN_SESSIONS_NEEDED = 10

# Live-side reject reasons that legitimately explain a BT pick not trading
# (live-only gates BT cannot model). pdr_veto is NOT here: BT applies the
# same veto, so a live pdr_veto on a BT-selected pick would be a parity bug.
EXPLAINED_LIVE_REJECTS = {'spread_gate', 'insufficient_bp',
                          'fcfs_other_strategy', 'halted'}


# ---------------------------------------------------------------------------
# Loaders (monkeypatch points)
# ---------------------------------------------------------------------------

def bt_book_csv_path() -> str:
    """Config-driven path to the nightly BT ground-truth book (B+ 2026-08-15).

    Single source of truth: orb.yaml::backtest.nightly_book_csv — the SAME key
    study_orb_pipeline_static_lock.py writes to, so the file the pipeline WRITES
    is always the file report_common READS (review P1-1: the old fixed glob on
    orb_static_lock_trades.csv would silently compare live-B+ against the stale
    $100K/N4/pdr8/no-G1 book). Falls back to the B+ book name if orb.yaml is
    unreadable or lacks the key."""
    default = str(ROOT / 'analysis_results' / 'orb_bplus_book.csv')
    try:
        import yaml
        cfg = yaml.safe_load(open(ROOT / 'orb.yaml')) or {}
        rel = (cfg.get('backtest') or {}).get('nightly_book_csv')
        if not rel:
            print("WARNING: orb.yaml lacks backtest.nightly_book_csv — "
                  "defaulting to orb_bplus_book.csv", flush=True)
            return default
        p = Path(rel)
        return str(p if p.is_absolute() else ROOT / p)
    except Exception as e:
        print(f"WARNING: bt_book_csv_path could not read orb.yaml ({e}) — "
              f"defaulting to orb_bplus_book.csv", flush=True)
        return default


def latest_bt_trades_csv() -> Optional[str]:
    path = bt_book_csv_path()
    return path if Path(path).exists() else None


def bt_filled_symbols(bt_rows: List[Dict]) -> set:
    """Symbols of BT picks whose breakout FIRED (`entered` != 0).

    Entered-inclusive book (2026-09-05): rows with entered == 0 are picks
    BT would have ordered but never filled (time_stop_canceled class);
    they are NOT fill-parity subjects. Rows without the column (legacy
    entered-only book) count as filled."""
    out = set()
    for r in bt_rows:
        v = r.get('entered', 1)
        try:
            if v is None or v != v:      # None / NaN → legacy → filled
                out.add(r['symbol']); continue
            if int(v) != 0:
                out.add(r['symbol'])
        except (TypeError, ValueError):
            out.add(r['symbol'])
    return out


def load_bt_selected(day: str) -> List[Dict]:
    """BT-defended selection rows (post Q1-filter, post PDR-veto) for a date."""
    import pandas as pd
    path = latest_bt_trades_csv()
    if not path:
        return []
    df = pd.read_csv(path)
    df = df[df['date'].astype(str).str.startswith(day)]
    return df.to_dict('records')


def bt_data_max_date() -> Optional[str]:
    """Latest date the nightly BT pipeline PROCESSED — staleness detector.

    Keys on the features CSV (regenerated nightly, one row per evaluated
    symbol-day) NOT the trades CSV: a zero-selection day (like 2026-07-22,
    quiet tape, BT picked nothing) leaves the trades CSV max date behind
    while the pipeline genuinely ran — reading that as 'stale' skipped
    parity on exactly the days it should trivially pass (7/22 incident).
    Falls back to the trades CSV when no features files exist."""
    import glob as _glob

    import pandas as pd
    feat_paths = [p for p in sorted(_glob.glob(
        str(ROOT / 'analysis_results' / 'orb_features_*.csv')))
        if 'corrmatrix' not in p]
    if feat_paths:
        try:
            return str(pd.read_csv(
                feat_paths[-1], usecols=['date'])['date'].max())[:10]
        except Exception as e:
            print(f"WARNING: features CSV unreadable for staleness "
                  f"({e}) — falling back to trades CSV", flush=True)
    path = latest_bt_trades_csv()
    if not path:
        return None
    try:
        return str(pd.read_csv(path, usecols=['date'])['date'].max())[:10]
    except Exception:
        return None


def load_live_rows(day: str, strategy: Optional[str] = None) -> List[Dict]:
    """All trade rows (any status — placements included) for a date."""
    conn = sqlite3.connect(ROOT / 'data' / 'trades.db', timeout=15)
    conn.row_factory = sqlite3.Row
    q = "SELECT * FROM trades WHERE trade_date = ?"
    args = [day]
    if strategy:
        q += " AND strategy = ?"
        args.append(strategy)
    rows = [dict(r) for r in conn.execute(q, args)]
    conn.close()
    return rows


def journal_grep(pattern: str, day: str) -> List[str]:
    """Journald lines matching `pattern` for onemil-trader on a UTC date.

    2026-07-10 fix: the old version piped the FULL day (DEBUG logging =
    hundreds of MB) through python and hit its 30s timeout, silently
    returning [] — which made every explained-miss check unreliable (the
    7/9 false-alarm red: IQMX's guard-skip line existed but was never
    seen). Filtering now happens journald-side (-g) — ~14s worst case —
    with a 60s ceiling and a WARNING (not silence) on failure.
    """
    try:
        res = subprocess.run(
            ['journalctl', '-u', 'onemil-trader', '--since', f'{day} 00:00',
             '--until', f'{day} 23:59', '-g', pattern, '--no-pager'],
            capture_output=True, text=True, timeout=60)
        return [ln for ln in res.stdout.splitlines()
                if not ln.startswith('-- ')]   # drop journald banners
    except Exception as e:
        print(f"WARNING: journal_grep({pattern!r}, {day}) failed: {e} — "
              f"treating as no matches (checks may over-flag)", flush=True)
        return []


def read_selection_audit(day: str) -> List[Dict]:
    recs = []
    if SELECTION_AUDIT.exists():
        for line in SELECTION_AUDIT.read_text().splitlines():
            try:
                r = json.loads(line)
                if str(r.get('ts_utc', '')).startswith(day):
                    recs.append(r)
            except Exception:
                continue
    return recs


# ---------------------------------------------------------------------------
# The daily verdict
# ---------------------------------------------------------------------------

def green_verdict(day: str) -> Dict:
    """Compute the operational-green verdict for one trading day.

    Hard gates (any failure = RED day for the ramp streak):
      A. every closed trade's exit_reason is known + attributed
      B. no rows stuck in exit_pending_verification
      C. every BT-selected pick was at least ORDERED live, or its absence
         is explained by a live-only gate (spread, buying power, conflict)
    Informational (reported, not gating — journald retention makes absence
    unprovable): touchgo REKEY / negative-age tripwires.
    """
    reasons: List[str] = []
    checks: Dict[str, str] = {}

    rows = load_live_rows(day)
    closed = [r for r in rows if r.get('exit_price') is not None]
    bad_exit = [r['symbol'] for r in closed
                if not is_known(r.get('exit_reason'))
                or needs_reconcile(r.get('exit_reason'))]
    checks['exits'] = 'OK' if not bad_exit else f"unattributed: {bad_exit}"
    if bad_exit:
        reasons.append(f"unattributed exits: {bad_exit}")

    stuck = [r['symbol'] for r in rows
             if r.get('order_status') == 'exit_pending_verification']
    checks['pending_verification'] = 'OK' if not stuck else f"stuck: {stuck}"
    if stuck:
        reasons.append(f"exit_pending_verification: {stuck}")

    bt_max = bt_data_max_date()
    bt_stale = bt_max is None or bt_max < day
    bt_rows = [] if bt_stale else load_bt_selected(day)
    # BT picks = everything BT would have ORDERED (entered-inclusive book,
    # 2026-09-05: no-fill picks included — live must show an order for
    # them too, filled or time_stop_canceled). Fill-parity below keys on
    # the ENTERED subset only.
    bt_syms = {r['symbol'] for r in bt_rows}
    bt_filled_syms = bt_filled_symbols(bt_rows)
    live_orb_syms = {r['symbol'] for r in rows if r.get('strategy') == 'orb'}
    missing = bt_syms - live_orb_syms
    unexplained = []
    if missing:
        audit = read_selection_audit(day)
        audited = set()
        for rec in audit:
            audited |= {e.get('sym') for e in rec.get('ranked', [])}
        # Live-only gates BT cannot model. 2026-07-10: added the buy-stop
        # guard skip ('breakout extended past limit ... Not chasing') — on
        # 7/9 IQMX broke out during the 44s ranking window, the guard
        # correctly refused to chase, and the checker scored a correctly-
        # explained miss as unexplained (false-alarm red; BT made +$5 on
        # the missed trade while live beat BT by ~$570 that day).
        # 2026-08-14 independent audit: the old 'spread' keyword matched
        # routine quote-DEBUG lines ("CIFU: quote bid=.. spread=$0.11"),
        # silently explaining ANY never-ordered pick that ever got a
        # quote logged — the gate was neutralized (reproduced on the
        # 8/11 CIFU miss). Explanations now require an ACTUAL skip
        # phrase, not a keyword coincidence.
        explained_lines = (journal_grep('skipped —', day)
                           + journal_grep('spread skip', day)
                           + journal_grep('ENTRY SKIPPED', day)
                           + journal_grep('Not chasing', day))
        _SKIP_PHRASES = ('spread skip', 'buying power', 'other strategy',
                         'halted', 'breakout extended past limit',
                         'Not chasing', 'ENTRY SKIPPED', 'skipped —')
        for sym in sorted(missing):
            if any(sym in ln and any(k in ln for k in _SKIP_PHRASES)
                   for ln in explained_lines):
                continue
            unexplained.append(sym)
    if bt_stale:
        checks['bt_parity'] = (f"SKIPPED — BT data stale (max={bt_max}); "
                               f"nightly orb-backtest hasn't covered {day} yet")
    elif unexplained:
        checks['bt_parity'] = f"BT picks never ordered: {unexplained}"
        reasons.append(f"BT picks never ordered live: {unexplained}")
    else:
        checks['bt_parity'] = 'OK'

    # Fill-parity (2026-08-14, IREX 7/30 lesson): a BT-FILLED pick that
    # live ORDERED but never FILLED passed the gate above silently — on
    # 7/30 that was the only >=$3K monster in 4 months (bar-1 breakout
    # outran the ~73s placement latency; the stop-limit was born
    # unfillable and hit time_stop_canceled while the BT rode +$4,361).
    # The edge is the tail: an unfilled BT-filled pick is a hard flag.
    if bt_stale:
        checks['fill_parity'] = 'SKIPPED — BT data stale'
    else:
        unfilled = sorted({
            r['symbol'] for r in rows
            if r.get('strategy') == 'orb' and r['symbol'] in bt_filled_syms
            and r.get('fill_price') is None
            and r.get('order_status') not in ('filled', 'closed')})
        checks['fill_parity'] = ('OK' if not unfilled else
                                 f"BT filled, live unfilled: {unfilled}")
        if unfilled:
            reasons.append(f"fill-parity: BT FILLED but live order never "
                           f"filled (tail-capture risk): {unfilled}")

    rekey = journal_grep('REKEY', day) + journal_grep('negative-age', day)
    checks['touchgo_tripwires'] = 'OK' if not rekey else f"{len(rekey)} tripwire line(s)"

    # Winner stack: floored-stop drift (HARD when a mismatch exists —
    # live monitored a stop the book wasn't validated on; review P0-6.2).
    # Inert while exit.atr_stop_floor is off (available=False).
    fsd = floored_stop_drift(day)
    if not fsd['available']:
        checks['floored_stop_drift'] = 'SKIPPED (flag off or check error)'
    elif fsd['mismatches']:
        checks['floored_stop_drift'] = f"DRIFT: {fsd['mismatches']}"
        reasons.append(f"floored-stop drift: {fsd['mismatches']}")
    else:
        checks['floored_stop_drift'] = f"OK ({fsd['n_checked']} checked)"

    # Service-uptime context (2026-07-06 incident follow-up): if the trader
    # was crash-looping / restarting around the 9:35 ET entry window, say so
    # in the red-day reasons — a missed pick from an outage is a different
    # investigation than a selection bug.
    starts = journal_grep('Started onemil-trader', day)
    fatal = journal_grep('Pre-start account validation failed', day)
    if unexplained and (len(starts) > 2 or fatal):
        reasons.append(
            f"CONTEXT: trader restarted {len(starts)}x today"
            + (", pre-start validation failures present" if fatal else "")
            + " — missed picks likely due to service outage, not selection")
        checks['service_uptime'] = f"{len(starts)} starts, fatal={bool(fatal)}"
    else:
        checks['service_uptime'] = 'OK'

    return {
        'day': day,
        'green': not reasons,
        'reasons': reasons,
        'checks': checks,
        'n_live_rows': len(rows),
        'n_bt_selected': len(bt_syms),
        'bt_stale': bt_stale,
    }


# ---------------------------------------------------------------------------
# Sizing attribution (2026-07-10 ship: quintile-mult correction + news-gated
# PM mult both go live 2026-07-13 — this is their EoD validation)
# ---------------------------------------------------------------------------

MULT_SHIP_DAY = '2026-07-13'   # first live day of corrected quintile mults
                               # + news-gated PM mult (A2)


def _orb_sizing_cfg() -> Dict:
    """Live sizing knobs from orb.yaml (for expected-mult recompute)."""
    import yaml
    try:
        with open(ROOT / 'orb.yaml') as f:
            cfg = yaml.safe_load(f) or {}
        return (cfg.get('sizing', {}) or {}).get('pm_dollar_vol_mult', {}) or {}
    except Exception as e:
        print(f"WARNING: orb.yaml unreadable for sizing recompute: {e}",
              flush=True)
        return {}


def eod_news_recheck(symbols: List[str], day: str) -> Dict[str, bool]:
    """Re-query the news API EoD for the day's traded symbols.

    Window: prev calendar day 15:00 ET → trade day 09:31 ET — a SUBSET of
    what live could have seen (live fetches 9:31+), so `EoD says news but
    live recorded none` = a real live fetch gap, while the reverse is just
    the timing tail and is NOT flagged. Returns {} on API failure (soft
    check — never blocks the green verdict on a news-API blip).
    """
    if not symbols:
        return {}
    try:
        import pandas as pd
        import requests
        # Cron runs with a bare env — load .env like send_telegram does,
        # or the drift/lag tripwires silently no-op every night
        # (final-review find 2026-07-10).
        from dotenv import load_dotenv
        load_dotenv(str(ROOT / '.env'))
        k = os.environ.get('ALPACA_API_KEY')
        s = os.environ.get('ALPACA_API_SECRET')
        if not k or not s:
            print("WARNING: eod_news_recheck skipped — no Alpaca creds in env",
                  flush=True)
            return {}
        d = pd.Timestamp(day)
        st = ((d - pd.Timedelta(days=1)).tz_localize('America/New_York')
              + pd.Timedelta(hours=15)).tz_convert('UTC').isoformat()
        en = (d.tz_localize('America/New_York')
              + pd.Timedelta(hours=9, minutes=31)).tz_convert('UTC').isoformat()
        newsy: Dict[str, bool] = {sym: False for sym in symbols}
        token = None
        for _ in range(6):
            params = {'symbols': ','.join(symbols), 'start': st, 'end': en,
                      'limit': 50, 'sort': 'desc'}
            if token:
                params['page_token'] = token
            r = requests.get('https://data.alpaca.markets/v1beta1/news',
                             params=params,
                             headers={'APCA-API-KEY-ID': k,
                                      'APCA-API-SECRET-KEY': s},
                             timeout=(5, 30))
            r.raise_for_status()
            j = r.json()
            for a in j.get('news', []):
                for sym in a.get('symbols', []):
                    if sym in newsy:
                        newsy[sym] = True
            token = j.get('next_page_token')
            if not token:
                break
        return newsy
    except Exception as e:
        print(f"WARNING: eod_news_recheck failed: {e} — news drift check "
              f"skipped today", flush=True)
        return {}


def sizing_attribution(day: str) -> Dict:
    """Per-trade mult attribution + drift checks for the day's ORB trades.

    Returns:
      trades:          [{symbol, quintile, adaptive_mult, pm_mult, has_news,
                         n_articles, pm_dollar_vol, pnl}]
      mult_mismatches: recorded pm_mult != recompute from recorded inputs
                       via the SHARED helper (HARD — that is a code bug)
      news_drift:      live recorded has_news=False/None but the EoD
                       re-query finds pre-9:31 articles (SOFT — fetch gap)
      cum:             realized P&L since MULT_SHIP_DAY by quintile group
                       + the news-boosted cell (the forward watch)
    """
    from trading.orb_asset_class import effective_has_news
    from trading.orb_pm_mult import (
        DEFAULT_HIGH_CUT_USD, DEFAULT_HIGH_MULT, DEFAULT_HIGH_MULT_NEWS,
        pm_size_multiplier,
    )
    cfg = _orb_sizing_cfg()
    high_cut = float(cfg.get('high_cut_usd', DEFAULT_HIGH_CUT_USD))
    high_mult = float(cfg.get('high_mult', DEFAULT_HIGH_MULT))
    high_mult_news = float(cfg.get('high_mult_news', DEFAULT_HIGH_MULT_NEWS))
    news_gate = bool(cfg.get('news_gate', True))

    trades, mult_mismatches = [], []
    for r in load_live_rows(day, strategy='orb'):
        try:
            pdata = json.loads(r.get('pattern_data') or '{}')
        except Exception:
            pdata = {}
        t = {'symbol': r['symbol'],
             'quintile': pdata.get('quintile'),
             'adaptive_mult': pdata.get('adaptive_mult'),
             'pm_mult': pdata.get('pm_mult'),
             'has_news': pdata.get('has_news'),
             'n_articles': pdata.get('n_articles'),
             'pm_dollar_vol': pdata.get('pm_dollar_vol'),
             'asset_class': pdata.get('asset_class'),
             'pnl': r.get('pnl'),
             'filled': r.get('fill_price') is not None,
             # Winner-stack scale attribution (2026-08-22). pnl above is
             # already COMBINED (single-writer) — these are display-only.
             'scaled': bool(r.get('scaled_at')),
             'scale_qty': r.get('scale_qty'),
             'scale_pnl': r.get('scale_pnl')}
        trades.append(t)
        if t['pm_mult'] is None:
            continue   # pre-ship row (no attribution recorded)
        # Class rule mirror (2026-07-11): news counts only for identified
        # stocks. Rows without a recorded class (pre-rule) pass through.
        eff_news = (effective_has_news(t['has_news'], t['asset_class'])
                    if t['asset_class'] is not None else t['has_news'])
        expected = pm_size_multiplier(
            t['pm_dollar_vol'], high_cut, high_mult,
            has_news=eff_news, high_mult_news=high_mult_news,
            news_gate=news_gate)
        if abs(float(t['pm_mult']) - expected) > 1e-9:
            mult_mismatches.append(
                f"{t['symbol']}: recorded pm_mult {t['pm_mult']} != "
                f"expected {expected} (pm$={t['pm_dollar_vol']}, "
                f"news={t['has_news']})")

    # News-drift soft check only makes sense when news actually SIZES a trade.
    # B+ 2026-08-15 disables the PM/news mult (pm_dollar_vol_mult.enabled=false)
    # — news feeds only the catalyst veto then, which fail-opens on missing
    # news (a drift there cannot mis-size anything), so gate the check off to
    # avoid meaningless nightly noise (review P2).
    pm_enabled = bool(cfg.get('enabled', True))
    news_drift = []
    check_syms = ([t['symbol'] for t in trades
                   if t['pm_mult'] is not None and t['has_news'] is not True]
                  if pm_enabled else [])
    eod_newsy = eod_news_recheck(check_syms, day)
    for t in trades:
        if (t['symbol'] in eod_newsy and eod_newsy[t['symbol']]
                and t['has_news'] is not True and t['pm_mult'] is not None):
            news_drift.append(
                f"{t['symbol']}: live recorded has_news={t['has_news']} but "
                f"EoD re-query finds pre-9:31 articles — live fetch gap"
                + ("" if (t['pm_dollar_vol'] or 0) > high_cut
                   else " (below PM$ cut, sizing unaffected)"))

    # Forward watch: realized P&L since the mult ship, by quintile group +
    # the news-boosted cell.
    conn = sqlite3.connect(ROOT / 'data' / 'trades.db', timeout=15)
    conn.row_factory = sqlite3.Row
    rows = [dict(x) for x in conn.execute(
        "SELECT symbol, pnl, pattern_data FROM trades WHERE strategy='orb' "
        "AND trade_date >= ? AND pnl IS NOT NULL", [MULT_SHIP_DAY])]
    conn.close()
    cum = {'Q2/Q3': [0.0, 0], 'Q4/Q5': [0.0, 0], 'news-boosted': [0.0, 0]}
    for x in rows:
        try:
            pdata = json.loads(x.get('pattern_data') or '{}')
        except Exception:
            pdata = {}
        q = pdata.get('quintile') or ''
        grp = 'Q2/Q3' if q in ('Q2', 'Q3') else \
              'Q4/Q5' if q in ('Q4', 'Q5') else None
        if grp:
            cum[grp][0] += float(x['pnl'])
            cum[grp][1] += 1
        if (pdata.get('pm_mult') or 1.0) > 1.0 and pdata.get('has_news'):
            cum['news-boosted'][0] += float(x['pnl'])
            cum['news-boosted'][1] += 1

    return {'trades': trades, 'mult_mismatches': mult_mismatches,
            'news_drift': news_drift, 'cum': cum}


# ---------------------------------------------------------------------------
# Winner stack: EoD floored-stop drift check (2026-08-22, review P0-6.2 —
# same spirit as the pm_mult recompute gate: recorded vs recomputed, HARD
# flag on mismatch)
# ---------------------------------------------------------------------------

FLOORED_STOP_DRIFT_TOL = 0.01    # $ — recorded stop vs BT recompute


def _winner_stack_cfg() -> Dict:
    """orb.yaml exit.atr_stop_floor (+scale_out) for the drift check."""
    import yaml
    try:
        with open(ROOT / 'orb.yaml') as f:
            cfg = yaml.safe_load(f) or {}
        return (cfg.get('exit') or {})
    except Exception as e:
        print(f"WARNING: orb.yaml unreadable for winner-stack cfg: {e}",
              flush=True)
        return {}


def floored_stop_drift(day: str) -> Dict:
    """Per-trade recorded `real_stop_loss_price` vs the BT-recomputed
    ATR floor (shared trading/orb_winner_stack over cache daily_bars,
    entry = recorded fill price).

    HARD mismatches (returned in `mismatches`) mean live monitored a
    different stop than the book was validated on — the P0-6.2 cache-gap
    class. Skipped (available=False) when the flag is off or on any
    infrastructure error (soft — never red a day on a cache blip; the
    mismatch itself IS red)."""
    exit_cfg = _winner_stack_cfg()
    atr_cfg = (exit_cfg.get('atr_stop_floor') or {})
    if not atr_cfg.get('enabled', False):
        return {'available': False, 'n_checked': 0, 'mismatches': []}
    try:
        from trading.orb_winner_stack import (DEFAULT_ATR_K, atr14_t1,
                                              floored_stop)
        k = float(atr_cfg.get('k', DEFAULT_ATR_K))
        import pandas as pd
        con = sqlite3.connect(ROOT / 'data' / 'cache.db', timeout=15)
        mismatches = []
        n = 0
        for r in load_live_rows(day, strategy='orb'):
            if r.get('fill_price') is None:
                continue
            try:
                pdata = json.loads(r.get('pattern_data') or '{}')
            except Exception:
                pdata = {}
            range_low = pdata.get('range_low')
            if range_low is None:
                continue
            d = pd.read_sql(
                "SELECT bar_date, high, low, close FROM daily_bars "
                "WHERE symbol=? AND bar_date < ? ORDER BY bar_date",
                con, params=(r['symbol'], day))
            atr = atr14_t1(d.tail(40))
            expected, status = floored_stop(
                float(range_low), float(r['fill_price']), atr, k)
            recorded = r.get('real_stop_loss_price')
            n += 1
            if recorded is None:
                mismatches.append(
                    f"{r['symbol']}: no real_stop_loss_price recorded with "
                    f"ATR floor enabled (expected {expected:.4f}, {status})")
            elif abs(float(recorded) - expected) > FLOORED_STOP_DRIFT_TOL:
                mismatches.append(
                    f"{r['symbol']}: recorded stop {float(recorded):.4f} != "
                    f"BT recompute {expected:.4f} ({status}, atr={atr}, "
                    f"k={k}) — live drifted from the validated floor")
        con.close()
        return {'available': True, 'n_checked': n, 'mismatches': mismatches}
    except Exception as e:
        print(f"WARNING: floored_stop_drift failed: {e} — check skipped",
              flush=True)
        return {'available': False, 'n_checked': 0, 'mismatches': []}


COMPOSITE_DRIFT_TOL = 0.005      # below this: measurement noise, ignore
NEAR_THRESHOLD_BAND = 0.5        # |comp| inside this band -> drift is
                                 # decision-RELEVANT (ASPI class: 0.317
                                 # vs 0.410 near the 0.0 gate)
HARD_DRIFT_ABS = 0.15            # drift this large means the feature
                                 # pipeline itself broke — hard even on
                                 # deep rejects


def classify_composite_drift(comp_live: float,
                             comp_bt: float) -> Optional[str]:
    """'hard' | 'soft' | None for a live-vs-BT composite pair.

    hard  -> red-day (z-param-desync class): selection flip across the
             0.0 threshold, either side near-threshold, or a drift too
             large to be a data revision.
    soft  -> warn-only (2026-07-21 VIVK class): tiny drift on a symbol
             both systems reject by a wide margin — vendor bar revisions
             on extreme movers land here; zero decisions changed, so it
             must not reset the ramp streak (owner-approved 2026-07-23).
    None  -> within tolerance.
    """
    delta = abs(comp_bt - comp_live)
    if delta <= COMPOSITE_DRIFT_TOL:
        return None
    if (comp_live >= 0) != (comp_bt >= 0):
        return 'hard'
    if min(abs(comp_live), abs(comp_bt)) < NEAR_THRESHOLD_BAND:
        return 'hard'
    if delta > HARD_DRIFT_ABS:
        return 'hard'
    return 'soft'


def decision_parity(day: str) -> Dict:
    """Field-level BT↔live decision parity (2026-07-17, born from the
    z-param desync): compare every composite LIVE actually computed
    (journal 'ORB SCORED' + 'below filter threshold' lines) against the
    composite the LIVE code path produces from the nightly features CSV
    for the same symbol-day. Decision-RELEVANT drift (see
    classify_composite_drift) = the 7/10 and 7/17 bug class — a HARD
    red-day reason; decision-irrelevant drift on deep rejects is a soft
    warning. Picks matching is not enough; NUMBERS must.

    Returns {available, n_compared, mismatches: [str], warnings: [str]}.
    """
    import re
    live: Dict[str, float] = {}
    for ln in journal_grep('ORB SCORED', day):
        m = re.search(r'ORB SCORED: (\S+) comp=(-?[\d.]+)', ln)
        if m:
            live[m.group(1)] = float(m.group(2))
    for ln in journal_grep('below filter threshold', day):
        m = re.search(r'ORB: (\S+) below filter threshold — comp=(-?[\d.]+)', ln)
        if m:
            live[m.group(1)] = float(m.group(2))
    if not live:
        return {'available': False, 'n_compared': 0, 'mismatches': [],
                'warnings': []}
    try:
        import glob as _glob
        import pandas as pd
        import yaml as _yaml
        from trading.orb_filter import composite_score, load_feature_params
        paths = [p for p in sorted(_glob.glob(
            str(ROOT / 'analysis_results' / 'orb_features_*.csv')))
            if 'corrmatrix' not in p]
        if not paths:
            return {'available': False, 'n_compared': 0, 'mismatches': [],
                    'warnings': []}
        feats = pd.read_csv(paths[-1])
        feats = feats[feats['date'].astype(str).str.startswith(day)]
        cfg = _yaml.safe_load(open(ROOT / 'orb.yaml'))
        params = load_feature_params((cfg.get('filter') or {}))
        mismatches = []
        warnings = []
        n = 0
        for sym, comp_live in live.items():
            row = feats[feats['symbol'] == sym]
            if row.empty:
                continue   # universe coverage gaps are bt_parity's job
            fdict = {k: row.iloc[0][k] for k in params.keys()
                     if k in row.columns}
            comp_bt = composite_score(fdict, params)
            if comp_bt is None:
                continue
            n += 1
            sev = classify_composite_drift(comp_live, comp_bt)
            if sev:
                txt = (f"{sym}: live comp {comp_live:.4f} vs BT "
                       f"{comp_bt:.4f} (Δ{comp_bt - comp_live:+.4f})")
                (mismatches if sev == 'hard' else warnings).append(txt)
        return {'available': True, 'n_compared': n,
                'mismatches': mismatches, 'warnings': warnings}
    except Exception as e:
        print(f"WARNING: decision_parity failed: {e} — check skipped",
              flush=True)
        return {'available': False, 'n_compared': 0, 'mismatches': [],
                'warnings': []}


def news_lag_audit(day: str) -> Dict:
    """Pool-level Benzinga indexing-lag audit (2026-07-10).

    The engine snapshots its live news view (logs/orb_news_flags_<day>.json,
    written after the 9:33 lag pass). Hours later the news API is fully
    indexed — any symbol whose snapshot says no-news but whose EoD re-query
    (window ending 09:31 ET) finds articles had a premarket article that
    was NOT VISIBLE while trading decisions were made. Persistent lag =
    live silently trading the worst sizing row (−$62K/18mo grid floor) —
    this audit is the tripwire, across the WHOLE candidate pool (not just
    the day's ~4 trades, so evidence accumulates ~10x faster).

    Returns {available, n_checked, lag_symbols: [...], material: [...]}.
    """
    snap_path = ROOT / 'logs' / f'orb_news_flags_{day}.json'
    if not snap_path.exists():
        return {'available': False, 'n_checked': 0,
                'lag_symbols': [], 'material': []}
    try:
        snap = json.loads(snap_path.read_text())
    except Exception as e:
        print(f"WARNING: unreadable news snapshot {snap_path}: {e}",
              flush=True)
        return {'available': False, 'n_checked': 0,
                'lag_symbols': [], 'material': []}
    flags = snap.get('flags', {})
    pm_vols = snap.get('pm_dollar_vols', {})
    no_news = [s for s, n in flags.items() if not n]   # 0 or None
    eod = eod_news_recheck(no_news, day)
    from trading.orb_pm_mult import DEFAULT_HIGH_CUT_USD
    cfg = _orb_sizing_cfg()
    high_cut = float(cfg.get('high_cut_usd', DEFAULT_HIGH_CUT_USD))
    lag = sorted(s for s, newsy in eod.items() if newsy)
    material = [s for s in lag if (pm_vols.get(s) or 0) > high_cut]
    return {'available': True, 'n_checked': len(flags),
            'lag_symbols': lag, 'material': material}


def news_lag_line(audit: Dict) -> str:
    """One Telegram line for the lag audit ('' pre-ship / no snapshot)."""
    if not audit['available']:
        return ''
    if not audit['lag_symbols']:
        return (f"news-lag audit: clean — 0 late-indexed premarket articles "
                f"({audit['n_checked']} symbols)")
    mat = (f" — SIZING-MATERIAL (above PM$ cut): {audit['material']}"
           if audit['material'] else " (none above PM$ cut)")
    return (f"⚠ NEWS LAG: {len(audit['lag_symbols'])} symbol(s) had "
            f"premarket articles invisible to live: "
            f"{audit['lag_symbols']}{mat}. Persistent lag → consider "
            f"direct Benzinga feed.")


def sizing_block(attr: Dict) -> str:
    """Compact Telegram block for the sizing attribution."""
    lines = []
    per = []
    for t in attr['trades']:
        if t['pm_mult'] is None:
            continue
        nf = ('news✓' if t['has_news'] is True
              else 'news✗' if t['has_news'] is False else 'news?')
        if t.get('asset_class') == 'wrapper':
            nf += '/wrap'
        pm = (f"${t['pm_dollar_vol'] / 1e6:.1f}M"
              if t['pm_dollar_vol'] else 'pm?')
        pnl = f" ${t['pnl']:+,.0f}" if t['pnl'] is not None else ''
        scale = ''
        if t.get('scaled'):
            sp = t.get('scale_pnl')
            scale = (f" [scaled {t.get('scale_qty')}sh"
                     + (f" ${sp:+,.0f}" if sp is not None else '') + "]")
        per.append(f"{t['symbol']} {t['quintile']} "
                   f"q{t['adaptive_mult']}×pm{t['pm_mult']} ({nf} {pm})"
                   f"{pnl}{scale}")
    if per:
        lines.append("sizing: " + ' | '.join(per))
    for d in attr['news_drift']:
        lines.append(f"⚠ {d}")
    c = attr['cum']
    if any(v[1] for v in c.values()):
        lines.append(
            f"since {MULT_SHIP_DAY}: " + '  '.join(
                f"{k} ${v[0]:+,.0f}(n={v[1]})" for k, v in c.items()))
    return '\n'.join(lines)


# ---------------------------------------------------------------------------
# Streak persistence — the number the ramp gate reads
# ---------------------------------------------------------------------------

def streak_update(day: str, green: bool, reasons: List[str],
                  path: Path = STREAK_PATH,
                  allow_downgrade: bool = False) -> int:
    """Record the day's verdict; return the current consecutive-green count.

    Streak counts consecutive green TRADING days (weekend/holiday gaps
    don't break it — only a recorded red does). Re-running the same day
    overwrites that day's record (idempotent) — EXCEPT green→red, which is
    blocked unless allow_downgrade: the parity checks lean on journald
    evidence that ROTATES OUT, so a later re-run of an already-green day
    can false-red purely from evidence decay (2026-07-10 incident: a
    smoke re-run of 7/9 reset the real ramp streak after IQMX's
    'Not chasing' line expired from the journal). Red→green re-adjudication
    stays allowed — that direction requires FINDING evidence, not losing it.
    """
    records: List[Dict] = []
    if path.exists():
        try:
            records = json.loads(path.read_text()).get('days', [])
        except Exception:
            records = []
    prior = next((r for r in records if r.get('day') == day), None)
    if (prior and prior.get('green') and not green and not allow_downgrade):
        print(f"WARNING: refusing green→red re-adjudication of {day} "
              f"(journal evidence decays; pass allow_downgrade to force). "
              f"New reasons were: {reasons}", flush=True)
        streak = 0
        for r in reversed(sorted(records, key=lambda r: r['day'])):
            if r['green']:
                streak += 1
            else:
                break
        return streak
    records = [r for r in records if r.get('day') != day]
    records.append({'day': day, 'green': bool(green), 'reasons': reasons})
    records.sort(key=lambda r: r['day'])
    records = records[-60:]
    streak = 0
    for r in reversed(records):
        if r['green']:
            streak += 1
        else:
            break
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(
        {'streak': streak, 'needed': GREEN_SESSIONS_NEEDED, 'days': records},
        indent=1))
    return streak


def read_streak(path: Path = STREAK_PATH) -> Optional[Dict]:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


# ---------------------------------------------------------------------------
# P&L helpers
# ---------------------------------------------------------------------------

def _row_pnl(r: Dict) -> Optional[float]:
    """One closed row's realized P&L. Prefer the recorded `pnl`; else
    recompute from the FILL price and filled qty. 8/14 audit P2: the old
    `(exit-entry_price)*shares` used the PLANNED entry — every daily
    Telegram P&L line was off by entry slippage (8/13 LUNL: reported
    +$34 for a −$29.44 trade) and ignored partial fills.

    Winner stack (2026-08-22, P0-3 defense-in-depth): on a SCALED row
    (scaled_at set) the exit covered only the runner qty — the recompute
    subtracts scale_qty and adds the banked scale_pnl. The recorded `pnl`
    is already combined (single-writer rule), so this only matters for the
    NULL-pnl fallback path."""
    if r.get('pnl') is not None:
        return float(r['pnl'])
    if r.get('exit_price') is None:
        return None
    entry = r.get('fill_price') or r.get('entry_price')
    if entry is None:
        return None
    qty = r.get('filled_qty') or r.get('shares') or 0
    scale_pnl = 0.0
    if r.get('scaled_at'):
        qty = max(0, qty - int(r.get('scale_qty') or 0))
        scale_pnl = float(r.get('scale_pnl') or 0.0)
    return (float(r['exit_price']) - float(entry)) * qty + scale_pnl


def realized_pnl(day: str) -> Dict[str, float]:
    """Realized P&L per strategy for a date (closed rows only)."""
    out: Dict[str, float] = {}
    for r in load_live_rows(day):
        pnl = _row_pnl(r)
        if pnl is None:
            continue
        out[r.get('strategy') or '?'] = out.get(r.get('strategy') or '?', 0.0) + pnl
    return out


def cumulative_orb_since(start: str) -> float:
    """Cumulative realized ORB P&L. The NULL-pnl fallback is scale-aware
    (winner stack P0-3c): runner qty = filled_qty − scale_qty and the
    banked scale_pnl is added — non-scaled rows are unchanged
    (COALESCE(scale_qty,0)=0)."""
    conn = sqlite3.connect(ROOT / 'data' / 'trades.db', timeout=15)
    row = conn.execute(
        """SELECT COALESCE(SUM(
                 COALESCE(pnl,
                          (exit_price - COALESCE(fill_price, entry_price))
                          * (COALESCE(filled_qty, shares)
                             - COALESCE(scale_qty, 0))
                          + COALESCE(scale_pnl, 0))), 0)
           FROM trades WHERE strategy='orb' AND trade_date >= ?
             AND exit_price IS NOT NULL""", (start,)).fetchone()
    conn.close()
    return float(row[0])


BF_RAIL_DEFAULTS = {'daily_usd': -800.0, 'weekly_usd': -1200.0,
                    'month_pause_usd': -2500.0}
BF_MONTH_PAUSE_FLAG = ROOT / 'data' / 'bf_month_pause.flag'


def _bf_rails_cfg() -> Dict:
    """BF kill-rail thresholds from config.yaml (trading.bull_flag.kill_rails).

    Falls back to the shipped defaults with a printed WARNING if the config
    is unreadable or lacks the block (fallback rule: never silent)."""
    try:
        import yaml
        cfg = yaml.safe_load(open(ROOT / 'config.yaml')) or {}
        kr = ((cfg.get('trading') or {}).get('bull_flag') or {}) \
            .get('kill_rails') or {}
        if not kr:
            print("WARNING: config.yaml lacks trading.bull_flag.kill_rails — "
                  "using shipped defaults", flush=True)
        return {**BF_RAIL_DEFAULTS, **kr}
    except Exception as e:
        print(f"WARNING: _bf_rails_cfg could not read config.yaml ({e}) — "
              f"using shipped defaults", flush=True)
        return dict(BF_RAIL_DEFAULTS)


def bf_rails_status(day: str) -> Dict:
    """BF kill-rail state for the EoD dive (Discipline Program Phase 1).

    Reads the SAME sums the live rails read (trades.db, strategy='bull_flag',
    pnl IS NOT NULL, ET trade_date keys: day / ISO-week Monday / month first)
    plus the persistent month-pause flag file. FAIL-CLOSED presentation: a DB
    error reports all rails breached with query_failed=True (mirrors the
    engine's -1e9 sentinel) rather than a green line on unknown data."""
    from datetime import date as _date
    kr = _bf_rails_cfg()
    d = _date.fromisoformat(day)
    week_start = (d - timedelta(days=d.weekday())).isoformat()
    month_start = day[:8] + '01'
    out = {'enabled': bool(kr.get('enabled', True)),
           'daily_limit': float(kr['daily_usd']),
           'weekly_limit': float(kr['weekly_usd']),
           'month_pause_limit': float(kr['month_pause_usd']),
           'month_pause_flag': BF_MONTH_PAUSE_FLAG.exists(),
           'query_failed': False}
    try:
        conn = sqlite3.connect(ROOT / 'data' / 'trades.db', timeout=15)
        sums = {}
        for key, since in (('daily', day), ('weekly', week_start),
                           ('monthly', month_start)):
            row = conn.execute(
                "SELECT COALESCE(SUM(pnl),0) FROM trades WHERE "
                "strategy='bull_flag' AND trade_date>=? AND pnl IS NOT NULL",
                (since,)).fetchone()
            sums[key] = float(row[0] or 0.0)
        conn.close()
        out.update(sums)
    except Exception as e:
        print(f"WARNING: bf_rails_status query failed ({e}) — reporting "
              f"FAIL-CLOSED (all rails breached)", flush=True)
        out.update({'daily': -1e9, 'weekly': -1e9, 'monthly': -1e9,
                    'query_failed': True})
    out['daily_breached'] = out['daily'] <= out['daily_limit']
    out['weekly_breached'] = out['weekly'] <= out['weekly_limit']
    out['month_breached'] = out['monthly'] <= out['month_pause_limit']
    out['month_paused'] = out['month_pause_flag'] or out['month_breached']
    return out


def bf_rails_line(st: Dict) -> str:
    """One EoD-dive line for the BF kill-rail state (grep: 'BF RAILS')."""
    if st.get('query_failed'):
        return "BF RAILS QUERY FAILED — fail-closed (assume all breached)"
    def _cell(val, limit, breached):
        return f"${val:+,.0f}/{limit:,.0f}{' BREACH' if breached else ''}"
    parts = [
        f"BF RAILS{'' if st['enabled'] else ' (DISABLED)'}",
        f"day {_cell(st['daily'], st['daily_limit'], st['daily_breached'])}",
        f"wk {_cell(st['weekly'], st['weekly_limit'], st['weekly_breached'])}",
        f"mo {_cell(st['monthly'], st['month_pause_limit'], st['month_breached'])}",
        f"pause={'YES' if st['month_paused'] else 'no'}",
    ]
    return ' | '.join(parts)


def send_telegram(msg: str) -> bool:
    """Send unless running under pytest. Returns True on success."""
    if os.environ.get('PYTEST_CURRENT_TEST'):
        return False
    try:
        from dotenv import load_dotenv
        load_dotenv(str(ROOT / '.env'))
        from notifications.telegram_notifier import TelegramNotifier
        n = TelegramNotifier(os.getenv('TELEGRAM_BOT_TOKEN'),
                             os.getenv('TELEGRAM_CHAT_ID'), enabled=True)
        n.send_message_sync(msg, parse_mode='HTML')
        return True
    except Exception:
        return False


def prev_trading_day_utc(today: Optional[date] = None) -> str:
    """Most recent weekday (approximation; holiday-running is harmless —
    a no-data day yields an empty-but-green verdict that's overwritten)."""
    d = today or datetime.utcnow().date()
    while d.weekday() >= 5:
        d -= timedelta(days=1)
    return d.isoformat()
