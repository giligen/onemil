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

def latest_bt_trades_csv() -> Optional[str]:
    paths = sorted(glob.glob(str(ROOT / 'analysis_results' / 'orb_static_lock_trades.csv')))
    return paths[-1] if paths else None


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
    """Latest date present in the BT trades CSV — staleness detector.
    The nightly orb-backtest regenerates it; if its max date < the day
    under check, BT-parity is UNKNOWN (skipped), not clean."""
    import pandas as pd
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
    bt_syms = set() if bt_stale else {r['symbol'] for r in load_bt_selected(day)}
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
        explained_lines = (journal_grep('skipped —', day)
                           + journal_grep('spread', day)
                           + journal_grep('ENTRY SKIPPED', day))
        for sym in sorted(missing):
            if any(sym in ln and any(k in ln for k in
                   ('spread', 'buying power', 'other strategy', 'halted',
                    'breakout extended past limit', 'Not chasing'))
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

    rekey = journal_grep('REKEY', day) + journal_grep('negative-age', day)
    checks['touchgo_tripwires'] = 'OK' if not rekey else f"{len(rekey)} tripwire line(s)"

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
             'filled': r.get('fill_price') is not None}
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

    news_drift = []
    check_syms = [t['symbol'] for t in trades
                  if t['pm_mult'] is not None and t['has_news'] is not True]
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
        per.append(f"{t['symbol']} {t['quintile']} "
                   f"q{t['adaptive_mult']}×pm{t['pm_mult']} ({nf} {pm}){pnl}")
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

def realized_pnl(day: str) -> Dict[str, float]:
    """Realized P&L per strategy for a date (closed rows only)."""
    out: Dict[str, float] = {}
    for r in load_live_rows(day):
        if r.get('exit_price') is None or r.get('entry_price') is None:
            continue
        pnl = (r['exit_price'] - r['entry_price']) * (r.get('shares') or 0)
        out[r.get('strategy') or '?'] = out.get(r.get('strategy') or '?', 0.0) + pnl
    return out


def cumulative_orb_since(start: str) -> float:
    conn = sqlite3.connect(ROOT / 'data' / 'trades.db', timeout=15)
    row = conn.execute(
        """SELECT COALESCE(SUM((exit_price-entry_price)*shares), 0)
           FROM trades WHERE strategy='orb' AND trade_date >= ?
             AND exit_price IS NOT NULL""", (start,)).fetchone()
    conn.close()
    return float(row[0])


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
