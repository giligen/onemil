"""ORB selection observer — catch the candidate-dropper in the act.

Context (2026-07-03): BT-selected winners (CRCD +$15.8K model 6/30, AVEX
6/30, FABC 6/11, RGNX 6/22) were NEVER SUBMITTED live — no DB row, i.e.
they died in the scanner→universe→scoring funnel, not at order time. All
log evidence for those days has rotated. This script runs OUTSIDE the
trader and captures, for ONE live session, everything needed to assign
blame per layer:

  09:29:30 ET  snapshot the seed the scanner WOULD build (same SQL on
               daily_bars) → layer-0 truth
  09:31:00 ET  independently fetch Alpaca snapshots for the full seed via
               the ORB account and apply BT-parity criteria (gap>=5%,
               open $3-30, prev_vol>=500K) → EXPECTED candidate set
  09:36:30 ET  read the journal for live's actual 'range complete' /
               'ORB SCORED' / 'ENTRY SUBMITTED' lines → LIVE set
  then         diff EXPECTED vs LIVE per layer; dump everything to
               logs/orb_selection_audit_<date>.json and print verdict.

Run: nohup python3 scripts/orb_selection_observer.py > /tmp/orb_observer.log 2>&1 &
(Requires being started BEFORE 13:29 UTC on a trading day.)
"""
from __future__ import annotations

import json
import os
import re
import sqlite3
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from monitoring.cron_guard import run_guarded  # noqa: E402
from dotenv import load_dotenv
load_dotenv(str(ROOT / '.env'))

from data_sources.alpaca_client import AlpacaClient

NY = ZoneInfo('America/New_York')
OPS_WARNING_TEXT = 'observer: engine log not parsed — no verdict'
OUT = ROOT / 'logs' / f"orb_selection_audit_{datetime.now(NY).strftime('%Y%m%d')}.json"
audit = {'date': datetime.now(NY).strftime('%Y-%m-%d')}


def log(msg):
    print(f"[{datetime.now(timezone.utc).strftime('%H:%M:%S')}Z] {msg}", flush=True)


def et_target(now, hh, mm, ss=0):
    """Return the America/New_York wall-clock instant hh:mm:ss on `now`'s ET date.

    Pure (no sleeping) so the DST behaviour is unit-testable: 09:29:30 ET is
    13:29:30 UTC in EDT and 14:29:30 UTC in EST. The observer used UTC literals
    until 2026-11-03 and ran an hour early after DST ended on 11/1.
    """
    now_et = now.astimezone(NY)
    return now_et.replace(hour=hh, minute=mm, second=ss, microsecond=0)


def wait_until_et(hh, mm, ss=0):
    """Sleep until hh:mm:ss America/New_York today (never a UTC literal)."""
    while True:
        now = datetime.now(timezone.utc)
        d = (et_target(now, hh, mm, ss) - now.astimezone(NY)).total_seconds()
        if d <= 0:
            return
        time.sleep(min(d, 20))


def build_seed():
    """Replicate _orb_universe_source layer-0 exactly."""
    conn = sqlite3.connect(ROOT / 'data' / 'cache.db', timeout=20)
    cur = conn.execute("""
        SELECT symbol FROM (
            SELECT symbol, close, volume,
                   ROW_NUMBER() OVER (PARTITION BY symbol ORDER BY bar_date DESC) AS rn
            FROM daily_bars
        ) WHERE rn = 1 AND close BETWEEN 1.0 AND 50.0 AND volume >= 500000
    """)
    syms = sorted({r[0] for r in cur})
    conn.close()
    return syms


def bt_parity_filter(client, seed):
    """Independently apply BT-parity criteria via snapshots (ORB account)."""
    snaps = client.get_snapshots(seed)
    kept, rejected = {}, {}
    for sym, s in (snaps or {}).items():
        op = float(s.get('open', 0) or 0) or float(s.get('latest_price', 0) or 0)
        pc = float(s.get('prev_close', 0) or 0)
        pv = int(s.get('prev_volume', 0) or 0)
        if op <= 0 or pc <= 0:
            rejected[sym] = 'no_price'
            continue
        gap = (op - pc) / pc * 100.0
        if not (3.0 <= op <= 30.0):
            rejected[sym] = f'price {op:.2f}'
        elif gap < 5.0:
            rejected[sym] = f'gap {gap:.1f}'
        elif pv < 500_000:
            rejected[sym] = f'prev_vol {pv}'
        else:
            kept[sym] = {'open': op, 'gap': round(gap, 2), 'prev_vol': pv}
    missing = sorted(set(seed) - set(snaps or {}))
    return kept, rejected, missing


def _live_submitted_symbols(submitted_lines, dry_lines):
    """Union of real 'ORB ENTRY SUBMITTED' picks and dry-run
    '[ORB DRY] WOULD BUY' picks for one day (docs/orb_dry_run_spec_20260925.md).

    Pure function (no journalctl I/O) so the dry-line parsing is unit
    testable directly. Addon-pool '[ORB+ DRY]' lines are never passed in
    here (main() filters on the literal '[ORB DRY] WOULD BUY' prefix), so
    they stay excluded exactly as before this mode existed.
    """
    return sorted(
        {l.split('ORB ENTRY SUBMITTED: ')[1].split()[0] for l in submitted_lines}
        | {l.split('WOULD BUY ')[1].split()[0] for l in dry_lines})


_JOURNAL = {'lines': None, 'ok': False}


def journal_since_arg(now, hh=9, mm=25):
    """journalctl --since string for hh:mm ET today, expressed in explicit UTC."""
    t = et_target(now, hh, mm).astimezone(timezone.utc)
    return t.strftime('%Y-%m-%d %H:%M:%S UTC')


def load_journal(timeout=180):
    """Read the onemil-trader journal ONCE since 09:25 ET into the module cache.

    One call (not one per pattern) so the 2.3M-line flood days cannot stack
    30 s timeouts. Sets _JOURNAL['ok'] False and logs WARNING on any failure.
    """
    since = journal_since_arg(datetime.now(timezone.utc))
    try:
        res = subprocess.run(
            ['journalctl', '-u', 'onemil-trader', '--since', since, '--no-pager'],
            capture_output=True, text=True, timeout=timeout)
        if res.returncode != 0:
            raise RuntimeError(f"journalctl rc={res.returncode}: {res.stderr[:200]}")
        _JOURNAL['lines'] = res.stdout.splitlines()
        _JOURNAL['ok'] = True
    except Exception as e:
        log(f"WARNING journalctl failed/timed out: {e}")
        _JOURNAL['lines'] = []
        _JOURNAL['ok'] = False
    return _JOURNAL['ok']


def journal_lines(pattern):
    """Lines of the cached journal containing `pattern` (loads it on first use)."""
    if _JOURNAL['lines'] is None:
        load_journal()
    return [l for l in _JOURNAL['lines'] if pattern in l]


_SYM = r"[A-Z][A-Z0-9.\-]{0,9}"


def preplace_symbols(lines):
    """Symbols the engine ranked on the preplace path, from its own log lines.

    The 10/30 failure: the engine ranked/vetoed names via '[ORB PREPLACE]'
    lines and logged neither 'range complete' nor 'ORB SCORED', so the parse
    came back empty although the engine worked. Sources: 'reconcile final
    top-N of M ranked: [...] | preplaced: [...]', 'provisional top-N ...: A(rh=',
    'provisional veto X'.
    """
    out = set()
    for l in lines:
        if 'reconcile final' in l:
            out |= set(re.findall(rf"'({_SYM})'", l.split('reconcile final', 1)[1]))
        elif 'provisional top-' in l:
            out |= set(re.findall(rf"({_SYM})\(rh=", l))
        elif 'provisional veto' in l:
            m = re.search(rf"provisional veto ({_SYM}) \(", l)
            if m:
                out.add(m.group(1))
    return out


def engine_active(journal_ok_lines):
    """True when the engine logged any decision line (submit/score/reconcile)."""
    keys = ('ENTRY SUBMITTED', 'ORB SCORED', 'reconcile final')
    return any(k in l for l in journal_ok_lines for k in keys)


def decide_verdict(live_range, live_scored, active, journal_ok, real_drops):
    """Pick the Telegram verdict: 'ops_warning' | 'caught' | 'clean'.

    CAUGHT is only legal when the live set is non-empty: an empty live set
    means the parse failed (or the engine did not run), never that every
    candidate was dropped (false alarm 2026-11-02). `active` is kept in the
    signature for the log line; the ops warning covers every empty-live case.
    """
    if not journal_ok or (not live_range and not live_scored):
        return 'ops_warning'
    return 'caught' if real_drops else 'clean'


def classify_drop(sym: str, now=None, http_get=None) -> str:
    """Classify one dropped symbol from its 09:30-09:34 SIP 1-min bars.

    The bars window is EXCLUSIVE of the 09:35 bar (end = 09:34:59 ET), so the
    count is out of 5, never "6/5" (2026-11-10). `now`/`http_get` are test
    seams (default: wall clock / requests.get)."""
    now = now or datetime.now(timezone.utc)
    try:
        if http_get is None:
            import requests as _rq
            http_get = _rq.get
        h = {'APCA-API-KEY-ID': os.environ['ALPACA_API_KEY'],
             'APCA-API-SECRET-KEY': os.environ['ALPACA_API_SECRET']}
        r = http_get('https://data.alpaca.markets/v2/stocks/bars',
                     params={'symbols': sym, 'timeframe': '1Min',
                             'feed': 'sip', 'limit': 10,
                             'start': et_target(now, 9, 30)
                             .astimezone(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'),
                             'end': et_target(now, 9, 34, 59)
                             .astimezone(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')},
                     headers=h, timeout=(5, 15))
        n_bars = len((r.json().get('bars') or {}).get(sym, []))
    except Exception as e:
        return f'UNKNOWN (bars query failed: {e})'
    if n_bars == 0:
        return 'benign: vendor-stale phantom (zero bars in 9:30-9:35)'
    if n_bars < 5:
        return (f'benign: gappy 5-min window ({n_bars}/5 bars) — '
                f'dropped by design (5-bar rule, BT parity)')
    return f'REAL DROP ({n_bars}/5 range bars present — investigate)'


def send_verdict(n, verdict, audit, kept, live_scored, real_drops, drop_class,
                 dropped, active) -> bool:
    """Deliver the verdict. Returns True when a Telegram message was sent.

    'ops_warning' and 'caught' send; 'clean' only logs (2026-11-10 owner: the
    daily 'No REAL drops' message was noise; the EOD report carries a
    'selection parity' line from the audit JSON instead)."""
    if verdict == 'ops_warning':
        log(f"WARNING {OPS_WARNING_TEXT} (journal_ok={_JOURNAL['ok']}, engine_active={active})")
        n.send_message_sync(f"<b>[ORB-SELECTION] {audit['date']}</b>\n{OPS_WARNING_TEXT}",
                            parse_mode='HTML')
        return True
    if verdict == 'caught':
        det = "\n".join(f"• {s}: gap {kept[s]['gap']}%, "
                        f"ranged={s in audit.get('live_range', [])} — {drop_class[s]}"
                        for s in real_drops[:6])
        n.send_message_sync(
            f"<b>[ORB-SELECTION] {audit['date']} — CAUGHT {len(real_drops)} REAL dropped candidate(s)</b>\n"
            f"{det}\nExpected {len(kept)}, live scored {len(live_scored)}. "
            f"Full: logs/{OUT.name}", parse_mode='HTML')
        return True
    benign = len(dropped) - len(real_drops)
    log(f"clean: No REAL drops — live scored all tradeable BT-parity "
        f"candidates ({benign} benign drop(s)); Telegram suppressed")
    return False


def main():
    log("observer armed — waiting for 09:29:30 ET")
    wait_until_et(9, 29, 30)

    seed = build_seed()
    audit['seed_n'] = len(seed)
    audit['seed'] = seed
    log(f"layer-0 seed: {len(seed)} symbols")

    wait_until_et(9, 31, 0)
    client = AlpacaClient(os.getenv('ALPACA_ORB_API_KEY'), os.getenv('ALPACA_ORB_API_SECRET'),
                          paper=os.getenv('ALPACA_ORB_PAPER', 'true').lower() == 'true')
    t0 = time.time()
    kept, rejected, missing = bt_parity_filter(client, seed)
    audit['expected_candidates'] = kept
    audit['rejected_931'] = rejected
    audit['snapshot_missing'] = missing
    audit['snapshot_secs'] = round(time.time() - t0, 1)
    log(f"EXPECTED candidates (BT-parity): {len(kept)} — {sorted(kept)}")
    log(f"snapshot coverage: {len(seed)-len(missing)}/{len(seed)} in {audit['snapshot_secs']}s")

    # SECOND capture at ~9:35:03 — the instant live ranks — including
    # bid/ask so the live-only spread gate (>300bps skip) is attributable.
    # (Gate is 300bps since 2026-07-04 — orb.yaml entry.max_spread_bps; the
    # observer previously flagged >150, over-flagging by construction.)
    wait_until_et(9, 35, 3)
    try:
        snaps935 = client.get_snapshots(sorted(kept))
        audit['snap_935'] = {
            s2: {'open': v.get('open'), 'prev_close': v.get('prev_close'),
                 'bid': v.get('bid_price'), 'ask': v.get('ask_price'),
                 'spread_bps': (round((v.get('ask_price', 0) - v.get('bid_price', 0))
                                / v.get('bid_price') * 10000, 1)
                                if v.get('bid_price') else None)}
            for s2, v in (snaps935 or {}).items()
        }
        wide = [s2 for s2, v in audit['snap_935'].items()
                if v['spread_bps'] is not None and v['spread_bps'] > 300]
        log(f"9:35 spreads captured for {len(audit['snap_935'])} expected; "
            f">300bps (live spread-gate would skip): {wide}")
        audit['spread_gate_would_skip'] = wide
    except Exception as e:
        log(f"9:35 snapshot capture failed: {e}")

    # give live until 9:36:30 to complete ranges + score
    wait_until_et(9, 36, 30)
    load_journal()
    preplace = preplace_symbols(journal_lines('ORB PREPLACE'))
    live_range = sorted({l.split('ORB: ')[1].split(' range complete')[0]
                         for l in journal_lines('range complete')
                         if 'ORB: ' in l and ' range complete' in l} | preplace)
    # 2026-07-13 fix: since the 7/10 below-threshold logging change,
    # composite-rejected candidates log as "ORB: X below filter threshold"
    # instead of "ORB SCORED: X" — they WERE scored (correct rejection).
    # The old parser counted them as dropped and false-alarmed
    # "CAUGHT 10 dropped" on 7/13 (8 of the 10 were threshold rejects).
    live_scored = sorted(
        {l.split('ORB SCORED: ')[1].split()[0]
         for l in journal_lines('ORB SCORED')}
        | {l.split('ORB: ')[1].split()[0]
           for l in journal_lines('below filter threshold')
           if 'ORB: ' in l}
        | preplace)
    # docs/orb_dry_run_spec_20260925.md: a dry-run week never emits 'ORB
    # ENTRY SUBMITTED' (no order is placed), so the parity observer must
    # also count '[ORB DRY] WOULD BUY <sym> ...' lines as picks, or every
    # dry day reads as zero live picks vs the BT book. '[ORB+ DRY]' (addon
    # pools) lines do not match this literal prefix, so they stay excluded
    # here exactly as before.
    live_submitted = _live_submitted_symbols(
        journal_lines('ORB ENTRY SUBMITTED'),
        journal_lines('[ORB DRY] WOULD BUY'))
    seed_lines = journal_lines('ORB universe seed')[-2:]
    snap_lines = journal_lines('Fetched snapshots for')[-4:]

    audit.update(live_range=live_range, live_scored=live_scored,
                 live_submitted=live_submitted,
                 live_seed_log=seed_lines, live_snap_log=snap_lines)

    exp = set(kept)
    dropped = sorted(exp - set(live_scored))
    extra = sorted(set(live_scored) - exp)
    audit['dropped_by_live'] = dropped
    audit['live_only'] = extra

    # Classify drops before alarming (2026-07-23: 6/6 "CAUGHT" symbols
    # were benign — 3 vendor-stale phantoms with months-old snapshots
    # (ORIS/CUK/MIGI), 3 thin names with gappy 5-min windows dropped BY
    # DESIGN in both live and BT via the 5-bar rule, orb_engine.py:977
    # == study_orb_features.py:237). Only unexplained drops alarm.
    _classify_drop = classify_drop

    drop_class = {s: _classify_drop(s) for s in dropped}
    real_drops = [s for s, c in drop_class.items() if c.startswith('REAL')
                  or c.startswith('UNKNOWN')]
    audit['drop_classification'] = drop_class

    log("=" * 60)
    log(f"LIVE ranged: {len(live_range)} | scored: {len(live_scored)} | submitted: {live_submitted}")
    log(f"*** DROPPED BY LIVE (expected but never scored): {dropped}")
    log(f"live-only (scored but not in expected): {extra}")
    for s in dropped:
        info = kept[s]
        in_range = s in live_range
        log(f"   {s}: open={info['open']} gap={info['gap']}% pv={info['prev_vol']:,} "
            f"| live_range_complete={in_range} -> died at "
            f"{'SCORING' if in_range else 'UNIVERSE/RANGE layer'} "
            f"| {drop_class[s]}")

    OUT.write_text(json.dumps(audit, indent=1))
    log(f"full audit -> {OUT}")

    # Telegram verdict
    active = engine_active(_JOURNAL['lines'] or [])
    verdict = decide_verdict(live_range, live_scored, active, _JOURNAL['ok'], real_drops)
    audit['verdict'] = verdict
    try:
        from notifications.telegram_notifier import TelegramNotifier
        n = TelegramNotifier(os.getenv('TELEGRAM_BOT_TOKEN'), os.getenv('TELEGRAM_CHAT_ID'), enabled=True)
        send_verdict(n, verdict, audit, kept, live_scored, real_drops, drop_class,
                     dropped, active)
    except Exception as e:
        log(f"telegram failed: {e}")


if __name__ == '__main__':
    run_guarded(main, 'orb_selection_observer')
