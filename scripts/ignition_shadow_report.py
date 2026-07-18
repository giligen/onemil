"""Ignition S1 daily shadow report → standalone Telegram (2026-07-19).

Reads logs/ignition_shadow_<day>.jsonl, resims exits for each
SHADOW_TRIGGER from the day's real bars (harness physics: static lock
1.75R/0.5R, 15:45 flat), and reports the S1 week-1 scoreboard:
  1. trigger rate vs harness expectation (bounds 1-8/day)
  2. spread reality (median/p90 bps vs the <=60/<=150 pass bars)
  3. detection latency (p90 vs <=90s pass bar)
  4. hypothetical P&L at ASK-entry vs harness-modeled entry
  5. shadow overhead (bars_fetch_s) — must stay trivial
  6. skip taxonomy (eval caps, no-catalyst, R-too-small, gap)
Week-to-date go/no-go tracker prints against the S1 gate.

Cron: 21:40 UTC weekdays. Usage: [--date YYYY-MM-DD] [--no-telegram]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import report_common as rc

ARM, LOCK = 1.75, 0.5
PASS = {'trig_lo': 1, 'trig_hi': 8, 'spread_med': 60, 'spread_p90': 150,
        'latency_p90': 90}


def resim_exit(bars, entry, stop, entry_min):
    """Harness-physics exit resim (static lock ARM/LOCK, 15:45 flat).
    Conservative intrabar ordering: stop checked before arming. A bar
    that OPENS below the live stop fills near its open, not at the stop
    (gap-down realism — stop-price fills there would overstate P&L)."""
    cur = stop
    armed = False
    R = entry - stop
    post = bars[bars['m'] > entry_min]
    for _, r in post.iterrows():
        if r['m'] >= 945:
            return (r['open'] - entry) / R, 'eod'
        if r['low'] <= cur:
            fill = min(cur, r['open'])
            return (fill * 0.999 - entry) / R, 'lock' if armed else 'stop'
        if not armed and r['high'] >= entry + ARM * R:
            armed = True
            cur = entry + LOCK * R
    if len(post):
        return (post.iloc[-1]['close'] - entry) / R, 'eod'
    return 0.0, 'none'


def day_bars(symbol, day):
    import os

    import pandas as pd
    import requests
    from dotenv import load_dotenv
    load_dotenv(str(rc.ROOT / '.env'))
    H = {'APCA-API-KEY-ID': os.environ['ALPACA_API_KEY'],
         'APCA-API-SECRET-KEY': os.environ['ALPACA_API_SECRET']}
    st = pd.Timestamp(day).tz_localize('America/New_York') \
        + pd.Timedelta(hours=9, minutes=30)
    en = pd.Timestamp(day).tz_localize('America/New_York') \
        + pd.Timedelta(hours=16)
    r = requests.get('https://data.alpaca.markets/v2/stocks/bars',
                     params={'symbols': symbol, 'timeframe': '1Min',
                             'feed': 'sip', 'limit': 10000,
                             'start': st.tz_convert('UTC').isoformat(),
                             'end': en.tz_convert('UTC').isoformat()},
                     headers=H, timeout=(5, 30))
    r.raise_for_status()
    bars = (r.json().get('bars') or {}).get(symbol, [])
    if not bars:
        return None
    df = pd.DataFrame(bars).rename(columns={
        'o': 'open', 'h': 'high', 'l': 'low', 'c': 'close', 'v': 'volume'})
    ts = pd.to_datetime(df['t'], utc=True).dt.tz_convert('America/New_York')
    df['m'] = ts.dt.hour * 60 + ts.dt.minute
    return df


def pct(vals, q):
    import numpy as np
    v = [x for x in vals if x is not None]
    return float(np.percentile(v, q)) if v else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--date', default=None)
    ap.add_argument('--no-telegram', action='store_true')
    args = ap.parse_args()
    day = args.date or rc.prev_trading_day_utc()
    path = rc.ROOT / 'logs' / f'ignition_shadow_{day}.jsonl'
    if not path.exists():
        # A dead shadow must be VISIBLE, not silent — no journal on a
        # weekday means shadow dead/disabled, holiday, or (rare) a day
        # with zero +10% movers before 10:30. Telegram either way.
        msg = (f"🟠 <b>[IGNITION SHADOW] {day}</b> — NO journal file. "
               f"Shadow dead/disabled, market holiday, or zero "
               f"qualifying movers? Check: journalctl | grep "
               f"IgnitionShadow")
        print(msg)
        if not args.no_telegram:
            rc.send_telegram(msg)
        return 0
    recs = [json.loads(ln) for ln in path.read_text().splitlines() if ln]
    trigs = [r for r in recs if r.get('verdict') == 'SHADOW_TRIGGER']
    skips: dict = {}
    for r in recs:
        v = r.get('verdict', '?')
        if v != 'SHADOW_TRIGGER':
            skips[v] = skips.get(v, 0) + 1
    spreads = [r.get('spread_bps') for r in recs if r.get('spread_bps')]
    lats = [r.get('latency_s') for r in recs if r.get('latency_s')]
    fetches = [r.get('bars_fetch_s') for r in recs if r.get('bars_fetch_s')]
    pnl_lines = []
    tot = 0.0
    for r in trigs:
        try:
            b = day_bars(r['symbol'], day)
            if b is None:
                continue
            entry = float(r.get('hypo_entry') or r['price'])
            stop = float(r['hypo_stop'])
            if entry <= stop:
                # inverted/stale quote at capture -> R<=0, resim math
                # is meaningless; surface instead of emitting nonsense
                pnl_lines.append(
                    f"• {r['symbol']}: SKIPPED resim — entry "
                    f"{entry} <= stop {stop} (bad quote at capture)")
                continue
            # complex_late triggers become actionable at confirm time,
            # not first sighting — resim exits from minute_final_et
            rr, reason = resim_exit(
                b, entry, stop,
                r.get('minute_final_et') or r['minute_et'])
            pos = float(r.get('hypo_position_usd') or 0)
            pnl = rr * (entry - stop) / entry * pos
            tot += pnl
            pnl_lines.append(
                f"• {r['symbol']} +{r['intraday_change_pct']}% "
                f"[{r.get('catalyst', '?')}] spread "
                f"{r.get('spread_bps', '?')}bps → {rr:+.2f}R "
                f"${pnl:+,.0f} ({reason})")
        except Exception as e:
            pnl_lines.append(f"• {r['symbol']}: resim failed {e}")
    med_sp = pct(spreads, 50)
    p90_sp = pct(spreads, 90)
    p90_lat = pct(lats, 90)
    checks = []
    checks.append(('trigger rate',
                   PASS['trig_lo'] <= len(trigs) <= PASS['trig_hi'],
                   f"{len(trigs)}/day (pass {PASS['trig_lo']}-{PASS['trig_hi']})"))
    if med_sp is not None:
        checks.append(('spread median', med_sp <= PASS['spread_med'],
                       f"{med_sp:.0f}bps (pass<= {PASS['spread_med']})"))
        checks.append(('spread p90', p90_sp <= PASS['spread_p90'],
                       f"{p90_sp:.0f}bps (pass<= {PASS['spread_p90']})"))
    if p90_lat is not None:
        checks.append(('latency p90', p90_lat <= PASS['latency_p90'],
                       f"{p90_lat:.0f}s (pass<= {PASS['latency_p90']})"))
    if fetches:
        checks.append(('shadow overhead', max(fetches) <= 6.5,
                       f"max bars fetch {max(fetches):.1f}s"))
    ok = all(c[1] for c in checks) if checks else False
    icon = '🟢' if ok else '🟠'
    msg = (f"{icon} <b>[IGNITION SHADOW] {day}</b> — "
           f"{len(trigs)} trigger(s), hypothetical ${tot:+,.0f}\n"
           + '\n'.join(pnl_lines[:8]) + ('\n' if pnl_lines else '')
           + 'checks: ' + '  '.join(
               f"{'✓' if c1 else '✗'}{c0}={c2}" for c0, c1, c2 in checks)
           + f"\nskips: {skips}")
    print(msg)
    if not args.no_telegram:
        rc.send_telegram(msg)
    return 0


if __name__ == '__main__':
    sys.exit(main())
