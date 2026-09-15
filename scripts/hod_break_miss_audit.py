#!/usr/bin/env python3
"""HOD-break MISS AUDIT — the two-sided parity the EOD check lacked (2026-09-15, after the CRWL miss).

The spec is run over the WHOLE scanner universe for the day (every symbol whose day-running bars show a
high >= 5% above the 09:30 open, price >= the live floor), producing the complete set of spec signals.
Each is then classified against the engine's journal:
  EVALUATED   the engine saw that break (a [HOD DRY]/BUY line or a rule skip on that symbol)
  MISSED      the spec signalled but the engine never evaluated the symbol at that break — with the cause:
              never_admitted / admitted_after_break (late admission) / admitted_before_break (engine-side bug)
The miss rate is THE input-parity number. Usage: python3 scripts/hod_break_miss_audit.py [--floor 20]
Read-only. Alpaca REST only (get_current_bars over the universe, then 1-min bars for the movers).
"""
import os, re, subprocess, sys
from datetime import datetime, timezone
from zoneinfo import ZoneInfo
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..')); sys.path.insert(0, ROOT); os.chdir(ROOT)
from config import Config                                   # noqa: E402
from data_sources.alpaca_client import AlpacaClient         # noqa: E402
from persistence.database import Database                   # noqa: E402
from trading.hod_break import HodBreakParams, detect, OPEN_MINUTE   # noqa: E402
ET = ZoneInfo('America/New_York')


def journal_events(since='12:29'):
    out = subprocess.run(['bash', '-c', f'journalctl -u onemil-trader --since "{since}" --no-pager -o short-iso | grep -E "\\[HOD"'], capture_output=True, text=True).stdout.splitlines()
    admitted, evaluated, stale = {}, {}, {}
    for ln in out:
        ts = re.match(r'(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}[+-]\d{4})', ln)
        m_et = None
        if ts:
            t = datetime.strptime(ts.group(1), '%Y-%m-%dT%H:%M:%S%z').astimezone(ET); m_et = t.hour * 60 + t.minute
        a = re.search(r'candidate (\S+) admitted', ln)
        if a and a.group(1) not in admitted: admitted[a.group(1)] = m_et
        e = re.search(r'\[HOD(?: DRY)?\] (?:WOULD BUY |BUY |ENTRY SUBMITTED )?(\S+?):? (?:level|ask|spread|stop|per-day|concurrency|notional)', ln)
        if e: evaluated.setdefault(e.group(1), []).append(m_et)
        st = re.search(r'\[HOD\] (\S+): MISSED the spec\'s break at bar (\d+)', ln)
        if st: stale[st.group(1)] = m_et
    return admitted, evaluated, stale


def streamed_symbols(day: str) -> set:
    """Symbols the engine streamed from the open (written at session start); empty when the file is absent."""
    try:
        with open(f'logs/hod_stream_universe_{day}.txt') as f: return {x.strip() for x in f if x.strip()}
    except FileNotFoundError:
        return set()


def main() -> int:
    floor = float(sys.argv[sys.argv.index('--floor') + 1]) if '--floor' in sys.argv else None
    cfg = Config(); hb = cfg.hod_break_cfg; p = HodBreakParams(**hb['params']); floor = floor or hb['min_price']
    alp = AlpacaClient(cfg.alpaca_api_key, cfg.alpaca_api_secret, paper=cfg.alpaca_paper); db = Database()
    uni = db.get_active_universe(); adv = {r['symbol']: float(r.get('avg_volume_daily') or 0) for r in uni}
    syms = [r['symbol'] for r in uni]                       # NO stale price_close screen — the day's bars decide
    day = alp.get_current_bars(syms)
    movers = [s for s, b in day.items() if b.get('open') and b['open'] > 0 and (b.get('high') or 0) >= b['open'] * (1 + p.min_dist_open_pct / 100.0) and (b.get('high') or 0) >= floor]
    no_adv = [s for s in movers if adv.get(s, 0) < hb['min_adv20']]
    print(f'universe {len(uni)} | day-running movers with high >= +{p.min_dist_open_pct:.0f}% above open and high >= ${floor:.0f}: {len(movers)} | of which adv < {hb["min_adv20"]:,.0f} or missing (engine would NOT admit): {len(no_adv)} {no_adv[:12]}', flush=True)
    bars = alp.get_1min_bars_multi(movers, lookback_minutes=420) if movers else {}
    admitted, evaluated, stale = journal_events()
    today = datetime.now(timezone.utc).astimezone(ET).strftime('%Y-%m-%d'); streamed = streamed_symbols(today)
    print(f'streamed from the open: {len(streamed)} symbols | scan-admitted today: {len(admitted)} | stale-break warnings: {len(stale)}')
    rows = []
    for sym in movers:
        df = bars.get(sym)
        if df is None or not len(df): continue
        ts = df['timestamp'] if 'timestamp' in df.columns else df.index.to_series(); r = []
        for i, t in enumerate(ts):
            t = t.to_pydatetime() if hasattr(t, 'to_pydatetime') else t
            if t.tzinfo is None: t = t.replace(tzinfo=timezone.utc)
            e = t.astimezone(ET); m = e.hour * 60 + e.minute
            if OPEN_MINUTE <= m < 960: x = df.iloc[i]; r.append((m, float(x['open']), float(x['high']), float(x['low']), float(x['close']), float(x.get('volume', 0) or 0)))
        if len(r) < 10: continue
        r.sort(); a = np.array(r, float); o, h, l, c, v, m = a[:, 1], a[:, 2], a[:, 3], a[:, 4], a[:, 5], a[:, 0].astype(int)
        sig = detect(o, h, l, v, m, adv.get(sym, 0.0), p)
        if sig is None or sig.level < floor: continue
        bm = int(m[sig.bar_idx]); adm = admitted.get(sym); ev = [x for x in evaluated.get(sym, []) if x is not None]
        at_break = [x for x in ev if bm <= x <= bm + 2]                  # the engine acts at the break bar's close (+0..2 min)
        if at_break: status = 'EVALUATED'
        elif ev: status = f'MISSED this break; engine evaluated a LATER break at {min(ev) // 60:02d}:{min(ev) % 60:02d} — ENGINE BUG (first break only)' + (f' (admitted {adm // 60:02d}:{adm % 60:02d})' if adm else '')
        elif sym in stale: status = f'MISSED stale_break (engine saw it {stale[sym] // 60:02d}:{stale[sym] % 60:02d}, after the fill window)' + (' — streamed: ENGINE BUG / stream outage' if sym in streamed else ' (scan-admitted late)')
        elif sym in streamed: status = 'MISSED streamed_never_evaluated — ENGINE BUG'
        elif adm is None: status = 'MISSED never_admitted' + (' (adv below floor/missing)' if adv.get(sym, 0) < hb['min_adv20'] else (' (prev close below the stream screen)' if streamed else ''))
        elif adm > bm + 1: status = f'MISSED admitted_after_break (+{adm - bm} min)'
        else: status = 'MISSED admitted_before_break — ENGINE BUG'
        rows.append((sym, bm, sig.level, sig.stop, sig.rv_profile, sig.dist_open_pct, status))
    rows.sort(key=lambda x: x[1])
    print(f"\nspec signals today (price >= ${floor:.0f}): {len(rows)} | evaluated by the engine {sum(1 for x in rows if x[6] == 'EVALUATED')} | MISSED {sum(1 for x in rows if x[6].startswith('MISSED'))}")
    for sym, bm, lvl, stp, rv, dist, st in rows:
        print(f"  {sym:6s} break {bm // 60:02d}:{bm % 60:02d} level {lvl:8.2f} stop {stp:8.2f} rv {rv:4.1f} +{dist:4.1f}%  {st}")
    n_miss = sum(1 for x in rows if x[6].startswith('MISSED')); n_bug = sum(1 for x in rows if 'ENGINE BUG' in x[6])
    print(f"\nMISS RATE {n_miss}/{len(rows)} = {(n_miss / len(rows) * 100) if rows else 0:.0f}% | engine-side bugs {n_bug} | engine-evaluated symbols not in the spec set: {sorted(set(evaluated) - {x[0] for x in rows})}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
