#!/usr/bin/env python3
"""HOD-break / red-to-green MISS AUDIT — the two-sided parity the EOD check lacked (2026-09-15, after the CRWL miss).

The spec is run over the WHOLE scanner universe for the day (for the HOD book: every symbol whose day-running bars
show a high >= 5% above the 09:30 open, price >= the live floor; for red-to-green: every symbol that opened BELOW its
prior close on a day whose PRIOR day ranged >= pdr_min_pct and whose day-running high reached the level), producing
the complete set of spec signals. Each is then classified against the engine's journal:
  EVALUATED   the engine saw that break (a [HOD DRY]/BUY line or a rule skip on that symbol)
  MISSED      the spec signalled but the engine never evaluated the symbol at that break — with the cause:
              never_admitted / admitted_after_break (late admission) / admitted_before_break (engine-side bug)
The miss rate is THE input-parity number.
Usage: python3 scripts/hod_break_miss_audit.py [--floor 20] [--book hod_break|red_to_green]
Default book `hod_break` — byte-identical to the pre-option script. Read-only. Alpaca REST only (get_current_bars
over the universe, then 1-min bars for the movers).

NOTE on the streamed-universe file: `logs/hod_stream_universe_<day>.txt` is written by whichever engine rolls its
session first, so with both books live the file may hold the OTHER book's list. For red-to-green the audit therefore
uses the book's OWN screen — exactly `HodBreakEngine._stream_the_universe` (ADV20, prev close, prior-day range) —
and reports whether the file agrees with it.
"""
import os, re, subprocess, sys
from datetime import datetime, timezone
from zoneinfo import ZoneInfo
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..')); sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'scripts')); os.chdir(ROOT)
from book_spec import book_from_argv, load_book             # noqa: E402
from config import Config                                   # noqa: E402
from data_sources.alpaca_client import AlpacaClient         # noqa: E402
from persistence.database import Database                   # noqa: E402
from trading.hod_break import OPEN_MINUTE                   # noqa: E402
from trading.hod_break_engine import load_adv20_from_daily_bars     # noqa: E402  (the engine's own ADV/universe definition)
ET = ZoneInfo('America/New_York')


def journal_events(since='12:29', tag='HOD'):
    out = subprocess.run(['bash', '-c', f'journalctl -u onemil-trader --since "{since}" --no-pager -o short-iso | grep -E "\\[{tag}"'], capture_output=True, text=True).stdout.splitlines()
    admitted, evaluated, stale = {}, {}, {}; stream_start = None
    for ln in out:
        ts = re.match(r'(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}[+-]\d{4})', ln)
        m_et = None
        if ts:
            t = datetime.strptime(ts.group(1), '%Y-%m-%dT%H:%M:%S%z').astimezone(ET); m_et = t.hour * 60 + t.minute
        a = re.search(r'candidate (\S+) admitted', ln)
        if a and a.group(1) not in admitted: admitted[a.group(1)] = m_et
        e = re.search(rf'\[{tag}(?: DRY)?\] (?:WOULD BUY |BUY |ENTRY SUBMITTED )?(\S+?):? (?:level|ask|spread|stop|per-day|concurrency|notional|no quote|quote is)', ln)
        if e: evaluated.setdefault(e.group(1), []).append(m_et)
        st = re.search(rf'\[{tag}\] (\S+): MISSED the spec\'s break at bar (\d+)', ln)
        if st: stale[st.group(1)] = m_et
        if 'universe symbols' in ln and 'streaming' in ln: stream_start = m_et      # the LAST boot wins (a restart after the open)
    return admitted, evaluated, stale, stream_start


def streamed_symbols(day: str) -> set:
    """Symbols the engine streamed from the open (written at session start); empty when the file is absent.
    SHARED between the books — whichever engine rolls first writes it (see the module docstring)."""
    try:
        with open(f'logs/hod_stream_universe_{day}.txt') as f: return {x.strip() for x in f if x.strip()}
    except FileNotFoundError:
        return set()


def movers_for(book, day: dict, floor: float) -> list:
    """The day's candidate symbols from the running DAILY bar — a superset of the book's signals (`detect` decides on
    the 1-min tape). HOD: high >= min_dist_open_pct above the open. Red-to-green: a gap-down open on a prior-day-range
    day whose running high reached the level and whose running range cleared the floor."""
    p = book.params
    if not book.is_r2g:
        return [s for s, b in day.items() if b.get('open') and b['open'] > 0
                and (b.get('high') or 0) >= b['open'] * (1 + p.min_dist_open_pct / 100.0) and (b.get('high') or 0) >= floor]
    out = []
    for s, b in day.items():
        if not book.eligible_prior_day(s): continue                # prior close + prior-day range >= pdr_min_pct
        o = b.get('open') or 0; hi = b.get('high') or 0; lo = b.get('low') or 0; lvl = book.level(s)
        if o <= 0 or lvl is None or hi < floor: continue
        pc, _ = book.prior(s)
        if o >= pc: continue                                       # not a gap-down open: outside the book
        if hi < lvl: continue                                      # the level was never reached today
        if lo <= 0 or (hi - lo) / lo * 100.0 < p.range_floor_pct: continue   # the day never ranged enough for the floor
        out.append(s)
    return out


def main() -> int:
    argv = list(sys.argv[1:]); book_name = book_from_argv(argv)
    floor = float(argv[argv.index('--floor') + 1]) if '--floor' in argv else None
    cfg = Config(); db = Database()
    book = load_book(book_name, cache_path=getattr(db, '_cache_path', None))
    hb = book.cfg; p = book.params; floor = floor or hb['min_price']
    alp = AlpacaClient(cfg.alpaca_api_key, cfg.alpaca_api_secret, paper=cfg.alpaca_paper)
    uni = db.get_active_universe(); adv = {r['symbol']: float(r.get('avg_volume_daily') or 0) for r in uni}
    adv20, last_close = load_adv20_from_daily_bars(db._cache_path); adv.update(adv20)   # EXACTLY the engine's ADV map (daily_bars over the universe field)
    syms = sorted(set(adv) | {r['symbol'] for r in uni})    # daily_bars names + the scanner universe; NO stale price screen — the day's bars decide
    if book.is_r2g:
        syms = sorted(s for s in syms if book.eligible_prior_day(s))   # the 09:30-known precondition: prior close + prior-day range >= pdr_min_pct
    day = alp.get_current_bars(syms)
    movers = movers_for(book, day, floor)
    no_adv = [s for s in movers if adv.get(s, 0) < hb['min_adv20']]
    if book.is_r2g:
        print(f'symbols with prior-day range >= {p.pdr_min_pct:.0f}% {len(syms)} | of those, gap-down opens whose day high reached prior close x {1 + p.level_buffer:.3f}, day range >= {p.range_floor_pct:.0f}% and high >= ${floor:.0f}: {len(movers)} | of which adv < {hb["min_adv20"]:,.0f} or missing (engine would NOT admit): {len(no_adv)} {no_adv[:12]}', flush=True)
    else:
        print(f'symbols {len(syms)} | day-running movers with high >= +{p.min_dist_open_pct:.0f}% above open and high >= ${floor:.0f}: {len(movers)} | of which adv < {hb["min_adv20"]:,.0f} or missing (engine would NOT admit): {len(no_adv)} {no_adv[:12]}', flush=True)
    now_et = datetime.now(timezone.utc).astimezone(ET); lookback = max(30, now_et.hour * 60 + now_et.minute - OPEN_MINUTE + 10)   # the WHOLE day from 09:30 (a fixed 420 min lost the morning when run after 16:30 UTC — found at the 9/15 EOD)
    bars = alp.get_1min_bars_multi(movers, lookback_minutes=lookback) if movers else {}
    admitted, evaluated, stale, stream_start = journal_events(tag=book.tag)
    today = datetime.now(timezone.utc).astimezone(ET).strftime('%Y-%m-%d'); streamed = streamed_symbols(today)
    ss = f'{stream_start // 60:02d}:{stream_start % 60:02d}' if stream_start is not None else 'n/a'
    if book.is_r2g:
        screen = book.screen_universe(adv, last_close)      # what the R2G engine subscribes — the file is shared, the screen is not
        agree = len(streamed & screen) / len(streamed) if streamed else 0.0
        print(f"streamed-universe FILE logs/hod_stream_universe_{today}.txt: {len(streamed)} symbols, {agree:.0%} of them in the red-to-green screen — the file is SHARED, written by whichever engine rolled first")
        streamed = screen
        print(f"red-to-green screen (ADV20 >= {hb['min_adv20']:,.0f}, prev close >= {hb['universe_min_prev_close']:.0f}, prior-day range >= {p.pdr_min_pct:.0f}%): {len(streamed)} symbols | stream start {ss} | scan-admitted today: {len(admitted)} | stale-break warnings: {len(stale)}")
    else:
        print(f'streamed universe: {len(streamed)} symbols from {ss} ET | scan-admitted today: {len(admitted)} | stale-break warnings: {len(stale)}')
    outside = []
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
        sig = book.detect(sym, o, h, l, v, m, adv.get(sym, 0.0))
        if sig is None or sig.level < floor: continue
        if adv.get(sym, 0.0) < hb['min_adv20']:
            outside.append(sym); continue                                # ADV20 below the universe floor: outside the spec's book too
        bm = int(m[sig.bar_idx]); adm = admitted.get(sym); ev = [x for x in evaluated.get(sym, []) if x is not None]
        at_break = [x for x in ev if bm <= x <= bm + 2]                  # the engine acts at the break bar's close (+0..2 min)
        if at_break: status = 'EVALUATED'
        elif stream_start is not None and bm < stream_start and not ev: status = f'MISSED before the stream started ({ss} boot/restart)'
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
    live = [x for x in rows if stream_start is None or x[1] >= stream_start]
    print(f"since the stream started: {len(live)} spec signals, evaluated {sum(1 for x in live if x[6] == 'EVALUATED')}, missed {sum(1 for x in live if x[6].startswith('MISSED'))} | outside the universe (ADV20 < {hb['min_adv20']:,.0f}): {len(outside)} {outside[:10]}")
    print(f"\nMISS RATE {n_miss}/{len(rows)} = {(n_miss / len(rows) * 100) if rows else 0:.0f}% | engine-side bugs {n_bug} | engine-evaluated symbols not in the spec set: {sorted(set(evaluated) - {x[0] for x in rows})}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
