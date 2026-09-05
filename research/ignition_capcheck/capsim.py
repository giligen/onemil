"""Ignition capcheck step 4: baseline + chase-cap variant sim over the
19-month universe. Trigger/entry/stop/exit mechanics IMPORTED from
trading/ignition_rules.py (single source of truth — replay-parity; the only
local computation is the cap-fill overlay, which by design cannot exist there
yet). NO catalyst gate (historically unreconstructable for never-scanned
names — shared bias, cancels in the cap DELTA); anchor recorded per symbol so
the complex-confirmed subset can be reported.

Cap fill model (per mission spec / evidence doc Study 2):
  cap_price = trigger_level * (1 + cap_bps/1e4)
  if baseline entry e0 = next_bar_open*1.003 <= cap_price -> fill at e0 on the
    entry bar (identical to baseline trade).
  else limit rests at cap_price: first bar from the entry bar onward (within
    trigger_m + max window 30min) whose low <= cap_price -> fill AT cap_price
    (no price-improvement credit). Track offset so 5/15/30-min cancel windows
    are derivable. No touch -> MISSED.
  Gates + sizing are evaluated at the BASELINE plan (the live decision is made
  at trigger time; the cap only changes the fill) — variants differ from
  baseline ONLY via fill price/miss.
  Exit: R.resim_exit from after the fill bar, structural stop
  min(pre_30min_low, fill*0.99) — same physics/convention as the book replay
  (exit sim starts after the entry bar).

Chunked per day, resumable (state file per part), bounded memory.
Usage: python3 capsim.py PART START END   e.g.  capsim.py 25H1 2025-01-01 2025-06-30
"""
import json, os, sqlite3, sys
import pandas as pd
sys.stdout.reconfigure(line_buffering=True)
ROOT = '/home/ec2-user/onemil'
sys.path.insert(0, ROOT)
from trading import ignition_rules as R

D = f'{ROOT}/research/ignition_capcheck'
PART, START, END = sys.argv[1], sys.argv[2], sys.argv[3]
OUT = f'{D}/trades_{PART}.csv'
STATE = f'{D}/simstate_{PART}.json'
CAPS = [50, 100, 200, 500]
MAX_WIN = 30

# CAPSIM_UNIVERSE (2026-09-05): alternate universe file with the same columns
# — used for the point-in-time add-on (universe_pit_addon.csv, the candidate
# symbol-days the cache-based universe never saw; bars in topup.db from
# fetch_missing_databento.py). Default = the original coverage file.
UNIVERSE = os.environ.get('CAPSIM_UNIVERSE', f'{D}/universe_coverage.csv')
print(f"[{PART}] universe file: {UNIVERSE}", flush=True)
u = pd.read_csv(UNIVERSE, dtype={'symbol': str}, keep_default_na=False)
u['prev_close'] = pd.to_numeric(u['prev_close'], errors='coerce')
u = u[(u['bar_date'] >= START) & (u['bar_date'] <= END)]
print(f"[{PART}] universe slice: {len(u):,} symbol-days "
      f"({START}..{END})", flush=True)

done = set(json.load(open(STATE))['done']) if os.path.exists(STATE) else set()
cache = sqlite3.connect(f'file:{ROOT}/data/cache.db?mode=ro', uri=True, timeout=120)
topup = sqlite3.connect(f'file:{D}/topup.db?mode=ro', uri=True, timeout=120)


def day_frames(day, cov_syms, miss_syms):
    """Load {symbol: bar df} for one day: cache for covered, topup for missing."""
    frames = {}
    if cov_syms:
        q = ("select symbol, timestamp as t, open, high, low, close, volume "
             "from intraday_bars_1min where bar_date=? and symbol in "
             f"({','.join('?' * len(cov_syms))})")
        b = pd.read_sql(q, cache, params=[day] + cov_syms)
        for s, g in b.groupby('symbol'):
            frames[s] = g
    if miss_syms:
        b = pd.read_sql("select symbol, t, o as open, h as high, l as low, "
                        "c as close, v as volume from bars where day=?",
                        topup, params=[day])
        b = b[b['symbol'].isin(miss_syms)]
        for s, g in b.groupby('symbol'):
            frames[s] = g
        # fallback: partially-cached keys the API returned nothing for
        left = [s for s in miss_syms if s not in frames]
        if left:
            q = ("select symbol, timestamp as t, open, high, low, close, volume "
                 "from intraday_bars_1min where bar_date=? and symbol in "
                 f"({','.join('?' * len(left))})")
            b2 = pd.read_sql(q, cache, params=[day] + left)
            for s, g in b2.groupby('symbol'):
                frames[s] = g
    return frames


def sim_one(g, prev_close):
    ts = pd.to_datetime(g['t'], utc=True).dt.tz_convert('America/New_York')
    g = g.assign(m=ts.dt.hour * 60 + ts.dt.minute)
    g = g[(g['m'] >= 570) & (g['m'] < 960)].sort_values('m').reset_index(drop=True)
    if len(g) < 20:
        return 'thin_day', None
    o = float(g.iloc[0]['open'])
    urej = R.universe_reject(o, prev_close if pd.notna(prev_close) else None)
    if urej:
        return urej, None
    if float((g['volume'] * g['close']).sum()) < R.DAY_DOLLAR_MIN:
        return 'u_dollar_2M', None
    tr = R.trigger_entry_stop(g, o)
    if 'reject' in tr:
        return tr['reject'], None
    entry, stop, rp = tr['entry'], tr['stop'], tr['r_pct']
    trig_m = tr['trigger_m']
    pos = R.position_usd(rp, tr['bar_dollar'])
    if R.position_reject(pos):
        return 'pos_lt_2k', None
    post = g[g.index > tr['next_idx']]
    rr, reason = R.resim_exit(post, entry, stop, trig_m)
    part = pos / max(tr['bar_dollar'], 1)
    fric = pos * R.FRICTION_BPS * min(part / R.PARTICIPATION, 1.0)
    row = {'trig_m': trig_m, 'day_open': o, 'level': R.level(o),
           'entry': entry, 'stop': stop, 'r_pct': round(rp, 3),
           'rr': round(rr, 4), 'reason': reason, 'pos': round(pos, 2),
           'pnl': round(pos * (rr * rp / 100.0) - fric, 2),
           'bar_dollar': round(tr['bar_dollar'], 0)}
    # --- cap overlay ---
    pre = g[(g['m'] >= trig_m - 30) & (g['m'] < trig_m)]
    pre_low = float(pre['low'].min())
    win = g[(g.index >= tr['next_idx']) & (g['m'] <= trig_m + MAX_WIN)]
    for cap in CAPS:
        cap_px = row['level'] * (1 + cap / 1e4)
        pfx = f'c{cap}_'
        if entry <= cap_px:                      # immediate fill == baseline
            row[pfx + 'off'] = int(g.loc[tr['next_idx'], 'm'] - trig_m)
            row[pfx + 'entry'] = entry
            row[pfx + 'rr'] = row['rr']
            row[pfx + 'pnl'] = row['pnl']
            continue
        touch = win[win['low'] <= cap_px]
        if touch.empty:                          # missed in every window
            row[pfx + 'off'] = -1
            row[pfx + 'entry'] = row[pfx + 'rr'] = row[pfx + 'pnl'] = None
            continue
        fb = touch.iloc[0]
        fill_m = int(fb['m'])
        e_f = cap_px
        s_f = R.stop_from_pre_lows(pre_low, e_f)
        rp_f = R.r_pct_from_stop(e_f, s_f)
        rr_f, _ = R.resim_exit(g[g['m'] > fill_m], e_f, s_f, fill_m)
        row[pfx + 'off'] = fill_m - trig_m
        row[pfx + 'entry'] = round(e_f, 4)
        row[pfx + 'rr'] = round(rr_f, 4)
        row[pfx + 'pnl'] = round(pos * (rr_f * rp_f / 100.0) - fric, 2)
    return None, row


rej_tot, n_tr = {}, 0
days = sorted(u['bar_date'].unique())
for di, day in enumerate(days):
    if day in done:
        continue
    sub = u[u['bar_date'] == day]
    frames = day_frames(day, sub[sub['covered']]['symbol'].tolist(),
                        sub[~sub['covered']]['symbol'].tolist())
    pcmap = dict(zip(sub['symbol'], sub['prev_close']))
    rows = []
    for sym in sub['symbol']:
        g = frames.get(sym)
        if g is None or len(g) < 20:
            rej_tot['no_bars'] = rej_tot.get('no_bars', 0) + 1
            continue
        rej, row = sim_one(g, pcmap.get(sym))
        if rej:
            rej_tot[rej] = rej_tot.get(rej, 0) + 1
        else:
            rows.append({'day': day, 'symbol': sym, **row})
    if rows:
        pd.DataFrame(rows).to_csv(OUT, mode='a',
                                  header=not os.path.exists(OUT), index=False)
        n_tr += len(rows)
    done.add(day)
    json.dump({'done': sorted(done)}, open(STATE, 'w'))
    if (di + 1) % 20 == 0 or di == len(days) - 1:
        print(f"[{PART}] {di+1}/{len(days)} days ({day}), trades so far {n_tr:,}",
              flush=True)
print(f"[{PART}] DONE. trades {n_tr:,}. rejects: "
      f"{json.dumps(dict(sorted(rej_tot.items(), key=lambda x: -x[1])))}", flush=True)
