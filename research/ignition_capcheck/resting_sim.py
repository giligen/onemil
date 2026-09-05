"""Ignition capcheck follow-up: RESTING-ORDER fill model over the 19-month
baseline book (9,515 trades from capsim.py), + K-study proximity ranks.

Model (pre-staged stop-limit resting BEFORE the cross):
  stop trigger = level*1.003, limit = level*(1+cap_bps/1e4).
  Effective election bar = baseline trigger bar (first in-window bar with
  high >= level) IF its high >= stop_px; else the first later bar whose high
  >= stop_px (30bps re-key nuance; if never -> order never elects = MISS).
  On the election bar with open O_e:
    O_e <= stop_px           -> CLEAN: fill at stop_px*(1+slip), slip {0,10,25}bps
    stop_px < O_e <= cap_px  -> GAP-INTO-BAND: fill at O_e (opening print lifts)
    O_e > cap_px             -> GAP-THROUGH: limit rests at cap_px; fill at
                                cap_px on first bar (incl. election bar) with
                                low <= cap_px before 15:45 (ADVERSE late fill);
                                else MISS.
  Names already above level at 9:35 (trig_m==575, O_t > level) flow through the
  same O-rules (their election bar IS the 9:35 bar); cohort flagged `preopen`.
  Sizing re-derived at the fill: stop_f = min(pre30_low, fill*0.99) (structural,
  capsim convention), rp_f = (fill-stop_f)/fill*100, pos_f =
  R.position_usd(rp_f, baseline bar_dollar), friction participation-scaled.
  Exits: PRIMARY = R.resim_exit from the fill bar (same physics as capsim /
  book replay). SECONDARY = fixed-exit-path approximation (exit price frozen
  from baseline row: exit_px = entry_b + rr_b*(entry_b-stop_b)).

K-study: per trade, rank of own distance-to-level = (level_s - last_close)/level_s
  among the day's stageable universe (universe_coverage rows passing the
  live-computable gates price-floor/gap, with >=1 bar by the ref minute) at
  T-5 and T-10 min before the trigger. Ties share the best rank. refs < 9:30
  (trig_m < 580/585) -> rank NA, counted.

Chunked per day, resumable. Usage: resting_sim.py PART START END
"""
import json, os, sqlite3, sys
import numpy as np
import pandas as pd
sys.stdout.reconfigure(line_buffering=True)
ROOT = '/home/ec2-user/onemil'
sys.path.insert(0, ROOT)
from trading import ignition_rules as R

D = f'{ROOT}/research/ignition_capcheck'
PART, START, END = sys.argv[1], sys.argv[2], sys.argv[3]
OUT = f'{D}/resting_{PART}.csv'
STATE = f'{D}/reststate_{PART}.json'
CAPS = [100, 200, 300, 500]
SLIPS = [0, 10, 25]
STOP_MULT = 1.003
EOD_M = R.EOD_FLAT_MIN  # 945; fills at/after 15:45 are not taken

# RESTING_TRADES / RESTING_UNIVERSE (2026-09-05): alternate inputs for the
# live-window roll-forward (trades_<PART>_annotated.csv + universe_live_window.csv)
# so the frozen 19-month files are never rewritten.
t_all = pd.read_csv(os.environ.get('RESTING_TRADES', f'{D}/trades_all_annotated.csv'), dtype={'symbol': str},
                    keep_default_na=False)
for c in t_all.columns:
    if c not in ('day', 'symbol', 'reason', 'era', 'ym', 'anchor',
                 'monster2', 'monster3', 'complex_conf'):
        t_all[c] = pd.to_numeric(t_all[c], errors='coerce')
t_all = t_all[(t_all['day'] >= START) & (t_all['day'] <= END)]
print(f"[{PART}] trades slice: {len(t_all):,} ({START}..{END})", flush=True)

u = pd.read_csv(os.environ.get('RESTING_UNIVERSE', f'{D}/universe_coverage.csv'), dtype={'symbol': str},
                keep_default_na=False)
u['prev_close'] = pd.to_numeric(u['prev_close'], errors='coerce')
u = u[(u['bar_date'] >= START) & (u['bar_date'] <= END)]

done = set(json.load(open(STATE))['done']) if os.path.exists(STATE) else set()
cache = sqlite3.connect(f'file:{ROOT}/data/cache.db?mode=ro', uri=True, timeout=120)
topup = sqlite3.connect(f'file:{D}/topup.db?mode=ro', uri=True, timeout=120)


def day_frames(day, cov_syms, miss_syms):
    """Load {symbol: bar df} for one day: cache for covered, topup for missing.
    (Verbatim pattern from capsim.py.)"""
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
        left = [s for s in miss_syms if s not in frames]
        if left:
            q = ("select symbol, timestamp as t, open, high, low, close, volume "
                 "from intraday_bars_1min where bar_date=? and symbol in "
                 f"({','.join('?' * len(left))})")
            b2 = pd.read_sql(q, cache, params=[day] + left)
            for s, g in b2.groupby('symbol'):
                frames[s] = g
    return frames


def prep(g):
    """Bars -> ET-minute frame, RTH from 9:30, sorted (capsim convention)."""
    ts = pd.to_datetime(g['t'], utc=True).dt.tz_convert('America/New_York')
    g = g.assign(m=ts.dt.hour * 60 + ts.dt.minute)
    return g[(g['m'] >= 570) & (g['m'] < 960)].sort_values('m').reset_index(drop=True)


def fill_class(g, ti, level):
    """Election-bar resolution. Returns (eff_idx or None, rekeyed: bool)."""
    stop_px = level * STOP_MULT
    tb = g.loc[ti]
    if float(tb['high']) >= stop_px:
        return ti, False
    later = g[(g.index > ti) & (g['high'] >= stop_px)]
    if later.empty:
        return None, True
    return later.index[0], True


def econ(fill, fill_m, g, row, cache_econ):
    """P&L at a model fill: resim (primary) + fixed-exit (secondary)."""
    key = (round(fill, 6), fill_m)
    if key in cache_econ:
        return cache_econ[key]
    pre = g[(g['m'] >= row['trig_m'] - 30) & (g['m'] < row['trig_m'])]
    stop_f = R.stop_from_pre_lows(float(pre['low'].min()), fill)
    rp_f = R.r_pct_from_stop(fill, stop_f)
    pos_f = R.position_usd(rp_f, row['bar_dollar'])
    pos_small = R.position_reject(pos_f)
    fric = pos_f * R.FRICTION_BPS * min((pos_f / max(row['bar_dollar'], 1))
                                        / R.PARTICIPATION, 1.0)
    rr_f, _ = R.resim_exit(g, fill, stop_f, fill_m)
    pnl = pos_f * (rr_f * rp_f / 100.0) - fric
    exit_px = row['entry'] + row['rr'] * (row['entry'] - row['stop'])
    pnl_fx = pos_f * ((exit_px - fill) / fill) - fric
    out = (round(pnl, 2), round(pnl_fx, 2), round(rr_f, 4), round(pos_f, 2),
           pos_small)
    cache_econ[key] = out
    return out


COLS = (['day', 'symbol', 'trig_m', 'level', 'o_t', 'preopen', 'rekeyed'] +
        [f'k{c}_{f}' for c in CAPS
         for f in ['cls', 'off', 'fill', 'pos', 'small', 'rr0',
                   'pnl0', 'pnl10', 'pnl25', 'fx0']] +
        ['rank5', 'nuniv5', 'rank10', 'nuniv10'])

rej_tot, n_out = {}, 0
days = sorted(t_all['day'].unique())
for di, day in enumerate(days):
    if day in done:
        continue
    usub = u[u['bar_date'] == day]
    frames = day_frames(day, usub[usub['covered']]['symbol'].tolist(),
                        usub[~usub['covered']]['symbol'].tolist())
    pcmap = dict(zip(usub['symbol'], usub['prev_close']))
    # --- stageable-universe arrays for K-study ---
    info = {}
    for sym, graw in frames.items():
        g = prep(graw)
        if g.empty:
            continue
        o = float(g.iloc[0]['open'])
        pc = pcmap.get(sym)
        if R.universe_reject(o, pc if pd.notna(pc) else None):
            continue
        info[sym] = (g['m'].to_numpy(), g['close'].to_numpy(), R.level(o), g)
    rows = []
    for row in t_all[t_all['day'] == day].to_dict('records'):
        sym = row['symbol']
        if sym not in info:
            rej_tot['no_bars_or_gate'] = rej_tot.get('no_bars_or_gate', 0) + 1
            continue
        g = info[sym][3]
        level = row['level']
        trig = g[(g['high'] >= level) & (g['m'] >= R.TRIGGER_MIN_START)
                 & (g['m'] <= R.TRIGGER_MIN_END)]
        if trig.empty or int(g.loc[trig.index[0], 'm']) != int(row['trig_m']):
            rej_tot['trigger_mismatch'] = rej_tot.get('trigger_mismatch', 0) + 1
            continue
        ti = trig.index[0]
        o_t = float(g.loc[ti, 'open'])
        stop_px = level * STOP_MULT
        eff, rekeyed = fill_class(g, ti, level)
        out = {'day': day, 'symbol': sym, 'trig_m': int(row['trig_m']),
               'level': level, 'o_t': o_t,
               'preopen': int(row['trig_m'] == R.TRIGGER_MIN_START
                              and o_t > level),
               'rekeyed': int(rekeyed)}
        cache_econ = {}
        for cap in CAPS:
            cap_px = level * (1 + cap / 1e4)
            pfx = f'k{cap}_'
            if eff is None:
                out[pfx + 'cls'] = 'miss_noelect'
                continue
            eb = g.loc[eff]
            o_e, m_e = float(eb['open']), int(eb['m'])
            if o_e <= stop_px:
                out[pfx + 'cls'] = 'clean'
                out[pfx + 'off'] = m_e - int(row['trig_m'])
                for sl in SLIPS:
                    f = stop_px * (1 + sl / 1e4)
                    pnl, fx, rr_f, pos_f, small = econ(f, m_e, g, row, cache_econ)
                    out[pfx + f'pnl{sl}'] = pnl
                    if sl == 0:
                        out[pfx + 'fill'] = round(f, 4)
                        out[pfx + 'fx0'], out[pfx + 'rr0'] = fx, rr_f
                        out[pfx + 'pos'], out[pfx + 'small'] = pos_f, int(small)
            elif o_e <= cap_px:
                out[pfx + 'cls'] = 'gap_into'
                out[pfx + 'off'] = m_e - int(row['trig_m'])
                pnl, fx, rr_f, pos_f, small = econ(o_e, m_e, g, row, cache_econ)
                out[pfx + 'fill'] = round(o_e, 4)
                for sl in SLIPS:
                    out[pfx + f'pnl{sl}'] = pnl
                out[pfx + 'fx0'], out[pfx + 'rr0'] = fx, rr_f
                out[pfx + 'pos'], out[pfx + 'small'] = pos_f, int(small)
            else:
                touch = g[(g.index >= eff) & (g['low'] <= cap_px)
                          & (g['m'] < EOD_M)]
                if touch.empty:
                    out[pfx + 'cls'] = 'miss'
                else:
                    fb = touch.iloc[0]
                    m_f = int(fb['m'])
                    out[pfx + 'cls'] = 'adverse'
                    out[pfx + 'off'] = m_f - int(row['trig_m'])
                    pnl, fx, rr_f, pos_f, small = econ(cap_px, m_f, g, row,
                                                       cache_econ)
                    out[pfx + 'fill'] = round(cap_px, 4)
                    for sl in SLIPS:
                        out[pfx + f'pnl{sl}'] = pnl
                    out[pfx + 'fx0'], out[pfx + 'rr0'] = fx, rr_f
                    out[pfx + 'pos'], out[pfx + 'small'] = pos_f, int(small)
        # --- K-study ranks ---
        for lag in (5, 10):
            ref = int(row['trig_m']) - lag
            rank = nuniv = None
            if ref >= 570:
                own = None
                ds = []
                for s2, (ms, cs, lv, _) in info.items():
                    i = np.searchsorted(ms, ref, side='right') - 1
                    if i < 0:
                        continue
                    d = (lv - cs[i]) / lv
                    ds.append(d)
                    if s2 == sym:
                        own = d
                if own is not None:
                    a = np.asarray(ds)
                    rank = int((a < own).sum()) + 1
                    nuniv = len(a)
            out[f'rank{lag}'] = rank
            out[f'nuniv{lag}'] = nuniv
        rows.append(out)
    if rows:
        pd.DataFrame(rows).reindex(columns=COLS).to_csv(
            OUT, mode='a', header=not os.path.exists(OUT), index=False)
        n_out += len(rows)
    done.add(day)
    json.dump({'done': sorted(done)}, open(STATE, 'w'))
    if (di + 1) % 20 == 0 or di == len(days) - 1:
        print(f"[{PART}] {di+1}/{len(days)} days ({day}), rows {n_out:,}",
              flush=True)
print(f"[{PART}] DONE. rows {n_out:,}. anomalies: {json.dumps(rej_tot)}",
      flush=True)
