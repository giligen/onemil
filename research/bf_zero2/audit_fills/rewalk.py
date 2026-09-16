#!/usr/bin/env python3
"""Adversarial fill audit — re-walk every F6 pool candidate from the SIP bars and recompute the exit
under each fill correction.  Writes audit_fills/pool_rewalk.csv (one row per pool candidate, all variants).

Variants (each identical to the study except the stated change):
  base : the study's own spec  (entry AT level*1.003, stops only from bar i+1, stop fill min(stop,open)*0.999,
         target = first bar CLOSE >= entry+2R filled AT the target, else flat at the open of the first bar >= 15:55)
  g    : entry = max(level*1.003, open of the entry bar)          -- a stop-buy cannot fill below the open
  eb   : a stop exit is charged INSIDE the entry bar when its low <= stop (pessimistic intrabar path)
  s25/s50/s100 : stop fill min(stop,open)*(1-bps)
  eod10: the 15:55 market exit fills 10 bps below that bar's open
  all  : g + eb + s50 + eod10 (the pessimistic book)

Also recorded per candidate: gap-through flag, entry-bar stop-hit flag, $ volume in the 5 min after entry,
the position notional at $100 of risk, tape-end minute (halts), the largest bar gap during the trade,
and whether a resting +2R limit would have filled earlier than the close-fill rule (a FAVOURABLE check).
"""
import json, os, sqlite3, sys, time
import numpy as np, pandas as pd

ROOT = '/home/ec2-user/onemil'; os.chdir(ROOT); sys.path.insert(0, ROOT)
A = 'research/bf_zero2/audit_fills'
SIP = f'{ROOT}/research/bf_zero/bars_sip.db'
EOD_M, OPEN_M = 955, 570

cache = sqlite3.connect(f'file:{ROOT}/data/cache.db?mode=ro', uri=True, timeout=180)
sip = sqlite3.connect(f'file:{SIP}?mode=ro', uri=True, timeout=180)


def load_bars(day, syms):
    """Same source order as research/bf_zero/build_candidates.load_bars with BFZ_SIP_STORE set."""
    out = {}
    q = ("select symbol, timestamp as t, open as o, high as h, low as l, close as c, volume as v "
         f"from intraday_bars_1min where bar_date=? and symbol in ({','.join('?' * len(syms))})")
    for s, gg in pd.read_sql(q, cache, params=[day] + list(syms)).groupby('symbol'): out[s] = gg
    left = [s for s in syms if s not in out]
    if left:
        q2 = f"select symbol, t, o, h, l, c, v from bars where day=? and symbol in ({','.join('?' * len(left))})"
        for s, gg in pd.read_sql(q2, sip, params=[day] + left).groupby('symbol'): out[s] = gg
    res = {}
    for s, gg in out.items():
        ts = pd.to_datetime(gg.t, utc=True).dt.tz_convert('America/New_York')
        gg = gg.assign(m=(ts.dt.hour * 60 + ts.dt.minute).values).sort_values('m').drop_duplicates('m')
        gg = gg[(gg.m >= OPEN_M) & (gg.m < 960)]
        res[s] = gg[['m', 'o', 'h', 'l', 'c', 'v']].reset_index(drop=True)
    return res


def sim(o, h, l, c, m, i, entry, stop, stop_bps=0.001, gap_entry=False, entry_bar_stop=False, eod_bps=0.0, mult=2.0):
    """Returns (rr, why, exit_m, entry_used, r_pct_used)."""
    if gap_entry: entry = max(entry, o[i])
    Rd = entry - stop
    if Rd <= 0: return (np.nan, 'invalid', int(m[i]), entry, np.nan)
    rp = Rd / entry * 100
    if entry_bar_stop and l[i] <= stop:
        return ((stop * (1 - stop_bps) - entry) / Rd, 'stop_entry_bar', int(m[i]), entry, rp)
    tgt = entry + mult * Rd
    oo, hh, ll, cc, mm = o[i + 1:], h[i + 1:], l[i + 1:], c[i + 1:], m[i + 1:]
    if len(oo) == 0: return (0.0, 'none', int(m[i]), entry, rp)
    for k in range(len(oo)):
        if mm[k] >= EOD_M: return ((oo[k] * (1 - eod_bps) - entry) / Rd, 'eod', int(mm[k]), entry, rp)
        if ll[k] <= stop: return ((min(stop, oo[k]) * (1 - stop_bps) - entry) / Rd, 'stop', int(mm[k]), entry, rp)
        if cc[k] >= tgt: return (mult, 'target', int(mm[k]), entry, rp)
    return ((cc[-1] * (1 - eod_bps) - entry) / Rd, 'eod_notape', int(mm[-1]), entry, rp)


VARIANTS = {
    'base':  dict(),
    'g':     dict(gap_entry=True),
    'eb':    dict(entry_bar_stop=True),
    's25':   dict(stop_bps=0.0025),
    's50':   dict(stop_bps=0.005),
    's100':  dict(stop_bps=0.01),
    'eod10': dict(eod_bps=0.001),
    'all':   dict(gap_entry=True, entry_bar_stop=True, stop_bps=0.005, eod_bps=0.001),
}

def main():
    p = pd.read_csv(f'{A}/pool.csv', dtype={'day': str, 'symbol': str}, keep_default_na=False, na_values=[''])
    print(f'pool {len(p):,} rows over {p.day.nunique()} days', flush=True)
    out_rows = []; t0 = time.time(); nmiss = 0
    days = sorted(p.day.unique())
    for di, day in enumerate(days):
        sub = p[p.day == day]
        B = load_bars(day, sorted(sub.symbol.unique()))
        for r in sub.itertuples():
            gg = B.get(r.symbol)
            if gg is None or len(gg) < 10: nmiss += 1; continue
            o, h, l, c, v = (gg[k].values.astype(float) for k in ('o', 'h', 'l', 'c', 'v')); m = gg.m.values.astype(int)
            idx = np.flatnonzero(m == int(r.entry_m))
            if not len(idx): nmiss += 1; continue
            i = int(idx[0]); entry = float(r.entry); stop = float(r.stop)
            # sanity: the study's own spec must reproduce the stored rr_e1c
            row = dict(day=r.day, symbol=r.symbol, entry_m=int(r.entry_m), split=r.split, wk=r.wk, mo=r.mo,
                       entry=entry, stop=stop, r_pct=float(r.r_pct), adv20=float(r.adv20) if r.adv20 == r.adv20 else np.nan,
                       rr_e1c_stored=float(r.rr_e1c), why_stored=str(r.why_e1c), xm_stored=int(r.exit_m_e1c),
                       o_i=o[i], h_i=h[i], l_i=l[i], c_i=c[i], v_i=v[i],
                       gap_through=int(o[i] > entry), gap_through_pct=(o[i] / entry - 1) * 100,
                       entry_bar_stop_hit=int(l[i] <= stop), last_m=int(m[-1]), n_bars=len(m))
            # liquidity: $ volume in the 5 minutes after entry, and over the whole trade
            w = slice(i + 1, i + 6)
            row['dv5'] = float((c[w] * v[w]).sum()); row['vol5'] = float(v[w].sum())
            Rd0 = entry - stop
            row['shares_100r'] = 100.0 / Rd0 if Rd0 > 0 else np.nan
            row['notional_100r'] = row['shares_100r'] * entry
            row['notional_frac_dv5'] = row['notional_100r'] / row['dv5'] if row['dv5'] > 0 else np.inf
            row['share_frac_vol5'] = row['shares_100r'] / row['vol5'] if row['vol5'] > 0 else np.inf
            for tag, kw in VARIANTS.items():
                rr, why, xm, ent, rp = sim(o, h, l, c, m, i, entry, stop, **kw)
                row[f'rr_{tag}'] = rr; row[f'why_{tag}'] = why; row[f'xm_{tag}'] = xm
                if tag in ('g', 'all'): row[f'entry_{tag}'] = ent; row[f'rpct_{tag}'] = rp
            # bar gaps during the trade (possible halt / no tape) + favourable check: resting +2R limit
            xb = np.flatnonzero(m == row['xm_base'])
            xi = int(xb[0]) if len(xb) else len(m) - 1
            seg = m[i:xi + 1]
            row['max_gap_min'] = int(np.diff(seg).max()) if len(seg) > 1 else 0
            tgt = entry + 2 * Rd0
            touch = np.flatnonzero(h[i + 1:xi + 1] >= tgt)
            row['limit_touch_before_exit'] = int(len(touch) > 0)
            row['limit_touch_m'] = int(m[i + 1 + touch[0]]) if len(touch) else -1
            out_rows.append(row)
        if di % 20 == 0 or di == len(days) - 1:
            el = time.time() - t0
            print(f'{di + 1}/{len(days)} {day} rows {len(out_rows):,} miss {nmiss} | {el / 60:.1f} min ({el / (di + 1):.2f} s/day)', flush=True)

    R = pd.DataFrame(out_rows)
    R.to_csv(f'{A}/pool_rewalk.csv', index=False)
    d = (R.rr_base - R.rr_e1c_stored).abs()
    print(f'\nrewalk rows {len(R):,} | missing bars {nmiss} | base-vs-stored rr: max |d| {d.max():.6g}, '
          f'>1e-6 on {int((d > 1e-6).sum())} rows, why mismatch {int((R.why_base != R.why_stored).sum())}', flush=True)
    print('DONE', flush=True)


if __name__ == '__main__':
    main()
