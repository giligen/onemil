"""Cell 1,737 — be the bid: M1 retest bid at the level, M2 coil bid under the level, M3 buy the failed touch
(PREREG_1737.md + Amendment 1). End-to-end scorer; reads replay_signals.csv arms and 1736_events.csv, walks SIP minute
bars (bars_sip.db read-only via walker.fetch_day_bars), writes 1737_orders.csv / 1737_summary.csv / 1737_diag.txt.
TEST split is never touched. Conventions where the PREREG is silent:
- bar minute m = bar START minute (ET). 15:30 = 930, 15:55 = 955. A missing bar at an entry/exit minute falls back to the
  next available row (positional indexing everywhere).
- j = first bar with m >= entry_m (the PREREG's arm bar). stop = min(l[j-4..j]); if (level-stop)/level < 1.5 % (live
  hod_break.min_r_pct, config.yaml) the stop is widened to level*(1-0.015).
- M1: C = first bar idx >= j with close >= level and m < 930. Bid rests from C+1 at `level`, fills AT the bid on the first bar
  with low < bid (even when the bar opens below it: fill-at-touch conservatism); runaway cancel on close >= bid*1.02 is
  checked AFTER the fill test in the same bar.
- FACT (a) (RESULT_1736): entry_m is the BREAK minute, so j = the break bar. Primary stop = the engine's own for a break
  at j: consolidation_low at the last closed bar (M1: bars j-5..j-1; M2: bars a-4..a at the arm bar a, causal). The
  PREREG-literal bars j-4..j are the sensitivities M1L / M2L (M2L contains look-ahead lows after the fill).
- M2: arm bar a = first i>=5 with running HOD == level and min(l[i-4..i]) in [0.96*HOD, HOD) (engine consolidation_low);
  bid rests from a+1 at level*0.996; the bar with h >= level cancels BEFORE the fill test (the breaking bar never fills).
- After a fill at bar k: stop checked from bar k (low <= stop -> stop, gap-through at open if open <= stop, + 13 bp slip,
  + half_spread); target checked from bar k+1 only (a resting sell placed after the fill); both in one bar = stop;
  15:55 open otherwise (last close if no 955 bar). Entry cost = 0 half-spread + 2 bp; target exit 0 half + 2 bp; stop/EOD
  exits pay half_spread (stop: 13 bp slip, EOD: 2 bp). R = (fill*1.0002) - stop.
- M3: long at open(F+1) + half_spread + 2 bp; stop = 1736 short entry - 2*R_short; target = 1736 short stop; same exits
  (the entry bar itself can stop or target; stop first).
- Placebo: every order's geometry is scaled by close[s-1]/close[ref bar] and replayed from a random start s after the
  signal window (M1: > C+1, M2: > break bar, M3: > F+1), starts m < 900, up to 40 sampled per order; a draw picks one
  start per order and averages the net R of those that fill; 300 draws.
"""
import os, sys, sqlite3, math
import numpy as np
import pandas as pd

sys.path.insert(0, '/home/ec2-user/onemil/research/hod_exit_lab')
import walker  # noqa: E402

ROOT = '/home/ec2-user/onemil'
OUT = f'{ROOT}/research/hod_entry'
SLIP, STOP_SLIP, EOD_M, LAST_M, PLAC_LAST_M = 0.0002, 0.0013, 955, 930, 900
FLOOR, CONSOL_PCT, TARGET_R = 0.015, 0.04, 2.0
N_PLACEBO, N_START = 300, 40
LIMIT = int(os.environ.get('H1737_LIMIT', '0'))
TAG = os.environ.get('H1737_TAG', '')
THRU = float(os.environ.get('H1737_THROUGH', '0'))  # sensitivity: bid fills need low < bid*(1-THRU), resting-sell targets need high >= target*(1+THRU) (queue / through-trade rule); 0 = PREREG


def log(msg):
    """Verbose progress line."""
    walker.log(msg)


def run_bid(A, s, bid, cancel_high, cancel_close, stop, tgt_abs, hs):
    """Rest a buy limit `bid` from bar index s. A = dict of lists o,h,l,c,m. Cancel: bar h >= cancel_high (checked before
    the fill test), close >= cancel_close (after it), or bar m >= 930. tgt_abs None -> 2R target, else an absolute
    resting-sell price. Returns dict(filled, fill_i, entry, R, exit_px, why, net_R, exit_i)."""
    o, h, l, c, m = A['o'], A['h'], A['l'], A['c'], A['m']
    n = len(m)
    fi = None
    for k in range(s, n):
        if m[k] >= LAST_M:
            break
        if cancel_high is not None and h[k] >= cancel_high:
            break
        if l[k] < bid * (1 - THRU):
            fi = k
            break
        if cancel_close is not None and c[k] >= cancel_close:
            break
    if fi is None:
        return dict(filled=False)
    return walk_long(A, fi, bid * (1 + SLIP), stop, tgt_abs, hs, target_from=fi + 1)


def walk_long(A, k0, entry, stop, tgt_abs, hs, target_from):
    """Walk a long opened at bar k0 with post-cost `entry`. Stop is checked from k0, the target from `target_from`.
    Returns the fill dict."""
    o, h, l, c, m = A['o'], A['h'], A['l'], A['c'], A['m']
    n = len(m)
    r_abs = entry - stop
    if r_abs <= 0:
        return dict(filled=False, void='r_nonpos')
    target = entry + TARGET_R * r_abs if tgt_abs is None else tgt_abs
    res = None
    for k in range(k0, n):
        if m[k] >= EOD_M:
            px = o[k]
            res = (px, 'eod', (px * (1 - SLIP) - hs - entry) / r_abs, k)
            break
        if l[k] <= stop:
            px = o[k] if o[k] <= stop else stop
            res = (px, 'stop', (px * (1 - STOP_SLIP) - hs - entry) / r_abs, k)
            break
        if k >= target_from and h[k] >= target * (1 + THRU):
            res = (target, 'target', (target * (1 - SLIP) - entry) / r_abs, k)
            break
    if res is None:
        px = c[n - 1]
        res = (px, 'eod_lastbar', (px * (1 - SLIP) - hs - entry) / r_abs, n - 1)
    return dict(filled=True, fill_i=k0, entry=entry, R=r_abs, r_pct=r_abs / entry, exit_px=res[0], why=res[1],
                net_R=res[2], exit_i=res[3])


def stats(df, nweeks):
    """Per-spec statistics for a trade frame with columns net_R, why, day, entry_m, r_pct."""
    n = len(df)
    if n == 0:
        return dict(n=0)
    r = df.net_R.values
    mean = r.mean()
    iid_t = mean / (r.std(ddof=1) / math.sqrt(n)) if n > 2 and r.std(ddof=1) > 0 else float('nan')
    g = df.assign(_r=r - mean).groupby('day')._r.sum()
    se = math.sqrt((g ** 2).sum()) / n
    day_t = mean / se if se > 0 else float('nan')
    k = int(math.ceil(0.05 * n))
    ex5 = np.sort(r)[:-k].mean() if n > k else float('nan')
    wk = df.assign(_w=pd.to_datetime(df.day).dt.strftime('%G-%V'), _r2=r).groupby('_w')._r2.sum()
    wk_full = np.concatenate([wk.values, np.zeros(max(nweeks - len(wk), 0))])
    ordr = df.assign(_r2=r).sort_values(['day', 'entry_m'])._r2.cumsum().values
    dd = float((np.maximum.accumulate(np.concatenate([[0], ordr]))[1:] - ordr).max())
    return dict(n=n, mean_R=mean, day_t=day_t, iid_t=iid_t, ex_top5=ex5, capped3=np.minimum(r, 3.0).mean(),
                stop_share=(df.why == 'stop').mean(), green_wk_share=(wk > 0).sum() / nweeks,
                wk_p10=np.percentile(wk_full, 10), max_dd_R=dd, mean_R_pct_price=(r * df.r_pct.values).mean(),
                fills_per_wk=n / nweeks)


def arrays(d):
    """Plain-list view of a bar frame for the fast walkers."""
    return {k: d[k].tolist() for k in ('o', 'h', 'l', 'c', 'm')}


def pool_bid(A, starts, ref_close, bid, cancel_high, cancel_close, stop, tgt_abs, hs):
    """Placebo pool for one bid order: net R (NaN = unfilled) per start bar s with geometry scaled by close[s-1]/ref_close."""
    out = []
    for s in starts:
        sc = A['c'][s - 1] / ref_close
        r = run_bid(A, s, bid * sc, None if cancel_high is None else cancel_high * sc,
                    None if cancel_close is None else cancel_close * sc, stop * sc,
                    None if tgt_abs is None else tgt_abs * sc, hs * sc)
        out.append(r['net_R'] if r.get('filled') else np.nan)
    return np.array(out)


def pool_entry(A, starts, ref_open, stop, tgt, hs):
    """Placebo pool for a market entry at the open of bar s (M3), geometry scaled by open[s]/ref_open."""
    out = []
    for s in starts:
        sc = A['o'][s] / ref_open
        px = A['o'][s]
        r = walk_long(A, s, px + hs * sc + SLIP * px, stop * sc, tgt * sc, hs * sc, target_from=s)
        out.append(r['net_R'] if r.get('filled') else np.nan)
    return np.array(out)


def sample_starts(rng, A, first_idx):
    """Up to N_START random bar indices >= first_idx with minute < 900 (placebo starts), sorted."""
    m = A['m']
    cand = [k for k in range(max(first_idx, 1), len(m)) if m[k] < PLAC_LAST_M]
    if not cand:
        return []
    if len(cand) > N_START:
        cand = sorted(rng.choice(cand, N_START, replace=False).tolist())
    return cand


def find_arm(h, l, level):
    """Engine arm condition: first i>=5 with running HOD == level and the last 5 lows within 4 % below the HOD (lo < HOD)."""
    hod = -1.0
    for i in range(len(h)):
        hod = max(hod, h[i])
        if i >= 5 and abs(hod - level) <= 1e-6:
            lo = min(l[i - 4:i + 1])
            if lo >= hod * (1 - CONSOL_PCT) and lo < hod:
                return i
    return None


def main():
    """Build orders for M1/M2/M2c/M3, score, placebo, write outputs."""
    walker.wait_for_db_window()
    arms = pd.read_csv(f'{OUT}/replay_signals.csv', dtype={'symbol': str, 'day': str}, keep_default_na=False, na_values=[''])
    arms = arms[(arms.void_reason != 'no_level') & arms.level.notna()].reset_index(drop=True)
    if LIMIT:
        arms = arms.sample(LIMIT, random_state=1).reset_index(drop=True)
    log(f'arms {len(arms)}; splits {arms.split.value_counts().to_dict()}')
    sig = pd.read_parquet(f'{ROOT}/research/hod_exit_lab/signals.parquet')[['day', 'symbol', 'half_spread']].drop_duplicates(['day', 'symbol'])
    arms = arms.merge(sig, on=['day', 'symbol'], how='left')
    arms['matched_spread'] = arms.half_spread.notna()
    arms['dec'] = pd.qcut(arms.level, 10, labels=False, duplicates='drop')
    med = arms[arms.matched_spread].groupby('dec').half_spread.median()
    arms.loc[~arms.matched_spread, 'half_spread'] = arms.loc[~arms.matched_spread, 'dec'].map(med)
    b0 = pd.read_csv(f'{ROOT}/research/hod_exit_lab/b0_trades.csv').drop_duplicates(['day', 'symbol'])[['day', 'symbol', 'net_R']].rename(columns={'net_R': 'b0_net_R'})
    b0d = {(r.day, r.symbol): r.b0_net_R for r in b0.itertuples()}
    ev = pd.read_csv(f'{OUT}/1736_events.csv', dtype={'symbol': str, 'day': str})
    ev3 = ev[(ev.event == 1) & (ev.rails == 'ok')]
    ev3d = {(r.day, r.symbol, r.entry_m): r for r in ev3.itertuples()}
    log(f'unmatched spread share {1 - arms.matched_spread.mean():.3f}; M3 events (rails ok) {len(ev3)}')

    con = sqlite3.connect('file:' + ROOT + '/research/bf_zero/bars_sip.db?mode=ro', uri=True)
    rng = np.random.default_rng(1737)
    rows, pools = [], {}
    diag = dict(no_bars=0, j_missing=0, floor_bind=0, j_high_ge_level=0, hodj_eq_level=0, m1_noC=0, m1_orders=0,
                m2_noarm=0, m2_arm_after_j=0, m2_window0=0, m2_orders=0, m2_window_min=[], stop_a_ne_j=0, m3_missing=0)
    for ix, a in enumerate(arms.itertuples()):
        d = walker.fetch_day_bars(con, a.symbol, a.day)
        base = dict(day=a.day, symbol=a.symbol, entry_m=a.entry_m, level=a.level, split=a.split, half_spread=a.half_spread,
                    matched_spread=a.matched_spread, b0_net_R=b0d.get((a.day, a.symbol), np.nan))
        if d is None:
            diag['no_bars'] += 1
            continue
        A = arrays(d)
        h, l, c, m = A['h'], A['l'], A['c'], A['m']
        n = len(m)
        idx = [k for k in range(n) if m[k] >= a.entry_m]
        if not idx:
            diag['j_missing'] += 1
            continue
        j = idx[0]
        diag['j_high_ge_level'] += h[j] >= a.level - 1e-9
        diag['hodj_eq_level'] += abs(max(h[:j + 1]) - a.level) <= 1e-6
        lvl, hs = a.level, a.half_spread
        floor_px = lvl * (1 - FLOOR)

        def stop_at(jj):
            lo = min(l[max(jj - 4, 0):jj + 1])
            return min(lo, floor_px)
        stop_j = stop_at(j)            # PREREG-literal: bars j-4..j (j = bar at entry_m = the BREAK bar, fact (a))
        stop_p = stop_at(max(j - 1, 0))  # engine's own for a break at j: consolidation_low at the last closed bar j-1
        diag['floor_bind'] += stop_p == floor_px
        key = (a.day, a.symbol, a.entry_m)

        def row(mech, c_m, bid_px, stop, r, fill_px, extra=None):
            """Order record."""
            rec = dict(base, mech=mech, c_m=c_m, bid_px=bid_px, stop=stop, filled=bool(r.get('filled')),
                       fill_m=m[r['fill_i']] if r.get('filled') else np.nan, fill_px=fill_px if r.get('filled') else np.nan,
                       **{k: r.get(k, np.nan) for k in ('R', 'r_pct', 'exit_px', 'why', 'net_R')})
            rec.update(extra or {})
            rows.append(rec)
        # ---- M1 ----
        C = next((k for k in range(j, n) if m[k] < LAST_M and c[k] >= lvl), None)
        if C is None or C + 1 >= n:
            diag['m1_noC'] += 1
        else:
            diag['m1_orders'] += 1
            row('M1', m[C], lvl, stop_p, run_bid(A, C + 1, lvl, None, lvl * 1.02, stop_p, None, hs), lvl)
            pools[len(rows) - 1] = pool_bid(A, sample_starts(rng, A, C + 2), c[C], lvl, None, lvl * 1.02, stop_p, None, hs)
            row('M1L', m[C], lvl, stop_j, run_bid(A, C + 1, lvl, None, lvl * 1.02, stop_j, None, hs), lvl)
        # ---- M2 / M2c ----
        arm_i = find_arm(h, l, lvl)
        if arm_i is None:
            diag['m2_noarm'] += 1
        else:
            diag['m2_arm_after_j'] += arm_i > j
            s0 = arm_i + 1
            b_idx = next((k for k in range(s0, n) if h[k] >= lvl), None)
            t_idx = next((k for k in range(s0, n) if m[k] >= LAST_M), n)
            end = min(b_idx, t_idx) if b_idx is not None else t_idx
            win = end - s0
            if s0 >= n or win <= 0:
                diag['m2_window0'] += 1
            else:
                diag['m2_orders'] += 1
                diag['m2_window_min'].append(win)
                bid, tgt = lvl * 0.996, lvl * 1.003
                stop_a = stop_at(arm_i)
                diag['stop_a_ne_j'] += abs(stop_a - stop_j) > 1e-9
                diag['stop_a_ne_p'] = diag.get('stop_a_ne_p', 0) + (abs(stop_a - stop_p) > 1e-9)
                after = (b_idx + 1) if b_idx is not None else n
                for mech, st in (('M2', stop_a), ('M2L', stop_j)):
                    row(mech, m[arm_i], bid, st, run_bid(A, s0, bid, lvl, None, st, tgt, hs), bid)
                    if mech == 'M2':
                        pools[len(rows) - 1] = pool_bid(A, sample_starts(rng, A, after), c[arm_i], bid, lvl, None, st, tgt, hs)
        # ---- M3 ----
        e = ev3d.get(key)
        if e is not None:
            fk = [k for k in range(n) if m[k] >= e.fail_m]
            if not fk or fk[0] + 1 >= n:
                diag['m3_missing'] += 1
            else:
                f1 = fk[0] + 1
                px = A['o'][f1]
                stop3, tgt3 = e.entry - 2 * e.R, e.stop
                r = walk_long(A, f1, px + hs + SLIP * px, stop3, tgt3, hs, target_from=f1)
                row('M3', e.fail_m, px, stop3, r, px + hs + SLIP * px, dict(etb=bool(e.etb)))
                pools[len(rows) - 1] = pool_entry(A, sample_starts(rng, A, f1 + 1), px, stop3, tgt3, hs)
        if ix % 500 == 0:
            log(f'{ix}/{len(arms)} arms, rows {len(rows)}')
    orders = pd.DataFrame(rows)
    orders.to_csv(f'{OUT}/1737_orders{TAG}.csv', index=False)
    wmin = np.array(diag.pop('m2_window_min'))
    lines = [f'{k}: {v}' for k, v in diag.items()]
    lines.append(f'M2 windows (bars) n={len(wmin)} median={np.median(wmin) if len(wmin) else None} p90={np.percentile(wmin, 90) if len(wmin) else None}')
    lines.append(f'spread unmatched share among orders: {1 - orders.matched_spread.mean():.3f}')
    lines.append('why counts: ' + str(orders[orders.filled].groupby('mech').why.value_counts().to_dict()))
    nweeks = {s: pd.to_datetime(arms[arms.split == s].day).dt.strftime('%G-%V').nunique() for s in ('TRAIN', 'VAL')}
    prng = np.random.default_rng(17370)
    out = []
    for mech in ('M1', 'M1L', 'M2', 'M2L', 'M3'):
        for univ in (('all', 'etb') if mech == 'M3' else ('all',)):
            for s in ('TRAIN', 'VAL'):
                sel = orders[(orders.mech == mech) & (orders.split == s)]
                if univ == 'etb':
                    sel = sel[sel.etb == True]  # noqa: E712
                fl = sel[sel.filled]
                st = stats(fl, nweeks[s]) if len(fl) else dict(n=0)
                st.update(mech=mech, univ=univ, split=s, n_arms=int((arms.split == s).sum()), n_orders=len(sel), n_filled=len(fl),
                          fill_share_orders=len(fl) / max(len(sel), 1), fill_share_arms=len(fl) / max((arms.split == s).sum(), 1))
                b0f, b0u = fl.b0_net_R.dropna(), sel[~sel.filled].b0_net_R.dropna()
                st.update(b0_filled=b0f.mean() if len(b0f) else np.nan, b0_filled_n=len(b0f),
                          b0_unfilled=b0u.mean() if len(b0u) else np.nan, b0_unfilled_n=len(b0u))
                st['adverse_selection_flag'] = bool(len(fl) and len(b0u) and (b0u.mean() - fl.net_R.mean()) > 0.3)
                ids = [i for i in sel.index if i in pools and len(pools[i])]
                if len(fl) > 5 and ids:
                    L = np.array([len(pools[i]) for i in ids])
                    P = np.full((len(ids), L.max()), np.nan)
                    for r_, i in enumerate(ids):
                        P[r_, :L[r_]] = pools[i]
                    pick = (prng.random((N_PLACEBO, len(ids))) * L).astype(int)
                    vals = P[np.arange(len(ids)), pick]
                    with np.errstate(all='ignore'):
                        draws = np.nanmean(vals, axis=1)
                    st.update(placebo_pctile=(draws < fl.net_R.mean()).mean() * 100, placebo_mean=np.nanmean(draws), placebo_orders=len(ids))
                out.append(st)
                log(f"{mech:4s} {univ:4s} {s:5s}: orders={len(sel)} filled={len(fl)} mean={st.get('mean_R', float('nan')):.3f} "
                    f"day_t={st.get('day_t', float('nan')):.2f} pct={st.get('placebo_pctile', float('nan'))}")
    pd.DataFrame(out).to_csv(f'{OUT}/1737_summary{TAG}.csv', index=False)
    open(f'{OUT}/1737_diag{TAG}.txt', 'w').write('\n'.join(lines) + '\n')
    log('\n'.join(lines))
    log('DONE')


if __name__ == '__main__':
    main()
