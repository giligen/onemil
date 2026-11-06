"""Cells 1,723-1,725: ORB loser anatomy on real ledgers (L1 live, L2 paper) and paired early-exit / stop rules.

Read-only on data/trades.db and research/bf_zero/bars_sip.db; extra symbol-days from the scratch bars_extra.db
(fetched by the scratch fetch.py, LOST 0). Writes per_fill_1723.csv and reads_1723.csv in this directory.
Usage: python anatomy_1723.py <scratch_dir>
"""
import json, math, sqlite3, sys
from zoneinfo import ZoneInfo
import numpy as np
import pandas as pd

ROOT = '/home/ec2-user/onemil'
OUT = f'{ROOT}/research/orb_loser_anatomy'
ET = ZoneInfo('America/New_York')
SLIP_MKT = 0.0035     # market early exit: next open - 35 bps
SL_BAND = 0.0020      # R4 limit 20 bps below the stop
SL_SLIP = 0.0013      # R4 measured stop-limit slip (HOD 1,463/1,466)
RISK_USD = 375.0
STOPS = ('stop_loss', 'stop_loss_market_fallback')


def load_fills():
    """Return the ORB fill population (L1 live + L2 paper) with split labels and R per share."""
    c = sqlite3.connect(f'file:{ROOT}/data/trades.db?mode=ro', uri=True)
    q = """select trade_date,symbol,entry_price,fill_price,stop_loss_price,shares,total_risk,pnl,exit_price,
           exit_reason,filled_at,exited_at,pattern_data from trades where strategy='orb' and fill_price is not null
           and exit_price is not null and order_status in ('filled','closed') order by filled_at"""
    df = pd.read_sql_query(q, c)
    pdj = df['pattern_data'].apply(lambda s: json.loads(s) if s else {})
    df['range_low'] = pdj.apply(lambda d: d.get('range_low'))
    df['range_high'] = pdj.apply(lambda d: d.get('range_high'))
    df['ledger'] = np.where(df.trade_date <= '2026-09-23', 'L1', 'L2')
    df['split'] = np.where((df.ledger == 'L1') & (df.trade_date <= '2026-08-15'), 'TRAIN', 'VAL')
    df['fill_ts'] = pd.to_datetime(df.filled_at, utc=True, format='ISO8601')
    df['exit_ts'] = pd.to_datetime(df.exited_at, utc=True, format='ISO8601')
    df['rps'] = df.fill_price - df.stop_loss_price
    df['r_floor_ok'] = df.rps >= 0.005 * df.fill_price
    df['R_actual'] = (df.exit_price - df.fill_price) / df.rps
    df['R_pnl'] = df.pnl / (df.shares * df.rps)
    return df.drop(columns=['pattern_data'])


def load_bars(sym, day, extra):
    """1-min bars for a symbol-day from bars_sip.db, else the scratch extra store; index = bar start (UTC)."""
    for path in (f'{ROOT}/research/bf_zero/bars_sip.db', extra):
        b = sqlite3.connect(f'file:{path}?mode=ro', uri=True)
        x = pd.read_sql_query("select t,o,h,l,c,v from bars where symbol=? and day=? order by t", b, params=(sym, day))
        if len(x) > 100:
            x['t'] = pd.to_datetime(x.t, utc=True, format='ISO8601')
            x = x.set_index('t')
            et = x.index.tz_convert(ET)
            mins = et.hour * 60 + et.minute
            x = x[(mins >= 570) & (mins < 960)].copy()
            x['et_min'] = (x.index.tz_convert(ET).hour * 60 + x.index.tz_convert(ET).minute)
            tp = (x.h + x.l + x.c) / 3
            x['vwap'] = (tp * x.v).cumsum() / x.v.cumsum().replace(0, np.nan)
            return x
    return None


def ceil_min(ts):
    """Start of the first full minute bar after timestamp ts."""
    f = ts.floor('min')
    return f if f == ts else f + pd.Timedelta(minutes=1)


def early_exit(bars, fill, rps, decision_bar_start, exit_ts):
    """Market exit at the open of the bar after decision_bar_start, minus 35 bps, if that bar starts before the
    actual exit; returns (rule_R or None)."""
    nxt = decision_bar_start + pd.Timedelta(minutes=1)
    if nxt >= exit_ts.floor('min') + pd.Timedelta(minutes=0) and nxt > exit_ts:
        return None
    if nxt > exit_ts or nxt not in bars.index:
        return None
    px = bars.loc[nxt, 'o'] * (1 - SLIP_MKT)
    return (px - fill) / rps


def walk(r, bars):
    """Compute the path anatomy and every rule's R for one fill. Returns a dict of new columns."""
    fill, rps, ex = r.fill_price, r.rps, r.exit_ts
    t0 = ceil_min(r.fill_ts)
    path = bars[(bars.index >= t0) & (bars.index <= ex.floor('min'))]
    o = {}
    if len(path) == 0:
        path = bars[(bars.index >= r.fill_ts.floor('min')) & (bars.index <= ex.floor('min'))]
    m_from_fill = ((path.index - r.fill_ts).total_seconds() / 60.0 + 1.0)  # minutes at each bar's close
    hi = (path.h - fill) / rps
    lo = (path.l - fill) / rps
    o['mfe_R'] = max(float(hi.max()) if len(hi) else 0.0, max(r.R_actual, 0))
    o['mae_R'] = min(float(lo.min()) if len(lo) else 0.0, min(r.R_actual, 0))
    o['min_mfe'] = float(m_from_fill[int(np.argmax(hi.values))]) if len(hi) else 0.0
    o['min_mae'] = float(m_from_fill[int(np.argmin(lo.values))]) if len(lo) else 0.0
    o['min_exit'] = (ex - r.fill_ts).total_seconds() / 60.0
    o['exit_et_min'] = ex.tz_convert(ET).hour * 60 + ex.tz_convert(ET).minute
    for k in (0.25, 0.5, 1.0):
        o[f'hit_{k}R'] = o['mfe_R'] >= k
    # causal within-5 / within-20 path reads
    w5 = path[m_from_fill <= 5.0]; w20 = path[m_from_fill <= 20.0]
    o['mae5_R'] = float(((w5.l - fill) / rps).min()) if len(w5) else 0.0
    o['mfe5_R'] = float(((w5.h - fill) / rps).max()) if len(w5) else 0.0
    o['mfe20_R'] = float(((w20.h - fill) / rps).max()) if len(w20) else 0.0
    o['bars_below_entry'] = int((path.c < fill).sum())
    for k in (5, 10, 20):
        tb = t0 + pd.Timedelta(minutes=k - 1)
        o[f'c_vs_vwap_{k}'] = (bars.loc[tb, 'c'] / bars.loc[tb, 'vwap'] - 1) * 1e4 if tb in bars.index else np.nan
    o['bar1_c_vs_entry_bps'] = (bars.loc[t0, 'c'] / fill - 1) * 1e4 if t0 in bars.index else np.nan
    rng = bars[(bars.et_min >= 570) & (bars.et_min < 575)].v.sum()
    v5 = bars[(bars.index >= t0) & (bars.index < t0 + pd.Timedelta(minutes=5))].v.sum()
    o['vol5_vs_range'] = v5 / rng if rng > 0 else np.nan
    # stop anatomy
    o['stop_slip_bps'] = (r.stop_loss_price - r.exit_price) / r.stop_loss_price * 1e4 if r.exit_reason in STOPS else np.nan
    after = bars[(bars.index > ex.floor('min')) & (bars.index <= ex.floor('min') + pd.Timedelta(minutes=30))]
    o['stop_and_reverse'] = bool(r.exit_reason in STOPS and len(after) and after.h.max() >= fill)
    # ---- rules: R of the rule exit (None = rule does not change the trade) ----
    rules = {}
    # R1 stagnation at 20 min
    dec = t0 + pd.Timedelta(minutes=19)
    rules['R1'] = early_exit(bars, fill, rps, dec, ex) if (dec in bars.index and o['mfe20_R'] < 0.25) else None
    # R2 bar-1 fail
    rules['R2'] = early_exit(bars, fill, rps, t0, ex) if (t0 in bars.index and bars.loc[t0, 'c'] < fill) else None
    # R3 VWAP loss at 10 and 11
    b10 = t0 + pd.Timedelta(minutes=9); b11 = b10 + pd.Timedelta(minutes=1)
    rules['R3'] = (early_exit(bars, fill, rps, b11, ex) if (b11 in bars.index and b10 in bars.index and
                   bars.loc[b10, 'c'] < bars.loc[b10, 'vwap'] and bars.loc[b11, 'c'] < bars.loc[b11, 'vwap']) else None)
    # force-close bar (15:45 ET) for carried exits
    fc = bars[bars.et_min >= 945]
    fc_px = float(fc.o.iloc[0]) if len(fc) else float(bars.c.iloc[-1])
    # R4 stop-limit band (mechanical): in band -> same fill as the market stop; through the band -> rests at L
    o['r4_unfilled'] = False; o['r4_carry_fc'] = False
    if r.exit_reason in STOPS:
        L = r.stop_loss_price * (1 - SL_BAND)
        if r.exit_price >= L:
            rules['R4'] = None
            rules['R4lit'] = (r.stop_loss_price * (1 - SL_SLIP) - fill) / rps
        else:
            o['r4_unfilled'] = True
            xb = ex.floor('min')
            fillpx = None
            if xb in bars.index and bars.loc[xb, 'c'] >= L:
                fillpx = L
            else:
                later = bars[(bars.index > xb) & (bars.et_min < 945)]
                hit = later[later.h >= L]
                fillpx = L if len(hit) else None
            if fillpx is None:
                o['r4_carry_fc'] = True; fillpx = fc_px
            rules['R4'] = (fillpx - fill) / rps
            rules['R4lit'] = rules['R4']
    else:
        rules['R4'] = None; rules['R4lit'] = None
    # R5 confirmation stop: a 1-min bar CLOSES below range_low -> next open - 35 bps; replaces the touch stop
    rl = r.range_low if r.range_low is not None and not pd.isna(r.range_low) else r.stop_loss_price
    seg = bars[(bars.index >= t0) & (bars.et_min < 945)]
    brk = seg[seg.c < rl]
    if r.exit_reason in STOPS:
        if len(brk):
            nb = brk.index[0] + pd.Timedelta(minutes=1)
            px = bars.loc[nb, 'o'] * (1 - SLIP_MKT) if nb in bars.index else fc_px
        else:
            px = fc_px; o['r5_to_fc'] = True
        rules['R5'] = (px - fill) / rps
    else:
        b = brk[brk.index < ex.floor('min')]
        rules['R5'] = early_exit(bars, fill, rps, b.index[0], ex) if len(b) else None
    # 1,725 trigger candidates (pre-declared mapping cluster -> trigger, chosen on TRAIN stop-$ share)
    d5 = t0 + pd.Timedelta(minutes=4)
    rules['T1'] = early_exit(bars, fill, rps, d5, ex) if (d5 in bars.index and o['mae5_R'] <= -0.5 and o['mfe5_R'] < 0.25) else None
    below = (seg.c < fill).astype(int).cumsum(); mfe_run = ((seg.h - fill) / rps).cummax()
    hit3 = seg[(below >= 20) & (mfe_run < 0.25)]
    rules['T3'] = early_exit(bars, fill, rps, hit3.index[0], ex) if len(hit3) else None
    armed = mfe_run >= 1.0
    hit2 = seg[armed & (seg.c <= fill + 0.25 * rps)]
    rules['T2'] = early_exit(bars, fill, rps, hit2.index[0], ex) if len(hit2) else None
    popped = mfe_run >= 0.25
    hit5 = seg[popped & (seg.c < fill)]
    rules['T5'] = early_exit(bars, fill, rps, hit5.index[0], ex) if len(hit5) else None
    noon = seg[seg.et_min == 719]
    rules['T4'] = (early_exit(bars, fill, rps, noon.index[0], ex) if len(noon) and noon.c.iloc[0] < fill else None)
    for k, v in rules.items():
        o[f'dR_{k}'] = 0.0 if v is None else float(v - r.R_actual)
    return o


def cluster(row):
    """Path-shape cluster for a loser (priority: give-back, immediate, slow bleed, late stop, other)."""
    if row.mfe_R >= 1.0: return 'C2_giveback'
    if row.mae5_R <= -0.5 and row.mfe_R < 0.25: return 'C1_immediate'
    if row.mfe_R < 0.25 and row.bars_below_entry >= 20: return 'C3_slow_bleed'
    if row.exit_et_min >= 720: return 'C4_late'
    if row.mfe_R >= 0.25: return 'C5a_pop_fade'
    if row.exit_reason == 'tag_bb' and row.min_exit <= 5: return 'C5b_tag_scratch'
    return 'C5c_other'


def cl_t(x, days):
    """Mean, day-clustered t, iid t, cluster SE for paired deltas x grouped by days."""
    x = np.asarray(x, float); n = len(x)
    if n < 2: return (np.nan,) * 4
    m = x.mean(); dev = pd.Series(x - m).groupby(np.asarray(days)).sum()
    se_c = math.sqrt((dev ** 2).sum()) / n
    se_i = x.std(ddof=1) / math.sqrt(n)
    return m, m / se_c if se_c > 0 else np.nan, m / se_i if se_i > 0 else np.nan, se_c


def read_rule(df, rule):
    """Paired read of one rule on a fill subset (R-floored rows)."""
    d = df[df.r_floor_ok]
    x = d[f'dR_{rule}'].values
    m, tc, ti, se = cl_t(x, d.trade_date.values)
    k = int(math.ceil(0.05 * len(x)))
    ex5 = np.sort(x)[:-k].mean() if k and len(x) > k else np.nan
    aff = d[d[f'dR_{rule}'] != 0]
    win = d[(d.R_actual > 0) & (d[f'dR_{rule}'] < 0)]
    return dict(rule=rule, n=len(d), n_aff=len(aff), dR=m, t_day=tc, t_iid=ti, MDE=2 * se if se == se else np.nan,
                dUSD=x.sum() * RISK_USD, ex_top5=ex5, saved_R=x[x > 0].sum(), forgone_R=x[x < 0].sum(),
                winners_clipped=len(win), winners_clipped_usd=win[f'dR_{rule}'].sum() * RISK_USD)


def main():
    """Build the per-fill table, the cluster tables and the rule reads."""
    extra = f'{sys.argv[1]}/bars_extra.db'
    df = load_fills()
    recs = []
    for _, r in df.iterrows():
        bars = load_bars(r.symbol, r.trade_date, extra)
        if bars is None:
            print('NO BARS', r.trade_date, r.symbol, flush=True); recs.append({}); continue
        recs.append(walk(r, bars))
    df = pd.concat([df.reset_index(drop=True), pd.DataFrame(recs)], axis=1)
    df['loser'] = df.pnl < 0
    df['cluster'] = np.where(df.loser, df.apply(cluster, axis=1), 'W')
    df.drop(columns=['fill_ts', 'exit_ts']).to_csv(f'{OUT}/per_fill_1723.csv', index=False)
    print('fills', len(df), 'R-floor fail', (~df.r_floor_ok).sum(), 'by ledger', df.ledger.value_counts().to_dict())
    print('R_actual vs R_pnl max abs diff', (df.R_actual - df.R_pnl).abs().max())
    st = df[df.exit_reason.isin(STOPS)]
    print('\n== STOP ANATOMY (n=%d) ==' % len(st))
    for s, g in st.groupby('split'):
        slip_usd = ((g.stop_loss_price - g.exit_price) * g.shares).sum()
        print(s, 'n', len(g), 'slip bps mean %.1f med %.1f p90 %.1f max %.1f' % (g.stop_slip_bps.mean(), g.stop_slip_bps.median(),
              g.stop_slip_bps.quantile(.9), g.stop_slip_bps.max()), 'R mean %.2f' % g.R_actual.mean(),
              'stop$ %.0f slip$ %.0f (%.1f%%)' % (g.pnl.sum(), -slip_usd, 100 * slip_usd / -g.pnl.sum()),
              'min-to-stop med %.1f' % g.min_exit.median(), 'stop&reverse30 %d' % g.stop_and_reverse.sum(),
              'R4 through-band %d carried-to-FC %d' % (g.r4_unfilled.sum(), g.r4_carry_fc.sum()))
    print('\n== LOSER CLUSTERS ==')
    for s in ('TRAIN', 'VAL'):
        g = df[(df.split == s) & df.loser]; stopd = -g[g.exit_reason.isin(STOPS)].pnl.sum()
        for c, h in g.groupby('cluster'):
            hs = h[h.exit_reason.isin(STOPS)]
            print(s, c, 'n %d stops %d $ %.0f R %.2f stop$share %.0f%% mfe %.2f mae5 %.2f min_exit med %.0f' % (
                len(h), len(hs), h.pnl.sum(), h.R_actual.mean(), 100 * -hs.pnl.sum() / stopd if stopd else 0,
                h.mfe_R.mean(), h.mae5_R.mean(), h.min_exit.median()))
    print('\n== WINNERS by causal shape (clip risk) ==')
    w = df[~df.loser]
    w = w.assign(shape=np.select([w.mfe5_R.lt(0.25) & w.mae5_R.le(-0.5), w.mfe20_R.lt(0.25), w.mfe_R.ge(1.0)],
                                 ['immediate_dip', 'stagnant20', 'hit1R'], 'other'))
    for (s, c), h in w.groupby(['split', 'shape']):
        print(s, c, 'n %d $ %.0f R %.2f' % (len(h), h.pnl.sum(), h.R_actual.mean()))
    print('\n== PATH SUMMARY by split x loser ==')
    print(df.groupby(['split', 'loser'])[['mfe_R', 'mae_R', 'min_mfe', 'min_mae', 'min_exit', 'hit_0.25R', 'hit_0.5R',
          'hit_1.0R', 'c_vs_vwap_5', 'c_vs_vwap_10', 'c_vs_vwap_20', 'bar1_c_vs_entry_bps', 'vol5_vs_range']].mean().round(2).T)
    out = []
    for rule in ('R1', 'R2', 'R3', 'R4', 'R4lit', 'R5', 'T1', 'T2', 'T3', 'T4', 'T5'):
        for lab, sub in (('TRAIN', df[df.split == 'TRAIN']), ('VAL', df[df.split == 'VAL']), ('L2', df[df.ledger == 'L2']),
                         ('VAL_L1', df[(df.split == 'VAL') & (df.ledger == 'L1')])):
            d = read_rule(sub, rule); d['set'] = lab; out.append(d)
    rd = pd.DataFrame(out)
    rd.to_csv(f'{OUT}/reads_1723.csv', index=False)
    pd.set_option('display.width', 250)
    sel = sys.argv[2] if len(sys.argv) > 2 else 'TRAIN'
    show = rd if sel == 'ALL' else rd[rd.set == 'TRAIN']
    print('\n== RULE READS (%s) ==' % sel)
    print(show.round(3).to_string(index=False))


if __name__ == '__main__':
    main()
