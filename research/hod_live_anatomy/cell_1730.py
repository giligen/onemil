#!/usr/bin/env python3
"""Cell 1,730 — the entry-cap ladder on the HOD BT population (PREREG_1730.md, frozen 2026-10-31).

Question: does a tighter resting stop-limit cap than the live +15 bps buy gross R per trade, or only un-fill?

Fill model (PREREG, identical on every rung): trigger bar T = the bar whose high first crosses the armed level;
d = (open_T - level) / level in bps (floored at 0: open_T < level fills AT the level). Rung c fills iff d <= c, at
max(open_T, level). Exits = the live rule on the raw SIP minute bars (research/bf_zero/bars_sip.db, read-only,
one query per (symbol, day)): stop = consolidation low, target = entry + 2 x (entry - stop), flat at the open of the
first bar >= 15:55 ET. R unit = level - stop (the live sizing basis, identical across rungs, so deltas are paired).
Cost: 12 bps of the exit price on stop / gap / EOD exits, nothing on target prints, no extra entry slip.

Populations:
  A (PRIMARY, the PREREG's) = the 5,506 live-rule BT fills (1663_features.csv) joined to causal_arming_causal.csv
    for level and the fractional fill minute; T = floor(fill_min) (the 1,438 fill window is bar j+1).
  B (report-only) = every armed crossing re-derived with research/hod_entry/causal_arming.armed_crossing_bars on
    the same bars for every symbol-day that had >= 1 armed crossing in the 1,438 build (fill or nofill); the FIRST
    armed crossing per symbol-day is the signal (B1); B2 lets a declined order re-arm at the next armed crossing.
    B exists because A is conditioned on the BT's own 15 bps NBBO fill, so A cannot price the looser rungs.

Cache: research/hod_live_anatomy/cache/opens_1730.parquet (one row per crossing incl. its outcome; resumable — the
symbol-days already present are skipped). Outputs: ladder_1730.csv, cell_1730_out.txt (RESULT_1730.md by hand).
"""
import os
import sqlite3
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
sys.path.insert(0, os.path.join(ROOT, 'research/hod_entry'))
sys.path.insert(0, ROOT)
import causal_arming as ca  # noqa: E402

CACHE = os.path.join(HERE, 'cache', 'opens_1730.parquet')
OUT_TXT = os.path.join(HERE, 'cell_1730_out.txt')
LADDER_CSV = os.path.join(HERE, 'ladder_1730.csv')
BARS_URI = 'file:' + os.path.join(ROOT, 'research/bf_zero/bars_sip.db') + '?mode=ro'
RUNGS = [0, 5, 10, 15, 25, 50, np.inf]
REF = 15
EXIT_COST = 0.0012
EOD_M = 955
N_BOOT = 2000
_out = open(OUT_TXT, 'a')


def log(*a):
    """Print flushed to stdout and append to the cell's output file."""
    s = ' '.join(str(x) for x in a)
    print(s, flush=True)
    _out.write(s + '\n')
    _out.flush()


# ------------------------------------------------------------------------------------------------ inputs
def load_pop_a() -> pd.DataFrame:
    """The 5,506 live-rule BT fills joined 1:1 to the 1,438 causal build for level and fill minute."""
    a = pd.read_csv(os.path.join(ROOT, 'research/hod_entry/1663_features.csv'))
    c = pd.read_csv(os.path.join(ROOT, 'research/hod_entry/causal_arming_causal.csv'), low_memory=False)
    c = c[c.status == 'fill'].rename(columns={'day': 'date', 'fill_min': 'fill_min_c', 'stop': 'stop_c'})
    a = a.merge(c[['date', 'symbol', 'level', 'fill', 'fill_min_c', 'stop_c', 'n_cross']], on=['date', 'symbol'],
                how='left', validate='1:1')
    miss = a.level.isna().sum()
    log(f'[A] {len(a)} BT fills; join misses {miss}; stop agreement |1663 - 1438| < 1c: '
        f'{(a.stop - a.stop_c).abs().lt(0.01).mean():.3f}')
    a = a[a.level.notna()].copy()
    a['T'] = np.floor(a.fill_min.astype(float)).astype(int)
    a['split'] = a.half
    return a


def load_pop_b_keys() -> pd.DataFrame:
    """Symbol-days with >= 1 armed crossing in the 1,438 build (status fill or nofill) + adv20 + split label."""
    c = pd.read_csv(os.path.join(ROOT, 'research/hod_entry/causal_arming_causal.csv'), low_memory=False)
    c = c[c.status.isin(['fill', 'nofill'])].rename(columns={'day': 'date'})
    u = pd.read_csv(ca.UNIVERSE_CSV, dtype={'symbol': str}, keep_default_na=False)
    u = u.rename(columns={'bar_date': 'date'})[['date', 'symbol', 'adv20']]
    u['adv20'] = pd.to_numeric(u.adv20, errors='coerce')
    k = c[['date', 'symbol', 'split', 'half', 'status']].merge(u.drop_duplicates(['date', 'symbol']),
                                                               on=['date', 'symbol'], how='left')
    k['split'] = np.where(k.split == 'TRAIN', 'TRAIN-H2', 'VAL')
    log(f'[B] {len(k)} symbol-days with an armed crossing (fill {int((k.status == "fill").sum())}, nofill '
        f'{int((k.status == "nofill").sum())}); adv20 missing {int(k.adv20.isna().sum())}')
    return k


def load_live() -> pd.DataFrame:
    """The 231 R-valid live paper fills (cell 1,721): entry_slip_bps, net R, gross R."""
    L = pd.read_csv(os.path.join(HERE, 'fills_1721.csv'), low_memory=False)
    L = L[L.valid_r.astype(str) == 'True'].copy()
    L['date'] = pd.to_datetime(L.trade_date).dt.strftime('%Y-%m-%d')
    return L


# ------------------------------------------------------------------------------------------------ bars + walk
def rth_bars(con, symbol: str, day: str) -> pd.DataFrame:
    """RTH 1-min SIP bars (m, o, h, l, c, v) for one (symbol, day), read-only."""
    g = pd.read_sql('select t, o, h, l, c, v from bars where symbol=? and day=?', con, params=[symbol, day])
    if g.empty:
        return g
    return ca._rth(g, 't')


def walk(b: pd.DataFrame, t0: int, entry: float, level: float, stop: float) -> tuple:
    """Live-rule exit from bar index t0 (the trigger bar) at `entry`. Bar t0: if the fill was at the open
    (open >= level) any low <= stop is after entry -> stopped; if the bar opened below the level the low may
    precede the cross, so it stops only if the bar CLOSES <= stop. Later bars: open <= stop -> gap exit at the
    open; low <= stop -> stop (stop first when both); high >= target -> target; first bar >= 15:55 -> its open.
    Returns (exit_m, exit_px, why)."""
    m, o, h, l, c = (b[k].values for k in ('m', 'o', 'h', 'l', 'c'))
    tgt = entry + 2.0 * (entry - stop)
    at_open = o[t0] >= level
    if (at_open and l[t0] <= stop) or (not at_open and c[t0] <= stop):
        return int(m[t0]), stop, 'stop'
    if h[t0] >= tgt:
        return int(m[t0]), tgt, 'target'
    for i in range(t0 + 1, len(m)):
        if m[i] >= EOD_M:
            return int(m[i]), float(o[i]), 'eod'
        if o[i] <= stop:
            return int(m[i]), float(o[i]), 'gap_stop'
        if l[i] <= stop:
            return int(m[i]), stop, 'stop'
        if h[i] >= tgt:
            return int(m[i]), tgt, 'target'
    return int(m[-1]), float(c[-1]), 'eod_last'


def crossing_row(b, t0, level, stop, extra) -> dict:
    """One crossing: trigger-bar OHLCV, d (bps), entry, outcome, net R on the level basis, entry-quality features."""
    o_t = float(b.o.iat[t0])
    entry = max(o_t, level)
    em, ex, why = walk(b, t0, entry, level, stop)
    u = level - stop
    gross = (ex - entry) / u
    cost = 0.0 if why == 'target' else EXIT_COST * ex / u
    j = t0 - 1
    v = b.v.values
    hod_idx = int(np.argmax(b.h.values[: t0])) if t0 > 0 else 0
    return dict(extra, m_T=int(b.m.iat[t0]), level=level, stop=stop, o_T=o_t, h_T=float(b.h.iat[t0]),
                l_T=float(b.l.iat[t0]), c_T=float(b.c.iat[t0]), v_T=float(v[t0]),
                v_cons10=float(np.mean(v[max(0, j - 9): j + 1])) if j >= 0 else np.nan,
                mins_since_level=int(b.m.iat[t0] - b.m.iat[hod_idx]),
                d_bps=max(0.0, (o_t / level - 1) * 1e4), d_raw_bps=(o_t / level - 1) * 1e4,
                entry=entry, exit_m=em, exit_px=ex, why=why, r_pct=u / level * 100,
                gross_R=gross, cost_R=cost, net_R=gross - cost)


def build_cache(A: pd.DataFrame, K: pd.DataFrame) -> pd.DataFrame:
    """Per (symbol, day): population-A row (the BT fill's own crossing) + every population-B armed crossing.
    Resumable: symbol-days already in the cache parquet are skipped; atomic rewrite every 500 symbol-days."""
    p, floor, _ = ca.live_params()
    log(f'[params] floor(min_price)={floor} min_r_pct={p.min_r_pct} last_entry_minute={p.last_entry_minute}')
    rows = []
    done = set()
    if os.path.exists(CACHE):
        old = pd.read_parquet(CACHE)
        rows = old.to_dict('records')
        done = set(zip(old.date, old.symbol))
        log(f'[cache] resume: {len(done)} symbol-days already cached')
    keys = pd.concat([A[['date', 'symbol']], K[['date', 'symbol']]]).drop_duplicates()
    keys = keys[[k not in done for k in zip(keys.date, keys.symbol)]].sort_values(['date', 'symbol'])
    Ai = A.set_index(['date', 'symbol'])
    Ki = K.drop_duplicates(['date', 'symbol']).set_index(['date', 'symbol'])
    con = sqlite3.connect(BARS_URI, uri=True, timeout=60)
    t_start, n_nobars, n_not_t = time.time(), 0, 0
    for n, (day, sym) in enumerate(zip(keys.date, keys.symbol), 1):
        b = rth_bars(con, sym, day)
        if b.empty:
            n_nobars += 1
            rows.append(dict(date=day, symbol=sym, pop='none', k=-1))
            continue
        got = False
        if (day, sym) in Ai.index:
            r = Ai.loc[(day, sym)]
            hit = np.nonzero(b.m.values == int(r['T']))[0]
            if len(hit):
                rows.append(crossing_row(b, int(hit[0]), float(r.level), float(r.stop),
                                         dict(date=day, symbol=sym, pop='A', k=0, split=r.split)))
                got = True
            else:
                n_not_t += 1
        if (day, sym) in Ki.index:
            kr = Ki.loc[(day, sym)]
            adv = float(kr.adv20) if pd.notna(kr.adv20) else 0.0
            for k, a in enumerate(ca.armed_crossing_bars(b, adv, p, floor)):
                rows.append(crossing_row(b, a['j'] + 1, float(a['level']), float(a['stop']),
                                         dict(date=day, symbol=sym, pop='B', k=k, split=kr.split)))
                got = True
        if not got:
            rows.append(dict(date=day, symbol=sym, pop='none', k=-1))
        if n % 500 == 0 or n == len(keys):
            tmp = CACHE + '.tmp'
            pd.DataFrame(rows).to_parquet(tmp, index=False)
            os.replace(tmp, CACHE)
            el = time.time() - t_start
            log(f'[bars] {n}/{len(keys)} symbol-days, {el:.0f}s, eta {el / n * (len(keys) - n):.0f}s, '
                f'no-bars {n_nobars}, A trigger bar missing {n_not_t}')
    con.close()
    if not len(keys):
        log('[cache] nothing to fetch')
    return pd.DataFrame(rows)


# ------------------------------------------------------------------------------------------------ statistics
def dclust_t(v: pd.Series, dates: pd.Series) -> float:
    """Day-clustered t of the mean of v."""
    if len(v) < 3:
        return np.nan
    dev = (v - v.mean()).groupby(dates.values).sum()
    se = np.sqrt((dev ** 2).sum()) / len(v)
    return v.mean() / se if se > 0 else np.nan


def ex_top5(v: pd.Series) -> float:
    """Mean after dropping the top 5 % of rows."""
    v = v.sort_values()
    return v.iloc[: int(np.floor(len(v) * 0.95))].mean() if len(v) else np.nan


def boot_delta(x: pd.DataFrame, c: float, rng) -> tuple:
    """Paired per-trade delta mean(net | d<=c) - mean(net | d<=REF) and the same for ex-top-5 %, with a
    day-cluster bootstrap SE (days resampled with replacement, both books recomputed on the same draw)."""
    a, r = x[x.d_bps <= c].net_R, x[x.d_bps <= REF].net_R
    d0, e0 = a.mean() - r.mean(), ex_top5(a) - ex_top5(r)
    kc, kr = (x.d_bps <= c).astype(float), (x.d_bps <= REF).astype(float)
    per_day = pd.DataFrame({'Sc': kc * x.net_R, 'Nc': kc, 'Sr': kr * x.net_R, 'Nr': kr}).groupby(x.date.values).sum()
    M = per_day.values                                           # days x 4
    W = rng.multinomial(len(M), np.full(len(M), 1.0 / len(M)), size=N_BOOT)   # day-resampling weights
    T = W @ M                                                    # N_BOOT x 4 resampled sums
    with np.errstate(invalid='ignore', divide='ignore'):
        ds = T[:, 0] / T[:, 1] - T[:, 2] / T[:, 3]
    se = np.nanstd(ds)
    return d0, (d0 / se if se > 0 else np.nan), e0


def ladder(x: pd.DataFrame, label: str, weeks: dict, rng) -> list:
    """Every rung x split: n, fills/wk, net R/trade, R/wk, day-clustered t, paired delta vs c=15 (bootstrap t),
    per-signal paired t, ex-top-5 %, winner-capped (+2 R), alternating-week halves."""
    out = []
    for s in ('TRAIN-H2', 'VAL'):
        xs = x[x.split == s].copy()
        xs['wkn'] = pd.to_datetime(xs.date).dt.isocalendar().week.astype(int)
        for c in RUNGS:
            k = xs[xs.d_bps <= c]
            row = dict(pop=label, split=s, cap=c, n=len(k), fills_wk=len(k) / weeks[s],
                       net=k.net_R.mean(), gross=k.gross_R.mean(), R_wk=k.net_R.sum() / weeks[s],
                       t=dclust_t(k.net_R, k.date), ex5=ex_top5(k.net_R), capped=k.net_R.clip(upper=2).mean(),
                       halfA=k[k.wkn % 2 == 0].net_R.mean(), halfB=k[k.wkn % 2 == 1].net_R.mean())
            if c != REF:
                d, tb, e = boot_delta(xs, c, rng)
                ref = xs[xs.d_bps <= REF]
                ra, rb = ref[ref.wkn % 2 == 0], ref[ref.wkn % 2 == 1]
                ka, kb = k[k.wkn % 2 == 0], k[k.wkn % 2 == 1]
                ps = np.where(xs.d_bps <= c, xs.net_R, 0) - np.where(xs.d_bps <= REF, xs.net_R, 0)
                row.update(delta=d, delta_t=tb, delta_ex5=e, dA=ka.net_R.mean() - ra.net_R.mean(),
                           dB=kb.net_R.mean() - rb.net_R.mean(),
                           persig_t=dclust_t(pd.Series(ps), xs.date))
            out.append(row)
    return out


def bucket_table(x: pd.DataFrame, L: pd.DataFrame) -> None:
    """L2: net R by d bucket at c = 15 per split, beside the live table by entry slip."""
    bins, lab = [-1e-9, 5, 10, 15], ['<=5', '5-10', '10-15']
    for s in ('TRAIN-H2', 'VAL'):
        xs = x[(x.split == s) & (x.d_bps <= REF)]
        g = xs.groupby(pd.cut(xs.d_bps, bins, labels=lab), observed=False)
        log(f'  L2 {s}: ' + '; '.join(f'{b} n {len(v)} net {v.net_R.mean():+.3f} gross {v.gross_R.mean():+.3f} '
                                     f't {dclust_t(v.net_R, v.date):+.2f} ex5 {ex_top5(v.net_R):+.3f}'
                                     for b, v in g))
        d0 = xs[xs.d_bps == 0]
        log(f'     d == 0 (opened at/below level) n {len(d0)} net {d0.net_R.mean():+.3f}; '
            f'0 < d <= 5 n {int(((xs.d_bps > 0) & (xs.d_bps <= 5)).sum())} '
            f'net {xs[(xs.d_bps > 0) & (xs.d_bps <= 5)].net_R.mean():+.3f}')
    g = L.groupby(pd.cut(L.entry_slip_bps, [-1e9, 5, 10, 15, 1e9], labels=['<=5', '5-10', '10-15', '>15']),
                  observed=False)
    log('  L2 LIVE (re-read, not re-fit): ' + '; '.join(f'{b} n {len(v)} net {v.R.mean():+.3f} gross '
                                                       f'{v.gross_R.mean():+.3f}' for b, v in g))


def declined(x: pd.DataFrame) -> None:
    """L3: at c = 5 and 10, the signals declined vs c = 15 (c < d <= 15): their net R if filled at open_T, and the
    retrace share (trigger-bar low <= level x (1 + c)) the live engine's working limit might still have filled."""
    for s in ('TRAIN-H2', 'VAL'):
        xs = x[x.split == s]
        for c in (5, 10):
            dc = xs[(xs.d_bps > c) & (xs.d_bps <= REF)]
            kept = xs[xs.d_bps <= c]
            rt = (dc.l_T <= dc.level * (1 + c / 1e4)).mean()
            log(f'  L3 {s} c={c}: declined n {len(dc)} net {dc.net_R.mean():+.3f} (t {dclust_t(dc.net_R, dc.date):+.2f}, '
                f'ex5 {ex_top5(dc.net_R):+.3f}) vs kept n {len(kept)} net {kept.net_R.mean():+.3f}; '
                f'retrace share {rt:.1%}')


def live_oos(L: pd.DataFrame, rng) -> None:
    """Live cohort with slip <= c vs the live mean, bootstrap 95 % CI of the cohort mean (per rung)."""
    log(f'  LIVE all n {len(L)} net {L.R.mean():+.3f}')
    for c in RUNGS:
        k = L[L.entry_slip_bps <= c].R.values
        if len(k) < 3:
            log(f'  LIVE slip <= {c}: n {len(k)}')
            continue
        bs = [rng.choice(k, len(k)).mean() for _ in range(N_BOOT)]
        log(f'  LIVE slip <= {c}: n {len(k)} net {k.mean():+.3f} CI [{np.percentile(bs, 2.5):+.3f}, '
            f'{np.percentile(bs, 97.5):+.3f}]  vs live mean {L.R.mean():+.3f}')


def features(x: pd.DataFrame) -> None:
    """Report-only, for the next hypotheses: net R by quintile of d (continuous, incl. d>15 on B1), trigger-bar
    volume vs the 10-bar consolidation mean, minutes since the level formed, per split."""
    x = x.assign(vratio=x.v_T / x.v_cons10.replace(0, np.nan))
    for col in ('d_raw_bps', 'vratio', 'mins_since_level'):
        for s in ('TRAIN-H2', 'VAL'):
            xs = x[(x.split == s) & x[col].notna()]
            q = pd.qcut(xs[col].rank(method='first'), 5, labels=False)
            log(f'  F {col} {s}: ' + '; '.join(
                f'Q{i + 1} [{v[col].min():.3g},{v[col].max():.3g}] n {len(v)} net {v.net_R.mean():+.3f}'
                for i, v in xs.groupby(q)))


def main():
    """Build/resume the cache, then score L1-L3, live OOS and the report-only feature reads."""
    rng = np.random.default_rng(1730)
    log(f'==== cell 1,730 run {time.strftime("%Y-%m-%d %H:%M:%S")}')
    A = load_pop_a()
    K = load_pop_b_keys()
    L = load_live()
    X = build_cache(A, K)
    c = pd.read_csv(os.path.join(ROOT, 'research/hod_entry/causal_arming_causal.csv'), usecols=['day', 'split'])
    weeks = {('TRAIN-H2' if s == 'TRAIN' else 'VAL'): pd.to_datetime(g.day).dt.to_period('W-FRI').nunique()
             for s, g in c.groupby('split')}
    log(f'[weeks] {weeks}')
    xa = X[X['pop'] == 'A'].copy()
    log(f'[A] crossings recovered {len(xa)} of {len(A)} (dropped from every rung: {len(A) - len(xa)}); '
        f'd==0 {(xa.d_bps == 0).mean():.1%}, d>15 {(xa.d_bps > 15).mean():.1%}')
    am = xa.merge(A[['date', 'symbol', 'gross_R', 'net_R', 'entry_price']].rename(
        columns={'gross_R': 'g1663', 'net_R': 'n1663'}), on=['date', 'symbol'])
    log(f'[A] walk check vs 1663: corr gross {am.gross_R.corr(am.g1663):.3f}, mean mine {am.gross_R.mean():+.3f} '
        f'vs 1663 gross {am.g1663.mean():+.3f} (1663 is on the fill-price R basis), entry - 1663 entry median '
        f'{((am.entry / am.entry_price - 1) * 1e4).median():+.1f} bps')
    xb = X[X['pop'] == 'B'].copy()
    b1 = xb[xb.k == 0].copy()
    log(f'[B] crossings {len(xb)} on {xb[["date", "symbol"]].drop_duplicates().shape[0]} symbol-days; B1 signals '
        f'{len(b1)}; d>15 {(b1.d_bps > 15).mean():.1%}')
    rows = ladder(xa, 'A', weeks, rng) + ladder(b1, 'B1', weeks, rng)
    # B2: re-arm — per symbol-day the first crossing with d <= c fills (report-only)
    b2rows = []
    for cc in RUNGS:
        f = xb[xb.d_bps <= cc].sort_values('k').drop_duplicates(['date', 'symbol'])
        for s in ('TRAIN-H2', 'VAL'):
            k = f[f.split == s]
            b2rows.append(dict(pop='B2', split=s, cap=cc, n=len(k), fills_wk=len(k) / weeks[s], net=k.net_R.mean(),
                               gross=k.gross_R.mean(), R_wk=k.net_R.sum() / weeks[s], t=dclust_t(k.net_R, k.date),
                               ex5=ex_top5(k.net_R), capped=k.net_R.clip(upper=2).mean()))
    lad = pd.DataFrame(rows + b2rows)
    lad.to_csv(LADDER_CSV, index=False)
    pd.set_option('display.width', 250)
    pd.set_option('display.max_columns', 30)
    log('L1 ladder:\n' + lad.round(3).to_string(index=False))
    log('L2 (population A, c = 15):')
    bucket_table(xa, L)
    log('L2 (population B1, c = 15):')
    bucket_table(b1, L)
    log('L3 (population A):')
    declined(xa)
    log('L3 (population B1):')
    declined(b1)
    log('LIVE OOS:')
    live_oos(L, rng)
    log(f'[live] weeks {pd.to_datetime(L.date).dt.to_period("W-FRI").nunique()}, fills {len(L)}')
    log('Features (population B1, report-only):')
    features(b1)
    log('==== done')


if __name__ == '__main__':
    main()
