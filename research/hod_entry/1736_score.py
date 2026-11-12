"""Cell 1,736 — short the FAILED TOUCH of the HOD (PREREG_1736.md). End-to-end scorer.

Reads replay_signals.csv arms, walks SIP minute bars (bars_sip.db, read-only, per (symbol, day) via
walker.fetch_day_bars), builds the failed-touch short event, scores E1 (2R target) and E2 (VWAP reclaim),
and writes 1736_events.csv / 1736_summary.csv. TEST split is never touched (only TRAIN/VAL rows exist).
Conventions where the PREREG is silent are stated in CONVENTIONS below and in RESULT_1736.md.
"""
import os, sys, sqlite3, math
import numpy as np
import pandas as pd

sys.path.insert(0, '/home/ec2-user/onemil/research/hod_exit_lab')
import walker  # noqa: E402

ROOT = '/home/ec2-user/onemil'
OUT = f'{ROOT}/research/hod_entry'
CONVENTIONS = """
- bar minute m = bar START minute (ET). Touch/failure scan uses bars with m >= entry_m+1.
- T..T+3 = bars with minute m in [T.m, T.m+3].  Entry bar = the row immediately after F (must exist).
- Short entry = open(F+1) - half_spread - 2bp*open  (this is the R base: R = stop - entry).
- Stop = max(h over T..F)*1.002.  Stop leg: price = bar open if open>=stop (gap-through) else stop;
  cost on that leg = half_spread + 13bp*price (13bp replaces the 2bp).  Target/EOD/reclaim legs: price + half_spread + 2bp*price.
- Bar F+1 itself can hit stop/target (entry is its open). Stop is checked before target in the same bar.
- E2: no target; after the close of any bar k >= F+1 with close >= session VWAP (and no stop that bar) cover at open(k+1).
- EOD = open of the 15:55 bar (m=955); if the path has no 955 bar the last bar's close is used (logged).
- half_spread $ from signals.parquet (day,symbol); unmatched -> median half_spread of matched arms in the same decile of level.
"""
SLIP, STOP_SLIP, EOD_M = 0.0002, 0.0013, 955
FAIL_FRAC, SUCCESS_FRAC, STOP_MULT, MIN_R_PCT, TARGET_R = 0.002, 0.006, 1.002, 0.01, 2.0
N_PLACEBO = 300
STRICT = os.environ.get('H1736_STRICT') == '1'   # diagnostic: touch = first bar with h > level (a real break), not h >= level
SUF = '_strict' if STRICT else ''


def log(msg):
    """Verbose progress line."""
    walker.log(msg)


def prior_close(cc, symbol, day):
    """Last daily close with bar_date < day, or NaN (fail closed)."""
    r = cc.execute('SELECT close FROM daily_bars WHERE symbol=? AND bar_date<? ORDER BY bar_date DESC LIMIT 1', (symbol, day)).fetchone()
    return float(r[0]) if r else float('nan')


def find_event(d, entry_m, level):
    """Return dict(touch_i, fail_i, status) for bars df d (columns m,o,h,l,c). status in ok/no_touch/no_failure/success."""
    pos = np.where(d.m.values >= entry_m + 1)[0]
    if len(pos) == 0:
        return dict(status='no_touch')
    hi = d.h.values
    ti = next((i for i in pos if (hi[i] > level if STRICT else hi[i] >= level)), None)
    if ti is None:
        return dict(status='no_touch')
    tm = d.m.values[ti]
    cl, mm = d.c.values, d.m.values
    i = ti
    while i < len(d) and mm[i] <= tm + 3:
        if cl[i] >= level * (1 + SUCCESS_FRAC):
            return dict(status='success', touch_i=ti)
        if cl[i] <= level * (1 - FAIL_FRAC):
            return dict(status='ok', touch_i=ti, fail_i=i)
        i += 1
    return dict(status='no_failure', touch_i=ti)


def short_exits(d, start_i, entry, stop, hs, r_abs, vw):
    """Walk a short opened at bar start_i's open. Returns dict with E1 / E2 (exit_px, why, net_R, exit_i).
    entry is the post-cost entry price; r_abs = stop-entry. d columns m,o,h,l,c; vw session VWAP aligned."""
    o, h, l, c, m = d.o.values, d.h.values, d.l.values, d.c.values, d.m.values
    target = entry - TARGET_R * r_abs
    n = len(d)

    def leg(px, stop_leg):
        return px + hs + (STOP_SLIP if stop_leg else SLIP) * px

    def net(px, stop_leg):
        return (entry - leg(px, stop_leg)) / r_abs

    res = {}
    # E1
    done = False
    for k in range(start_i, n):
        if m[k] >= EOD_M:
            res['e1'] = (o[k], 'eod', net(o[k], False), k); done = True; break
        if h[k] >= stop:
            px = o[k] if o[k] >= stop else stop
            res['e1'] = (px, 'stop', net(px, True), k); done = True; break
        if l[k] <= target:
            res['e1'] = (target, 'target', net(target, False), k); done = True; break
    if not done:
        res['e1'] = (c[n - 1], 'eod_lastbar', net(c[n - 1], False), n - 1)
    # E2
    done = False
    for k in range(start_i, n):
        if m[k] >= EOD_M:
            res['e2'] = (o[k], 'eod', net(o[k], False), k); done = True; break
        if h[k] >= stop:
            px = o[k] if o[k] >= stop else stop
            res['e2'] = (px, 'stop', net(px, True), k); done = True; break
        if c[k] >= vw[k]:
            if k + 1 < n:
                kk = k + 1
                if m[kk] >= EOD_M:
                    res['e2'] = (o[kk], 'eod', net(o[kk], False), kk)
                else:
                    res['e2'] = (o[kk], 'reclaim', net(o[kk], False), kk)
            else:
                res['e2'] = (c[k], 'eod_lastbar', net(c[k], False), k)
            done = True; break
    if not done:
        res['e2'] = (c[n - 1], 'eod_lastbar', net(c[n - 1], False), n - 1)
    return res


def placebo_pool(d, vw_unused, after_i, r_pct, hs):
    """For one trade: E1 net R of the same geometry (R as pct of entry, 2R target, same half-spread) started at the
    open of every bar index > after_i (the entry bar) with minute < 900. Returns array of net R (E1)."""
    out = []
    n = len(d)
    for s in range(after_i + 1, n):
        if d.m.values[s] >= 900:
            break
        px = d.o.values[s]
        entry = px - hs - SLIP * px
        r_abs = r_pct * entry
        stop = entry + r_abs
        sub = d.iloc[s:]
        e = short_exits(sub.reset_index(drop=True), 0, entry, stop, hs, r_abs, np.zeros(len(sub)))
        out.append(e['e1'][2])
    return np.array(out)


def stats(df, col, nweeks):
    """Per-spec statistics for a trade frame df with net R in col."""
    n = len(df)
    if n == 0:
        return dict(n=0)
    r = df[col].values
    mean = r.mean()
    iid_t = mean / (r.std(ddof=1) / math.sqrt(n)) if n > 2 and r.std(ddof=1) > 0 else float('nan')
    g = df.assign(_r=r - mean).groupby('day')._r.sum()
    se = math.sqrt((g ** 2).sum()) / n
    day_t = mean / se if se > 0 else float('nan')
    k = int(math.ceil(0.05 * n))
    ex5 = np.sort(r)[:-k].mean() if n > k else float('nan')
    cap = np.minimum(r, 3.0).mean()
    why = df[col.replace('net_R', 'why')]
    wk = df.assign(_w=pd.to_datetime(df.day).dt.strftime('%G-%V'), _r2=r).groupby('_w')._r2.sum()
    green = (wk > 0).sum() / nweeks
    wk_full = np.concatenate([wk.values, np.zeros(max(nweeks - len(wk), 0))])
    ordr = df.assign(_r2=r).sort_values(['day', 'entry_m'])._r2.cumsum().values
    dd = float((np.maximum.accumulate(np.concatenate([[0], ordr]))[1:] - ordr).max())
    return dict(n=n, mean_R=mean, day_t=day_t, iid_t=iid_t, ex_top5=ex5, capped3=cap, stop_share=(why == 'stop').mean(),
                green_wk_share=green, wk_p10=np.percentile(wk_full, 10), max_dd_R=dd,
                mean_R_pct_price=(r * df.r_pct.values).mean(), trades_per_wk=n / nweeks)


def main():
    """Build events, score, placebo, write outputs."""
    walker.wait_for_db_window()
    arms = pd.read_csv(f'{OUT}/replay_signals.csv', dtype={'symbol': str, 'day': str}, keep_default_na=False, na_values=[''])
    n_all = len(arms)
    arms = arms[(arms.void_reason != 'no_level') & arms.level.notna()].reset_index(drop=True)
    log(f'arms {n_all} -> {len(arms)} with a level; splits {arms.split.value_counts().to_dict()}')
    sig = pd.read_parquet(f'{ROOT}/research/hod_exit_lab/signals.parquet')[['day', 'symbol', 'half_spread']].drop_duplicates(['day', 'symbol'])
    arms = arms.merge(sig, on=['day', 'symbol'], how='left')
    arms['matched_spread'] = arms.half_spread.notna()
    arms['dec'] = pd.qcut(arms.level, 10, labels=False, duplicates='drop')
    med = arms[arms.matched_spread].groupby('dec').half_spread.median()
    arms.loc[~arms.matched_spread, 'half_spread'] = arms.loc[~arms.matched_spread, 'dec'].map(med)
    log(f'spread unmatched share {1 - arms.matched_spread.mean():.3f}')
    sh = pd.read_csv(f'{OUT}/1736_shortable.csv')
    shd = dict(zip(sh.symbol, sh.shortable)); etb = dict(zip(sh.symbol, sh.easy_to_borrow))
    b0 = pd.read_csv(f'{ROOT}/research/hod_exit_lab/b0_trades.csv').drop_duplicates(['day', 'symbol'])[['day', 'symbol', 'net_R']].rename(columns={'net_R': 'b0_net_R'})

    con = sqlite3.connect('file:' + ROOT + '/research/bf_zero/bars_sip.db?mode=ro', uri=True)
    cc = sqlite3.connect('file:' + ROOT + '/data/cache.db?mode=ro', uri=True)
    rng = np.random.default_rng(1736)
    verif_idx = set(rng.choice(len(arms), 20, replace=False).tolist())
    verif = []
    rows, pool = [], {}
    nbars_missing = 0
    for ix, a in enumerate(arms.itertuples()):
        d = walker.fetch_day_bars(con, a.symbol, a.day)
        rec = dict(day=a.day, symbol=a.symbol, entry_m=a.entry_m, level=a.level, split=a.split, event=0, rails='', e1_fill=a.e1_fill,
                   shortable=bool(shd.get(a.symbol, False)), etb=bool(etb.get(a.symbol, False)), half_spread=a.half_spread,
                   matched_spread=a.matched_spread)
        if d is None:
            nbars_missing += 1; rec['rails'] = 'no_bars'; rows.append(rec); continue
        if ix in verif_idx:
            j = d[d.m == a.entry_m]
            after = d[d.m > a.entry_m]
            verif.append((a.day, a.symbol, bool(len(j) and j.h.iloc[0] < a.level), bool(len(after) and after.h.max() >= a.level)))
        ev = find_event(d, a.entry_m, a.level)
        rec['status'] = ev['status']
        if 'touch_i' in ev:
            rec['touch_m'] = int(d.m.iloc[ev['touch_i']])
        if ev['status'] != 'ok':
            rows.append(rec); continue
        ti, fi = ev['touch_i'], ev['fail_i']
        rec['fail_m'] = int(d.m.iloc[fi]); rec['event'] = 1
        pc = prior_close(cc, a.symbol, a.day)
        if not (pc == pc):
            rec['rails'] = 'no_prior_close'
        elif d.c.iloc[fi] < 0.90 * pc:
            rec['rails'] = 'ssr'
        elif d.m.iloc[fi] > 930:
            rec['rails'] = 'late'
        elif fi + 1 >= len(d):
            rec['rails'] = 'no_entry_bar'
        else:
            hs = a.half_spread
            px = d.o.iloc[fi + 1]
            entry = px - hs - SLIP * px
            stop = d.h.iloc[ti:fi + 1].max() * STOP_MULT
            r_abs = stop - entry
            rec.update(entry=entry, stop=stop, R=r_abs, r_pct=r_abs / entry)
            if r_abs < MIN_R_PCT * entry:
                rec['rails'] = 'r_too_small'
            else:
                rec['rails'] = 'ok'
                vw = walker.session_vwap(d)
                e = short_exits(d, fi + 1, entry, stop, hs, r_abs, vw)
                for k in ('e1', 'e2'):
                    px_, why, nr, xi = e[k]
                    rec[f'{k}_exit'] = px_; rec[f'{k}_why'] = why; rec[f'{k}_net_R'] = nr; rec[f'{k}_exit_m'] = int(d.m.iloc[xi])
                pool[len(rows)] = placebo_pool(d, None, fi + 1, rec['r_pct'], hs)
        rows.append(rec)
        if ix % 500 == 0:
            log(f'{ix}/{len(arms)} arms, events so far {sum(r["event"] for r in rows)}')
    ev = pd.DataFrame(rows).merge(b0, on=['day', 'symbol'], how='left')
    ev.to_csv(f'{OUT}/1736_events{SUF}.csv', index=False)
    log(f'arm-bar verification (20 random): bar j high<level {sum(v[2] for v in verif)}/{len(verif)}; later bar high>=level {sum(v[3] for v in verif)}/{len(verif)}; missing-bars arms {nbars_missing}')

    trades = ev[ev.rails == 'ok'].copy()
    # placebo pools aligned by row index -> trade
    trades['pool_key'] = trades.index
    nweeks = {s: pd.to_datetime(arms[arms.split == s].day).dt.strftime('%G-%V').nunique() for s in ('TRAIN', 'VAL')}
    univ = {'all': trades.index == trades.index, 'shortable_now': trades.shortable.values, 'etb_now': trades.etb.values}
    out = []
    prng = np.random.default_rng(17360)
    for u, mask in univ.items():
        for s in ('TRAIN', 'VAL'):
            sub = trades[mask & (trades.split == s).values]
            narms = int(((arms.split == s) & (arms.symbol.map(lambda x: True if u == 'all' else bool((shd if u == 'shortable_now' else etb).get(x, False))))).sum())
            evs = ev[(ev.split == s) & (ev.event == 1) & (True if u == 'all' else ev.symbol.map(lambda x: bool((shd if u == 'shortable_now' else etb).get(x, False))))]
            tch = ev[(ev.split == s) & ev.touch_m.notna() & (True if u == 'all' else ev.symbol.map(lambda x: bool((shd if u == 'shortable_now' else etb).get(x, False))))]
            for ex in ('e1', 'e2'):
                st = stats(sub, f'{ex}_net_R', nweeks[s]) if len(sub) else dict(n=0)
                st.update(universe=u, split=s, exit=ex, n_arms=narms, n_touch=len(tch), n_events=len(evs),
                          event_rate=len(evs) / max(narms, 1), n_trades=len(sub), events_per_wk=len(evs) / nweeks[s])
                if ex == 'e1' and len(sub) > 5:
                    keys = [k for k in sub.index if k in pool and len(pool[k])]
                    obs = sub.e1_net_R.mean()
                    draws = np.array([np.mean([prng.choice(pool[k]) for k in keys]) for _ in range(N_PLACEBO)])
                    st['placebo_pctile'] = (draws < obs).mean() * 100
                    st['placebo_mean'] = draws.mean()
                if len(sub):
                    m_ = sub.b0_net_R.dropna()
                    st['b0_cf_mean_R'] = m_.mean(); st['b0_cf_n'] = len(m_)
                out.append(st)
                log(f'{u:14s} {s:5s} {ex}: n={st.get("n")} mean={st.get("mean_R", float("nan")):.3f} day_t={st.get("day_t", float("nan")):.2f} pct={st.get("placebo_pctile", float("nan"))}')
    summ = pd.DataFrame(out)
    summ.to_csv(f'{OUT}/1736_summary{SUF}.csv', index=False)
    log('rails: ' + str(ev[ev.event == 1].rails.value_counts().to_dict()))
    log('r_too_small / spread unmatched share among trades: ' + str(1 - trades.matched_spread.mean()))
    log('events per month: ' + str(ev[ev.event == 1].assign(mo=lambda x: x.day.str[:7]).groupby('mo').size().to_dict()))
    log('DONE')


if __name__ == '__main__':
    main()
