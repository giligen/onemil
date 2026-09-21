#!/usr/bin/env python3
"""PREREG_RANK — attention-rank filter for HOD-break, cells 1,351-1,352 (owner 2026-09-21).

Adds two causal features to the existing causal-filter machinery and scores them with the SAME
scoring functions (`cells.load`, `cells.book`, `cells.stats`) and the SAME TRAIN/VAL split as
`CAUSAL_FILTER_PREREG.md`'s 12-cell study. No new population, no new split, no other feature.

  rank_cand — among candidates_full.csv rows with the same day & entry_m <= the signal's entry_m,
              the signal's rank by dist_open_pct descending (1 = strongest mover so far). Computed
              with an offline dominance count: candidates inserted into a sorted list in entry_m
              order, queries answered once all candidates at entry_m <= E are inserted (ties at the
              query's own dist_open_pct do NOT count as "greater than").
  rank_mkt  — among the day's stream universe (logs/hod_stream_universe_<day>.txt where it exists,
              else every symbol with an RTH bar in bars_sip.db that day), the signal's rank by
              (asof close at/just before entry_m / session open - 1) descending. A day is VOID for
              rank_mkt if bar coverage (symbols with an RTH bar that day / distinct symbols in
              data/cache.db daily_bars that day) is < 80%.

Pass bar (PREREG_RANK.md): causal-filter G1 (TRAIN t>=2 AND TRAIN >=5 trades/week) on the measured
(NBBO) arm, PLUS kept cohort meanR > 0 on BOTH TRAIN halves (H1/H2 2025), dropped cohort meanR <= 0
on BOTH TRAIN and VAL, AND cadence C4 (VAL green share >= 55%) / C5 (VAL >= 3 fills/week) — C4/C5
read directly off cells.stats()'s own 'green'/'tpw' fields, not re-derived.

Forward check: `[HOD DRY] WOULD BUY` lines since 2026-09-18 (60s journalctl bound exceeded for
--since 2026-09-14, so the PREREG's fallback date was used), ranked by "+x% from open" among that
day's dry signals fired at/before the same minute. Small n; R is the line's own nominal book R
(no EOD spec re-simulation was run for these 9 lines) -- reported as-is per PREREG, sign only.
"""
import bisect, os, sqlite3, sys, time
import numpy as np
import pandas as pd

ROOT = '/home/ec2-user/onemil'
os.chdir(ROOT)
sys.path.insert(0, ROOT)
sys.path.insert(0, f'{ROOT}/research/bf_zero/causal_filter')
import cells as CF                                                    # noqa: E402  reuse load/book/stats

D = f'{ROOT}/research/bf_zero/causal_filter'
CAND = f'{ROOT}/research/bf_zero/candidates_full.csv'
BARS_DB = f'{ROOT}/research/bf_zero/bars_sip.db'
CACHE_DB = f'{ROOT}/data/cache.db'
LOGDIR = f'{ROOT}/logs'
FWD_LINES = '/tmp/claude-1000/-home-ec2-user-onemil/257c3e2d-cf38-45d5-94e7-4877f8170f44/scratchpad/hod_dry_lines.txt'


def log(msg):
    print(f'[{time.strftime("%H:%M:%S")}] {msg}', flush=True)


# ---------------------------------------------------------------- rank_cand
def compute_rank_cand(feat):
    cand = pd.read_csv(CAND, usecols=['day', 'symbol', 'entry_m', 'dist_open_pct'],
                        dtype={'day': str, 'symbol': str})
    cand = cand.sort_values(['day', 'entry_m'])
    sig_by_day = {d: g for d, g in feat.groupby('day')}
    ranks = {}
    for day, g in cand.groupby('day', sort=False):
        sig = sig_by_day.get(day)
        if sig is None:
            continue
        q_by_em = {}
        for r in sig.itertuples():
            q_by_em.setdefault(int(r.entry_m), []).append(r)
        ems = g.entry_m.values
        dists = g.dist_open_pct.values
        i, n = 0, len(g)
        sorted_list = []
        for em in sorted(q_by_em):
            while i < n and ems[i] <= em:
                bisect.insort(sorted_list, dists[i])
                i += 1
            for r in q_by_em[em]:
                N = len(sorted_list)
                gt = N - bisect.bisect_right(sorted_list, r.dist_open_pct)
                ranks[(day, r.symbol, em)] = gt + 1
    log(f'rank_cand: {len(ranks)} signals ranked')
    return ranks


# ----------------------------------------------------------------- rank_mkt
def compute_rank_mkt(feat):
    con_bars = sqlite3.connect(f'file:{BARS_DB}?mode=ro', uri=True, timeout=180)
    con_cache = sqlite3.connect(f'file:{CACHE_DB}?mode=ro', uri=True, timeout=180)
    ranks, cov_rows = {}, []
    days = sorted(feat.day.unique())
    for n_i, day in enumerate(days):
        g = feat[feat.day == day]
        bars = pd.read_sql('select symbol, t, o, c from bars where day=?', con_bars, params=(day,))
        ref = pd.read_sql('select distinct symbol from daily_bars where bar_date=?', con_cache, params=(day,))
        ref_syms = set(ref.symbol.astype(str))
        if not len(bars) or not ref_syms:
            cov_rows.append((day, 0.0, len(g)))
            continue
        ts = pd.to_datetime(bars.t, utc=True).dt.tz_convert('America/New_York')
        bars = bars.assign(m=(ts.dt.hour * 60 + ts.dt.minute).values)
        bars = bars[(bars.m >= 570) & (bars.m < 960)].sort_values(['symbol', 'm'])
        sf = f'{LOGDIR}/hod_stream_universe_{day}.txt'
        if os.path.exists(sf):
            uni = set(l.strip() for l in open(sf) if l.strip())
            bars = bars[bars.symbol.isin(uni)]
        cov = len(set(bars.symbol.unique()) & ref_syms) / len(ref_syms)
        cov_rows.append((day, cov, len(g)))
        if cov < 0.8:
            continue
        opens = bars.groupby('symbol').o.first()
        for em, gg in g.groupby('entry_m'):
            asof = bars[bars.m <= em].groupby('symbol').c.last()
            ret = (asof / opens.reindex(asof.index) - 1).dropna()
            if not len(ret):
                continue
            order = ret.rank(ascending=False, method='min')
            for r in gg.itertuples():
                if r.symbol in order.index:
                    ranks[(day, r.symbol, int(em))] = int(order.loc[r.symbol])
        if n_i % 50 == 0:
            log(f'rank_mkt: {n_i + 1}/{len(days)} days')
    con_bars.close()
    con_cache.close()
    cov_df = pd.DataFrame(cov_rows, columns=['day', 'coverage', 'n_signals'])
    log(f'rank_mkt: {len(ranks)} signals ranked, {len(cov_df)} days scanned')
    return ranks, cov_df


# ------------------------------------------------------------- cell scoring
def score_cell(c, kept_mask, dropped_mask, name, rule):
    """Kept/dropped halves of one cell, both arms, both splits -- via cells.book/cells.stats.
    Both masks are boolean with NaN-feature rows excluded from BOTH sides (VOID rank_mkt days)."""
    rows = []
    for tag, m in (('kept', kept_mask), ('dropped', dropped_mask)):
        d = c[m.fillna(False)]
        for arm, col in (('band', 'net_band'), ('meas', 'net_meas')):
            dd = d[d.obtainable == True] if arm == 'meas' else d          # noqa: E712  NO-FILL rail
            row = {'cell': name, 'rule': rule, 'side': tag, 'arm': arm}
            for sp in ('TRAIN', 'VAL'):
                st = CF.stats(CF.book(dd[dd.split == sp], col), sp)
                for k, v in (st or {}).items():
                    row[f'{sp}_{k}'] = v
            rows.append(row)
    return rows


def half_check(c, kept_mask, dropped_mask, col='net_meas'):
    """TRAIN H1/H2 mean R for kept vs dropped (meas arm, obtainable rows only)."""
    tr = c[(c.split == 'TRAIN') & (c.obtainable == True)]              # noqa: E712
    out = {}
    for tag, m in (('kept', kept_mask), ('dropped', dropped_mask)):
        d = tr[m.reindex(tr.index).fillna(False)]
        for h in ('H1', 'H2'):
            dd = d[d.half == h]
            out[f'{tag}_{h}'] = round(float(dd[col].mean()), 3) if len(dd) else np.nan
            out[f'{tag}_{h}_n'] = len(dd)
    return out


def decile_table(c, feat_col):
    tr = c[(c.split == 'TRAIN') & c[feat_col].notna() & (c.obtainable == True)]  # noqa: E712
    if not len(tr):
        return pd.DataFrame()
    dec = pd.qcut(tr[feat_col].rank(method='first'), 10, labels=[f'D{i+1}' for i in range(10)])
    g = tr.groupby(dec, observed=True).net_meas.agg(['mean', 'count'])
    return g.round(3)


def forward_check():
    if not os.path.exists(FWD_LINES):
        return 'no forward-check lines captured', None
    import re
    rx = re.compile(r'^(\w+ \d+) (\d\d:\d\d:\d\d).*WOULD BUY (\S+) level.*R ([\d.]+) .*\+([\d.]+)% from open')
    rows = []
    for line in open(FWD_LINES):
        m = rx.search(line)
        if not m:
            continue
        mon_day, hhmmss, sym, rr, pct = m.groups()
        rows.append(dict(day=mon_day, t=hhmmss, symbol=sym, R=float(rr), pct=float(pct)))
    if not rows:
        return 'no WOULD BUY lines parsed', None
    f = pd.DataFrame(rows)
    f['minute'] = f.t.str.slice(0, 5)
    out = []
    for day, g in f.groupby('day'):
        g = g.sort_values('t').reset_index(drop=True)
        for i in range(len(g)):
            upto = g.iloc[:i + 1]
            rank = int((upto.pct >= g.pct.iloc[i]).sum())
            out.append({'day': day, 't': g.t.iloc[i], 'symbol': g.symbol.iloc[i],
                        'R': g.R.iloc[i], 'pct': g.pct.iloc[i], 'rank': rank})
    fr = pd.DataFrame(out)
    le10 = fr[fr['rank'] <= 10].R.mean() if (fr['rank'] <= 10).any() else np.nan
    gt10 = fr[fr['rank'] > 10].R.mean() if (fr['rank'] > 10).any() else np.nan
    return fr, dict(n=len(fr), n_le10=int((fr['rank'] <= 10).sum()), n_gt10=int((fr['rank'] > 10).sum()),
                     meanR_le10=round(float(le10), 3) if pd.notna(le10) else None,
                     meanR_gt10=round(float(gt10), 3) if pd.notna(gt10) else None)


def main():
    log('loading features via cells.load()')
    c = CF.load()
    c = c[c.split != 'TEST'].reset_index(drop=True)                     # PREREG_RANK: TRAIN/VAL only
    log(f'{len(c)} rows loaded (TRAIN+VAL)')

    log('computing rank_cand ...')
    rc = compute_rank_cand(c[['day', 'symbol', 'entry_m', 'dist_open_pct']])
    c['rank_cand'] = [rc.get((d, s, int(e))) for d, s, e in zip(c.day, c.symbol, c.entry_m)]

    log('computing rank_mkt ...')
    rm, cov_df = compute_rank_mkt(c[['day', 'symbol', 'entry_m']])
    c['rank_mkt'] = [rm.get((d, s, int(e))) for d, s, e in zip(c.day, c.symbol, c.entry_m)]

    void_days = cov_df[cov_df.coverage < 0.8]
    void_share_days = round(len(void_days) / len(cov_df), 3) if len(cov_df) else np.nan
    void_share_signals = round(void_days.n_signals.sum() / cov_df.n_signals.sum(), 3) if len(cov_df) else np.nan

    kept_1351 = (c.rank_cand <= 10) & c.rank_cand.notna()
    drop_1351 = (c.rank_cand > 10) & c.rank_cand.notna()
    kept_1352 = (c.rank_mkt <= 10) & c.rank_mkt.notna()
    drop_1352 = (c.rank_mkt > 10) & c.rank_mkt.notna()

    cell_rows = []
    cell_rows += score_cell(c, kept_1351, drop_1351, '1351 rank_cand<=10', 'rank_cand<=10 vs >10')
    cell_rows += score_cell(c, kept_1352, drop_1352, '1352 rank_mkt<=10', 'rank_mkt<=10 vs >10 (VOID excluded)')
    R = pd.DataFrame(cell_rows)
    R.to_csv(f'{D}/rank_cells.csv', index=False)

    halves = {'1351': half_check(c, kept_1351, drop_1351),
              '1352': half_check(c, kept_1352, drop_1352)}

    diag = {}
    for thr in (5, 20):
        k1 = (c.rank_cand <= thr) & c.rank_cand.notna()
        d1 = (c.rank_cand > thr) & c.rank_cand.notna()
        k2 = (c.rank_mkt <= thr) & c.rank_mkt.notna()
        d2 = (c.rank_mkt > thr) & c.rank_mkt.notna()
        diag[f'1351_le{thr}'] = score_cell(c, k1, d1, f'1351 diag <= {thr}', f'rank_cand<={thr}')
        diag[f'1352_le{thr}'] = score_cell(c, k2, d2, f'1352 diag <= {thr}', f'rank_mkt<={thr}')

    dec_cand = decile_table(c, 'rank_cand')
    dec_mkt = decile_table(c, 'rank_mkt')

    fr_result = forward_check()

    # ---------------------------------------------------------- verdicts
    def get(rows, side, arm, sp, k):
        for r in rows:
            if r['side'] == side and r['arm'] == arm:
                return r.get(f'{sp}_{k}')
        return None

    def cell_rows_for(name):
        return [r for r in cell_rows if r['cell'] == name]

    def verdict(name, halfkey):
        rows = cell_rows_for(name)
        train_t = get(rows, 'kept', 'meas', 'TRAIN', 't')
        train_tpw = get(rows, 'kept', 'meas', 'TRAIN', 'tpw')
        val_green = get(rows, 'kept', 'meas', 'VAL', 'green')
        val_tpw = get(rows, 'kept', 'meas', 'VAL', 'tpw')
        drop_train = get(rows, 'dropped', 'meas', 'TRAIN', 'meanR')
        drop_val = get(rows, 'dropped', 'meas', 'VAL', 'meanR')
        h = halves[halfkey]
        g1 = (train_t is not None and train_t >= 2 and train_tpw is not None and train_tpw >= 5)
        halves_ok = (h[f'kept_H1'] is not None and h['kept_H1'] > 0 and h['kept_H2'] is not None and h['kept_H2'] > 0)
        dropped_ok = (drop_train is not None and drop_train <= 0 and drop_val is not None and drop_val <= 0)
        c4 = val_green is not None and val_green >= 0.55
        c5 = val_tpw is not None and val_tpw >= 3
        passed = g1 and halves_ok and dropped_ok and c4 and c5
        return dict(g1=g1, halves_ok=halves_ok, dropped_ok=dropped_ok, c4=c4, c5=c5, passed=passed)

    v1351 = verdict('1351 rank_cand<=10', '1351')
    v1352 = verdict('1352 rank_mkt<=10', '1352')

    # ---------------------------------------------------------- write report
    lines = []
    lines.append('# RANK_REPORT — attention-rank filter, cells 1,351-1,352 (PREREG_RANK.md)\n')
    lines.append(f'Rows scored: {len(c)} (TRAIN {len(c[c.split=="TRAIN"])}, VAL {len(c[c.split=="VAL"])}). '
                 f'Program count: 1,352.\n')
    lines.append('## rank_mkt VOID share')
    lines.append(f'Days with bar coverage < 80% (vs data/cache.db daily_bars that day): '
                 f'{len(void_days)}/{len(cov_df)} days = {void_share_days:.1%} of days, '
                 f'{void_share_signals:.1%} of signals VOID.\n')

    def fmt_table(rows):
        cols = ['cell', 'side', 'arm'] + [f'{sp}_{k}' for sp in ('TRAIN', 'VAL')
                for k in ('n', 'tpw', 'meanR', 't', 'wkR', 'green')]
        df = pd.DataFrame(rows)
        cols = [x for x in cols if x in df.columns]
        return df[cols].to_string(index=False)

    lines.append('## Cell 1,351 -- rank_cand <= 10 vs > 10')
    lines.append('```\n' + fmt_table(cell_rows_for('1351 rank_cand<=10')) + '\n```')
    lines.append(f"TRAIN halves (meas, obtainable): {halves['1351']}\n")
    lines.append(f'Verdict 1351: {v1351}\n')

    lines.append('## Cell 1,352 -- rank_mkt <= 10 vs > 10 (VOID days excluded)')
    lines.append('```\n' + fmt_table(cell_rows_for('1352 rank_mkt<=10')) + '\n```')
    lines.append(f"TRAIN halves (meas, obtainable): {halves['1352']}\n")
    lines.append(f'Verdict 1352: {v1352}\n')

    lines.append('## Diagnostics (report-only): <=5, <=20 thresholds')
    for k, rows in diag.items():
        lines.append(f'### {k}')
        lines.append('```\n' + fmt_table(rows) + '\n```')

    lines.append('## Monotone decile table (TRAIN, meas arm, obtainable), mean net_meas by decile')
    lines.append('### rank_cand deciles (D1 = lowest rank number = strongest)')
    lines.append('```\n' + dec_cand.to_string() + '\n```')
    lines.append('### rank_mkt deciles')
    lines.append('```\n' + dec_mkt.to_string() + '\n```')

    lines.append('## Forward check')
    if isinstance(fr_result, tuple) and fr_result[1] is not None:
        fr, summ = fr_result
        lines.append(f'`[HOD DRY] WOULD BUY` lines since 2026-09-18 (60s bound exceeded for --since '
                     f'2026-09-14, PREREG fallback date used). n={summ["n"]} '
                     f'(rank<=10: {summ["n_le10"]}, rank>10: {summ["n_gt10"]}).')
        lines.append(f'Mean nominal book R: rank<=10 = {summ["meanR_le10"]}, rank>10 = {summ["meanR_gt10"]} '
                     f'(R taken from the WOULD BUY line itself; no EOD spec re-simulation run for this n).')
    else:
        lines.append(str(fr_result))

    with open(f'{D}/RANK_REPORT.md', 'w') as fh:
        fh.write('\n\n'.join(lines) + '\n')
    log('wrote RANK_REPORT.md')
    log('DONE')


if __name__ == '__main__':
    main()
