"""Scorer for cell 1,439 (low-of-day mirror SHORT) — research/hod_entry/PREREG_WEEKEND.md
"## 1,439" section + the 2026-09-25 20:10 UTC amendment (Reg SHO SSR obtainability rail).

Run ONCE on the full `cell_1439_fills.csv` (research/hod_entry/cell_1439.py --workers N).
Computes, per holdout (TRAIN = TRAIN-H2, VAL):
  1. Coverage (1 - share of crossed symbol-days that are 'no_tape') and the missingness gap
     (|no_tape share among proxy winners b0_net_R>0 - proxy losers| in pp).
  2. Per-fill SSR flag: SSR_today (a bar low <= 0.9 x prior close at or before fill_min, using
     minute bars from causal_arming.load_day_bars) OR SSR_carried (previous trading day's low
     <= 0.9 x the close of the day before it). Prior close from data/cache.db::daily_bars
     (read-only). Missing prior close -> SSR unknown, treated as SSR=True (conservative) in the
     primary book; the unknown count is reported.
  3. Shortable flag from cell_1439.load_borrow_flags(); unknown symbol = not shortable.
  4. PRIMARY book = fills with SSR false AND shortable true. SECONDARY = all fills (report-only).
     Per book: n, fill rate, mean net_R, mean net_R_stopslip, day-clustered t (statsmodels OLS,
     cluster-robust SE by day), ex-top-5% net_R, fills/week (slot-simulated at 4-concurrent /
     12-per-day via research/hod_consol/run_consol.simulate_slots, entry_m=fill_min; falls back
     to raw n/distinct-weeks if the slot sim cannot be applied).
  5. Verdict per book vs the pre-registered bar (PRIMARY only): mean net R >= +0.10 both holdouts,
     VAL t >= 2, ex-top-5% > 0, coverage >= 80%, gap <= 5pp, >= 3 fills/week, >= 60% shortable.
     VOID overrides PASS/FAIL if coverage < 80% or gap > 5pp on either holdout.

Usage: nice -n 19 python3 research/hod_entry/score_1439.py [--fills-csv PATH]
"""
import argparse
import os
import sqlite3
import sys

import numpy as np
import pandas as pd
import statsmodels.api as sm

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'research/hod_entry'))

import cell_1439 as c1439  # noqa: E402
import causal_arming as ca  # noqa: E402

CACHE_DB_URI = 'file:' + os.path.join(ROOT, 'data/cache.db') + '?mode=ro'
FILLS_CSV = os.path.join(ROOT, 'research/hod_entry/cell_1439_fills.csv')

PASS_BAR = dict(mean_net_r=0.10, val_t=2.0, ex_top5_gt0=True, coverage=0.80, gap_pp=5.0,
                fills_per_week=3.0, shortable_share=0.60)


def log(msg):
    print(f'[score_1439] {msg}', flush=True)


# --------------------------------------------------------------------------------------- coverage
def coverage_and_gap(df_split):
    """(coverage, gap_pp) for one holdout's crossed symbol-days (status in fill/nofill/no_tape)."""
    crossed = df_split[df_split.status.isin(['fill', 'nofill', 'no_tape'])]
    if not len(crossed):
        return float('nan'), float('nan')
    coverage = 1.0 - (crossed.status == 'no_tape').mean()
    winners = crossed[crossed.b0_net_R > 0]
    losers = crossed[crossed.b0_net_R <= 0]
    w_share = (winners.status == 'no_tape').mean() if len(winners) else float('nan')
    l_share = (losers.status == 'no_tape').mean() if len(losers) else float('nan')
    gap_pp = abs(w_share - l_share) * 100 if len(winners) and len(losers) else float('nan')
    return coverage, gap_pp


# -------------------------------------------------------------------------------------------- SSR
def ssr_today_hit(bars, fill_min, prior_close):
    """True iff any bar with m <= fill_min has low <= 0.9 x prior_close (fill bar included)."""
    if bars is None or not len(bars) or prior_close is None:
        return False
    early = bars[bars.m <= fill_min]
    return bool(len(early) and (early.l <= 0.9 * prior_close).any())


def ssr_carried_hit(prev_bars, close_prev2):
    """True iff the previous trading day's bars have a low <= 0.9 x the close two days back."""
    if prev_bars is None or not len(prev_bars) or close_prev2 is None:
        return False
    return bool((prev_bars.l <= 0.9 * close_prev2).any())


def prior_close_map(con, symbols, days):
    """{(symbol, bar_date): close} from daily_bars for the given symbols over [min(days)-5, max(days)]."""
    syms = sorted(set(symbols))
    if not syms:
        return {}
    q = (f"select symbol, bar_date, close from daily_bars where symbol in "
         f"({','.join('?' * len(syms))}) and bar_date <= ?")
    d = pd.read_sql(q, con, params=syms + [max(days)])
    return {(r.symbol, r.bar_date): r.close for r in d.itertuples()}


def ssr_flags(fills, con, sipcon):
    """Returns fills with an added 'ssr' (bool) and 'ssr_unknown' (bool) column.

    SSR_today: any minute bar with m <= fill_min has low <= 0.9 x prior close (fill bar included).
    SSR_carried: the previous trading day's low <= 0.9 x the close of the day before it.
    Unknown prior close -> ssr_unknown=True, ssr=True (conservative).
    """
    days_all = sorted(fills.day.unique())
    day_idx = {d: i for i, d in enumerate(days_all)}
    closes = prior_close_map(con, fills.symbol.unique(), days_all)
    ssr_vals, unknown_vals = [], []
    n_unknown = 0
    bars_cache = {}
    for day, g in fills.groupby('day'):
        syms = g.symbol.tolist()
        bars = ca.load_day_bars(con, day, syms, sipcon, {})
        for r in g.itertuples():
            unknown = False
            ssr = False
            # SSR_today: prior close for `day`
            prior_close = None
            # find previous calendar day present in daily_bars just before `day` for this symbol
            cand = [dt for (s, dt) in closes if s == r.symbol and dt < day]
            if cand:
                pdate = max(cand)
                prior_close = closes.get((r.symbol, pdate))
            if prior_close is None:
                unknown = True
                ssr = True
            else:
                b = bars.get(r.symbol)
                ssr = ssr_today_hit(b, r.fill_min, prior_close)
            # SSR_carried: previous trading day's low <= 0.9 x close of the day before it
            if not ssr:
                cand2 = sorted([dt for (s, dt) in closes if s == r.symbol and dt < day], reverse=True)
                if len(cand2) >= 2:
                    d_prev, d_prev2 = cand2[0], cand2[1]
                    close_prev2 = closes.get((r.symbol, d_prev2))
                    prev_bars = ca.load_day_bars(con, d_prev, [r.symbol], sipcon, {}).get(r.symbol)
                    ssr = ssr_carried_hit(prev_bars, close_prev2)
            if unknown:
                n_unknown += 1
            ssr_vals.append(ssr)
            unknown_vals.append(unknown)
    fills = fills.copy()
    fills['ssr'] = ssr_vals
    fills['ssr_unknown'] = unknown_vals
    log(f'[ssr] {n_unknown}/{len(fills)} fills with unknown prior close (flagged SSR, conservative)')
    return fills


def shortable_flags(fills):
    borrow = c1439.load_borrow_flags()
    fills = fills.copy()
    fills['shortable'] = fills.symbol.map(lambda s: bool(borrow.loc[s, 'shortable']) if s in borrow.index else False)
    return fills


# ------------------------------------------------------------------------------------------ stats
def day_clustered_t(y, day):
    y = np.asarray(y, dtype=float)
    n_days = pd.Series(day).nunique()
    if len(y) < 2 or n_days < 2:
        return float('nan'), n_days
    X = np.ones((len(y), 1))
    model = sm.OLS(y, X).fit(cov_type='cluster', cov_kwds={'groups': np.asarray(day)})
    return float(model.tvalues[0]), n_days


def ex_top5(x):
    x = pd.Series(x).sort_values()
    if not len(x):
        return float('nan')
    cut = max(1, int(round(0.05 * len(x)))) if len(x) >= 20 else max(1, int(round(0.05 * len(x))))
    return float(x.iloc[:len(x) - cut].mean()) if cut < len(x) else float(x.mean())


def fills_per_week(g):
    """Slot-simulated fills/week (4-concurrent, 12/day) via research/hod_consol/run_consol.simulate_slots;
    falls back to raw n / distinct weeks with a note if the slot frame cannot be built."""
    try:
        sys.path.insert(0, os.path.join(ROOT, 'research/hod_consol'))
        import run_consol as rc  # noqa
        trades = g.rename(columns={'fill_min': 'entry_m'})[['day', 'entry_m', 'exit_m']].dropna()
        if not len(trades):
            raise ValueError('no rows with entry_m/exit_m')
        keep = rc.simulate_slots(trades)
        n_kept = int(keep.sum())
        n_weeks = g.wk.nunique()
        return (n_kept / n_weeks if n_weeks else float('nan')), False
    except Exception as exc:
        log(f'[slots] WARNING falling back to raw fills/week: {exc}')
        n_weeks = g.wk.nunique()
        return (len(g) / n_weeks if n_weeks else float('nan')), True


def score_book(fills_split, crossed_n):
    n = len(fills_split)
    out = dict(n=n, fill_rate=n / crossed_n if crossed_n else float('nan'))
    out['mean_net_r'] = float(fills_split.net_R.mean()) if n else float('nan')
    out['mean_net_r_stopslip'] = float(fills_split.net_R_stopslip.mean()) if n else float('nan')
    t, ndays = day_clustered_t(fills_split.net_R, fills_split.day) if n else (float('nan'), 0)
    out['t'] = t
    out['ex_top5'] = ex_top5(fills_split.net_R) if n else float('nan')
    fpw, fallback = fills_per_week(fills_split) if n else (float('nan'), True)
    out['fills_per_week'] = fpw
    out['fpw_fallback'] = fallback
    out['ssr_share'] = float(fills_split.ssr.mean()) if n else float('nan')
    out['shortable_share'] = float(fills_split.shortable.mean()) if n else float('nan')
    return out


def verdict(primary_train, primary_val, cov_train, gap_train, cov_val, gap_val):
    void = (cov_train < PASS_BAR['coverage'] or gap_train > PASS_BAR['gap_pp'] or
            cov_val < PASS_BAR['coverage'] or gap_val > PASS_BAR['gap_pp'])
    if void:
        return 'VOID'
    checks = [
        primary_train['mean_net_r'] >= PASS_BAR['mean_net_r'],
        primary_val['mean_net_r'] >= PASS_BAR['mean_net_r'],
        primary_val['t'] >= PASS_BAR['val_t'],
        primary_train['ex_top5'] > 0,
        primary_val['ex_top5'] > 0,
        primary_train['fills_per_week'] >= PASS_BAR['fills_per_week'],
        primary_val['fills_per_week'] >= PASS_BAR['fills_per_week'],
        primary_train['shortable_share'] >= PASS_BAR['shortable_share'],
        primary_val['shortable_share'] >= PASS_BAR['shortable_share'],
    ]
    return 'PASS' if all(checks) else 'FAIL'


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--fills-csv', default=FILLS_CSV)
    a = ap.parse_args(argv)
    log(f'reading {a.fills_csv}')
    df = pd.read_csv(a.fills_csv, dtype={'symbol': str, 'day': str})
    log(f'{len(df)} symbol-day rows, splits={df.split.value_counts().to_dict()}')

    con = sqlite3.connect(CACHE_DB_URI, uri=True, timeout=120)
    sipcon = sqlite3.connect(ca.BARS_SIP_URI, uri=True, timeout=120)

    results = {}
    for split_name, split_key in (('TRAIN', 'TRAIN'), ('VAL', 'VAL')):
        sub = df[df.split == split_key]
        cov, gap = coverage_and_gap(sub)
        fills = sub[sub.status == 'fill'].copy()
        crossed_n = sub.status.isin(['fill', 'nofill', 'no_tape']).sum()
        log(f'[{split_name}] coverage={cov:.3f} gap={gap:.2f}pp fills={len(fills)} crossed={crossed_n}')
        if len(fills):
            fills = ssr_flags(fills, con, sipcon)
            fills = shortable_flags(fills)
        else:
            fills['ssr'] = []
            fills['shortable'] = []
        primary = fills[(~fills.ssr) & (fills.shortable)] if len(fills) else fills
        secondary = fills
        results[split_name] = dict(
            coverage=cov, gap=gap, crossed_n=crossed_n,
            primary=score_book(primary, crossed_n),
            secondary=score_book(secondary, crossed_n),
        )
    con.close()
    sipcon.close()

    v = verdict(results['TRAIN']['primary'], results['VAL']['primary'],
                results['TRAIN']['coverage'], results['TRAIN']['gap'],
                results['VAL']['coverage'], results['VAL']['gap'])
    log(f'VERDICT (primary book, pre-registered bar): {v}')
    for split_name in ('TRAIN', 'VAL'):
        r = results[split_name]
        for book in ('primary', 'secondary'):
            b = r[book]
            log(f'[{split_name}/{book}] n={b["n"]} fill_rate={b["fill_rate"]:.3f} '
                f'mean_net_r={b["mean_net_r"]:.3f} stopslip={b["mean_net_r_stopslip"]:.3f} t={b["t"]:.2f} '
                f'ex_top5={b["ex_top5"]:.3f} fills_wk={b["fills_per_week"]:.2f}'
                f'{"(raw)" if b["fpw_fallback"] else ""} ssr_share={b["ssr_share"]:.3f} '
                f'shortable_share={b["shortable_share"]:.3f}')
        log(f'[{split_name}] coverage={r["coverage"]:.3f} gap={r["gap"]:.2f}pp')
    return results, v


if __name__ == '__main__':
    main()
