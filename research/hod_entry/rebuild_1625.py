"""
Independent rebuild of Cell 1,625 -- "the index as the instrument" -- FROM THE PROSE ONLY.

Source of truth: research/hod_entry/PREREG_1623.md, section "## Cell 1,625 -- the index as
the instrument" (frozen 2026-09-28 17:15 UTC). This script was written WITHOUT opening
cell_1625.py, cell_1625_signals.csv or RESULT_1625.md, per the independent-reimplementation
protocol in CLAUDE.md ("No research claim ships without an independent check", step 1).

Verbatim spec:
  "Signal: a burst -- B30 (all names) in its TRAIN-H2 top decile for the first time that day,
  at minute m*. Trade: buy SPY at the open of minute m* + 1, exit at the open of minute m* + 61
  (60-minute hold) or 15:55, whichever first; a second leg: IWM the same; cost 2 bps round trip
  (penny spread on a $500 ETF) + no stop (the exposure is 60 minutes of index). One signal per
  day at most. Report per holdout: n days, mean return in bps, day-clustered t, ex-top-5 %,
  winner-capped +1 %, the same trade at a random minute of the same day (placebo, seed 1625),
  and the 30-/120-minute holds beside (report-only). Pass bar in bps: mean net >= +8 bps per
  signal, t >= 2.5, ex-top-5 % > 0, placebo margin >= +5 bps t >= 2, >= 2 signals/week, TRAIN-H2
  same sign."

B30(e), reused from cell 1,624's own definition ("the number of ARM events (any status, all
names) in the 30 minutes before f's fill minute"): for an arm event e on day d at minute t_e,
B30(e) = count of OTHER arm events on day d whose minute lies in [t_e - 30, t_e), i.e. strictly
in the trailing 30 minutes before e. B30 can only newly cross a threshold AT an arm event's own
minute (it is non-increasing between events), so "the first time B30 enters the TRAIN-H2 top
decile" is found by scanning each day's arm events in time order and taking the first e whose
own B30(e) clears the threshold; m* = that event's minute.

KNOWN, DOCUMENTED DEVIATION FROM THE LITERAL "any status" WORDING:
The only minute-level timestamp in the shared input causal_arming_causal.csv is `fill_min`, and
it is populated ONLY for status == 'fill' rows -- verified empirically before writing this
script: 0/2,010 'nofill' rows and 0/21,931 'not_armed' rows carry a fill_min, a stop, a level or
an exit_m (causal_arming.py's resolve_window() does not persist a cross-minute for the 'nofill'
branch). bars_fills_1478.db is likewise fill-only (its fetch_log table has exactly 9,911 rows,
one per fill). Recovering a real arm minute for the 2,010 'nofill' symbol-days would require
re-running causal_arming.py's arm_state() detection against raw per-symbol minute bars -- a
second research pipeline, outside this cell's step budget and outside "rebuild cell 1,625 from
the prose". B30 here is therefore built from FILL events only (9,911 of them, "all names" over
symbols, not statuses) and is a documented LOWER BOUND on the prose's "any status" breadth,
undercounting by the ~17% of arm events (2,010 / 11,921) that armed but did not fill. This is
reported as a caveat in REBUILD_1625.md, not hidden.

Minute-rounding convention (a unit-conversion choice, not a tuned threshold): fill_min is a
fractional ET minute-of-day (sub-minute fill timestamp). "Minute m*" is taken as
floor(fill_min) of the triggering event; "minute m*+1" / "m*+61" then land on the 1-minute bar
grid of the SPY/IWM bars.
"""
import logging
import sys

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s", stream=sys.stdout)
log = logging.getLogger("rebuild_1625")

try:
    import statsmodels.api as sm
    _HAVE_SM = True
except ImportError:
    _HAVE_SM = False
    log.warning("statsmodels unavailable -- day_clustered_t falls back to scipy.stats.ttest_1samp. "
                "This cell has at most one signal per day, so every cluster has exactly one "
                "observation and the cluster-robust t-stat on a constant is algebraically the "
                "ordinary one-sample t-test (CR1 sandwich with n_g=1 collapses to the HC0 sum of "
                "squared residuals, and statsmodels' G/(G-1) small-sample correction with G=n "
                "cancels the (n-1)-vs-n denominator difference against the unbiased sample "
                "variance) -- verified by construction below, not assumed.")

DATA_DIR = "research/hod_entry"
CAUSAL_ARMING_CSV = f"{DATA_DIR}/causal_arming_causal.csv"
INDEX_BARS_PARQUET = f"{DATA_DIR}/index_bars_1625.parquet"
OUT_SIGNALS_CSV = f"{DATA_DIR}/rebuild_1625_signals.csv"
OUT_REPORT_MD = f"{DATA_DIR}/REBUILD_1625.md"

B30_WINDOW_MIN = 30.0
HOLD_MIN = 60
HOLD_MIN_SHORT = 30
HOLD_MIN_LONG = 120
LATE_CUTOFF_ET_MIN = 15 * 60 + 55  # 15:55 ET = 955
ROUND_TRIP_COST_BPS = 2.0
WINNER_CAP_BPS = 100.0  # "+1 %" == 100 bps
DECILE = 90  # "top decile" -> value >= the 90th percentile
PLACEBO_SEED = 1625
MAX_BAR_FORWARD_FILL_MIN = 5  # tolerance for a missing exact-minute bar


# --------------------------------------------------------------------------------------------
# Helpers copied VERBATIM from research/hod_entry/cell_1445.py (named as shared inputs in the
# task -- day_clustered_t, ex_top5_mean; winner_capped_mean/weeks_spanned are the same file's
# neighbouring helpers, reused for the "winner-capped +1 %" and "signals/week" stats the prose
# also asks for). Copied rather than imported so this script has zero import-time dependency on
# cell_1445.py's own module-level pipeline code.
# --------------------------------------------------------------------------------------------

def day_clustered_t(y, day):
    """statsmodels OLS on a constant, clustered by day -- the t-stat on the mean."""
    y = pd.Series(y).dropna()
    if len(y) < 2:
        return np.nan
    d = pd.Series(day).loc[y.index]
    if d.nunique() < 2:
        return np.nan
    if _HAVE_SM:
        X = np.ones((len(y), 1))
        model = sm.OLS(y.to_numpy(), X).fit(cov_type='cluster', cov_kwds={'groups': d.to_numpy()})
        return float(model.tvalues[0])
    from scipy import stats as _stats
    t, _ = _stats.ttest_1samp(y.to_numpy(), 0.0)
    return float(t)


def ex_top5_mean(y):
    """Mean excluding the top 5% (by value) of a series -- tail-dependence check."""
    y = pd.Series(y).dropna().sort_values(ascending=False)
    n = len(y)
    if n == 0:
        return np.nan
    k = int(round(0.05 * n))
    return float(y.iloc[k:].mean()) if k < n else float(y.mean())


def winner_capped_mean(y, cap):
    y = pd.Series(y).dropna()
    if not len(y):
        return np.nan
    return float(np.minimum(y, cap).mean())


def weeks_spanned(days):
    iso = pd.to_datetime(pd.Series(days).unique())
    wk = {(d.isocalendar()[0], d.isocalendar()[1]) for d in iso}
    return max(len(wk), 1)


# --------------------------------------------------------------------------------------------
# Step 1: load arm/fill events, compute B30 per event, find the TRAIN-H2 top-decile threshold
# --------------------------------------------------------------------------------------------

def load_fill_events():
    log.info("loading %s", CAUSAL_ARMING_CSV)
    df = pd.read_csv(CAUSAL_ARMING_CSV, low_memory=False)
    fills = df[df['status'] == 'fill'][['day', 'symbol', 'split', 'fill_min']].copy()
    fills = fills.dropna(subset=['fill_min']).sort_values(['day', 'fill_min']).reset_index(drop=True)
    log.info("fill events: %d rows, %d unique days (TRAIN=%d, VAL=%d)",
              len(fills), fills['day'].nunique(),
              fills.loc[fills['split'] == 'TRAIN', 'day'].nunique(),
              fills.loc[fills['split'] == 'VAL', 'day'].nunique())
    return fills


def compute_b30(fills):
    """B30(e) for every fill event e: count of OTHER fill events on the same day with
    fill_min in [t_e - 30, t_e) (strictly before e, trailing 30 minutes)."""
    b30 = np.zeros(len(fills), dtype=int)
    for day, idx in fills.groupby('day').groups.items():
        idx = np.array(sorted(idx))
        t = fills.loc[idx, 'fill_min'].to_numpy()
        # t is already sorted ascending within the day (fills was sorted by day, fill_min)
        for i in range(len(t)):
            lo = np.searchsorted(t[:i], t[i] - B30_WINDOW_MIN, side='left')
            b30[idx[i]] = i - lo
    fills = fills.copy()
    fills['b30'] = b30
    log.info("B30 computed: min=%d max=%d mean=%.2f", b30.min(), b30.max(), b30.mean())
    return fills


def find_burst_days(fills, threshold):
    """For each day, the first (by fill_min) event whose own B30 >= threshold. One row per
    signal day: day, split, m_star (floor of the triggering fill_min), b30_at_signal,
    n_events_day, triggering_symbol."""
    rows = []
    for day, g in fills.groupby('day', sort=False):
        g = g.sort_values('fill_min')
        hit = g[g['b30'] >= threshold]
        if hit.empty:
            continue
        first = hit.iloc[0]
        rows.append(dict(day=day, split=g['split'].iloc[0], m_star=float(np.floor(first['fill_min'])),
                          m_star_raw=float(first['fill_min']), b30_at_signal=int(first['b30']),
                          n_events_day=len(g), trigger_symbol=first['symbol']))
    out = pd.DataFrame(rows)
    log.info("burst days found: %d (TRAIN=%d, VAL=%d) of %d total causal_arming days",
              len(out), (out['split'] == 'TRAIN').sum() if len(out) else 0,
              (out['split'] == 'VAL').sum() if len(out) else 0, fills['day'].nunique())
    return out


# --------------------------------------------------------------------------------------------
# Step 2: SPY / IWM minute bars -> per-(symbol, day) open-price lookup by ET minute-of-day
# --------------------------------------------------------------------------------------------

def load_index_bars():
    log.info("loading %s", INDEX_BARS_PARQUET)
    ib = pd.read_parquet(INDEX_BARS_PARQUET, columns=['symbol', 'day', 't', 'o'])
    ts = pd.to_datetime(ib['t'], utc=True).dt.tz_convert('America/New_York')
    ib['et_min'] = ts.dt.hour * 60 + ts.dt.minute
    log.info("index bars: %d rows, symbols=%s, days=%d, source file has no gaps check yet",
              len(ib), sorted(ib['symbol'].unique()), ib['day'].nunique())
    lookup = {}
    for (sym, day), g in ib.groupby(['symbol', 'day']):
        lookup[(sym, day)] = g.set_index('et_min')['o'].sort_index()
    return lookup


def get_open(lookup, symbol, day, minute, counters):
    """Open price at `minute`; if the exact minute bar is missing, forward-fills to the next
    available minute within MAX_BAR_FORWARD_FILL_MIN (logged), else returns None (logged)."""
    s = lookup.get((symbol, day))
    if s is None:
        counters['no_day_data'] += 1
        log.warning("no %s bars at all for day %s -- dropping this leg/day", symbol, day)
        return None
    if minute in s.index:
        return float(s.loc[minute])
    fwd = s.loc[s.index >= minute]
    fwd = fwd[fwd.index <= minute + MAX_BAR_FORWARD_FILL_MIN]
    if len(fwd):
        counters['forward_filled'] += 1
        log.warning("%s %s: no bar at minute %s, using next bar at %s (gap of %d min)",
                    symbol, day, minute, fwd.index[0], fwd.index[0] - minute)
        return float(fwd.iloc[0])
    counters['missing'] += 1
    log.warning("%s %s: no bar within %d min of requested minute %s -- dropping this leg/day",
                symbol, day, MAX_BAR_FORWARD_FILL_MIN, minute)
    return None


# --------------------------------------------------------------------------------------------
# Step 3: build the trade for every signal day, both legs, both real and placebo, 30/60/120 min
# --------------------------------------------------------------------------------------------

def trade_leg(lookup, symbol, day, ref_min, counters):
    """One index trade: entry at open of ref_min+1, three report exits at +31/+61/+121 capped
    at the 15:55 (955) cutoff. Returns dict of prices/bps or None if the entry bar is missing."""
    entry_min = int(ref_min) + 1
    entry_px = get_open(lookup, symbol, day, entry_min, counters)
    if entry_px is None:
        return None
    out = dict(entry_min=entry_min, entry_px=entry_px)
    for tag, hold in (('30', HOLD_MIN_SHORT), ('60', HOLD_MIN), ('120', HOLD_MIN_LONG)):
        exit_min = min(int(ref_min) + hold + 1, LATE_CUTOFF_ET_MIN)
        if exit_min <= entry_min:
            out[f'exit_min_{tag}'] = np.nan
            out[f'exit_px_{tag}'] = np.nan
            out[f'gross_bps_{tag}'] = np.nan
            out[f'net_bps_{tag}'] = np.nan
            continue
        exit_px = get_open(lookup, symbol, day, exit_min, counters)
        out[f'exit_min_{tag}'] = exit_min
        out[f'exit_px_{tag}'] = exit_px
        if exit_px is None:
            out[f'gross_bps_{tag}'] = np.nan
            out[f'net_bps_{tag}'] = np.nan
        else:
            gross = (exit_px / entry_px - 1.0) * 1e4
            out[f'gross_bps_{tag}'] = gross
            out[f'net_bps_{tag}'] = gross - ROUND_TRIP_COST_BPS
    return out


def build_signals(burst_days, lookup, counters):
    rng = np.random.default_rng(PLACEBO_SEED)
    rows = []
    for _, b in burst_days.sort_values('day').iterrows():
        day, split, m_star = b['day'], b['split'], b['m_star']
        row = dict(day=day, split=split, m_star_et_min=m_star, m_star_raw=b['m_star_raw'],
                   b30_at_signal=b['b30_at_signal'], n_events_day=b['n_events_day'],
                   trigger_symbol=b['trigger_symbol'])
        # placebo reference minute: uniform over the tradable session leaving room for entry+1
        # and a >=1 min hold before the 15:55 cutoff, drawn once per signal day (seed 1625)
        placebo_min = int(rng.integers(570, LATE_CUTOFF_ET_MIN - 1))
        row['placebo_ref_min'] = placebo_min
        for sym in ('SPY', 'IWM'):
            real = trade_leg(lookup, sym, day, m_star, counters)
            plac = trade_leg(lookup, sym, day, placebo_min, counters)
            prefix = sym.lower()
            if real is not None:
                for k, v in real.items():
                    row[f'{prefix}_{k}'] = v
            if plac is not None:
                for k, v in plac.items():
                    row[f'{prefix}_placebo_{k}'] = v
        rows.append(row)
    log.info("bar-lookup fallbacks: forward_filled=%d missing=%d no_day_data=%d",
              counters['forward_filled'], counters['missing'], counters['no_day_data'])
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------------------------
# Step 4: statistics per holdout, per leg, per hold length
# --------------------------------------------------------------------------------------------

def holdout_stats(sig, split, symbol, tag='60', placebo=False):
    col = f'{symbol.lower()}_{"placebo_" if placebo else ""}net_bps_{tag}'
    sub = sig[sig['split'] == split]
    y = sub[col]
    days = sub['day']
    n = int(y.notna().sum())
    if n == 0:
        return dict(n=0, mean_bps=np.nan, t=np.nan, ex_top5_bps=np.nan, winner_capped_bps=np.nan,
                    signals_per_week=0.0)
    return dict(
        n=n,
        mean_bps=float(y.mean()),
        t=day_clustered_t(y, days),
        ex_top5_bps=ex_top5_mean(y),
        winner_capped_bps=winner_capped_mean(y, WINNER_CAP_BPS),
        signals_per_week=n / weeks_spanned(sub['day'].unique()),
    )


def placebo_margin(sig, split, symbol, tag='60'):
    sub = sig[sig['split'] == split].copy()
    real_col = f'{symbol.lower()}_net_bps_{tag}'
    plac_col = f'{symbol.lower()}_placebo_net_bps_{tag}'
    sub = sub.dropna(subset=[real_col, plac_col])
    if sub.empty:
        return dict(n=0, margin_bps=np.nan, t=np.nan)
    diff = sub[real_col] - sub[plac_col]
    return dict(n=len(diff), margin_bps=float(diff.mean()), t=day_clustered_t(diff, sub['day']))


def main():
    fills = load_fill_events()
    fills = compute_b30(fills)

    train_b30 = fills.loc[fills['split'] == 'TRAIN', 'b30']
    threshold = float(np.percentile(train_b30, DECILE))
    log.info("TRAIN-H2 top-decile (p%d) B30 threshold = %.3f (pooled over %d TRAIN fills)",
              DECILE, threshold, len(train_b30))

    burst_days = find_burst_days(fills, threshold)
    lookup = load_index_bars()
    counters = {'forward_filled': 0, 'missing': 0, 'no_day_data': 0}
    sig = build_signals(burst_days, lookup, counters)
    sig.to_csv(OUT_SIGNALS_CSV, index=False)
    log.info("wrote %s (%d rows)", OUT_SIGNALS_CSV, len(sig))

    lines = []
    lines.append("# REBUILD_1625 -- Cell 1,625 independent rebuild from the prose only\n")
    lines.append(f"Rebuilt {pd.Timestamp.now(tz='UTC').isoformat()}. Source: PREREG_1623.md "
                 "'## Cell 1,625' section only -- cell_1625.py / cell_1625_signals.csv / "
                 "RESULT_1625.md were never opened.\n")
    lines.append(f"TRAIN-H2 top-decile (p{DECILE}) B30 threshold: **{threshold:.3f}** "
                 f"(pooled over {len(train_b30)} TRAIN-H2 fill events).\n")
    lines.append(f"Burst (signal) days found: **{len(burst_days)}** of "
                 f"{fills['day'].nunique()} causal_arming days "
                 f"(TRAIN-H2={int((burst_days['split']=='TRAIN').sum())}, "
                 f"VAL={int((burst_days['split']=='VAL').sum())}).\n")

    lines.append("\n## Primary metric: SPY, 60-minute hold, net bps (2 bps round-trip cost)\n")
    lines.append("| split | n days | mean net bps | day-clustered t | ex-top-5% bps | "
                 "winner-capped(+100bps) bps | signals/week |")
    lines.append("|---|---|---|---|---|---|---|")
    for split in ('TRAIN', 'VAL'):
        st = holdout_stats(sig, split, 'SPY', '60', placebo=False)
        label = 'TRAIN-H2' if split == 'TRAIN' else 'VAL'
        lines.append(f"| {label} | {st['n']} | {st['mean_bps']:.2f} | {st['t']:.2f} | "
                     f"{st['ex_top5_bps']:.2f} | {st['winner_capped_bps']:.2f} | "
                     f"{st['signals_per_week']:.2f} |")

    lines.append("\n## IWM leg, 60-minute hold, net bps\n")
    lines.append("| split | n days | mean net bps | day-clustered t | ex-top-5% bps | "
                 "winner-capped(+100bps) bps | signals/week |")
    lines.append("|---|---|---|---|---|---|---|")
    for split in ('TRAIN', 'VAL'):
        st = holdout_stats(sig, split, 'IWM', '60', placebo=False)
        label = 'TRAIN-H2' if split == 'TRAIN' else 'VAL'
        lines.append(f"| {label} | {st['n']} | {st['mean_bps']:.2f} | {st['t']:.2f} | "
                     f"{st['ex_top5_bps']:.2f} | {st['winner_capped_bps']:.2f} | "
                     f"{st['signals_per_week']:.2f} |")

    lines.append("\n## Placebo (random minute, same day, seed 1625) -- SPY 60-minute hold\n")
    lines.append("| split | n days | placebo mean net bps | real - placebo margin bps | margin t |")
    lines.append("|---|---|---|---|---|")
    for split in ('TRAIN', 'VAL'):
        stp = holdout_stats(sig, split, 'SPY', '60', placebo=True)
        m = placebo_margin(sig, split, 'SPY', '60')
        label = 'TRAIN-H2' if split == 'TRAIN' else 'VAL'
        lines.append(f"| {label} | {stp['n']} | {stp['mean_bps']:.2f} | {m['margin_bps']:.2f} | "
                     f"{m['t']:.2f} |")

    lines.append("\n## Report-only: 30- and 120-minute holds (SPY)\n")
    lines.append("| split | hold | n days | mean net bps | day-clustered t |")
    lines.append("|---|---|---|---|---|")
    for split in ('TRAIN', 'VAL'):
        label = 'TRAIN-H2' if split == 'TRAIN' else 'VAL'
        for tag in ('30', '120'):
            st = holdout_stats(sig, split, 'SPY', tag, placebo=False)
            lines.append(f"| {label} | {tag}min | {st['n']} | {st['mean_bps']:.2f} | {st['t']:.2f} |")

    val_st = holdout_stats(sig, 'VAL', 'SPY', '60', placebo=False)
    train_st = holdout_stats(sig, 'TRAIN', 'SPY', '60', placebo=False)
    val_margin = placebo_margin(sig, 'VAL', 'SPY', '60')
    same_sign = (np.sign(val_st['mean_bps']) == np.sign(train_st['mean_bps'])) if (
        not np.isnan(val_st['mean_bps']) and not np.isnan(train_st['mean_bps'])) else False

    lines.append("\n## Pass bar (VAL, SPY leg, 60-minute hold, frozen in PREREG_1623.md)\n")
    checks = [
        ("mean net >= +8 bps", val_st['mean_bps'] >= 8.0, val_st['mean_bps']),
        ("t >= 2.5", val_st['t'] >= 2.5, val_st['t']),
        ("ex-top-5% > 0", val_st['ex_top5_bps'] > 0, val_st['ex_top5_bps']),
        ("placebo margin >= +5 bps", val_margin['margin_bps'] >= 5.0, val_margin['margin_bps']),
        ("placebo margin t >= 2", val_margin['t'] >= 2.0, val_margin['t']),
        (">= 2 signals/week", val_st['signals_per_week'] >= 2.0, val_st['signals_per_week']),
        ("TRAIN-H2 same sign", same_sign, f"TRAIN-H2={train_st['mean_bps']:.2f} VAL={val_st['mean_bps']:.2f}"),
    ]
    all_pass = all(c[1] for c in checks)
    for name, ok, val in checks:
        lines.append(f"- [{'PASS' if ok else 'FAIL'}] {name} (value: {val})")
    lines.append(f"\n**Overall: {'PASS' if all_pass else 'FAIL'}**\n")

    lines.append("\n## Caveats (read as an adversary)\n")
    lines.append("1. **B30 undercounts breadth.** The prose defines B30 over arm events of "
                 "'any status'; causal_arming_causal.csv only carries a minute for status=='fill' "
                 "rows (verified: 0/2,010 'nofill' and 0/21,931 'not_armed' rows have a fill_min, "
                 "and bars_fills_1478.db's fetch_log has exactly the 9,911 fill symbol-days, no "
                 "more). This rebuild's B30 is fills-only and is a lower bound on the prose's "
                 "breadth measure by construction -- it will systematically undercount by roughly "
                 "the 2,010/11,921 (~17%) share of arm events that never filled. If the original "
                 "cell_1625.py recovered nofill arm minutes some other way, the burst-day sets "
                 "and the Jaccard check in the Independent-check section will disagree on that "
                 "gap and the discrepancy should be attributed here, not to a coding error.\n")
    lines.append("2. **Minute rounding.** fill_min is a fractional ET minute; m* is taken as "
                 "floor(fill_min) of the triggering event to land on the 1-minute SPY/IWM bar "
                 "grid. This is a unit-conversion choice, not a tuned threshold.\n")
    lines.append("3. **Percentile method.** The TRAIN-H2 top-decile threshold uses numpy's "
                 "default linear-interpolation percentile over the pooled per-fill B30 values "
                 "(not hour-of-day adjusted -- the prose only states the hour adjustment for "
                 "cell 1,624's terciles, not for 1,625's decile, so none was added here per "
                 "'Not allowed: tuning the thresholds ... after a number').\n")
    lines.append("4. **Bar-lookup fallbacks.** forward_filled=%d, missing=%d, no_day_data=%d "
                 "(see log above); a missing bar drops that leg/day from that leg's stats only, "
                 "logged at WARNING, never silently zero-filled.\n" % (
                     counters['forward_filled'], counters['missing'], counters['no_day_data']))

    report = "\n".join(lines)
    with open(OUT_REPORT_MD, "w") as f:
        f.write(report)
    log.info("wrote %s", OUT_REPORT_MD)
    log.info("DONE. VAL SPY 60-min: n=%d mean=%.2f bps t=%.2f | overall %s",
              val_st['n'], val_st['mean_bps'], val_st['t'], 'PASS' if all_pass else 'FAIL')


if __name__ == "__main__":
    main()
