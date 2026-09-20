import pandas as pd, numpy as np

def dclustered_t(daily):
    daily = np.asarray(daily, dtype=float)
    n = len(daily)
    if n < 2:
        return np.nan
    m, s = daily.mean(), daily.std(ddof=1)
    return np.nan if s == 0 else m / (s / np.sqrt(n))

def mdd(pnl_series_sorted):
    cum = np.cumsum(pnl_series_sorted)
    peak = np.maximum.accumulate(cum)
    return float((cum - peak).min())

def green_week_share(dates, pnl):
    df = pd.DataFrame({'date': pd.to_datetime(dates), 'pnl': pnl})
    df['week'] = df.date.dt.isocalendar().week.astype(str) + '-' + df.date.dt.isocalendar().year.astype(str)
    wk = df.groupby('week').pnl.sum()
    return float((wk > 0).mean()), len(wk)

def split_report(dates, pnl, Rmult=None):
    d = pd.to_datetime(dates)
    out = {}
    for name, mask in [('TRAIN-2025', (d.dt.year == 2025)),
                        ('VAL-Jan..May26', (d >= '2026-01-01') & (d <= '2026-05-31'))]:
        idx = mask.values
        p = np.asarray(pnl)[idx]
        dd = d[idx]
        if len(p) == 0:
            out[name] = None
            continue
        order = np.argsort(dd.values)
        daily = pd.Series(p, index=dd).groupby(level=0).sum()
        gw, nwk = green_week_share(dd, p)
        rec = dict(n=len(p), total=float(p.sum()), mdd=mdd(p[order]),
                   t_day=dclustered_t(daily.values), green_week=gw, n_weeks=nwk)
        if Rmult is not None:
            rm = np.asarray(Rmult)[idx]
            rec['R_mean'] = float(np.nanmean(rm))
        out[name] = rec
    return out

# ---------------- BF ----------------
b = pd.read_csv('research/mature_method/frames14/f45_bf_book.csv')
raw_fill = b.entry_price / 1.005
implied_entry_bps = (0.5 * b.sp_e / raw_fill) * 1e4
med_bps_e = implied_entry_bps.median()
print(f"[BF] median implied entry half-spread bps (current/P) = {med_bps_e:.2f}")

stopish = b.exit_reason.astype(str).str.contains('stop', na=False)
charged_x = np.where(stopish, 0.003 * b.exit_price, 0.0)

BF_TARGETS = {'P': med_bps_e, 'M': 23.235, 'O': 12.55}
bf_results = {}
for setting, tgt in BF_TARGETS.items():
    scale = tgt / med_bps_e
    sp_e_new = b.sp_e * scale
    sp_x_half = b.sp_x * 0.5   # exit at 50% of current charge, fixed across settings
    d_entry_local = b.shares * ((raw_fill + 0.5 * sp_e_new) - b.entry_price)
    d_entry_stored = -d_entry_local.fillna(0)
    d_exit_local = b.ex_shares * (0.5 * sp_x_half - charged_x)
    d_exit_stored = -d_exit_local.fillna(0)
    pnl_new = b.pnl + d_entry_stored + d_exit_stored
    rep = split_report(b.date, pnl_new, Rmult=(pnl_new / b.R_dollar))
    bf_results[setting] = (float(pnl_new.sum()), rep)
    print(f"[BF-{setting}] target_entry_bps={tgt:.2f} total_ALL=${pnl_new.sum():,.0f}")
    for k, v in rep.items():
        if v: print(f"    {k}: n={v['n']} total=${v['total']:,.0f} MDD=${v['mdd']:,.0f} "
                     f"t={v['t_day']:.2f} R={v['R_mean']:+.3f} greenwk={v['green_week']:.0%} (n={v['n_weeks']})")

# ---------------- ORB ----------------
o = pd.read_csv('analysis_results/orb_bplus_book.csv')
o = o[o.entered.astype(int) == 1].copy()
notional = o._sized_pnl / (o.pnl_pct / 100.0)
notional = notional.abs()
ORB_TARGETS = {'P': 13.50, 'M': 8.29, 'O': 3.08}
orb_results = {}
for setting, tgt in ORB_TARGETS.items():
    delta_bps = (tgt - 13.50) / 1e4
    pnl_new = o._sized_pnl - notional * delta_bps  # more bps = more cost = lower pnl
    Rproxy = o.pnl_pct / o.range_size_pct.replace(0, np.nan)
    rep = split_report(o.date, pnl_new, Rmult=Rproxy)
    orb_results[setting] = (float(pnl_new.sum()), rep)
    print(f"[ORB-{setting}] target_entry_bps={tgt:.2f} total_ALL=${pnl_new.sum():,.0f}")
    for k, v in rep.items():
        if v: print(f"    {k}: n={v['n']} total=${v['total']:,.0f} MDD=${v['mdd']:,.0f} "
                     f"t={v['t_day']:.2f} R={v['R_mean']:+.3f} greenwk={v['green_week']:.0%} (n={v['n_weeks']})")

import json
with open('research/exec_cost/recal_out.json', 'w') as f:
    json.dump({'BF': {k: v[1] for k, v in bf_results.items()},
               'ORB': {k: v[1] for k, v in orb_results.items()}}, f, indent=2, default=str)
print("done")
