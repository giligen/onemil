"""ORB per-regime multiplier study: rule-regime (A/B/C1/C2) vs HMM-regime (0/1/2).
Fits state multipliers on TRAIN 2025, applies to VAL 2026-01..05, reports vs flat 1.0x.
TEST (>=2026-06-01) sealed - counted only, never touched.
"""
import sqlite3
import sys
import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, '/home/ec2-user/onemil')
from trading.regime_helpers import build_regime_lookup

BOOK = '/home/ec2-user/onemil/analysis_results/orb_bplus_book.csv'
HMM = '/home/ec2-user/onemil/research/regime/hmm_labels.csv'
DB = '/home/ec2-user/onemil/data/cache.db'
OUTDIR = '/home/ec2-user/onemil/research/regime/'

# ---- load book ----
df = pd.read_csv(BOOK)
print(f"raw rows: {len(df)}  entered==1: {(df['entered']==1).sum()}")
df = df[df['entered'] == 1].copy()
df['pnl_dollar'] = df['_sized_pnl'] if '_sized_pnl' in df.columns else df['pnl'] / 15.0
df['date'] = pd.to_datetime(df['date'])

print("_rp_position describe:\n", df['_rp_position'].describe())
r_denom = df['_rp_position'].median()
print(f"R denom (median _rp_position): {r_denom:.4f}  "
      f"(min={df['_rp_position'].min():.4f} max={df['_rp_position'].max():.4f})")
df['pnl_R'] = df['pnl_dollar'] / r_denom

# ---- rule regime lookup ----
conn = sqlite3.connect(DB)
spy = pd.read_sql_query(
    "SELECT bar_date, close FROM daily_bars WHERE symbol='SPY' ORDER BY bar_date", conn)
conn.close()
rule_lookup = build_regime_lookup(spy)
df['rule_regime'] = df['date'].dt.strftime('%Y-%m-%d').map(rule_lookup).fillna('unknown')

# ---- hmm regime lookup ----
hmm = pd.read_csv(HMM)
hmm_lookup = dict(zip(hmm['bar_date'], hmm['hmm_state']))
df['hmm_regime'] = df['date'].dt.strftime('%Y-%m-%d').map(hmm_lookup)
df['hmm_regime'] = df['hmm_regime'].apply(lambda x: f"hmm{int(x)}" if pd.notna(x) else 'unknown')

TRAIN = (df['date'] >= '2025-01-01') & (df['date'] <= '2025-12-31')
VAL = (df['date'] >= '2026-01-01') & (df['date'] <= '2026-05-31')
TEST = df['date'] >= '2026-06-01'
print(f"\nTRAIN n={TRAIN.sum()}  VAL n={VAL.sum()}  TEST(sealed, not used) n={TEST.sum()}")


def clustered_t(sub):
    """day-clustered t-stat: t-test on per-day mean R across days."""
    if len(sub) == 0:
        return np.nan
    day_means = sub.groupby('date')['pnl_R'].mean()
    if len(day_means) < 2:
        return np.nan
    t, _ = stats.ttest_1samp(day_means.values, 0)
    return t


def iid_t(sub):
    if len(sub) < 2:
        return np.nan
    t, _ = stats.ttest_1samp(sub['pnl_R'].values, 0)
    return t


def fit_multiplier(sub2025):
    h1 = sub2025[sub2025['date'] <= '2025-06-30']
    h2 = sub2025[sub2025['date'] > '2025-06-30']
    r1 = h1['pnl_R'].sum() if len(h1) else np.nan
    r2 = h2['pnl_R'].sum() if len(h2) else np.nan
    pooled_r = sub2025['pnl_R'].sum()
    t_pooled = iid_t(sub2025)
    both_neg = (not np.isnan(r1) and not np.isnan(r2) and r1 < 0 and r2 < 0)
    both_pos = (not np.isnan(r1) and not np.isnan(r2) and r1 > 0 and r2 > 0)
    if both_neg:
        return 0.0, 'both halves negative'
    if both_pos and (not np.isnan(t_pooled)) and t_pooled >= 1.5:
        return 1.5, 'both halves positive, t>=1.5'
    mixed = not (both_neg or both_pos)
    if mixed and pooled_r < 0:
        return 0.5, 'mixed halves, pooled negative'
    return 1.0, 'default (mixed-pooled-positive or low-t positive)'


def analyze_system(colname, label):
    print(f"\n===== {label} ({colname}) =====")
    states = sorted(df[colname].unique())
    mults = {}
    for st in states:
        sub_all = df[df[colname] == st]
        sub_tr = sub_all[TRAIN]
        n = len(sub_tr)
        net_r = sub_tr['pnl_R'].sum()
        t_i = iid_t(sub_tr)
        t_c = clustered_t(sub_tr)
        h1 = sub_tr[sub_tr['date'] <= '2025-06-30']['pnl_R'].sum() if n else np.nan
        h2 = sub_tr[sub_tr['date'] > '2025-06-30']['pnl_R'].sum() if n else np.nan
        mult, reason = (0.0, 'no TRAIN trades') if n == 0 else fit_multiplier(sub_tr)
        mults[st] = mult
        val_n = TRAIN if False else (df[colname] == st) & VAL
        print(f"  state={st:6s} TRAIN n={n:3d} netR={net_r:7.2f} t_iid={t_i:6.2f} "
              f"t_clust={t_c:6.2f} H1={h1:7.2f} H2={h2:7.2f} -> mult={mult} ({reason}) "
              f"| VAL n={val_n.sum()}")
    return mults


rule_mults = analyze_system('rule_regime', 'RULE REGIME A/B/C1/C2')
hmm_mults = analyze_system('hmm_regime', 'HMM REGIME 0/1/2')


def equity_mdd(sub):
    if len(sub) == 0:
        return 0.0, 0.0
    s = sub.sort_values('date')
    cum = s['pnl_dollar_applied'].cumsum()
    peak = cum.cummax()
    dd = (cum - peak).min()
    return cum.iloc[-1], dd


def green_week_share(sub):
    if len(sub) == 0:
        return np.nan, 0
    s = sub.copy()
    s['iso_week'] = s['date'].dt.strftime('%G-W%V')
    wk = s.groupby('iso_week')['pnl_dollar_applied'].sum()
    return (wk > 0).mean(), len(wk)


def apply_and_report(colname, mults, label):
    val = df[VAL].copy()
    val['mult'] = val[colname].map(mults).fillna(1.0)
    val['pnl_dollar_applied'] = val['pnl_dollar'] * val['mult']
    val['pnl_R_applied'] = val['pnl_R'] * val['mult']
    flat = val.copy()
    flat['pnl_dollar_applied'] = flat['pnl_dollar']
    flat['pnl_R_applied'] = flat['pnl_R']

    tot_flat, mdd_flat = equity_mdd(flat)
    tot_reg, mdd_reg = equity_mdd(val)
    gw_flat, nwk_flat = green_week_share(flat)
    gw_reg, nwk_reg = green_week_share(val)
    r_flat = flat['pnl_R_applied'].sum()
    r_reg = val['pnl_R_applied'].sum()

    print(f"\n----- {label} VAL: per-regime vs flat -----")
    print(f"  FLAT     : ${tot_flat:9.2f}  MDD=${mdd_flat:9.2f}  netR={r_flat:7.2f}  "
          f"green-week={gw_flat:.2%} ({nwk_flat} wks)")
    print(f"  PER-REG  : ${tot_reg:9.2f}  MDD=${mdd_reg:9.2f}  netR={r_reg:7.2f}  "
          f"green-week={gw_reg:.2%} ({nwk_reg} wks)")
    print("  states exercised in VAL (n, mult):")
    for st, m in mults.items():
        n_val = (val[colname] == st).sum()
        if n_val > 0:
            print(f"    {st}: n={n_val} mult={m}")
    non1_exercised = any((val[colname] == st).sum() >= 10 for st, m in mults.items() if m != 1.0)
    pass_bar = (tot_reg > tot_flat) and (mdd_reg >= mdd_flat) and non1_exercised
    print(f"  PASS BAR: {pass_bar}  ($_up={tot_reg > tot_flat}, "
          f"mdd_not_worse={mdd_reg >= mdd_flat}, state_n>=10_exercised={non1_exercised})")

    out = val[['date', colname, 'symbol', 'pnl_R_applied']].rename(
        columns={'pnl_R_applied': 'pnl_R'})
    out['date'] = out['date'].dt.strftime('%Y-%m-%d')
    outpath = f"{OUTDIR}val_per_regime_{colname}.csv"
    out[['date', 'pnl_R', 'symbol']].to_csv(outpath, index=False)
    print(f"  wrote {outpath} ({len(out)} rows)")
    return dict(tot_flat=tot_flat, mdd_flat=mdd_flat, tot_reg=tot_reg, mdd_reg=mdd_reg,
                gw_flat=gw_flat, gw_reg=gw_reg, pass_bar=pass_bar)


res_rule = apply_and_report('rule_regime', rule_mults, 'RULE')
res_hmm = apply_and_report('hmm_regime', hmm_mults, 'HMM')

print("\n===== SUMMARY =====")
print("rule_mults:", rule_mults)
print("hmm_mults:", hmm_mults)
print("res_rule:", res_rule)
print("res_hmm:", res_hmm)
print(f"\nR denom used: {r_denom:.4f}")
print(f"hmm state2 VAL days present: {(df[VAL]['hmm_regime']=='hmm2').sum()}")
