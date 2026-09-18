#!/usr/bin/env python3
"""BF decay part 2: reconcile the quoted +0.61R -> +0.06R against the SHIPPED P1 stack,
test the outcome (exit) side, and do the power/MDE arithmetic."""
import sys, math, json
import numpy as np, pandas as pd
sys.path.insert(0, '/home/ec2-user/onemil')
OUT = '/home/ec2-user/onemil/research/bf_decay'
CACHE = '/home/ec2-user/onemil/data/bull_flag_cache_causal_full_20260905.csv'
def log(*a): print(*a); sys.stdout.flush()

df = pd.read_csv(CACHE, keep_default_na=False, na_values=[''], dtype={'symbol': str})
df = df[(df['date'] >= '2025-01-01') & (df['date'] <= '2026-08-31')].copy()
df['year'] = df['date'].str[:4]; df['month'] = df['date'].str[:7]
for c in ['entry_price','stop_loss','pnl','shares','qf_pole_gain_pct','qf_vwap_dist_pct',
          'conviction_mult','macd_zone_mult','intraday_change_at_entry','avg_volume_20d',
          'qf_pole_bars','exit_price']:
    df[c] = pd.to_numeric(df[c], errors='coerce')
df['R'] = df['pnl'] / ((df['entry_price']-df['stop_loss'])*df['shares'])
df = df[np.isfinite(df['R'])].copy()

from trading.bf_universe_filter import filter_trades as bf_univ
recs = df.to_dict('records'); keep = {id(r) for r in bf_univ(recs)}
df['univ_ok'] = [id(r) in keep for r in recs]
base = df[df.univ_ok & (df['avg_volume_20d'].fillna(0) >= 200000)].copy()

from trading.two_tier_filter import classify_tier, build_features_from_trade, should_keep
from trading.bf_vwap_gate import load_vwap_gate_config, filter_trades as vwap_gate
from config import Config
cfg = Config._load_yaml_only(); bf = cfg['trading']['bull_flag']; tt = bf['two_tier_filter']

def apply_ttf(sub):
    rs = sub.to_dict('records'); ok = []
    for r in rs:
        ic = r.get('intraday_change_at_entry')
        ic = None if (ic is None or (isinstance(ic,float) and math.isnan(ic))) else float(ic)
        tl = classify_tier(ic, a_tier_lower=float(tt['a_tier_lower']), extras_lower=float(tt['extras_lower']))
        k,_ = should_keep(tier=tl, macd_zone_mult=float(r.get('macd_zone_mult') or 0.0),
                          features=build_features_from_trade(r), cfg=tt)
        ok.append(k)
    return sub[np.array(ok)]

def apply_vwap(sub):
    rs = sub.to_dict('records'); k = {id(r) for r in vwap_gate(rs, load_vwap_gate_config(bf))}
    return sub[[id(r) in k for r in rs]]

def stack(sub, price_max, pole_min, vwap_on):
    s = sub
    if price_max: s = s[s['entry_price'] <= price_max]
    if pole_min:  s = s[s['qf_pole_gain_pct'] >= pole_min]
    s = s[s['conviction_mult'].fillna(1.0) >= 1.8]
    s = s[(s['qf_pole_bars'] > 0) & (s['qf_pole_bars'] <= 3)]
    if vwap_on: s = apply_vwap(s)
    return apply_ttf(s)

def rep(name, s):
    o = {'stack': name}
    for y in ('2025','2026'):
        v = s[s.year==y]['R'].values
        o[f'n{y[2:]}'] = len(v); o[f'R{y[2:]}'] = float(v.mean()) if len(v) else np.nan
        o[f'WR{y[2:]}'] = float((v>0).mean()*100) if len(v) else np.nan
        o[f'se{y[2:]}'] = float(v.std(ddof=1)/math.sqrt(len(v))) if len(v)>1 else np.nan
    o['dR'] = o['R26']-o['R25']
    a = s[s.year=='2025']['R'].values; b = s[s.year=='2026']['R'].values
    se = math.sqrt(a.var(ddof=1)/len(a)+b.var(ddof=1)/len(b)) if len(a)>1 and len(b)>1 else np.nan
    o['se_diff'] = se; o['t'] = o['dR']/se if se==se else np.nan
    sdp = math.sqrt(((len(a)-1)*a.var(ddof=1)+(len(b)-1)*b.var(ddof=1))/(len(a)+len(b)-2))
    o['mde80'] = 2.8*sdp*math.sqrt(1/len(a)+1/len(b))
    return o

variants = [
    ('AS-IS (pre-P1: no price cap, pole>=3, no gate)', stack(base, 0, 3.0, False)),
    ('P1 SHIPPED (price<=20, pole>=5, gate on)',        stack(base, 20.0, 5.0, True)),
    ('+price cap only',                                 stack(base, 20.0, 3.0, False)),
    ('+pole>=5 only',                                   stack(base, 0, 5.0, False)),
    ('+VWAP gate only',                                 stack(base, 0, 3.0, True)),
]
tab = [rep(n, s) for n, s in variants]
log("\n=== STACK RECONCILIATION (R per pick) ===")
log(pd.DataFrame(tab).to_string(index=False, float_format=lambda x: f'{x:8.3f}'))
pd.DataFrame(tab).to_csv(f'{OUT}/stack_reconciliation.csv', index=False)

asis = stack(base, 0, 3.0, False); p1 = stack(base, 20.0, 5.0, True)

# ---- half-year / rolling stability of BOTH stacks
def halves(s, name):
    s = s.copy(); s['half'] = s['date'].str[:4]+'H'+((s['date'].str[5:7].astype(int)-1)//6+1).astype(str)
    g = s.groupby('half')['R'].agg(['size','mean','sum'])
    log(f"\n--- {name} by half ---"); log(g.to_string())
    return g
halves(asis, 'AS-IS'); halves(p1, 'P1')

# ---- OUTCOME SIDE: same simulator both years; did the winners get shorter?
log("\n=== OUTCOME SIDE (raw pool, n=886 — high power) ===")
for tag, s in [('RAW', df), ('P1 picks', p1)]:
    for y in ('2025','2026'):
        v = s[s.year==y]
        w = v[v.R>0]['R'].values; l = v[v.R<=0]['R'].values
        log(f"{tag} {y}: n={len(v)} WR={len(w)/len(v)*100:5.1f}% avgWin={w.mean():5.3f} "
            f"avgLoss={l.mean():6.3f} >=2R={(v.R>=2).mean()*100:4.1f}% >=3R={(v.R>=3).mean()*100:4.1f}%")

def welch(a, b):
    se = math.sqrt(a.var(ddof=1)/len(a)+b.var(ddof=1)/len(b))
    return b.mean()-a.mean(), se, (b.mean()-a.mean())/se

log("\n--- winner size, raw pool, per exit reason ---")
for er in ['trail_stop','exhaust+trail_stop','stop']:
    a = df[(df.year=='2025')&(df.exit_reason==er)]['R'].values
    b = df[(df.year=='2026')&(df.exit_reason==er)]['R'].values
    if len(a)>1 and len(b)>1:
        d,se,t = welch(a,b)
        log(f"{er:22s} 2025 n={len(a):3d} R={a.mean():6.3f} | 2026 n={len(b):3d} R={b.mean():6.3f} | d={d:+.3f} se={se:.3f} t={t:+.2f}")
log("\n--- exit-reason MIX share, raw pool ---")
mix = (df.groupby(['year','exit_reason']).size()/df.groupby('year').size()*100).unstack(fill_value=0)
log(mix.to_string(float_format=lambda x: f'{x:5.1f}'))

# winners only, all exits
a = df[(df.year=='2025')&(df.R>0)]['R'].values; b = df[(df.year=='2026')&(df.R>0)]['R'].values
d,se,t = welch(a,b); log(f"\nRAW winners: 2025 {a.mean():.3f} (n={len(a)}) vs 2026 {b.mean():.3f} (n={len(b)}) d={d:+.3f} t={t:+.2f}")
a = df[(df.year=='2025')&(df.R<=0)]['R'].values; b = df[(df.year=='2026')&(df.R<=0)]['R'].values
d,se,t = welch(a,b); log(f"RAW losers : 2025 {a.mean():.3f} (n={len(a)}) vs 2026 {b.mean():.3f} (n={len(b)}) d={d:+.3f} t={t:+.2f}")

# ---- ENTRY-vs-EXIT split of the P1 delta
p25, p26 = p1[p1.year=='2025']['R'].values, p1[p1.year=='2026']['R'].values
wr25 = (p25>0).mean(); wr26 = (p26>0).mean()
w25, w26 = p25[p25>0].mean(), p26[p26>0].mean()
l25, l26 = p25[p25<=0].mean(), p26[p26<=0].mean()
# Additive decomposition of meanR = wr*W + (1-wr)*L
contrib_wr   = (wr26-wr25)*w25 - (wr26-wr25)*l25
contrib_win  = wr26*(w26-w25)
contrib_loss = (1-wr26)*(l26-l25)
log(f"\n=== DELTA DECOMPOSITION (P1 picks, 2026 - 2025 = {p26.mean()-p25.mean():+.3f}R) ===")
log(f" WR effect      ({wr25*100:.1f}% -> {wr26*100:.1f}%): {contrib_wr:+.3f}R")
log(f" win-size effect({w25:.3f} -> {w26:.3f})     : {contrib_win:+.3f}R")
log(f" loss-size eff. ({l25:.3f} -> {l26:.3f})   : {contrib_loss:+.3f}R")

# ---- POWER / MDE arithmetic
log("\n=== POWER ===")
for name, s in [('RAW pool', df), ('P1 picks', p1), ('AS-IS picks', asis)]:
    a = s[s.year=='2025']['R'].values; b = s[s.year=='2026']['R'].values
    sdp = math.sqrt(((len(a)-1)*a.var(ddof=1)+(len(b)-1)*b.var(ddof=1))/(len(a)+len(b)-2))
    se = math.sqrt(a.var(ddof=1)/len(a)+b.var(ddof=1)/len(b))
    mde = 2.8*sdp*math.sqrt(1/len(a)+1/len(b))
    log(f"{name:12s} n={len(a):4d}/{len(b):4d} sd={sdp:.2f} se(diff)={se:.3f} "
        f"observed d={b.mean()-a.mean():+.3f} t={(b.mean()-a.mean())/se:+.2f} MDE80={mde:.3f}R")

# how many picks to detect the observed -0.18R at 80% power?
sdp = math.sqrt(((len(p25)-1)*p25.var(ddof=1)+(len(p26)-1)*p26.var(ddof=1))/(len(p25)+len(p26)-2))
for eff in (0.18, 0.30, 0.50, 0.75):
    n = 2*(2.8*sdp/eff)**2
    log(f"  to detect {eff:.2f}R at 80%/5%: n={n:.0f} per year ({n/2.8:.0f} months at 2.8 picks/mo)")

# Q3-2026 alone
q3 = p1[p1.date.str[:7].isin(['2026-07','2026-08'])]
log(f"\nQ3-2026 (Jul+Aug): n={len(q3)} meanR={q3['R'].mean():.3f} pnl=${q3['pnl'].sum():,.0f} "
    f"| one-sample t vs 0: t={q3['R'].mean()/(q3['R'].std(ddof=1)/math.sqrt(len(q3))):.2f}")
log(f"2026H1 P1: n={len(p1[p1.date<'2026-07-01'][p1.year=='2026'])} "
    f"meanR={p1[(p1.year=='2026')&(p1.date<'2026-07-01')]['R'].mean():.3f}")

# ---- regime / market variable gate test on P1 picks
p1 = p1.copy()
import sqlite3
con = sqlite3.connect('file:/home/ec2-user/onemil/data/cache.db?mode=ro', uri=True)
spy = pd.read_sql_query("select bar_date,open,high,low,close,volume from daily_bars where symbol='SPY' order by bar_date", con); con.close()
spy['bar_date'] = pd.to_datetime(spy['bar_date'])
from trading.regime_helpers import build_regime_lookup
reg = build_regime_lookup(spy)
p1['regime'] = p1['date'].map(reg).fillna('unknown')
log("\n=== P1 picks by regime ===")
log(p1.groupby(['year','regime'])['R'].agg(['size','mean']).to_string())
log("\n=== P1 picks by regime, pooled ===")
log(p1.groupby('regime')['R'].agg(['size','mean','sum']).to_string())
log("\n=== RAW by regime, pooled ===")
df['regime'] = df['date'].map(reg).fillna('unknown')
log(df.groupby(['year','regime'])['R'].agg(['size','mean']).to_string())
p1.to_csv(f'{OUT}/p1_picks.csv', index=False)
asis.to_csv(f'{OUT}/asis_picks.csv', index=False)
