#!/usr/bin/env python3
"""Ignition-from-zero — hypothesis scoring (DESIGN.md steps 2-3).

Base book (the "as-is" definition, so every hypothesis is a delta from a
known point): level 10, trigger inside 9:35-10:30, BT gates pass, exit =
partial 50%@+1R / breakeven (rr_p1be). A hypothesis is a feature split of
that book (or a different level / window / exit where the hypothesis IS the
definition). Metrics per split, per group: n, meanR ex-tail (rr<2), meanR,
WR, weeks green, worst week (sum of R at unit risk).

Verdict rule (pre-registered): the effect direction on TRAIN must agree with
VALIDATE, and only then is TEST read; a group is a FINDING if TEST meanR
ex-tail >= +0.05 with >= half its weeks green. Everything is printed —
failures included — and written to score_tables.md.
"""
import os, sys
import numpy as np, pandas as pd
ROOT = '/home/ec2-user/onemil'; os.chdir(ROOT)
D = 'research/ignition_zero'
c = pd.read_csv(f'{D}/candidates_full.csv', low_memory=False)
c['split'] = np.where(c.day < '2026-01-01', 'TRAIN', np.where(c.day < '2026-06-01', 'VAL', 'TEST'))
c['wk'] = pd.to_datetime(c.day).dt.to_period('W-FRI').astype(str)
GATED = (c.flag_chase == 0) & (c.flag_prebars == 0) & (c.flag_rmin == 0)
BASE = GATED & (c.level == 10) & (c.in_bt_window == 1)
EXIT = 'rr_p1be'
out = []


def stats(d, rr=EXIT):
    r = d[rr]; ex = r[r < 2]
    w = d.groupby('wk')[rr].sum()
    return dict(n=len(d), meanR_ex=round(ex.mean(), 3) if len(ex) else np.nan, meanR=round(r.mean(), 3) if len(d) else np.nan,
                WR=round((r > 0).mean() * 100, 1) if len(d) else np.nan, wk_green=f"{int((w > 0).sum())}/{len(w)}" if len(w) else '',
                wk_green_frac=(w > 0).mean() if len(w) else np.nan, worst_wk=round(w.min(), 1) if len(w) else np.nan)


def table(h, title, groups, mask=None, rr=EXIT):
    """groups: dict name -> boolean mask (on c). Prints TRAIN/VAL/TEST per group with the verdict."""
    base = BASE if mask is None else mask
    rows = []
    for g, gm in groups.items():
        for s in ('TRAIN', 'VAL', 'TEST'):
            d = c[base & gm & (c.split == s)]
            rows.append(dict(H=h, group=g, split=s, **stats(d, rr)))
    t = pd.DataFrame(rows)
    # verdict per group: sign of (group meanR_ex - base meanR_ex) agrees TRAIN/VAL; TEST >= 0.05 & >= half weeks green
    bstat = {s: stats(c[base & (c.split == s)], rr)['meanR_ex'] for s in ('TRAIN', 'VAL', 'TEST')}
    verdicts = {}
    for g in groups:
        tr = t[(t.group == g) & (t.split == 'TRAIN')].iloc[0]; va = t[(t.group == g) & (t.split == 'VAL')].iloc[0]; te = t[(t.group == g) & (t.split == 'TEST')].iloc[0]
        d_tr = tr.meanR_ex - bstat['TRAIN']; d_va = va.meanR_ex - bstat['VAL']
        consistent = np.sign(d_tr) == np.sign(d_va) and abs(d_tr) >= 0.02 and abs(d_va) >= 0.02
        finding = consistent and d_tr > 0 and te.meanR_ex >= 0.05 and te.wk_green_frac >= 0.5 and te.n >= 30
        verdicts[g] = 'FINDING' if finding else ('consistent' if consistent else 'no signal')
    t['verdict'] = t.group.map(verdicts)
    print(f"\n### {h} — {title}   (base meanR_ex TRAIN {bstat['TRAIN']:+.3f} / VAL {bstat['VAL']:+.3f} / TEST {bstat['TEST']:+.3f})")
    print(t.drop(columns=['wk_green_frac']).to_string(index=False))
    out.append((h, title, bstat, t))
    return t


pd.set_option('display.width', 250)
print(f"candidates_full: {len(c):,} rows | base book rows: {int(BASE.sum()):,} | splits: {c[BASE].split.value_counts().to_dict()}")

# --- the gates themselves (a decision, so a hypothesis) ---
table('H0-gates', 'BT gates: chase / pre-bars / R-min as a hypothesis (level 10, window)',
      {'all gates pass (base)': GATED, 'chase-rejected only': (c.flag_chase == 1) & (c.flag_prebars == 0) & (c.flag_rmin == 0),
       'rmin-rejected only': (c.flag_rmin == 1) & (c.flag_chase == 0), 'prebars-rejected only': (c.flag_prebars == 1) & (c.flag_chase == 0) & (c.flag_rmin == 0)},
      mask=(c.level == 10) & (c.in_bt_window == 1))
table('H1', 'speed: minutes from open to +5% (level-10 book)', {'fast <=15 min': c.min_to_5pct <= 15, 'mid 15-30': (c.min_to_5pct > 15) & (c.min_to_5pct <= 30), 'slow >30': c.min_to_5pct > 30})
table('H2', 'cross level (gated, window)', {f'level {L}': c.level == L for L in (5, 7, 10, 15)}, mask=GATED & (c.in_bt_window == 1))
table('H4', 'trigger-bar close position', {'top third': c.tb_close_pos >= 0.67, 'middle': (c.tb_close_pos > 0.33) & (c.tb_close_pos < 0.67), 'bottom third': c.tb_close_pos <= 0.33})
table('H5', 'ignition volume: trigger-bar vol vs prior per-minute avg', {'>=3x': c.tb_vol_x_prior >= 3, '1-3x': (c.tb_vol_x_prior >= 1) & (c.tb_vol_x_prior < 3), '<1x': c.tb_vol_x_prior < 1})
table('H6', 'price band (entry)', {'$1-2': c.entry < 2, '$2-5': (c.entry >= 2) & (c.entry < 5), '$5-10': (c.entry >= 5) & (c.entry < 10), '$10-20': (c.entry >= 10) & (c.entry < 20), '$20+': c.entry >= 20})
table('H7', 'float (SNAPSHOT, coverage limited)', {'float <20M': (c.float_shares > 0) & (c.float_shares < 20e6), 'float 20-100M': (c.float_shares >= 20e6) & (c.float_shares < 100e6), 'float >=100M': c.float_shares >= 100e6, 'unknown': ~(c.float_shares > 0)})
table('H8', 'prior-day range (day-2 continuation)', {'prev range >=10%': c.prev_range_pct >= 10, 'prev range 5-10%': (c.prev_range_pct >= 5) & (c.prev_range_pct < 10), 'prev range <5%': c.prev_range_pct < 5})
table('H9', 'distance from 20-day high at the level', {'breakout (>= 20d high)': c.dist_20d_high_pct >= 0, 'within 10% below': (c.dist_20d_high_pct < 0) & (c.dist_20d_high_pct >= -10), 'more than 10% below': c.dist_20d_high_pct < -10})
table('H10', 'wrapper vs common', {'wrapper': c.is_wrapper == 1, 'common': c.is_wrapper == 0})
table('H11', 'sibling cohort at trigger', {'cohort >=1 other': c.coh_by_t >= 1, 'alone': c.coh_by_t == 0})
table('H12', 'sympathy: lagging sibling (NOTE: the leader cell uses coh_day = whole-day cohort = LOOK-AHEAD; see REPORT.md §3 — never a finding)', {'leader LOOKAHEAD (sibling follows later)': (c.coh_day >= 1) & c.sympathy_lag_min.isna(), 'leader causal (anchor, no sibling yet)': c.anchor.notna() & (c.coh_by_t == 0), 'lagger <=30 min': (c.sympathy_lag_min > 0) & (c.sympathy_lag_min <= 30), 'lagger >30 min': c.sympathy_lag_min > 30})
table('H13', 'theme heat: other +10% crosses in the prior 30 min (market-wide)', {'>=3 others': c.theme_n30 >= 3, '1-2 others': (c.theme_n30 >= 1) & (c.theme_n30 < 3), 'isolated': c.theme_n30 == 0})
table('H14', 'headline class (news-covered rows only)', {'dilution': c.headline_class == 'dilution', 'positive': c.headline_class == 'positive', 'other news': c.headline_class == 'other', 'no news (covered)': (c.news_covered == 1) & (c.has_news_pre == False)}, mask=BASE & (c.news_covered == 1))
table('H15', 'news recency (covered rows)', {'<=60 min before': c.news_recency_min <= 60, '1-16h before (premarket/overnight)': (c.news_recency_min > 60), 'no news': (c.news_covered == 1) & (c.has_news_pre == False)}, mask=BASE & (c.news_covered == 1))
table('H16', 'premarket dollar volume (coverage limited)', {'pm >= $1M': c.pm_dollar >= 1e6, 'pm < $1M': (c.pm_dollar < 1e6) & c.pm_dollar.notna(), 'no premarket bars': c.pm_dollar.isna()})
sr_med = c[BASE & (c.split == 'TRAIN')].spy_range3.median()
table('H17', 'market context at trigger', {'SPY 5m up': c.spy_5m_ret >= 0, 'SPY 5m down': c.spy_5m_ret < 0, f'SPY 3d range calm (<= TRAIN median {sr_med:.2f})': c.spy_range3 <= sr_med, 'SPY 3d range violent': c.spy_range3 > sr_med})
table('H18', 'time of day of the trigger (window itself)', {'9:35-9:50': c.trig_m < 590, '9:50-10:10': (c.trig_m >= 590) & (c.trig_m < 610), '10:10-10:30': (c.trig_m >= 610) & (c.trig_m <= 630), '10:30-11:30 (outside BT window)': c.trig_m > 630}, mask=GATED & (c.level == 10))
for ex in ('rr_v0', 'rr_hold', 'rr_strail', 'rr_t60'):
    table('H19/20-exit', f'exit = {ex} vs the p1be base', {ex: BASE}, rr=ex)
table('H20', 'time-60 exit by speed', {'fast <=15 & t60': c.min_to_5pct <= 15, 'slow >15 & t60': c.min_to_5pct > 15}, rr='rr_t60')
table('H21', 'short interest (FINRA, point-in-time)', {'SI >= 3x ADV20': c.si_ratio_adv20 >= 3, 'SI 1-3x': (c.si_ratio_adv20 >= 1) & (c.si_ratio_adv20 < 3), 'SI < 1x': c.si_ratio_adv20 < 1, 'days-to-cover >= 5': c.si_dtc >= 5, 'no SI record': c.si_qty.isna()})
table('H22', 'open gap (ORB overlap now allowed)', {'gap <= 0': c.open_gap_pct <= 0, 'gap 0-5%': (c.open_gap_pct > 0) & (c.open_gap_pct < 5), 'gap 5-15%': (c.open_gap_pct >= 5) & (c.open_gap_pct < 15), 'gap >= 15%': c.open_gap_pct >= 15})

with open(f'{D}/score_tables.md', 'w') as fh:
    fh.write('# Ignition-from-zero — hypothesis tables (auto-generated by score.py)\n\n')
    for h, title, bstat, t in out:
        fh.write(f"## {h} — {title}\nbase meanR_ex TRAIN {bstat['TRAIN']:+.3f} / VAL {bstat['VAL']:+.3f} / TEST {bstat['TEST']:+.3f}\n\n")
        fh.write(t.drop(columns=['wk_green_frac']).to_markdown(index=False) + '\n\n')
print('\nFINDINGS:', [(h, g) for h, _, _, t in out for g in t[t.verdict == 'FINDING'].group.unique()])
