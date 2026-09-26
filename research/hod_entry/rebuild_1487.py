"""
Independent rebuild of cells 1,487-1,488 (CONFIRMATION ENTRY) from PREREG_1487.md prose alone.
Written WITHOUT reading cell_1487.py / test_cell_1487.py / RESULT_1487.md.

Rule (1,487): for every base fill (causal_arming_causal.csv, status=='fill') still open at the
end of the 15th RTH minute after the fill bar (base exit_m > m0+15, m0=floor(fill_min)), and with
NO bar in (m0, m0+15] whose low <= level-0.01: enter LONG at the ask of the bar at m0+16 (ask =
that bar's open + the fill's measured half-spread from features_1478_A.csv's half_entry column),
stop = level-0.01, R'' = entry-stop, target = entry+2R'', 15:55 EOD exit. Path walked on minute
bars with sip_rebuild.walk_path semantics (stop-first on a bar touching both, gap-through at the
open). Costs: entry half-spread paid (baked into the ask); stop-limit exit slip = expected-value
0.88*filled-stop bps + 0.12*no-fill-tail bps (SLIP_STOP_BPS, PREREG_1478 amendment, filled bps
2.9/3.2 TRAIN/VAL, tail bps 94/76); target = exact limit fill (no extra cost); EOD exit slipped at
1,443's measured EOD-holdout mean bps (TRAIN-H2 11.5, VAL 9.7). R'' < 0.5% of price -> excluded
from the primary book (below_floor) but still reported.

Rule (1,488): the SAME eligibility. If eligible and entered, add 2/3 of the position at the ask of
the same m0+16 bar (base fill already held 1/3 from its own entry); move stop for the WHOLE
position to level-0.01; target = base fill + 2*R_original (R_original = causal book's own R column,
i.e. the ORIGINAL risk unit, not R''); walk the same path from m0+16; net R booked in ORIGINAL R
units against the blended entry (1/3*base_fill + 2/3*ask); paired delta = net_1488 - base_outcome_R.
"""
import math
import sqlite3
import sys
import time

import numpy as np
import pandas as pd

HERE = '/home/ec2-user/onemil/research/hod_entry'
OPEN_M, EOD_M = 570, 955
SLIP_STOP_BPS = {'TRAIN': 0.88 * 2.9 + 0.12 * 94.0, 'VAL': 0.88 * 3.2 + 0.12 * 76.0}
EOD_BPS = {'TRAIN': 11.5, 'VAL': 9.7}


def log(msg):
    print(f'[{time.strftime("%H:%M:%S")}] {msg}', flush=True)


def load_base():
    """Base book: causal_arming_causal.csv rows status=='fill', joined with half_entry
    (features_1478_A.csv) and base_outcome_R (model_1478_predictions.csv's outcome_R -- the
    materialized output of cell_1478.py's build_outcome). Both joins are on (day, symbol), each
    verified 1:1 with zero unmatched rows before this script was written."""
    df = pd.read_csv(f'{HERE}/causal_arming_causal.csv', low_memory=False)
    fills = df[df.status == 'fill'].copy()
    log(f'base fills: {len(fills)}')

    feat = pd.read_csv(f'{HERE}/features_1478_A.csv', usecols=['day', 'symbol', 'half_entry'])
    fills = fills.merge(feat, on=['day', 'symbol'], how='left', validate='one_to_one')
    assert fills['half_entry'].isna().sum() == 0, 'half_entry join gap'

    pred = pd.read_csv(f'{HERE}/model_1478_predictions.csv', usecols=['day', 'symbol', 'outcome_R'])
    fills = fills.merge(pred.rename(columns={'outcome_R': 'base_outcome_R'}), on=['day', 'symbol'],
                         how='left', validate='one_to_one')
    assert fills['base_outcome_R'].isna().sum() == 0, 'base_outcome_R join gap'

    fills['m0'] = np.floor(fills['fill_min']).astype(int)
    return fills.reset_index(drop=True)


def load_bars(symbols_days):
    """Load only the bars for the (symbol, day) pairs the base book needs, with an ET minute
    column, grouped into per-(symbol,day) arrays sorted by minute for O(1) walk lookups."""
    con = sqlite3.connect(f'{HERE}/bars_fills_1478.db')
    days = sorted(set(d for _, d in symbols_days))
    log(f'loading bars for {len(days)} distinct days...')
    chunks = []
    q = ('SELECT symbol, day, t, o, h, l, c FROM bars WHERE day IN ({})'
         .format(','.join('?' * len(days))))
    for chunk in pd.read_sql_query(q, con, params=days, chunksize=500_000):
        chunks.append(chunk)
    bars = pd.concat(chunks, ignore_index=True)
    con.close()
    log(f'bars loaded: {len(bars)} rows')

    ts = pd.to_datetime(bars['t'], utc=True).dt.tz_convert('America/New_York')
    bars['m'] = ts.dt.hour * 60 + ts.dt.minute
    bars = bars.sort_values(['symbol', 'day', 'm'])
    dup = bars.duplicated(subset=['symbol', 'day', 'm']).sum()
    if dup:
        log(f'WARNING: {dup} duplicate (symbol,day,minute) bars -- keeping first')
        bars = bars.drop_duplicates(subset=['symbol', 'day', 'm'], keep='first')

    groups = {}
    for (sym, day), g in bars.groupby(['symbol', 'day'], sort=False):
        groups[(sym, day)] = g[['m', 'o', 'h', 'l', 'c']].to_numpy()
    log(f'grouped into {len(groups)} (symbol,day) bar series')
    return groups


def walk_path(entry_m, entry_px, stop, target, arr):
    """sip_rebuild.walk_path semantics reproduced from its docstring/code: iterate bars with
    m >= entry_m in order; m >= EOD_M -> exit at that bar's open, 'eod'; low <= stop -> stop
    (gap-through at the open if the open itself is already <= stop), 'stop'; high >= target ->
    target, 'target'. If the path runs out before EOD_M, exit at the last bar's close, 'eod_fallback'
    (logged, per the original's own WARNING)."""
    rows = arr[arr[:, 0] >= entry_m]
    if len(rows) == 0:
        return None
    for m, o, h, l, c in rows:
        if m >= EOD_M:
            return int(m), float(o), 'eod'
        if l <= stop:
            return int(m), float(o if o <= stop else stop), 'stop'
        if h >= target:
            return int(m), float(target), 'target'
    last = rows[-1]
    return int(last[0]), float(last[3 + 1]), 'eod_fallback'  # last[4] = c


def slip_bps(why, split):
    if why in ('stop', 'stop_bar'):
        return SLIP_STOP_BPS[split]
    if why in ('eod', 'eod_fallback'):
        return EOD_BPS[split]
    return 0.0  # target: exact limit fill, no extra cost


def net_r(entry, exit_price, risk, why, split):
    raw = (exit_price - entry) / risk
    slip = exit_price * slip_bps(why, split) / 1e4 / risk
    return raw - slip, raw, slip


def process(fills, bar_groups):
    out_rows = []
    n_no_bar_group = 0
    n_no_entry_bar = 0
    n_eod_fallback = 0
    for r in fills.itertuples():
        m0 = r.m0
        key = (r.symbol, r.day)
        arr = bar_groups.get(key)
        row = dict(day=r.day, symbol=r.symbol, split=r.split, fill_min=r.fill_min,
                   base_exit_m=r.exit_m, base_why=r.why, base_outcome_R=r.base_outcome_R,
                   level=r.level, m0=m0)
        if arr is None:
            n_no_bar_group += 1
            row.update(eligible=False, reason='no_bar_store_row', entry=np.nan, stop2=np.nan,
                       R2=np.nan, exit_m2=np.nan, exit_price2=np.nan, why2=None,
                       net_R_dprime=np.nan, r2_pct_price=np.nan, below_floor=None)
            out_rows.append(row)
            continue

        window = arr[(arr[:, 0] > m0) & (arr[:, 0] <= m0 + 15)]
        dip = bool((window[:, 3] <= (r.level - 0.01)).any()) if len(window) else False
        if dip:
            row.update(eligible=False, reason='dip_in_window', entry=np.nan, stop2=np.nan,
                       R2=np.nan, exit_m2=np.nan, exit_price2=np.nan, why2=None,
                       net_R_dprime=np.nan, r2_pct_price=np.nan, below_floor=None)
            out_rows.append(row)
            continue
        if not (r.exit_m > m0 + 15):
            row.update(eligible=False, reason='base_exited_by_15', entry=np.nan, stop2=np.nan,
                       R2=np.nan, exit_m2=np.nan, exit_price2=np.nan, why2=None,
                       net_R_dprime=np.nan, r2_pct_price=np.nan, below_floor=None)
            out_rows.append(row)
            continue

        entry_bar = arr[arr[:, 0] == m0 + 16]
        if len(entry_bar) == 0:
            n_no_entry_bar += 1
            row.update(eligible=True, reason='no_entry_bar', entry=np.nan, stop2=np.nan, R2=np.nan,
                       exit_m2=np.nan, exit_price2=np.nan, why2=None, net_R_dprime=np.nan,
                       r2_pct_price=np.nan, below_floor=None)
            out_rows.append(row)
            continue

        entry = float(entry_bar[0, 1]) + float(r.half_entry)
        stop2 = r.level - 0.01
        R2 = entry - stop2
        if R2 <= 0:
            row.update(eligible=True, reason='nonpositive_R2', entry=entry, stop2=stop2, R2=R2,
                       exit_m2=np.nan, exit_price2=np.nan, why2=None, net_R_dprime=np.nan,
                       r2_pct_price=np.nan, below_floor=None)
            out_rows.append(row)
            continue

        target = entry + 2 * R2
        wr = walk_path(m0 + 16, entry, stop2, target, arr)
        exit_m2, exit_price2, why2 = wr
        if why2 == 'eod_fallback':
            n_eod_fallback += 1
        net_R2, raw2, slip2 = net_r(entry, exit_price2, R2, why2, r.split)
        r2_pct = R2 / entry * 100.0
        row.update(eligible=True, reason='entered', entry=entry, stop2=stop2, R2=R2,
                   exit_m2=exit_m2, exit_price2=exit_price2, why2=why2, net_R_dprime=net_R2,
                   r2_pct_price=r2_pct, below_floor=bool(r2_pct < 0.5))
        out_rows.append(row)

    log(f'no bar-store row for the (symbol,day): {n_no_bar_group}')
    log(f'no entry bar at m0+16: {n_no_entry_bar}')
    log(f'eod_fallback (path ran out before 15:55): {n_eod_fallback}')
    return pd.DataFrame(out_rows), fills


def process_1488(fills_df, out1487, bar_groups):
    """1,488 pyramid: only for rows eligible & reason=='entered' in 1,487 (position was still open,
    no dip, an entry bar existed, R2 positive). Base fill weight 1/3 (already open at its own
    entry/R since the fill bar), add-on 2/3 at the SAME m0+16 ask; stop moves to level-0.01 for the
    whole position; target = base_fill + 2*R_original (the ORIGINAL causal-book risk unit); walked
    from m0+16 with the same physics; net R booked in ORIGINAL R units vs the blended entry."""
    base = fills_df.set_index(['day', 'symbol'])
    rows = []
    entered = out1487[out1487.reason == 'entered']
    for r in entered.itertuples():
        b = base.loc[(r.day, r.symbol)]
        base_fill = float(b.fill)
        R_orig = float(b.R)
        arr = bar_groups[(r.symbol, r.day)]
        blended_entry = (1.0 / 3.0) * base_fill + (2.0 / 3.0) * r.entry
        stop_w = r.stop2  # same level-0.01
        target_w = base_fill + 2.0 * R_orig
        exit_m3, exit_price3, why3 = walk_path(int(r.m0) + 16, blended_entry, stop_w, target_w, arr)
        net_1488, raw3, slip3 = net_r(blended_entry, exit_price3, R_orig, why3, r.split)
        delta = net_1488 - r.base_outcome_R
        rows.append(dict(day=r.day, symbol=r.symbol, split=r.split, fill_min=r.fill_min,
                          exit_m3=exit_m3, exit_price3=exit_price3, why3=why3,
                          net_1488=net_1488, base_outcome_R=r.base_outcome_R, delta=delta))
    return pd.DataFrame(rows)


def compare(mine, theirs, key_cols, val_col_mine, val_col_theirs, label):
    """Cohort Jaccard on `key_cols`, share within 0.01 R on the joined rows, VAL means both sides."""
    mkeys = set(map(tuple, mine[key_cols].round(6).to_numpy()))
    tkeys = set(map(tuple, theirs[key_cols].round(6).to_numpy()))
    inter = mkeys & tkeys
    union = mkeys | tkeys
    jacc = len(inter) / len(union) if union else float('nan')

    j = mine.merge(theirs, on=key_cols, suffixes=('_mine', '_theirs'))
    col_mine = val_col_mine + '_mine' if val_col_mine + '_mine' in j.columns else val_col_mine
    col_theirs = val_col_theirs + '_theirs' if val_col_theirs + '_theirs' in j.columns else val_col_theirs
    diff = (j[col_mine] - j[col_theirs]).abs()
    share = float((diff <= 0.01).mean()) if len(j) else float('nan')

    val_mine = mine[mine.split == 'VAL'][val_col_mine].mean()
    val_theirs = theirs[theirs.split == 'VAL'][val_col_theirs].mean()
    log(f'[{label}] mine n={len(mine)} theirs n={len(theirs)} joined n={len(j)} '
        f'jaccard={jacc:.4f} share_within_0.01R={share:.4f} '
        f'VAL_mean_mine={val_mine:.4f} VAL_mean_theirs={val_theirs:.4f}')
    return dict(label=label, jaccard=jacc, share_0_01=share, val_mine=val_mine,
                val_theirs=val_theirs, n_mine=len(mine), n_theirs=len(theirs), n_joined=len(j),
                max_abs_diff=float(diff.max()) if len(j) else float('nan'))


def main():
    fills = load_base()
    bar_groups = load_bars(list(zip(fills.symbol, fills.day)))
    out1487, fills = process(fills, bar_groups)
    out1488 = process_1488(fills, out1487, bar_groups)

    out1487.to_csv(f'{HERE}/rebuild_1487_fills.csv', index=False)
    out1488.to_csv(f'{HERE}/rebuild_1488_fills.csv', index=False)
    log(f'wrote rebuild_1487_fills.csv ({len(out1487)}) and rebuild_1488_fills.csv ({len(out1488)})')

    log('--- eligibility/reason breakdown (mine) ---')
    log(str(out1487['reason'].value_counts()))
    entered_primary = out1487[(out1487.reason == 'entered') & (out1487.below_floor == False)]
    log(f'primary book (entered, R>=0.5% floor): n={len(entered_primary)}')

    theirs1487 = pd.read_csv(f'{HERE}/cell_1487_fills.csv', low_memory=False)
    theirs1488 = pd.read_csv(f'{HERE}/cell_1488_fills.csv', low_memory=False)

    # Primary-book comparison (entered & not below_floor) on both sides.
    mine_primary = out1487[(out1487.reason == 'entered') & (~out1487.below_floor.astype('boolean').fillna(False))]
    theirs_primary = theirs1487[(theirs1487.reason == 'entered') & (theirs1487.below_floor == False)]
    r1 = compare(mine_primary, theirs_primary, ['day', 'symbol'], 'net_R_dprime', 'net_R2',
                 '1487 primary (entered, not below_floor)')

    # All 'entered' rows (incl. below_floor) for a wider check.
    mine_entered = out1487[out1487.reason == 'entered']
    theirs_entered = theirs1487[theirs1487.reason == 'entered']
    r2 = compare(mine_entered, theirs_entered, ['day', 'symbol'], 'net_R_dprime', 'net_R2',
                 '1487 all entered')

    r3 = compare(out1488, theirs1488, ['day', 'symbol'], 'net_1488', 'net_1488', '1488 pyramid')

    # eligible/reason agreement (categorical), for the caveats section
    merged_reason = out1487.merge(theirs1487[['day', 'symbol', 'eligible', 'reason']],
                                   on=['day', 'symbol'], suffixes=('_mine', '_theirs'))
    reason_agree = (merged_reason.reason_mine == merged_reason.reason_theirs).mean()
    elig_agree = (merged_reason.eligible_mine.astype(bool) ==
                  merged_reason.eligible_theirs.astype(bool)).mean()
    log(f'reason agreement (all 9911): {reason_agree:.4f}; eligible agreement: {elig_agree:.4f}')

    disagree = merged_reason[merged_reason.reason_mine != merged_reason.reason_theirs]
    log(f'reason DISAGREEMENTS: {len(disagree)}')
    if len(disagree):
        log(disagree[['day', 'symbol', 'reason_mine', 'reason_theirs']].head(15).to_string())

    import json
    summary = dict(reason_agree=float(reason_agree), elig_agree=float(elig_agree),
                    primary=r1, all_entered=r2, pyramid_1488=r3,
                    n_disagree_reason=int(len(disagree)))
    with open(f'{HERE}/rebuild_1487_summary.json', 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    log('wrote rebuild_1487_summary.json')
    log('DONE')


if __name__ == '__main__':
    main()
