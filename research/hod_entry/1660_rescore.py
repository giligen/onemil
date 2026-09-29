"""
Cell 1,660: re-score the HOD exit lab's 35 cells (X1-X12/X5b/X7b, D1-D5, W1-W2, T1-T2, M1-M2,
E1, O1-O2, S1-S2) on the SAME fills under the MEASURED cost (PREREG_1660.md):
  entry leg   = 7 bps of entry price   (always charged)
  stop exit   = 6 bps of exit price
  target exit = 0 bps (resting limit fills at the touch)
  eod exit    = 11 bps of exit price (bid line)  OR  1 bp (MOC line, second column)
  overnight exit (O1 held subset only) = same eod bucket, flagged as an explicit assumption
    (the PREREG's cost table has no "overnight" leg; the next-day open is a crossing exit, not a
    resting order, so it is closest in character to the EOD leg -- NOT one of the 4 defined legs).
Each leg's bps-of-price is converted to R with the fill's OWN stop distance: R$ = entry - stop.
TEST split is never touched (sealed). Read-only on every input.
"""
import sys
import numpy as np
import pandas as pd

LAB = '/home/ec2-user/onemil/research/hod_exit_lab'
OUT = '/home/ec2-user/onemil/research/hod_entry'
sys.path.insert(0, LAB)
from score_cells import day_clustered_t, ex_top5  # noqa: E402  (reuse exact lab conventions)

ENTRY_BPS = 0.0007
EXIT_BPS_FIXED = {'stop': 0.0006, 'target': 0.0}
EOD_LINES = {'bid': 0.0011, 'moc': 0.0001}
FLOOR_PCTS = {'none': 0.0, 'f15': 1.5, 'f30': 3.0}
UNKNOWN_WHY_SEEN = set()


def exit_rate(why, eod_bps):
    if why in EXIT_BPS_FIXED:
        return EXIT_BPS_FIXED[why]
    if why in ('eod', 'overnight'):
        return eod_bps
    UNKNOWN_WHY_SEEN.add(why)
    return eod_bps  # conservative fallback, logged below


def score_population(entry, stop, exit_price, why, day, wk, floor_mask=None):
    """Return dict of {line: {metric: value}} for one population (already the 'kept' set)."""
    R = entry - stop
    raw_rr = (exit_price - entry) / R
    out = {}
    for line, eod_bps in EOD_LINES.items():
        rate = why.map(lambda w: exit_rate(w, eod_bps))
        cost_R = (entry * ENTRY_BPS + exit_price * rate) / R
        net_R = raw_rr - cost_R
        n = len(net_R)
        if n == 0:
            out[line] = dict(n=0, mean=np.nan, t=np.nan, ex_top5=np.nan, fills_wk=0.0)
            continue
        t, ndays = day_clustered_t(net_R, day)
        et5 = ex_top5(net_R)
        nwk = wk.nunique()
        out[line] = dict(n=n, mean=float(net_R.mean()), t=t, ex_top5=et5,
                          fills_wk=float(n / nwk) if nwk else 0.0, net_R=net_R)
    return out, R


def split_rows(df, entry_c, stop_c, exit_c, why_c):
    rows = []
    for split in ('TRAIN', 'VAL'):
        pop = df[df.split == split]
        r_pct = (pop[entry_c] - pop[stop_c]) / pop[entry_c] * 100.0
        for floor_name, floor_val in FLOOR_PCTS.items():
            kept = pop[r_pct >= floor_val]
            if len(kept) == 0:
                for line in EOD_LINES:
                    rows.append(dict(split=split, floor=floor_name, line=line, n=0, mean=np.nan,
                                      t=np.nan, ex_top5=np.nan, fills_wk=0.0))
                continue
            res, R = score_population(kept[entry_c], kept[stop_c], kept[exit_c], kept[why_c],
                                       kept.day, kept.wk)
            for line, m in res.items():
                rows.append(dict(split=split, floor=floor_name, line=line, n=m['n'], mean=m['mean'],
                                  t=m['t'], ex_top5=m['ex_top5'], fills_wk=m['fills_wk'],
                                  _net_R=m.get('net_R'), _idx=kept.index))
    return rows


def main():
    b0 = pd.read_csv(f'{LAB}/b0_trades.csv')
    b0 = b0[b0.split.isin(['TRAIN', 'VAL'])].copy()
    b0_key = b0.set_index(['day', 'symbol', 'entry_m'])

    all_rows = []   # summary grid rows (one per cell x split x floor x line)
    fill_dump = {}  # cell -> per-fill DataFrame (for CSV export), bid line, floor=none only
    per_fill_parts = []  # cell-level per-fill rows (floor=none, both cost lines), for the CSV export

    def record(cell_id, kind, df, entry_c, stop_c, exit_c, why_c, base_net=None, base_idx_key=None):
        rows = split_rows(df, entry_c, stop_c, exit_c, why_c)
        none_bid = [r for r in rows if r['floor'] == 'none' and r['line'] == 'bid' and r.get('_net_R') is not None]
        none_moc = [r for r in rows if r['floor'] == 'none' and r['line'] == 'moc' and r.get('_net_R') is not None]
        for rb, rm in zip(none_bid, none_moc):  # one pair per split (TRAIN, VAL), same idx within a pair
            idx = rb['_idx']
            pf = df.loc[idx, ['day', 'symbol', 'entry_m', 'split', 'wk', entry_c, stop_c, exit_c, why_c]].copy()
            pf.columns = ['day', 'symbol', 'entry_m', 'split', 'wk', 'entry', 'stop', 'exit_price', 'why']
            pf['cell'] = cell_id
            pf['r_pct'] = (pf.entry - pf.stop) / pf.entry * 100.0
            pf['net_R_bid'] = rb['_net_R'].reindex(idx).values
            pf['net_R_moc'] = rm['_net_R'].reindex(idx).values
            per_fill_parts.append(pf)
        for r in rows:
            r2 = dict(cell=cell_id, kind=kind, **{k: v for k, v in r.items() if not k.startswith('_')})
            if base_net is not None and r.get('_net_R') is not None:
                # paired dR against the re-read B0 on the exact same rows
                idx = r['_idx']
                base_sub = base_net.reindex(idx)
                dR = r['_net_R'] - base_sub
                dR = dR.dropna()
                if len(dR):
                    day_sub = df.loc[dR.index, 'day']
                    t, _ = day_clustered_t(dR, day_sub)
                    r2['dR_mean'] = float(dR.mean())
                    r2['dR_t'] = t
                    r2['dR_ex_top5'] = ex_top5(dR)
                else:
                    r2['dR_mean'] = np.nan; r2['dR_t'] = np.nan; r2['dR_ex_top5'] = np.nan
            all_rows.append(r2)
        # stash the bid/floor-none per-fill net_R for CSV dump
        for r in rows:
            if r['line'] == 'bid' and r['floor'] == 'none' and r.get('_net_R') is not None:
                fill_dump.setdefault(cell_id, []).append(r['_net_R'])

    # ---- B0 itself (reference line for the report; also its own paired-vs-self = 0) ----
    b0_res_rows = split_rows(b0, 'entry', 'stop', 'exit_price', 'why')
    for r in b0_res_rows:
        all_rows.append(dict(cell='B0', kind='base', **{k: v for k, v in r.items() if not k.startswith('_')}))
    # b0 net_R (bid line, floor=none) per split, indexed by original row index, for pairing
    b0_bid_none = {}
    for split in ('TRAIN', 'VAL'):
        pop = b0[b0.split == split]
        R = pop.entry - pop.stop
        raw_rr = (pop.exit_price - pop.entry) / R
        rate = pop.why.map(lambda w: exit_rate(w, EOD_LINES['bid']))
        cost_R = (pop.entry * ENTRY_BPS + pop.exit_price * rate) / R
        b0_bid_none[split] = raw_rr - cost_R
    b0_bid_none_all = pd.concat(b0_bid_none.values())

    # ---- X-cells (exit variants): join entry/stop/R from B0, use own exit_price/why ----
    x_cells = ['X1', 'X2', 'X3', 'X4', 'X5', 'X5b', 'X6', 'X7', 'X7b', 'X8', 'X9', 'X10', 'X11', 'X12']
    for cid in x_cells:
        df = pd.read_csv(f'{LAB}/trades/{cid}.csv')
        df = df[df.split.isin(['TRAIN', 'VAL'])].copy()
        j = df.set_index(['day', 'symbol', 'entry_m']).join(
            b0_key[['entry', 'stop']], how='inner', rsuffix='_b0')
        j = j.reset_index()
        record(cid, 'exit', j, 'entry', 'stop', 'exit_price', 'why', base_net=b0_bid_none_all)

    # ---- D/W/T/M/S cohort cells: own entry/stop/exit_price/why/R + 'keep' ----
    cohort_cells = [('D1', 'trades'), ('D2', 'trades'), ('D3', 'trades'), ('D4', 'trades'),
                     ('D5', 'trades'), ('W1', 'trades'), ('W2', 'trades'),
                     ('T1', 'trades2'), ('T2', 'trades2'), ('M1', 'trades2'), ('M2', 'trades2'),
                     ('S1', 'trades2'), ('S2', 'trades2')]
    for cid, sub in cohort_cells:
        df = pd.read_csv(f'{LAB}/{sub}/{cid}.csv')
        df = df[df.split.isin(['TRAIN', 'VAL'])].copy()
        df = df[df['keep'] == True].copy()  # noqa: E712  cohort cells report the KEPT rows only
        if len(df) == 0:
            continue
        record(cid, 'cohort', df, 'entry', 'stop', 'exit_price', 'why', base_net=b0_bid_none_all)

    # ---- E1: retest entry, own entry/stop (new_entry, R_own) ----
    e1 = pd.read_csv(f'{LAB}/trades2/E1.csv')
    e1 = e1[e1.split.isin(['TRAIN', 'VAL'])].copy()
    e1['stop_own'] = e1.new_entry - e1.R_own
    record('E1', 'exit', e1, 'new_entry', 'stop_own', 'exit_price', 'why', base_net=b0_bid_none_all)

    # ---- O1: overnight hold; held subset exits at next_open (assumption: eod-bucket cost) ----
    o1 = pd.read_csv(f'{LAB}/trades2/O1.csv')
    o1 = o1[o1.split.isin(['TRAIN', 'VAL'])].copy()
    j = o1.set_index(['day', 'symbol', 'entry_m']).join(
        b0_key[['entry', 'stop', 'exit_price', 'why']], how='inner', rsuffix='_b0').reset_index()
    j['exit_use'] = np.where(j.held, j.next_open, j.exit_price)
    j['why_use'] = np.where(j.held, 'overnight', j.why)
    record('O1', 'exit', j, 'entry', 'stop', 'exit_use', 'why_use', base_net=b0_bid_none_all)
    # O2 VOID: no held rows exist in permitted data (see REPORT_PASS2.md) -- not scored.

    # B0 itself into the per-fill dump (cell='B0')
    pf_b0 = b0[['day', 'symbol', 'entry_m', 'split', 'wk', 'entry', 'stop', 'exit_price', 'why']].copy()
    pf_b0['cell'] = 'B0'
    pf_b0['r_pct'] = (pf_b0.entry - pf_b0.stop) / pf_b0.entry * 100.0
    pf_b0['net_R_bid'] = b0_bid_none_all.reindex(pf_b0.index).values
    rate_moc = b0.why.map(lambda w: exit_rate(w, EOD_LINES['moc']))
    R_b0 = b0.entry - b0.stop
    cost_moc_b0 = (b0.entry * ENTRY_BPS + b0.exit_price * rate_moc) / R_b0
    pf_b0['net_R_moc'] = ((b0.exit_price - b0.entry) / R_b0 - cost_moc_b0).values
    per_fill_parts.append(pf_b0)

    per_fill = pd.concat(per_fill_parts, ignore_index=True)
    per_fill.to_csv(f'{OUT}/1660_per_fill.csv', index=False)
    print('wrote', f'{OUT}/1660_per_fill.csv', 'rows=', len(per_fill))

    grid = pd.DataFrame(all_rows)
    grid.to_csv(f'{OUT}/1660_full_grid.csv', index=False)

    # per-fill CSVs (bid line, floor=none) for the cells with the largest |dR_mean| on VAL, cap file count
    for cid, series_list in fill_dump.items():
        pass  # per-fill already embedded in source trade CSVs; full_grid carries the aggregates

    print('UNKNOWN why values encountered (fell back to eod bucket):', UNKNOWN_WHY_SEEN)
    print('wrote', f'{OUT}/1660_full_grid.csv', 'rows=', len(grid))
    print('B0 bid/none TRAIN mean', b0_bid_none['TRAIN'].mean(), 'n', len(b0_bid_none['TRAIN']))
    print('B0 bid/none VAL   mean', b0_bid_none['VAL'].mean(), 'n', len(b0_bid_none['VAL']))


if __name__ == '__main__':
    main()
