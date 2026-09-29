"""
Cell 1,661: re-score the HOD dry-run entry limit width on the dry cross records since 2026-09-26.

logs/hod_dry_entry_ledger.csv stopped being written after 2026-09-25 (174 rows total, all dated
2026-09-25; every row from line 176 on is also malformed -- 14 fields where the header has 13 --
a logging regression, not a data choice). The ONLY source for 9/26+ arms is the journal:
`journalctl -u onemil-trader --since 2026-09-26 | grep "[HOD DRY]"` (3,051 ARMED + 129 CROSS lines,
captured to the scratchpad). ARMED carries level/trigger/limit/stop; CROSS carries the print/ask and
the FILL/NO-FILL outcome under the LIVE limit (entry_limit_pct=0.0015, trading/hod_break.py:44/127:
limit = round(level * (1 + entry_limit_pct), 6), fill iff ask <= limit, fill_px = ask). Re-scoring a
different limit_pct is a pure counterfactual on the SAME (level, ask) pair: hyp_limit =
level*(1+pct/100); filled iff ask <= hyp_limit; fill_px = ask (unchanged -- no chase).

Outcomes (for net R of the filled set) come from logs/hod_dry_counterfactuals.csv, which only has
rows for arms that filled under today's live 0.15% limit AND only from 2026-09-28 on (cf-watch
logging is newer than the 9/26 journal window) -- a real coverage gap, reported not hidden.
"""
import re
import numpy as np
import pandas as pd

SCRATCH = '/tmp/claude-1000/-home-ec2-user-onemil/257c3e2d-cf38-45d5-94e7-4877f8170f44/scratchpad'
LOGF = f'{SCRATCH}/1661_hod_dry_lines.txt'
OUT = '/home/ec2-user/onemil/research/hod_entry'

TS_RE = re.compile(r'(\d{4}-\d{2}-\d{2}) (\d{2}:\d{2}:\d{2})')
ARM_RE = re.compile(r'ARMED (\S+) level ([\d.]+) trigger ([\d.]+) limit ([\d.]+) stop ([\d.]+)')
CROSS_RE = re.compile(
    r'CROSS (\S+) \(tape\) at (\S+) print ([\d.]+) size (\d+) ask ([\d.]+) -> '
    r'(?:FILL ([\d.]+)|NO FILL)')

LIMIT_PCTS = [0.05, 0.10, 0.15, 0.25]  # matches PREREG_1660 section 1,661


def parse():
    last_arm = {}  # symbol -> dict(date, level, trigger, limit, stop)
    crosses = []
    n_cross_no_arm = 0
    with open(LOGF) as f:
        for line in f:
            m = TS_RE.search(line)
            if not m:
                continue
            date = m.group(1)
            a = ARM_RE.search(line)
            if a:
                sym, level, trigger, limit, stop = a.groups()
                last_arm[sym] = dict(date=date, level=float(level), trigger=float(trigger),
                                      limit=float(limit), stop=float(stop))
                continue
            c = CROSS_RE.search(line)
            if c:
                sym, at, prnt, size, ask, fillpx = c.groups()
                arm = last_arm.get(sym)
                if arm is None or arm['date'] != date:
                    n_cross_no_arm += 1
                    continue
                crosses.append(dict(date=date, symbol=sym, at=at, print_px=float(prnt),
                                     ask=float(ask), live_filled=fillpx is not None,
                                     live_fill_px=float(fillpx) if fillpx else np.nan,
                                     level=arm['level'], trigger=arm['trigger'],
                                     live_limit=arm['limit'], stop=arm['stop']))
    return pd.DataFrame(crosses), n_cross_no_arm


def main():
    df, n_orphan = parse()
    print(f'parsed {len(df)} CROSS events with a same-day ARMED match ({n_orphan} orphaned)')
    # sanity check against the live 0.15% rule the journal itself already resolved
    df['limit_015_check'] = (df.level * 1.0015).round(6)
    mismatch = (df.live_filled != (df.ask <= df.limit_015_check + 1e-9)).sum()
    print(f'live-rule reproduction mismatches: {mismatch} / {len(df)}')

    cf = pd.read_csv('/home/ec2-user/onemil/logs/hod_dry_counterfactuals.csv',
                      on_bad_lines='warn', engine='python')
    cf = cf.rename(columns={'symbol': 'symbol'})
    # one exit outcome per (date, symbol); if several cf watches on the same symbol-day, keep the first (arm order)
    cf = cf.sort_values('fill_ts').drop_duplicates(subset=['date', 'symbol'], keep='first')
    cf_idx = cf.set_index(['date', 'symbol'])

    rows = []
    detail = []
    for pct in LIMIT_PCTS:
        hyp_limit = (df.level * (1.0 + pct / 100.0)).round(6)
        filled = df.ask <= hyp_limit + 1e-9
        fill_px = df.ask.where(filled)
        slip_pct = (fill_px - df.trigger) / df.trigger * 100.0
        sub = df[filled].copy()
        sub['fill_px'] = fill_px[filled]
        sub['slip_pct'] = slip_pct[filled]
        # join realized outcome
        joined = sub.set_index(['date', 'symbol']).join(
            cf_idx[['actual_stop', 'exit_px', 'exit_reason']], how='left').reset_index()
        stop_use = joined['actual_stop'].where(joined['actual_stop'].notna(), joined['stop'])
        joined['R_dollar'] = joined.fill_px - stop_use  # stop distance at THIS hypothetical fill price
        has_outcome = joined['exit_px'].notna() & (joined.R_dollar > 0)
        wo = joined[has_outcome].copy()
        wo['net_R'] = (wo.exit_px - wo.fill_px) / wo.R_dollar
        rows.append(dict(
            limit_pct=pct, n_arms=len(df), n_filled=int(filled.sum()),
            fill_rate=float(filled.mean()), mean_slip_pct=float(slip_pct[filled].mean()) if filled.sum() else np.nan,
            n_with_outcome=len(wo), net_R_filled=float(wo.net_R.mean()) if len(wo) else np.nan,
        ))
        for _, r in joined.iterrows():
            detail.append(dict(limit_pct=pct, date=r.date, symbol=r.symbol, level=r.level,
                                trigger=r.trigger, ask=r.ask, fill_px=r.fill_px, slip_pct=r.slip_pct,
                                exit_px=r.get('exit_px'), exit_reason=r.get('exit_reason'),
                                stop=r.stop))

    res = pd.DataFrame(rows)
    res.to_csv(f'{OUT}/1661_limit_rescore.csv', index=False)
    pd.DataFrame(detail).to_csv(f'{OUT}/1661_filled_detail.csv', index=False)
    print(res.to_string(index=False))

    # verdict vs the frozen reading rule: narrower limit beats 0.15% net R by >=0.03R with fill count >= 70% of 0.15%'s
    base = res[res.limit_pct == 0.15].iloc[0]
    print('\nreading-rule check (narrower than 0.15% only):')
    for _, r in res[res.limit_pct < 0.15].iterrows():
        ok_r = (not np.isnan(r.net_R_filled)) and (not np.isnan(base.net_R_filled)) and \
               (r.net_R_filled >= base.net_R_filled + 0.03)
        ok_n = r.n_filled >= 0.70 * base.n_filled
        print(f"  {r.limit_pct}%: dNetR={r.net_R_filled - base.net_R_filled if not np.isnan(r.net_R_filled) else np.nan}, "
              f"n_filled {r.n_filled} vs base {base.n_filled} ({ok_n}), candidate={ok_r and ok_n}")


if __name__ == '__main__':
    main()
