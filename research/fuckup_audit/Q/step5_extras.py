#!/usr/bin/env python3
"""Step 5 — the checks CLAUDE.md requires that steps 1-4 did not cover.

  a) price-scale / split check on the three symbols (a raw-bar split would fabricate bands)
  b) gross reference + breakeven cost per leg
  c) tail: both tails, winner cap, top-5-day share, concentration
  d) power: the smallest per-day effect the OOS window could have seen
  e) report-only variants: flat at 15:30 (the last-30-min window is a drag OOS), entry-minute P&L
  f) trades/day, hold time, and the obtainability restatement

Writes Q/step5_extras.md and Q/step5_*.csv
"""
import sys
import math
import numpy as np
import pandas as pd

sys.path.insert(0, '/home/ec2-user/onemil/research/fuckup_audit/Q')
import zsim as Z

Q = '/home/ec2-user/onemil/research/fuckup_audit/Q/'
EQUITY = 60_000.0
L = []


def say(s=''):
    print(s, flush=True)
    L.append(s)


def split_check(data):
    """prev_close -> open ratio outliers: a raw (unadjusted) split shows as a ~0.5x / 2x jump."""
    r = data['dopen'] / data['prevclose']
    bad = np.where((r < 0.75) | (r > 1.33))[0]
    return [(str(data['days'][i])[:10], float(data['prevclose'][i]), float(data['dopen'][i]),
             float(r[i])) for i in bad]


def main():
    say('# Step 5 — split/price-scale, breakeven cost, tails, power, report-only variants')
    say()

    # ---------------- a) split / price-scale -------------------------------------
    say('## a) Price-scale check (all bars come from ONE source: Alpaca SIP 1-min, `etf_1min.db`)')
    say()
    say('| symbol | prev_close -> open jumps outside 0.75x..1.33x | dates |')
    say('|---|---|---|')
    keep = {}
    for sym in ('SPY', 'QQQ', 'TQQQ'):
        d = Z.load_symbol(sym)
        j = split_check(d)
        say(f'| {sym} | {len(j)} | ' + ('; '.join(f'{a} {b:.2f}->{c:.2f} ({r:.3f}x)' for a, b, c, r in j)
                                        if j else '—') + ' |')
        keep[sym] = d
    say()
    say('This table is the RESIDUAL after `zsim.load_symbol` back-adjusts the five known TQQQ splits')
    say('(2017-01-12 2:1, 2018-05-24 3:1, 2021-01-21 2:1, 2022-01-13 2:1, 2025-11-20 2:1 -- factors derived as')
    say('`(TQQQ open/prev_close) / (1 + 3 x QQQ gap)` = 0.4996..0.5000 / 0.3333). The one remaining row,')
    say('2020-03-16 at 1.0225 implied, is the COVID crash gap and is correctly NOT treated as a split.')
    say()
    say('No daily-file / intraday-file price comparison exists in this rule (bands, VWAP, signal and fill all')
    say('come from the same 1-min table), so the split risk is confined to the 14-day sigma and prev-close')
    say('anchor crossing a split date.')
    say()

    # ---------------- b) gross + breakeven cost ----------------------------------
    say('## b) Gross reference and the breakeven cost per leg (QQQ, live fill, MOC flat)')
    say()
    say('| period | cost/leg (bp) | bps/traded day 1x | t | ann % 1x | SR 1x | MDD % 1x |')
    say('|---|---|---|---|---|---|---|')
    rows = []
    d = keep['QQQ']
    UB, LB = Z.bands(d, 1.0)
    for slip in (0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0):
        tr, _ = Z.simulate(d, UB, LB, fill='next_open', eod='moc', slip_bp=slip)
        df = Z.daily_returns(d, tr, 'gross')
        for lab, part in (('IS 2016-2023', Z.split(df)[0]), ('OOS 2024-2026', Z.split(df)[1])):
            m = Z.metrics(part, 'r1x')
            rows.append(dict(period=lab, slip_bp=slip, **{k: m[k] for k in
                                                          ('bps', 't', 'ann', 'sharpe', 'mdd', 'trades_day')}))
            say(f'| {lab} | {slip} | {m["bps"]:.2f} | {m["t"]:.2f} | {m["ann"]:.2f} | {m["sharpe"]:.2f} '
                f'| {m["mdd"]:.1f} |')
    pd.DataFrame(rows).to_csv(Q + 'step5_costcurve.csv', index=False)
    rr = pd.DataFrame(rows)
    for lab in ('IS 2016-2023', 'OOS 2024-2026'):
        g = rr[rr['period'] == lab]
        # linear interpolation of bps vs slip to zero
        x, y = g['slip_bp'].values, g['bps'].values
        be = np.interp(0.0, y[::-1], x[::-1]) if y[0] > 0 > y[-1] else float('nan')
        say(f'\nBreakeven cost per leg, {lab}: **{be:.2f} bp** (gross {y[0]:.2f} bps/traded day).')
    say()

    # ---------------- c) tails ---------------------------------------------------
    say('## c) Tail dependence (QQQ, live fill + 0.5 bp/leg, 1x)')
    say()
    tr, _ = Z.simulate(d, UB, LB, fill='next_open', eod='moc', slip_bp=0.5)
    df = Z.daily_returns(d, tr, 'gross')
    trows = []
    for lab, part in (('IS 2016-2023', Z.split(df)[0]), ('OOS 2024-2026', Z.split(df)[1])):
        base = part['r1x'].values
        tot = base.sum()
        traded = part[part['ntr'] > 0]['r1x'].values
        srt = np.sort(traded)[::-1]
        variants = {
            'full': base,
            'top 1% days removed': part[part['r1x'] < part['r1x'].quantile(0.99)]['r1x'].values,
            'top 5% days removed': part[part['r1x'] < part['r1x'].quantile(0.95)]['r1x'].values,
            'bottom 5% days removed': part[part['r1x'] > part['r1x'].quantile(0.05)]['r1x'].values,
            'both 5% tails removed': part[(part['r1x'] > part['r1x'].quantile(0.05)) &
                                          (part['r1x'] < part['r1x'].quantile(0.95))]['r1x'].values,
            'daily return capped at +1%': np.minimum(base, 0.01),
            'daily return capped at +0.5%': np.minimum(base, 0.005),
        }
        for name, v in variants.items():
            m = v.mean() * 1e4
            t = m / (v.std(ddof=1) * 1e4) * math.sqrt(len(v)) if v.std(ddof=1) > 0 else np.nan
            trows.append(dict(period=lab, variant=name, days=len(v), bps_per_calendar_day=m, t=t,
                              usd_per_month_60k=v.mean() * EQUITY * 21))
        say(f'**{lab}** — total return {100*tot:.1f}%; top 5 days = {100*srt[:5].sum()/tot:.0f}% of it; '
            f'top 1% of days = {100*srt[:max(1,int(0.01*len(srt)))].sum()/tot:.0f}%; '
            f'top 5% = {100*srt[:max(1,int(0.05*len(srt)))].sum()/tot:.0f}%.')
    say()
    tdf = pd.DataFrame(trows)
    tdf.to_csv(Q + 'step5_tails.csv', index=False)
    say('| period | variant | days | bps/calendar day | t | $/month at $60K |')
    say('|---|---|---|---|---|---|')
    for _, r in tdf.iterrows():
        say(f'| {r["period"]} | {r["variant"]} | {r["days"]} | {r["bps_per_calendar_day"]:.2f} | '
            f'{r["t"]:.2f} | {r["usd_per_month_60k"]:.0f} |')
    say()

    # ---------------- d) power ---------------------------------------------------
    say('## d) Power — the smallest daily effect this OOS window could have seen')
    say()
    for lab, part in (('IS 2016-2023', Z.split(df)[0]), ('OOS 2024-2026', Z.split(df)[1])):
        sd = part['r1x'].std(ddof=1) * 1e4
        n = len(part)
        mde = 2.8 * sd / math.sqrt(n)
        say(f'- {lab}: n = {n} days, daily sd = {sd:.1f} bps; MDE at 80% power / 5% two-sided = '
            f'**{mde:.2f} bps/day** = {mde*1e-4*EQUITY*21:.0f} $/month at $60K 1x. '
            f'Observed = {part["r1x"].mean()*1e4:.2f} bps/day.')
    say()

    # ---------------- e) report-only variants ------------------------------------
    say('## e) Report-only variants (NOT adopted — each is an extra cell)')
    say()
    say('| variant | period | bps/traded day 1x | t | ann % 1x | SR 1x | MDD % 1x | trades/day |')
    say('|---|---|---|---|---|---|---|---|')
    for name, checks in (('flat at 15:30 (skip the last 30 min)', list(range(30, 361, 30))),
                         ('paper cadence (flat 15:59)', Z.CHECKS_SEMI)):
        # 'flat at 15:30' = no new entries after 15:00 and the EOD exit forced at 15:30
        tr2, _ = Z.simulate(d, UB, LB, checks=checks, fill='next_open', eod='moc', slip_bp=0.5)
        if name.startswith('flat at 15:30'):
            tr2 = [t for t in tr2]
            # force any position still open to close at 15:30's next open instead of 15:59
            for t in tr2:
                if t['why'] == 'eod':
                    k = 361
                    px = d['O'][t['d'], k]
                    if np.isnan(px):
                        px = d['C'][t['d'], 360]
                    if np.isnan(px):
                        px = d['C'][t['d'], d['last'][t['d']]]
                        k = int(d['last'][t['d']])
                    t['x'] = px * (1 - 0.5e-4) if t['side'] > 0 else px * (1 + 0.5e-4)
                    t['xk'] = k
        df2 = Z.daily_returns(d, tr2, 'gross')
        for lab, part in (('IS 2016-2023', Z.split(df2)[0]), ('OOS 2024-2026', Z.split(df2)[1])):
            m = Z.metrics(part, 'r1x')
            say(f'| {name} | {lab} | {m["bps"]:.2f} | {m["t"]:.2f} | {m["ann"]:.2f} | {m["sharpe"]:.2f} '
                f'| {m["mdd"]:.1f} | {m["trades_day"]:.2f} |')
    say()

    # entry-minute P&L
    pe = Z.pnl_by_entry_minute(tr)
    pe['date'] = [d['days'][t['d']] for t in tr]
    pe['oos'] = pe['date'] >= Z.OOS_START
    g = pe.groupby(['oos', 'ek'])['pnl'].agg(['count', 'sum', 'mean']).reset_index()
    g.to_csv(Q + 'step5_entry_minute.csv', index=False)
    say('## f) Gross $/share P&L by ENTRY minute index (k = ET minute − 570; 31 = 10:01 fill)')
    say()
    say('| period | k | ET | trades | sum $/sh | mean $/sh |')
    say('|---|---|---|---|---|---|')
    for _, r in g.iterrows():
        k = int(r['ek'])
        et = f'{(570+k)//60:02d}:{(570+k)%60:02d}'
        say(f'| {"OOS" if r["oos"] else "IS"} | {k} | {et} | {int(r["count"])} | {r["sum"]:.2f} | {r["mean"]:.4f} |')
    say()

    # hold time
    hold = np.array([t['xk'] - t['ek'] for t in tr])
    say(f'Hold time: median {np.median(hold):.0f} min, mean {hold.mean():.0f} min, '
        f'p90 {np.percentile(hold,90):.0f} min. Round trips/day OOS '
        f'{Z.split(df)[1]["ntr"].sum()/len(Z.split(df)[1]):.2f}.')
    say()

    with open(Q + 'step5_extras.md', 'w') as f:
        f.write('\n'.join(L) + '\n')
    print('wrote', Q + 'step5_extras.md')


if __name__ == '__main__':
    main()
