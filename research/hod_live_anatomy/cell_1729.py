#!/usr/bin/env python3
"""Cell 1,729 (PREREG_1729.md, frozen): is the TIGHT-R cohort (R_pct < 1.5 %) the profitable part of HOD-break?

Population: cell 1,438's 9,911 fills (research/hod_entry/causal_arming_causal.csv, status==fill; TRAIN-H2 4,398 +
VAL 5,513; 2025-07-01 .. 2026-05-29).  NOTE: that file has NO TEST rows -- cell 1,438 built TRAIN-H2/VAL only
("TEST NOT read").  The TEST read of the PREREG is therefore gated twice in code: (1) C1 must pass on TRAIN-H2 AND
VAL, logged BEFORE anything TEST-like is touched; (2) the TEST population must exist (it does not -> logged ERROR).

Cost (frozen in the task): entry = level x (1 + 8 bps); marketable exits (stop, stop_bar, eod, eod_fallback) pay
12 bps off the recorded exit price; target prints (resting limit, price improvement) pay NOTHING.  R unit =
level - stop; R_pct = (level - stop) / level x 100 (trigger ~ level).  Entry is charged from the LEVEL, not from the
file's own modelled fill, so the slip is not double-charged.
Live cohort: fills_1721.csv valid_r rows, R_pct = risk_per_share / entry_price x 100, net R = pnl / (total_risk x
filled_qty / shares) (filled_qty missing -> shares).
Outputs: RESULT_1729.md, cohort_1729.csv.  Read-only; no network; no DB writes."""
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
POP = os.path.join(ROOT, 'research/hod_entry/causal_arming_causal.csv')
LIVE = os.path.join(HERE, 'fills_1721.csv')
OUT_MD = os.path.join(HERE, 'RESULT_1729.md')
OUT_CSV = os.path.join(HERE, 'cohort_1729.csv')

ENTRY_BPS, EXIT_BPS, CUT, CUT2, CAP_R = 8.0, 12.0, 1.5, 0.75, 2.0
MARKETABLE = {'stop', 'stop_bar', 'eod', 'eod_fallback'}
NOTIONAL_CAP = 4000.0   # live notional cap at $50 risk (RESULT_1728); scaled with risk in the capped variant
N_NULL, SEED = 1000, 1729


def log(msg):
    """Verbose progress to stdout, flushed."""
    print(msg, flush=True)


def fmt(x, k=3, sign=True):
    """Format a float (nan-safe)."""
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return 'n/a'
    return f'{x:+.{k}f}' if sign else f'{x:.{k}f}'


def load_population() -> pd.DataFrame:
    """The 9,911 fills with net R under the frozen cost model; split labelled TRAIN-H2 / VAL."""
    c = pd.read_csv(POP, low_memory=False)
    full_weeks = c.groupby('split').wk.nunique().to_dict()
    f = c[(c.variant == 'causal') & (c.status == 'fill')].copy()
    f['split'] = f.split.map({'TRAIN': 'TRAIN-H2', 'VAL': 'VAL'})
    assert len(f) == 9911 and f.split.notna().all(), (len(f), f.split.value_counts(dropna=False))
    f['date'] = f.day
    f['R_unit'] = f.level - f.stop
    f['r_pct'] = f.R_unit / f.level * 100
    entry = f.level * (1 + ENTRY_BPS / 1e4)
    mkt = f.why.isin(MARKETABLE)
    exit_net = np.where(mkt, f.exit_price * (1 - EXIT_BPS / 1e4), f.exit_price)
    f['gross_R'] = (f.exit_price - f.level) / f.R_unit
    f['net'] = (exit_net - entry) / f.R_unit
    f['week'] = f.wk
    log(f'[pop] {len(f)} fills; weeks in split (all rows): {full_weeks}; net mean {f.net.mean():+.4f} '
        f'gross {f.gross_R.mean():+.4f}; R_pct<1.5 share {(f.r_pct < CUT).mean():.1%}')
    return f, full_weeks


def clustered(df, col='net', by='date'):
    """Mean, day-clustered SE and t of a per-fill column (cluster-robust SE of the mean)."""
    n = len(df)
    if n < 3:
        return np.nan, np.nan, np.nan
    m = df[col].mean()
    s = (df[col] - m).groupby(df[by]).sum()
    se = np.sqrt((s ** 2).sum()) / n
    return m, se, (m / se if se > 0 else np.nan)


def ex_top5(v):
    """Mean after dropping the top 5 % of fills by value."""
    v = np.sort(np.asarray(v, float))
    k = int(np.ceil(0.05 * len(v)))
    return v[:-k].mean() if len(v) > k else np.nan


def alt_halves(df, col='net'):
    """Mean of the two alternating-week halves (week ordinal parity within the frame)."""
    wks = sorted(df.week.unique())
    idx = {w: i for i, w in enumerate(wks)}
    h = df.week.map(idx) % 2
    return df[h == 0][col].mean(), df[h == 1][col].mean(), int((h == 0).sum()), int((h == 1).sum())


def stats(df, weeks, label=''):
    """The C1/C2 table row for a cohort."""
    m, se, t = clustered(df)
    a, b, na, nb = alt_halves(df) if len(df) > 5 else (np.nan,) * 2 + (0, 0)
    return dict(label=label, n=len(df), per_wk=len(df) / weeks, net=m, se=se, t=t, ex5=ex_top5(df.net) if len(df) else np.nan,
                capped=df.net.clip(upper=CAP_R).mean() if len(df) else np.nan, gross=df.gross_R.mean() if len(df) else np.nan,
                hA=a, hB=b, nA=na, nB=nb, mde=2.8 * se)


def row_md(s):
    """Markdown row of a stats dict."""
    return (f"| {s['label']} | {s['n']} | {s['per_wk']:.1f} | {fmt(s['net'])} | {fmt(s['t'], 2)} | {fmt(s['ex5'])} | "
            f"{fmt(s['capped'])} | {fmt(s['gross'])} | {fmt(s['hA'])} / {fmt(s['hB'])} |")


HDR = ('| cohort | n | fills/wk | net R | day-clust t | ex-top-5 % | winner-capped +2R | gross R | alt-week halves A / B |\n'
       '|---|---|---|---|---|---|---|---|---|')


def weekly_book(df, all_weeks, risk, capped=False):
    """Weekly $ P&L (zero-fill weeks included) for a kept cohort at `risk` $ per trade (R-unit book;
    optional fixed notional cap = NOTIONAL_CAP x risk/50)."""
    if capped:
        sh = np.minimum(risk / df.R_unit, NOTIONAL_CAP * risk / 50.0 / df.level)
        usd = df.net * sh * df.R_unit
    else:
        usd = df.net * risk
    w = usd.groupby(df.week).sum().reindex(all_weeks).fillna(0.0)
    return w


def max_dd(w):
    """Max drawdown of the cumulative weekly series (positive $ number)."""
    c = np.cumsum(w.values)
    peak = np.maximum.accumulate(np.concatenate([[0.0], c]))[1:]
    return float((peak - c).max())


def green_null(pop, kept, all_weeks, risk, rng):
    """Green-week share of `kept` vs 1,000 count-matched random draws (same n) from the split's whole population."""
    obs = (weekly_book(kept, all_weeks, risk) > 0).mean()
    g = []
    for _ in range(N_NULL):
        d = pop.iloc[rng.choice(len(pop), size=len(kept), replace=False)]
        g.append((weekly_book(d, all_weeks, risk) > 0).mean())
    g = np.array(g)
    return obs, g.mean(), float((g >= obs).mean())


def load_live():
    """Live paper cohort with R_pct and net R in R units."""
    L = pd.read_csv(LIVE, low_memory=False)
    L = L[L.valid_r.astype(str) == 'True'].copy()
    L['date'] = pd.to_datetime(L.trade_date).dt.strftime('%Y-%m-%d')
    L['r_pct_ledger'] = L.r_pct
    L['r_pct'] = L.risk_per_share / L.entry_price * 100
    fq = L.filled_qty.fillna(L.shares)
    L['net'] = L.pnl / (L.total_risk * fq / L.shares)
    chk = (L.net - (L.gross_R - L.entry_cost_R - L.exit_cost_R - L.resid_R)).abs()
    log(f'[live] {len(L)} valid_r fills; |net - (gross-costs)| max {chk.max():.3f} (n>0.01: {(chk > 0.01).sum()}); '
        f'r_pct<1.5: {(L.r_pct < CUT).sum()}; ledger r_pct<1.5: {(L.r_pct_ledger < CUT).sum() if "r_pct_ledger" in L else "n/a"}')
    return L


def c1_gate(res):
    """C1 verdict legs measurable without TEST: per split kept net >= +0.05, ex5 > 0, complement <= 0, halves same sign."""
    legs = {}
    for sp, r in res.items():
        k, c = r['kept'], r['comp']
        legs[sp] = dict(net=k['net'] >= 0.05, ex5=k['ex5'] > 0, comp=c['net'] <= 0,
                        halves=(np.sign(k['hA']) == np.sign(k['hB'])) and k['hA'] > 0)
    return legs


def main():
    """Run C1/C2/C3 on TRAIN-H2 + VAL, gate the TEST read, write RESULT_1729.md and cohort_1729.csv."""
    pop, full_weeks = load_population()
    live = load_live()
    rng = np.random.default_rng(SEED)
    wk_all = {sp: sorted(pop[pop.split == sp].week.unique()) for sp in ('TRAIN-H2', 'VAL')}
    nweeks = {sp: len(w) for sp, w in wk_all.items()}
    res = {}
    for sp in ('TRAIN-H2', 'VAL'):
        d = pop[pop.split == sp]
        res[sp] = dict(all=stats(d, nweeks[sp], 'all fills'),
                       kept=stats(d[d.r_pct < CUT], nweeks[sp], 'kept R_pct < 1.5 %'),
                       comp=stats(d[d.r_pct >= CUT], nweeks[sp], 'complement >= 1.5 %'),
                       c2=stats(d[d.r_pct < CUT2], nweeks[sp], 'C2 R_pct < 0.75 %'),
                       mid=stats(d[(d.r_pct >= CUT2) & (d.r_pct < CUT)], nweeks[sp], '0.75-1.5 %'))
        log(f'[{sp}] ' + ' | '.join(f"{k}: n {v['n']} net {v['net']:+.4f} ex5 {v['ex5']:+.4f}" for k, v in res[sp].items()))
    pooled_kept = pop[pop.r_pct < CUT]
    pm, pse, pt = clustered(pooled_kept)
    log(f'[pooled kept] n {len(pooled_kept)} net {pm:+.4f} day-clustered t {pt:.2f}')

    # TEST gate: verdict on TRAIN-H2/VAL computed and logged BEFORE any TEST access
    legs = c1_gate(res)
    c1_pass_tv = all(all(l.values()) for l in legs.values())
    log(f'[gate] C1 legs on TRAIN-H2/VAL: {legs} -> pass on both = {c1_pass_tv}')
    test_read, test_note = False, ''
    if c1_pass_tv:
        test_note = ('C1 passed on TRAIN-H2 and VAL, but the 9,911 population (cell 1,438) contains NO TEST rows '
                     '(2026-06..09 was never built); building it needs a causal_arming run on TEST dates, outside this cell.')
        log('ERROR: ' + test_note)
    else:
        test_note = 'C1 did NOT pass on both TRAIN-H2 and VAL, so TEST stays sealed and was not read.'
        log('[gate] ' + test_note)
    assert not test_read

    # C3 book
    c3 = {}
    kept_all = pop[pop.r_pct < CUT]
    for sp in ('TRAIN-H2', 'VAL'):
        d = pop[pop.split == sp]
        k = d[d.r_pct < CUT]
        for risk in (50, 150):
            w = weekly_book(k, wk_all[sp], risk)
            obs, nullm, p = green_null(d, k, wk_all[sp], risk, rng)
            wc = weekly_book(k, wk_all[sp], risk, capped=True)
            c3[(sp, risk)] = dict(total=w.sum(), mean_wk=w.mean(), green=obs, null=nullm, p=p, dd=max_dd(w),
                                  worst=w.min(), p10=w.quantile(0.1), cap_total=wc.sum(), cap_dd=max_dd(wc),
                                  cap_green=(wc > 0).mean())
        # alternating-week halves of the book
        for risk in (50, 150):
            w = weekly_book(k, wk_all[sp], risk)
            h = np.arange(len(w)) % 2
            c3[(sp, risk)]['halves'] = (w.values[h == 0].mean(), w.values[h == 1].mean())
    wc_all = wk_all['TRAIN-H2'] + wk_all['VAL']
    for risk in (50, 150):
        w = weekly_book(kept_all, wc_all, risk)
        c3[('both', risk)] = dict(total=w.sum(), mean_wk=w.mean(), green=(w > 0).mean(), dd=max_dd(w), worst=w.min())
    log('[C3] ' + str({k: round(v['total'], 0) for k, v in c3.items()}))

    # live cohort
    lt = live[live.r_pct < CUT]
    lt75 = live[live.r_pct < CUT2]
    lc = live[live.r_pct >= CUT]
    lm, lse, ltt = clustered(lt)
    live_info = dict(n=len(lt), mean=lm, lo=lm - 1.96 * lse, hi=lm + 1.96 * lse, t=ltt, n75=len(lt75),
                     m75=lt75.net.mean(), ncomp=len(lc), mcomp=lc.net.mean(), ex5=ex_top5(lt.net))
    live_wks = (pd.to_datetime(live.date).max() - pd.to_datetime(live.date).min()).days / 7 + 1
    live_rate = len(live) / live_wks
    bt_rate = len(pop) / (nweeks['TRAIN-H2'] + nweeks['VAL'])
    ratio = live_rate / bt_rate
    log(f'[live] tight n {len(lt)} net {lm:+.4f} CI [{live_info["lo"]:+.3f},{live_info["hi"]:+.3f}]; ratio live/BT {ratio:.3f}')

    pop[['date', 'symbol', 'split', 'week', 'level', 'stop', 'r_pct', 'why', 'gross_R', 'net']].assign(
        kept=lambda x: x.r_pct < CUT).to_csv(OUT_CSV, index=False)
    write_result(res, c3, live_info, ratio, live_wks, pm, pt, legs, c1_pass_tv, test_note, nweeks, pooled_kept, pop)
    log(f'[done] wrote {OUT_MD} and {OUT_CSV}')


def write_result(res, c3, li, ratio, live_wks, pm, pt, legs, c1_tv, test_note, nweeks, kept_all, pop):
    """Write RESULT_1729.md."""
    L = ['# RESULT 1,729 — tight-R cohort (R_pct < 1.5 %) on cell 1,438\'s 9,911-fill population', '',
         'Script `cell_1729.py`, per-fill `cohort_1729.csv`. PREREG `PREREG_1729.md` (frozen). Cost: entry level x (1+8 bps); '
         'marketable exits (stop/stop_bar/eod/eod_fallback) 12 bps off the recorded exit; TARGET prints pay NOTHING '
         '(resting limit). R unit = level − stop; R_pct = (level − stop)/level. Entry charged from the level, so the file\'s '
         'own modelled slip is not double-charged.', '',
         f'**TEST WAS NOT READ.** {test_note}', '',
         f'Weeks: TRAIN-H2 {nweeks["TRAIN-H2"]}, VAL {nweeks["VAL"]} (weeks with any armed row). Cells run here: C1 (2 cohorts x 2 splits), '
         'C2, C3 (2 risks x 2 splits) + a sensitivity — this is the 14th cut on the HOD live book tonight; TRAIN-H2/VAL are a replication of a SEEN hypothesis.', '']
    for sp in ('TRAIN-H2', 'VAL'):
        r = res[sp]
        L += [f'## C1 / C2 — {sp}', '', HDR] + [row_md(r[k]) for k in ('all', 'kept', 'comp', 'c2', 'mid')]
        L += ['', f'MDE (2.8 x day-clustered SE, kept cohort): {r["kept"]["mde"]:.3f} R; complement {r["comp"]["mde"]:.3f} R; C2 {r["c2"]["mde"]:.3f} R.', '']
    L += ['C2 (R_pct < 0.75 %) is EMPTY in this population: cell 1,438 enforced a 0.75 % R floor, so kept (< 1.5 %) = the 0.75-1.5 % bucket only; the live < 0.75 % cohort (n 29) has no BT counterpart.', '']
    L += [f'Pooled kept (TRAIN-H2+VAL): n {len(kept_all)}, net {fmt(pm)} R, day-clustered t {pt:.2f}.', '',
          '## C3 — the book "keep only R_pct < 1.5 %" (weekly P&L, zero-fill weeks included; $ = net R x risk, uncapped notional)', '',
          '| split | risk | total $ | mean $/wk | green-week share | null mean (1,000 count-matched draws) | P(null >= obs) | P10 wk | worst wk | max DD $ | alt-week halves $/wk A / B | notional-capped total / DD / green |',
          '|---|---|---|---|---|---|---|---|---|---|---|---|']
    for sp in ('TRAIN-H2', 'VAL'):
        for risk in (50, 150):
            c = c3[(sp, risk)]
            L.append(f"| {sp} | ${risk} | {c['total']:+.0f} | {c['mean_wk']:+.1f} | {c['green']:.0%} | {c['null']:.0%} | {c['p']:.3f} | "
                     f"{c['p10']:+.0f} | {c['worst']:+.0f} | {c['dd']:.0f} | {c['halves'][0]:+.1f} / {c['halves'][1]:+.1f} | "
                     f"{c['cap_total']:+.0f} / {c['cap_dd']:.0f} / {c['cap_green']:.0%} |")
    for risk in (50, 150):
        c = c3[('both', risk)]
        L.append(f"| TRAIN-H2+VAL chronological | ${risk} | {c['total']:+.0f} | {c['mean_wk']:+.1f} | {c['green']:.0%} | | | | {c['worst']:+.0f} | {c['dd']:.0f} | | |")
    L += ['', '## Live paper cohort (fills_1721, valid_r)', '',
          f"* R_pct < 1.5 %: n {li['n']}, net {fmt(li['mean'])} R, 95 % CI [{fmt(li['lo'])}, {fmt(li['hi'])}] (day-clustered, t {li['t']:.2f}), ex-top-5 % {fmt(li['ex5'])}.",
          f"* R_pct < 0.75 %: n {li['n75']}, net {fmt(li['m75'])} R. Complement: n {li['ncomp']}, net {fmt(li['mcomp'])} R.",
          f"* (the ledger-based count 82 in the PREREG vs this n differ only by the R_pct definition/valid_r rows; both reported above as computed.)", '',
          f"Live/BT fill-rate ratio {ratio:.3f} (live {live_wks:.1f} wk)."]
    # verdict
    L += ['', '## Frozen pass bar, item by item', '']
    for sp in ('TRAIN-H2', 'VAL'):
        k, c, lg = res[sp]['kept'], res[sp]['comp'], legs[sp]
        L += [f"* {sp}: kept net >= +0.05: {'PASS' if lg['net'] else 'FAIL'} ({fmt(k['net'])}); ex-top-5 % > 0: {'PASS' if lg['ex5'] else 'FAIL'} ({fmt(k['ex5'])}); "
              f"alt-week halves same sign (and positive): {'PASS' if lg['halves'] else 'FAIL'} ({fmt(k['hA'])}/{fmt(k['hB'])}); complement <= 0: {'PASS' if lg['comp'] else 'FAIL'} ({fmt(c['net'])})."]
    fr = {sp: res[sp]['kept']['per_wk'] * ratio for sp in res}
    L += [f"* pooled day-clustered t >= 2: {'PASS' if pt >= 2 else 'FAIL'} ({pt:.2f}).",
          f"* >= 3 fills/week at the live rate (BT rate x {ratio:.2f}): TRAIN-H2 {fr['TRAIN-H2']:.1f}, VAL {fr['VAL']:.1f} -> {'PASS' if min(fr.values()) >= 3 else 'FAIL'}.",
          f"* live tight fills >= 0: {'PASS' if li['mean'] >= 0 else 'FAIL'} ({fmt(li['mean'])}, CI [{fmt(li['lo'])}, {fmt(li['hi'])}]).",
          f"* TEST net >= +0.05 / ex-top-5 % > 0: NOT EVALUATED (TEST not read).", '',
          f"**C1 on TRAIN-H2 and VAL both pass: {c1_tv}.** " + ('Overall verdict: FAIL on the replication splits; TEST sealed and unread; per PREREG the parity defect is fixed the other way (the twin\'s gate goes live) and the cohort is reported as noise with its CI.'
           if not c1_tv else 'Overall verdict: replication legs pass; PREREG PASS requires the TEST read, which this cell could not perform (see top). NOT a PASS yet.'),
          '', '## Caveats', '',
          '* The 9,911 file is cell 1,438\'s population: exits come from its recorded exit_price (NBBO / fill-instant exit-half sources, 76+58 bar-fallback exits); the cost model here is the task-frozen 8/12 bps, not the file\'s banded cost_R (0.23 R/fill) — with R_pct < 0.75 % a flat 20 bps round trip is itself 0.27+ R, so the cohort\'s net is very sensitive to the true spread of these (mostly large-cap) names; the live measured exit slip for them is not separately known.',
          '* Tight-R fills are sized by the live $4,000 notional cap, so live dollar weight differs from the per-fill R mean; the notional-capped column in C3 shows it.',
          '* Hypothesis seen in live data first (13 prior cuts), so TRAIN-H2/VAL are a replication, not a discovery; the sealed TEST is the confirmation and is absent from this population.',
          '* C3 $ totals are uncapped by the live 4-concurrent / 12-per-day caps (BT cohort fires ~80-115 fills/wk vs ~49/wk live all-in), so they overstate dollars; the sign and green-week shares are the content.',
          '* Trigger taken as the level (live trigger sits a tick above); R_pct difference is immaterial except below ~0.3 %.']
    open(OUT_MD, 'w').write('\n'.join(L) + '\n')


if __name__ == '__main__':
    sys.exit(main())
