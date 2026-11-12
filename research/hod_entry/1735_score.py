"""Cell 1,735 scorer (PREREG_1735.md): no-stop / soft-stop / dead-trade variants of the HOD-break long.

End to end: reproduce B0 from the cached 1-minute paths (exit-lab fill physics), walk five pre-committed
variants on TRAIN + VAL signals only (TEST is never loaded into the walk), write 1735_per_signal.csv and
1735_summary.csv, print every table.  Read-only on the inputs; no DB; no network.

Run:  nice -n 19 python3 research/hod_entry/1735_score.py
"""
import math
import numpy as np
import pandas as pd

LAB = 'research/hod_exit_lab/'
OUT = 'research/hod_entry/'
EOD_M = 955          # 15:55 ET, minutes since midnight
SLIP_BP = 0.0002     # 2 bp per side
RNG_SEED = 1735
N_SHUFFLE = 300
VARIANTS = ['v1', 'v2', 'v3', 'v4', 'v5']
SIZE = {'v1': 0.5, 'v2': 0.5, 'v3': 1.0, 'v4': 1.0, 'v5': 1.0}
LABEL = {'v1': 'V1 no-stop half', 'v2': 'V2 no-stop+target half', 'v3': 'V3 soft stop 2R close',
         'v4': 'V4 dead-trade 20m', 'v5': 'V5 dead-trade 60m'}


def log(msg):
    """Flushed progress line."""
    print(msg, flush=True)


def first_true(mask):
    """Index of the first True in a boolean array, or -1."""
    return int(mask.argmax()) if mask.any() else -1


def cost_R(entry, exit_px, half_spread, R):
    """Baseline-R cost at size 1: half-spread both legs + 2 bp per side (exit-lab convention)."""
    return 2.0 * half_spread / R + SLIP_BP * (entry + exit_px) / R


def walk_signal(r, a):
    """Walk one signal.  `a` = arrays (m,o,h,l,c) of bars strictly after entry_m up to the 15:55 bar.

    Returns {variant or 'b0': (exit_px, reason)}.  Index e = first bar at/after 15:55 (the EOD open).
    """
    m, o, h, l, c = a
    n = len(m)
    e = first_true(m >= EOD_M)
    eod_missing = e < 0
    if eod_missing:
        e = n                      # defensive fallback (no 15:55 bar): last close, as the exit lab does
    eod_px = float(c[-1]) if eod_missing else float(o[e])
    pre = slice(0, e)
    i_stop = first_true(l[pre] <= r.stop)
    i_tgt = first_true(h[pre] >= r.target)

    def nxt_open(i):
        """Open of the bar after bar i (the EOD bar at the latest)."""
        return float(o[i + 1]) if i + 1 < n else float(c[-1])

    # B0 (stop first on ties, gap-through at the open, target at the target)
    if i_stop >= 0 and (i_tgt < 0 or i_stop <= i_tgt):
        px = float(o[i_stop]) if o[i_stop] <= r.stop else float(r.stop)
        b0 = (px, 'stop'); b0_i = i_stop
    elif i_tgt >= 0:
        b0 = (float(r.target), 'target'); b0_i = i_tgt
    else:
        b0 = (eod_px, 'eod'); b0_i = e
    out = {'b0': b0}
    out['v1'] = (eod_px, 'eod')
    out['v2'] = (float(r.target), 'target') if i_tgt >= 0 else (eod_px, 'eod')
    # V3 soft stop: first CLOSE <= entry - 2R -> next bar open; target kept; same-bar tie = soft stop
    i_soft = first_true(c[pre] <= r.entry - 2.0 * r.R)
    if i_soft >= 0 and (i_tgt < 0 or i_tgt >= i_soft):
        out['v3'] = (nxt_open(i_soft), 'softstop')
    elif i_tgt >= 0:
        out['v3'] = (float(r.target), 'target')
    else:
        out['v3'] = (eod_px, 'eod')
    # V4/V5 dead-trade time stop on top of B0
    for name, k in (('v4', 20), ('v5', 60)):
        j = int(np.searchsorted(m, r.entry_m + k, side='right')) - 1   # last bar at/before entry+k
        if 0 <= j < e and j < b0_i and c[j] < r.entry + 0.3 * r.R:
            out[name] = (nxt_open(j), 'timestop')
        else:
            out[name] = b0
    return out


def build_per_signal():
    """B0 reproduction + five variants per TRAIN/VAL signal -> DataFrame in baseline-R."""
    sig = pd.read_parquet(LAB + 'signals.parquet')
    sig = sig[sig.split.isin(['TRAIN', 'VAL'])].reset_index(drop=True)      # TEST never walked
    log(f'[load] signals TRAIN/VAL: {len(sig)} (TEST excluded)')
    paths = pd.read_parquet(LAB + 'paths.parquet', columns=['day', 'symbol', 'm', 'o', 'h', 'l', 'c'])
    days = set(sig.day)
    paths = paths[paths.day.isin(days)].sort_values(['day', 'symbol', 'm']).reset_index(drop=True)
    grp = paths.groupby(['day', 'symbol']).indices
    marr = {k: None for k in ()}
    M = paths.m.to_numpy(); O = paths.o.to_numpy(); H = paths.h.to_numpy()
    L = paths.l.to_numpy(); C = paths.c.to_numpy()
    rows, nopath = [], 0
    for i, r in enumerate(sig.itertuples()):
        idx = grp.get((r.day, r.symbol))
        if idx is None:
            nopath += 1; continue
        sel = idx[M[idx] > r.entry_m]
        if len(sel) == 0:
            nopath += 1; continue
        a = (M[sel], O[sel], H[sel], L[sel], C[sel])
        res = walk_signal(r, a)
        row = dict(day=r.day, symbol=r.symbol, split=r.split, half=r.half, wk=r.wk, entry_m=r.entry_m,
                   R=r.R)
        for name, (px, why) in res.items():
            raw = (px - r.entry) / r.R
            s = SIZE.get(name, 1.0)
            cst = cost_R(r.entry, px, r.half_spread, r.R) if pd.notna(r.half_spread) else np.nan
            row[f'{name}_raw'] = s * raw
            row[f'{name}_cost'] = s * cst
            row[f'{name}_net_R'] = s * (raw - cst)
            row[f'{name}_why'] = why
        rows.append(row)
        if i % 3000 == 0:
            log(f'[walk] {i}/{len(sig)}')
    log(f'[walk] done rows={len(rows)} nopath={nopath}')
    df = pd.DataFrame(rows)
    return df


def reproduce_b0(df):
    """Compare my B0 against b0_trades.csv per signal; returns dict of deltas."""
    b = pd.read_csv(LAB + 'b0_trades.csv', usecols=['day', 'symbol', 'net_R', 'split'])
    m = df.merge(b, on=['day', 'symbol', 'split'], how='left', validate='one_to_one')
    ok = m.net_R.notna() & m.b0_net_R.notna()
    d = (m.b0_net_R - m.net_R)[ok]
    out = {}
    for sp in ('TRAIN', 'VAL'):
        k = ok & (m.split == sp)
        out[sp] = (float(m.b0_net_R[k].mean()), float(m.net_R[k].mean()))
    out['rows'] = int(ok.sum()); out['maxabs'] = float(d.abs().max()); out['nbad'] = int((d.abs() > 1e-6).sum())
    return out


def cluster_t(x, days):
    """Mean and day-clustered t of x (cluster = day)."""
    x = np.asarray(x, float); n = len(x)
    if n < 3:
        return float('nan'), float('nan')
    mu = x.mean()
    s = pd.Series(x - mu).groupby(np.asarray(days)).sum()
    se = math.sqrt((s ** 2).sum()) / n
    return mu, (mu / se if se > 0 else float('nan'))


def iid_t(x):
    """iid t of the mean."""
    x = np.asarray(x, float); n = len(x)
    sd = x.std(ddof=1)
    return x.mean() / (sd / math.sqrt(n)) if sd > 0 else float('nan')


def ex_top(x, frac=0.05):
    """Mean after dropping the `frac` largest values."""
    x = np.sort(np.asarray(x, float))
    k = int(math.ceil(frac * len(x)))
    return float(x[:len(x) - k].mean())


def placebo_pct(d, days, rng):
    """Percentile of the observed mean dR among N_SHUFFLE day-level sign-flip randomisations.

    The day label is shuffled in the sense of re-assigning each day's block of paired dR a random sign
    (null: the variant is exchangeable with B0 within a day); the pairing itself is kept.
    """
    s = pd.Series(np.asarray(d, float)).groupby(np.asarray(days)).agg(['sum', 'count'])
    tot = s['sum'].to_numpy(); n = s['count'].sum()
    obs = tot.sum() / n
    flips = rng.choice([-1.0, 1.0], size=(N_SHUFFLE, len(tot)))
    null = (flips * tot).sum(axis=1) / n
    return 100.0 * float((null < obs).mean())


def own_book(x, sub, why):
    """Own-book metrics of the variant's net R series (sub = df slice sorted by day, entry_m)."""
    mu, t = cluster_t(x, sub.day)
    wk = pd.Series(np.asarray(x)).groupby(sub.wk.to_numpy()).sum()
    cum = np.cumsum(np.asarray(x)); dd = float((np.maximum.accumulate(cum) - cum).max())
    return dict(own_mean=mu, own_t=t, own_extop5=ex_top(x), stop_share=float(np.isin(why, ['stop', 'softstop']).mean()),
                green_wk=float((wk > 0).mean()), wk_p10=float(wk.quantile(0.10)), maxdd_R=dd, weeks=len(wk))


def score(df):
    """Summary table: variant x read."""
    rng = np.random.default_rng(RNG_SEED)
    df = df.dropna(subset=['b0_net_R']).sort_values(['day', 'entry_m', 'symbol']).reset_index(drop=True)
    reads = {'TRAIN_H1': (df.split == 'TRAIN') & (df.half == 'H1'), 'TRAIN_H2': (df.split == 'TRAIN') & (df.half == 'H2'),
             'TRAIN': df.split == 'TRAIN', 'VAL': df.split == 'VAL'}
    recs = []
    for v in VARIANTS:
        df[f'{v}_ok'] = df[f'{v}_net_R'].notna()
        for rd, mask in reads.items():
            sub = df[mask & df[f'{v}_ok']]
            x = sub[f'{v}_net_R'].to_numpy(); b = sub.b0_net_R.to_numpy(); d = x - b
            mu, t = cluster_t(d, sub.day)
            dcap = np.minimum(x, 3) - np.minimum(b, 3)
            rec = dict(variant=v, read=rd, n=len(sub), dR=mu, dR_day_t=t, dR_iid_t=iid_t(d),
                       dR_extop5=ex_top(d), dR_wincap=float(dcap.mean()), mean_cost_R=float(sub[f'{v}_cost'].mean()),
                       b0_mean=float(b.mean()), placebo_pct=placebo_pct(d, sub.day, rng) if rd == 'VAL' else np.nan)
            rec.update(own_book(x, sub, sub[f'{v}_why'].to_numpy()))
            recs.append(rec)
    return pd.DataFrame(recs)


def verdict(sm):
    """PASS/FAIL per variant against the PREREG bar, with the failing clauses listed."""
    out = {}
    for v in VARIANTS:
        g = sm[sm.variant == v].set_index('read')
        tr, va, h1, h2 = g.loc['TRAIN'], g.loc['VAL'], g.loc['TRAIN_H1'], g.loc['TRAIN_H2']
        clauses = {
            'dR>=.05&t>=2 TRAIN': tr.dR >= 0.05 and tr.dR_day_t >= 2,
            'dR>=.05&t>=2 VAL': va.dR >= 0.05 and va.dR_day_t >= 2,
            'extop5>=0 both': tr.dR_extop5 >= 0 and va.dR_extop5 >= 0,
            'halves same sign': np.sign(h1.dR) == np.sign(h2.dR),
            'placebo>=95 VAL': va.placebo_pct >= 95,
            'own VAL mean>=.05&t>=2': va.own_mean >= 0.05 and va.own_t >= 2}
        out[v] = (all(clauses.values()), [k for k, ok in clauses.items() if not ok])
    return out


def main():
    """Entry point."""
    df = build_per_signal()
    rep = reproduce_b0(df)
    log(f'[B0 repro] rows matched {rep["rows"]}; TRAIN mine {rep["TRAIN"][0]:+.4f} vs b0_trades {rep["TRAIN"][1]:+.4f} '
        f'(delta {rep["TRAIN"][0]-rep["TRAIN"][1]:+.5f}); VAL mine {rep["VAL"][0]:+.4f} vs {rep["VAL"][1]:+.4f}; '
        f'max|row diff| {rep["maxabs"]:.2e}; rows >1e-6: {rep["nbad"]}')
    keep = ['day', 'symbol', 'split', 'half', 'b0_net_R', 'b0_why'] + \
           [c for v in VARIANTS for c in (f'{v}_net_R', f'{v}_why')]
    df[keep].to_csv(OUT + '1735_per_signal.csv', index=False)
    sm = score(df)
    sm.to_csv(OUT + '1735_summary.csv', index=False)
    pd.set_option('display.width', 250); pd.set_option('display.max_columns', 40)
    log(sm.round(3).to_string())
    for v, (ok, bad) in verdict(sm).items():
        log(f'{LABEL[v]}: {"PASS" if ok else "FAIL"}  failing clauses: {bad}')
    log(f'[reasons] ' + '; '.join(f'{v}: ' + str(df[f"{v}_why"].value_counts().to_dict()) for v in VARIANTS))


if __name__ == '__main__':
    main()
