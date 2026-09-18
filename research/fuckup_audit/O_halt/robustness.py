"""S1 step 6 — the two checks that decide whether the survivor is a real move or a quote artefact.

R1  open-to-open. The booked cell buys the OPEN of bar T and sells the CLOSE of bar T+5. If the edge
    is the bid-ask (a post-resume first print at the offer, a later print at the bid) it dies when
    both legs are bar OPENS.
R2  one-minute-later entry. If the edge lives only in the first minute after the resume — the minute
    an engine may not reliably reach — it dies when the fill is the open of bar T+1.
Both keep every other rule identical, including the 0.6% cap measured against the same `ref`.
"""
import numpy as np
import pandas as pd

import score_cells as S

HERE = '/home/ec2-user/onemil/research/fuckup_audit/O_halt'


def main():
    tr = pd.read_parquet(f'{HERE}/trades.parquet')
    for c in ('halt_ts', 'resume_ts', 'entry_t'):
        tr[c] = pd.to_datetime(tr[c], utc=True).dt.tz_convert('America/New_York')
    bars = S.Bars()
    cache = {}
    add = {'px_h5_open': [], 'fill_t1': [], 'px_h5_t1': []}
    for r in tr.itertuples():
        key = (r.symbol, r.day)
        if key not in cache:
            cache[key] = bars.get(r.symbol, r.day)[0]
        bd = cache[key]
        post = bd[bd.index >= r.entry_t]
        def op(k):
            w = post[post.index == r.entry_t + pd.Timedelta(minutes=k)]
            return float(w['o'].iloc[0]) if len(w) else np.nan
        def cl(k):
            w = post[post.index <= min(r.entry_t + pd.Timedelta(minutes=k),
                                       r.entry_t.normalize() + pd.Timedelta(hours=15, minutes=55))]
            return float(w['c'].iloc[-1]) if len(w) else np.nan
        add['px_h5_open'].append(op(5))
        add['fill_t1'].append(op(1))
        add['px_h5_t1'].append(cl(6))
    for k, v in add.items():
        tr[k] = v

    for nm, (side, rule) in {'A up-fade': ('up', 'fade'), 'B down-cont': ('down', 'continuation')}.items():
        ss = tr[tr['side'] == side].reset_index(drop=True)
        base = S.score(ss, rule, 'h5')
        d = 1.0 if (side == 'up') == (rule == 'continuation') else -1.0
        mod = ss['entry_t'].dt.hour * 60 + ss['entry_t'].dt.minute
        sb = np.array([S.SPREAD_BPS[S.band(p)][S.hourbucket(m)] for p, m in zip(ss['fill'], mod)])
        half = 0.5 * (sb / 100.0) / S.R_PCT
        for label, fill, px in (('R1 open->open ', ss['fill'], ss['px_h5_open']),
                                ('R2 +1min entry', ss['fill_t1'], ss['px_h5_t1'])):
            ok = base['ok'].values & fill.notna().values & px.notna().values
            if label.startswith('R2'):   # the cap is re-measured against the LATER fill
                ok = ok & np.where(d > 0, fill <= ss['ref'] * (1 + S.CAP), fill >= ss['ref'] * (1 - S.CAP)).astype(bool)
            raw = d * (px.values / fill.values - 1.0) * 100.0
            net = raw / S.R_PCT - 0.25 * half - 0.875 * half
            for split, a, b in S.SPLITS:
                sel = ok & (ss['day'] >= a).values & (ss['day'] <= b).values
                st = S.stats(net[sel])
                print(f'{nm:11s} {label} {split:5s} n={st["n"]:4d} mean {st["mean"]:+.4f} t {st["t"]:+.2f}')
        st = S.stats(base[base['ok']]['net_R'].values)
        print(f'{nm:11s} BASE (all)     n={st["n"]:4d} mean {st["mean"]:+.4f} t {st["t"]:+.2f}\n')


if __name__ == '__main__':
    main()
