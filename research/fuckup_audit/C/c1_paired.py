#!/usr/bin/env python3
"""Stage C — the three PRE-REGISTERED paired comparisons (C/PREREG.md §7). These are DELTAS on matched signals, not
gate cells: the same signal is scored two ways and the difference is a paired statistic, which is far more powerful
than differencing two independent cell means.

  P1  H2   resting fill vs next-open fill, per family x outcome:
           (a) on the signals BOTH models fill  -> paired mean delta and its t
           (b) on the RESTING-ONLY signals (the bursts the next-open cap threw away) -> their own mean net R
           (c) the same two, with the resting side restricted to rest_queue_ok == 1 (the honest H2 claim)
  P2  H1   each stop variant vs the touch stop, per fill x family:  `2R close-stop` and `2R stop-1%` vs
           `2R close-fill`.  Both are in R units of THEIR OWN stop, so at a constant $100 of risk the $ P&L is
           100 x rr for either — the R difference IS the constant-dollar-risk difference (PLAN §3 H1).  The stop
           RATE is reported next to it, because H1's adoption rule requires it to fall.
  P3  H8   the lock exit (1.75R arm -> +0.5R stop, hold) vs hold-to-close, per fill x family.

Cost contract, population and column semantics are imported from C/score5c.py — ONE implementation, so a paired
delta can never drift from the cell table it explains.

Usage: ulimit -v 1800000; nice -n 10 python3 research/fuckup_audit/C/c1_paired.py
"""
import os, sys
import numpy as np, pandas as pd

ROOT = '/home/ec2-user/onemil'
sys.argv = ['c1_paired']                     # score5c parses sys.argv at import; give it its defaults
sys.path.insert(0, f'{ROOT}/research/fuckup_audit/C')
import score5c as S                                                              # noqa: E402

SRC = f'{ROOT}/research/fuckup_audit/C/pop_c.csv'
OUT = f'{ROOT}/research/fuckup_audit/C/c1_paired.md'
WINDOWS = [('all-day', 0), ('>=10:00', 600)]
NON_CLOSE = [k for k in S.FAM_KEYS if k.split(' ')[0] not in S.CLOSE_TRIGGERED]


def tstat(v):
    v = np.asarray(v, dtype=float)
    v = v[~np.isnan(v)]
    if len(v) < 5:
        return np.nan, np.nan, len(v)
    sd = v.std(ddof=1)
    return float(v.mean()), (float(v.mean() / (sd / np.sqrt(len(v)))) if sd > 0 else np.nan), len(v)


def load():
    cols = (S.BASE_COLS + ['rest_obtain'] + [f'{t}_{c}' for t in ('next', 'rest') for c in S.FILL_COLS])
    out = []
    for ch in pd.read_csv(SRC, usecols=cols, dtype={'symbol': str, 'day': str, 'fam': str, 'cfg': str},
                          keep_default_na=False, na_values=[''], chunksize=250_000, low_memory=True):
        ch = ch[ch.fam.isin(S.EXEMPT) | (ch.range_so_far_pct >= 5)]
        if len(ch):
            out.append(ch)
    d = pd.concat(out, ignore_index=True)
    d['key'] = d.fam + ' ' + d.cfg
    d['split'] = S.split_of(d.day.values)
    print(f'loaded {len(d):,} causal signal rows', flush=True)
    return d


def fill_frame(d, t, rp, min_m):
    """The population mask and the per-fill frame named the way score5c's net_r expects."""
    e, em = d[f'{t}_entry'], d[f'{t}_entry_m']
    ok = e.notna() & (e >= 5) & (em <= 841) & (em >= min_m) & (d[f'{t}_{rp}'] >= 1.0)
    x = d[[c for c in S.BASE_COLS]].copy()
    for c in S.FILL_COLS:
        x[c] = d[f'{t}_{c}']
    x['split'] = d.split
    x['key'] = d.key
    return ok.fillna(False).values, x


def main():
    d = load()
    L = ['# Stage C — P1/P2/P3: the pre-registered paired comparisons', '',
         'Deltas on MATCHED signals (signal level, before the 12/4 book). `n` is the number of matched signals.', '']

    # ---------------- P1 : resting vs next-open ----------------
    L += ['## P1 — H2: the resting stop-limit fill vs the engine\'s next-open fill', '']
    rows = []
    for wname, mm in WINDOWS:
        for oname, (rr, why, xm, rp) in S.OUTCOMES.items():
            okn, xn = fill_frame(d, 'next', rp, mm)
            okr, xr = fill_frame(d, 'rest', rp, mm)
            netn = S.net_r(xn, rr, why, rp, 'next')
            netr = S.net_r(xr, rr, why, rp, 'rest')
            grn, grr = xn[rr], xr[rr]          # GROSS, to separate the fill from the cost convention
            for key in NON_CLOSE + ['ALL (non-close-triggered)']:
                km = (d.key == key).values if key != 'ALL (non-close-triggered)' else d.key.isin(NON_CLOSE).values
                for sp in ('TRAIN', 'VAL'):
                    sm = (d.split == sp).values & km
                    both = sm & okn & okr
                    ronly = sm & okr & ~okn
                    nonly = sm & okn & ~okr
                    qok = (d.rest_queue_ok == 1).values
                    m_g, t_g, _ = tstat((grr - grn)[both])
                    m_gr, _, _ = tstat(grr[ronly])
                    m_d, t_d, n_d = tstat((netr - netn)[both])
                    m_dq, t_dq, n_dq = tstat((netr - netn)[both & qok])
                    m_r, t_r, n_r = tstat(netr[ronly])
                    m_rq, t_rq, n_rq = tstat(netr[ronly & qok])
                    m_n, _, n_n = tstat(netn[nonly])
                    rows.append(dict(window=wname, outcome=oname, key=key, split=sp,
                                     n_both=n_d, paired_d=round(m_d, 4) if m_d == m_d else np.nan,
                                     t_d=round(t_d, 2) if t_d == t_d else np.nan,
                                     paired_d_GROSS=round(m_g, 4) if m_g == m_g else np.nan,
                                     t_d_GROSS=round(t_g, 2) if t_g == t_g else np.nan,
                                     restonly_GROSS=round(m_gr, 4) if m_gr == m_gr else np.nan,
                                     n_both_q=n_dq, paired_d_q=round(m_dq, 4) if m_dq == m_dq else np.nan,
                                     t_dq=round(t_dq, 2) if t_dq == t_dq else np.nan,
                                     n_restonly=n_r, restonly_meanR=round(m_r, 4) if m_r == m_r else np.nan,
                                     t_restonly=round(t_r, 2) if t_r == t_r else np.nan,
                                     n_restonly_q=n_rq, restonly_q_meanR=round(m_rq, 4) if m_rq == m_rq else np.nan,
                                     n_nextonly=n_n, nextonly_meanR=round(m_n, 4) if m_n == m_n else np.nan))
    P1 = pd.DataFrame(rows)
    P1.to_csv(OUT.replace('.md', '_P1.csv'), index=False)
    for wname, _ in WINDOWS:
        L += [f'### {wname} — ALL non-close-triggered families pooled, per outcome',
              P1[(P1.window == wname) & (P1.key == 'ALL (non-close-triggered)')]
              .drop(columns=['window', 'key']).to_string(index=False), '']
    L += ['### per family, PRIMARY window >=10:00, outcome `2R close-fill` and `hold-to-close`',
          P1[(P1.window == '>=10:00') & (P1.outcome.isin(['2R close-fill', 'hold-to-close']))
             & (P1.key != 'ALL (non-close-triggered)')].drop(columns=['window']).to_string(index=False), '',
          'Queue check (`rest_queue_ok`, signal-bar volume >= 5 x the shares $100 of risk buys) — share of resting '
          'fills that pass it, by family:', '']
    qs = []
    for key in NON_CLOSE:
        ok, _ = fill_frame(d, 'rest', 'r_pct', 0)
        m = (d.key == key).values & ok
        qs.append(dict(key=key, n=int(m.sum()),
                       queue_ok=round(float((d.rest_queue_ok.values[m] == 1).mean()), 3) if m.sum() else np.nan,
                       obtainable=round(float((d.rest_obtain.values[m] == 1).mean()), 3)
                       if 'rest_obtain' in d.columns and m.sum() else np.nan))
    L += [pd.DataFrame(qs).to_string(index=False), '']

    # ---------------- P2 / P3 : stop variants and the lock ----------------
    L += ['## P2 — H1: stop variants vs the touch stop (`2R close-fill`), paired per signal',
          '## P3 — H8: the lock exit vs hold-to-close, paired per signal', '',
          'All deltas are in net R **of each variant\'s own R**, which at a constant $100 of risk is the $ delta / 100.',
          '']
    rows = []
    for wname, mm in WINDOWS:
        for fill in ('next', 'rest'):
            ok, x = fill_frame(d, fill, 'r_pct', mm)          # the touch stop's population is the paired base
            base = S.net_r(x, 'rr_2r', 'why_2r', 'r_pct', fill)
            hold = S.net_r(x, 'rr_hold', 'why_hold', 'r_pct', fill)
            cs = S.net_r(x, 'rr_2r_closestop', 'why_2r_closestop', 'r_pct', fill)
            m1 = S.net_r(x, 'rr_2r_stopm1', 'why_2r_stopm1', 'r_pct_m1', fill)
            lk = S.net_r(x, 'rr_lock', 'why_lock', 'r_pct', fill)
            keys = NON_CLOSE if fill == 'rest' else S.FAM_KEYS
            for key in keys + ['ALL']:
                km = (d.key == key).values if key != 'ALL' else d.key.isin(keys).values
                for sp in ('TRAIN', 'VAL'):
                    sm = ok & km & (d.split == sp).values
                    if sm.sum() < 40:
                        continue
                    r = dict(window=wname, fill=fill, key=key, split=sp, n=int(sm.sum()),
                             base_meanR=round(float(base[sm].mean()), 4),
                             base_GROSS=round(float(x.rr_2r[sm].mean()), 4),
                             base_stopP=round(float((x.why_2r[sm] == 'stop').mean() * 100), 1),
                             d_stopm1_GROSS=round(float((x.rr_2r_stopm1 - x.rr_2r)[sm].mean()), 4),
                             d_lock_hold_GROSS=round(float((x.rr_lock - x.rr_hold)[sm].mean()), 4))
                    for nm, v, wc in (('closestop', cs, 'why_2r_closestop'), ('stopm1', m1, 'why_2r_stopm1'),
                                      ('lock_vs_hold', lk - hold + base - base, None)):
                        if nm == 'lock_vs_hold':
                            md, td, _ = tstat((lk - hold)[sm])
                            r |= {'d_lock_vs_hold': round(md, 4), 't_lock': round(td, 2),
                                  'lock_stopP': round(float(x.why_lock[sm].isin(['stop', 'lock']).mean() * 100), 1),
                                  'hold_stopP': round(float((x.why_hold[sm] == 'stop').mean() * 100), 1)}
                        else:
                            md, td, _ = tstat((v - base)[sm])
                            r |= {f'd_{nm}': round(md, 4), f't_{nm}': round(td, 2),
                                  f'{nm}_stopP': round(float((x[wc][sm] == 'stop').mean() * 100), 1)}
                    rows.append(r)
    P2 = pd.DataFrame(rows)
    P2.to_csv(OUT.replace('.md', '_P2.csv'), index=False)
    for wname, _ in WINDOWS:
        L += [f'### {wname} — pooled (`ALL`) and per family, next-open fill',
              P2[(P2.window == wname) & (P2.fill == 'next')].drop(columns=['window', 'fill']).to_string(index=False),
              '', f'### {wname} — resting fill',
              P2[(P2.window == wname) & (P2.fill == 'rest')].drop(columns=['window', 'fill']).to_string(index=False),
              '']
    open(OUT, 'w').write('\n'.join(L))
    print('wrote', OUT, flush=True)


if __name__ == '__main__':
    main()
