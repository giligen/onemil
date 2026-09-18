#!/usr/bin/env python3
"""Multi-day DATA stage, step 5 — quarterly EPS facts (SEC XBRL) + the availability column.

Source: `companyconcept/us-gaap/EarningsPerShareDiluted` (fallback
`EarningsPerShareBasic`) for every CIK that has at least one Item-2.02 8-K.
Every fact carries its own `filed` date, which is what makes a point-in-time SUE
possible without a consensus vendor (Bernard–Thomas 1989 seasonal random walk).

What is kept
  * quarterly facts: period length 80–100 days (`fp` in Q1..Q4 or 10-Q/10-K);
  * derived Q4: where only the fiscal-year figure is reported, Q4 = FY − (Q1+Q2+Q3)
    of the SAME fiscal year, with the ANNUAL filing's `filed` date (that is when
    the number became public) and `source='derived_q4'`;
  * duplicates across amendments are kept — a restatement is a different fact with
    a later `filed`, and the availability logic must see both.

Nothing is computed from these facts here (SUE is a family-stage computation),
EXCEPT the availability column the audit needs: for every earnings event,
`eps_prior_filed` = the value of the latest quarterly fact FILED STRICTLY BEFORE
the event's acceptance time, plus `eps_prior_end`, `eps_prior_filed_date` and
`n_prior_qfacts` (distinct period-ends available then — SUE needs ≥ 9 for the
8 seasonal differences).

Resumable: one parquet per shard of CIKs under `edgar_eps/`.

    python3 edgar_eps.py
    python3 edgar_eps.py --finalize
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path('/home/ec2-user/onemil')
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from edgar_common import company_concept, shard_paths  # noqa: E402

HERE = Path(__file__).resolve().parent
EVENTS = HERE / 'earnings_events.parquet'
SHARDS = HERE / 'edgar_eps'
FACTS_OUT = HERE / 'eps_facts.parquet'
SHARD_SIZE = 200
TAGS = ('EarningsPerShareDiluted', 'EarningsPerShareBasic')
MIN_Q_DAYS, MAX_Q_DAYS = 80, 100
MIN_A_DAYS, MAX_A_DAYS = 350, 380


def _facts_for(cik: int) -> list[dict]:
    """All usable EPS facts for one CIK, quarterly + annual (annual only to derive Q4)."""
    for tag in TAGS:
        doc = company_concept(cik, tag)
        if not doc:
            continue
        units = doc.get('units', {})
        entries = []
        for unit_name, items in units.items():
            if 'shares' not in unit_name.lower():
                continue
            entries += items
        if not entries:
            continue
        rows = []
        for e in entries:
            start, end, filed = e.get('start'), e.get('end'), e.get('filed')
            if not (start and end and filed) or e.get('val') is None:
                continue
            days = (pd.Timestamp(end) - pd.Timestamp(start)).days
            if MIN_Q_DAYS <= days <= MAX_Q_DAYS:
                period = 'Q'
            elif MIN_A_DAYS <= days <= MAX_A_DAYS:
                period = 'A'
            else:
                continue
            rows.append({'cik': cik, 'tag': tag, 'period': period,
                         'start': start, 'end': end, 'days': days,
                         'val': float(e['val']), 'filed': filed,
                         'form': e.get('form', ''), 'fy': e.get('fy'),
                         'fp': e.get('fp', ''), 'accn': e.get('accn', '')})
        if rows:
            return rows
    return []


def fetch_shards(ciks: list[int]) -> None:
    """Pull EPS facts for every CIK, one checkpoint parquet per shard."""
    shards = [ciks[i:i + SHARD_SIZE] for i in range(0, len(ciks), SHARD_SIZE)]
    cols = ['cik', 'tag', 'period', 'start', 'end', 'days', 'val', 'filed',
            'form', 'fy', 'fp', 'accn']
    for si, shard in enumerate(shards):
        path = shard_paths(SHARDS, 'eps', si)
        if path.exists():
            continue
        rows = []
        for cik in shard:
            rows += _facts_for(cik)
        pd.DataFrame(rows, columns=cols).to_parquet(path, index=False)
        print(f'eps shard {si + 1}/{len(shards)}: {len(rows)} facts from {len(shard)} CIKs',
              flush=True)


def derive_q4(facts: pd.DataFrame) -> pd.DataFrame:
    """Q4 = FY − (Q1+Q2+Q3) for fiscal years where only the annual figure is reported."""
    ann = facts[facts['period'] == 'A']
    qs = facts[facts['period'] == 'Q']
    if ann.empty or qs.empty:
        return pd.DataFrame(columns=facts.columns)
    q_by_cik = {c: g.sort_values('end') for c, g in qs.groupby('cik')}
    out = []
    for cik, g in ann.groupby('cik'):
        qg = q_by_cik.get(cik)
        if qg is None:
            continue
        q_start = qg['start'].values
        q_end = qg['end'].values
        for _, a in g.iterrows():
            inside = qg[(q_start >= a['start']) & (q_end <= a['end'])]
            inside = inside.drop_duplicates('end', keep='last')
            if len(inside) != 3:
                continue
            if (inside['end'] == a['end']).any():          # Q4 already reported
                continue
            out.append({'cik': cik, 'tag': a['tag'], 'period': 'Q',
                        'start': inside['end'].max(), 'end': a['end'], 'days': 90,
                        'val': float(a['val']) - float(inside['val'].sum()),
                        'filed': a['filed'], 'form': a['form'], 'fy': a['fy'],
                        'fp': 'Q4', 'accn': a['accn']})
    df = pd.DataFrame(out, columns=list(facts.columns))
    return df


def attach_availability(events: pd.DataFrame, q: pd.DataFrame) -> pd.DataFrame:
    """Per event: the latest EPS fact filed strictly before the acceptance instant."""
    q = q.copy()
    q['filed_ts'] = pd.to_datetime(q['filed']).dt.tz_localize('America/New_York') + \
        pd.Timedelta(hours=23, minutes=59)   # conservative: a filing is public that day
    q = q.sort_values(['cik', 'filed_ts', 'end'])

    cols = {'eps_prior_filed': np.nan, 'eps_prior_end': None,
            'eps_prior_filed_date': None, 'n_prior_qfacts': 0}
    for c, v in cols.items():
        events[c] = v

    by_cik = {c: g for c, g in q.groupby('cik')}
    acc = pd.to_datetime(events['acceptance_utc'], utc=True).dt.tz_convert('America/New_York')
    events = events.assign(_acc=acc)
    parts = []
    for cik, eg in events.groupby('cik'):
        g = by_cik.get(cik)
        if g is None or g.empty:
            parts.append(eg)
            continue
        filed = g['filed_ts'].values
        ends = pd.to_datetime(g['end']).values
        vals = g['val'].values
        filed_d = g['filed'].values
        # running best = the fact with the largest period end seen so far
        best_idx = np.maximum.accumulate(np.arange(len(g)) *
                                         (ends == np.maximum.accumulate(ends)))
        # running count of distinct period ends
        seen_counts, seen = [], set()
        for e in ends:
            seen.add(e)
            seen_counts.append(len(seen))
        seen_counts = np.asarray(seen_counts)

        pos = np.searchsorted(filed, eg['_acc'].values, side='left') - 1
        ok = pos >= 0
        eg = eg.copy()
        if ok.any():
            bi = best_idx[pos[ok]]
            eg.loc[ok, 'eps_prior_filed'] = vals[bi]
            eg.loc[ok, 'eps_prior_end'] = pd.to_datetime(ends[bi]).strftime('%Y-%m-%d')
            eg.loc[ok, 'eps_prior_filed_date'] = filed_d[bi]
            eg.loc[ok, 'n_prior_qfacts'] = seen_counts[pos[ok]]
        parts.append(eg)
    out = pd.concat(parts).sort_index().drop(columns=['_acc'])
    return out


def finalize() -> None:
    facts = pd.concat([pd.read_parquet(p) for p in sorted(SHARDS.glob('eps_*.parquet'))],
                      ignore_index=True)
    print(f'raw EPS facts: {len(facts)} on {facts["cik"].nunique()} CIKs', flush=True)
    derived = derive_q4(facts)
    q = pd.concat([facts[facts['period'] == 'Q'], derived], ignore_index=True)
    q['source'] = np.where(q['fp'].eq('Q4') & q['days'].eq(90) & q['form'].str.startswith('10-K'),
                           'derived_q4', 'reported')
    q = q.drop_duplicates(['cik', 'end', 'filed', 'val']).reset_index(drop=True)
    q.to_parquet(FACTS_OUT, index=False)
    print(f'quarterly EPS facts: {len(q)} ({len(derived)} derived Q4) on '
          f'{q["cik"].nunique()} CIKs -> {FACTS_OUT}', flush=True)

    events = pd.read_parquet(EVENTS)
    events = attach_availability(events, q)
    events.to_parquet(EVENTS, index=False)
    cov = events['n_prior_qfacts']
    print(f'events with ≥1 prior EPS fact: {100 * (cov >= 1).mean():.1f}%', flush=True)
    print(f'events with ≥9 prior EPS facts (SUE-ready): {100 * (cov >= 9).mean():.1f}%',
          flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--finalize', action='store_true')
    args = ap.parse_args()
    ciks = sorted(pd.read_parquet(EVENTS, columns=['cik'])['cik'].unique().tolist())
    print(f'{len(ciks)} CIKs with at least one item-2.02 8-K', flush=True)
    if not args.finalize:
        fetch_shards(ciks)
    finalize()


if __name__ == '__main__':
    main()
